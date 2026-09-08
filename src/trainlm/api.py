"""Public Hugging Face-like TrainLM training surface.

The classes in this module deliberately hide backend construction and the
training engine's internal lifecycle. TPU worker/coordinator integration can
be added behind the same API without changing user code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, IterableDataset

from trainlm.config import (
    CheckpointConfig,
    DatasetConfig,
    LoggingConfig,
    LossConfig,
    ModelSourceConfig,
    MonitoringConfig,
    OptimizationConfig,
    OptimizerConfig,
    ParallelismConfig,
    RuntimeConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlm.optimization import create_optimizer
from trainlm.runtime import Runtime
from trainlm.tasks import CausalLMTask
from trainlm.training import Trainer as EngineTrainer
from trainlm.training import TrainerCallback, create_scheduler

if TYPE_CHECKING:
    from trainlm.model import LoadedCausalLM


@dataclass(frozen=True, slots=True)
class TrainLMTrainingArguments:
    """Small, HF-recognizable training argument surface.

    Advanced TrainLM configuration remains available through ``TrainConfig``
    and the lower-level engine. Defaults favor a portable first run; backend
    implementations may select more efficient values automatically later.
    """

    output_dir: str | Path = Path("runs")
    max_steps: int | None = 1000
    max_tokens: int | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    sequence_length: int = 2048
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    lr_scheduler_type: Literal["constant", "linear", "cosine"] = "cosine"
    warmup_steps: int = 0
    bf16: bool = False
    fp16: bool = False
    accelerator: Literal["auto", "cpu", "cuda", "tpu"] = "auto"
    logging_steps: int = 10
    save_steps: int | None = None
    eval_steps: int | None = None
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    seed: int = 42
    report_to: str | tuple[str, ...] = "none"

    def __post_init__(self) -> None:
        if self.max_steps is None and self.max_tokens is None:
            raise ValueError("Set max_steps or max_tokens.")
        for name in (
            "max_steps",
            "max_tokens",
            "save_steps",
            "eval_steps",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be positive when configured.")
        for name in (
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "sequence_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive.")
        if isinstance(self.dataloader_num_workers, bool) or self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative.")
        if not isinstance(self.dataloader_pin_memory, bool):
            raise ValueError("dataloader_pin_memory must be boolean.")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled.")
        if self.accelerator not in {"auto", "cpu", "cuda", "tpu"}:
            raise ValueError(f"Unsupported accelerator: {self.accelerator}")
        if self.lr_scheduler_type not in {"constant", "linear", "cosine"}:
            raise ValueError(f"Unsupported scheduler: {self.lr_scheduler_type}")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be non-negative.")


def _default_collator(features: Sequence[Any]) -> dict[str, torch.Tensor]:
    if not features:
        raise ValueError("Cannot collate an empty batch.")
    if not all(isinstance(feature, Mapping) for feature in features):
        raise TypeError("TrainLM datasets must yield mappings.")
    keys = tuple(features[0])
    if any(tuple(feature) != keys for feature in features):
        raise ValueError("All features in a batch must expose the same fields.")
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [feature[key] for feature in features]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = torch.as_tensor(values)
    return batch


class TrainLMTrainer:
    """HF-like facade over TrainLM's backend-neutral trainer engine."""

    def __init__(
        self,
        *,
        model: nn.Module | str | Path | ModelSourceConfig,
        args: TrainLMTrainingArguments | None = None,
        train_dataset: Dataset | DataLoader | Any,
        eval_dataset: Dataset | DataLoader | Any | None = None,
        tokenizer: Any | None = None,
        processing_class: Any | None = None,
        data_collator: Callable[[Sequence[Any]], Mapping[str, Any]] | None = None,
        callbacks: Sequence[TrainerCallback] | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.args = args or TrainLMTrainingArguments()
        if tokenizer is not None and processing_class is not None:
            raise ValueError("Set either tokenizer or processing_class, not both.")
        self.processing_class = processing_class if processing_class is not None else tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or _default_collator
        self.callbacks = tuple(callbacks or ())
        self._model_source: ModelSourceConfig | None = None
        self.loaded: LoadedCausalLM | None = None
        self._last_metrics: dict[str, Any] = {}
        self._tpu_coordinator = None
        if self.args.accelerator == "tpu":
            if any(value is not None for value in (optimizer, scheduler, runtime)):
                raise ValueError(
                    "TPU execution constructs runtime, optimizer, and scheduler inside "
                    "each worker; do not pass local instances."
                )
            self._model_source = self._coerce_model_source(model)
            if self.args.max_steps is None:
                raise NotImplementedError(
                    "The TPU coordinator currently requires max_steps; max_tokens-only "
                    "execution will be added with lifecycle parity."
                )
            if self.args.max_tokens is not None:
                raise NotImplementedError(
                    "TPU max_tokens stopping will be added with lifecycle parity."
                )
            if self.args.fp16:
                raise ValueError("TPU execution supports fp32 or bf16, not fp16.")
            if self.args.save_steps is not None or self.args.eval_steps is not None:
                raise NotImplementedError(
                    "TPU save_steps and eval_steps will be added with lifecycle parity."
                )
            from trainlm._tpu_coordinator import _TPUCoordinator

            self._tpu_coordinator = _TPUCoordinator()
            self.model = None
            self.runtime = None
            self.optimizer = None
            self.scheduler = None
            self.engine = None
            return
        self.model = self._resolve_model(model)
        self.runtime = runtime or self._make_runtime()
        self.optimizer = optimizer or self._make_optimizer()
        self.scheduler = scheduler or self._make_scheduler()
        self.engine = EngineTrainer(
            config=self._make_config(),
            model=self.model,
            runtime=self.runtime,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            task=CausalLMTask(loss_implementation="causal_lm"),
            train_dataloader=self._make_loader(train_dataset, self.args.per_device_train_batch_size),
            eval_dataloader=(
                self._make_loader(eval_dataset, self.args.per_device_eval_batch_size)
                if eval_dataset is not None
                else None
            ),
            callbacks=self.callbacks,
        )

    def _resolve_model(self, model: nn.Module | str | Path | ModelSourceConfig) -> nn.Module:
        if isinstance(model, nn.Module):
            return model
        model = self._coerce_model_source(model)
        from trainlm.model import load_huggingface_causal_lm

        self._model_source = model
        self.loaded = load_huggingface_causal_lm(model)
        return self.loaded.model

    @staticmethod
    def _coerce_model_source(
        model: nn.Module | str | Path | ModelSourceConfig,
    ) -> ModelSourceConfig:
        if isinstance(model, nn.Module):
            raise TypeError(
                "TPU execution requires a model ID/path or ModelSourceConfig so each "
                "worker can construct its own model."
            )
        if isinstance(model, (str, Path)):
            return ModelSourceConfig(
                provider="huggingface",
                initialization="pretrained",
                name_or_path=str(model),
            )
        if not isinstance(model, ModelSourceConfig):
            raise TypeError("model must be a torch module, model ID/path, or ModelSourceConfig.")
        if model.provider != "huggingface":
            raise ValueError("The public facade currently resolves Hugging Face models only.")
        return model

    def _make_runtime(self):
        accelerator = self.args.accelerator
        if accelerator == "tpu":
            raise RuntimeError(
                "TPU coordinator integration is private and is being wired behind "
                "TrainLMTrainer; use accelerator='auto' for local execution."
            )
        if accelerator == "auto":
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        if accelerator == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("accelerator='cuda' requested but CUDA is unavailable.")
        precision = "bf16" if self.args.bf16 else "fp16" if self.args.fp16 else "fp32"
        return Runtime(device=accelerator, precision=precision)

    def _make_loader(self, source: Dataset | DataLoader | Any, batch_size: int) -> DataLoader:
        if isinstance(source, DataLoader):
            return source
        if not isinstance(source, (Dataset, IterableDataset)) and not (
            hasattr(source, "__getitem__") or hasattr(source, "__iter__")
        ):
            raise TypeError(
                "Datasets must be DataLoader, torch Dataset, IterableDataset, "
                "or a dataset-compatible object."
            )
        is_iterable = isinstance(source, IterableDataset) or (
            hasattr(source, "__iter__") and not hasattr(source, "__getitem__")
        )
        return DataLoader(
            source,
            batch_size=batch_size,
            shuffle=not is_iterable,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=self.data_collator,
        )

    def _make_optimizer(self) -> Optimizer:
        return create_optimizer(
            self.model.parameters(),
            OptimizerConfig(
                learning_rate=self.args.learning_rate,
                betas=self.args.betas,
                eps=self.args.eps,
                weight_decay=self.args.weight_decay,
                fused=False,
            ),
        )

    def _make_scheduler(self) -> LRScheduler:
        horizon = self.args.max_steps or 1
        return create_scheduler(
            self.optimizer,
            SchedulerConfig(
                name=self.args.lr_scheduler_type,
                horizon_steps=horizon if self.args.lr_scheduler_type != "constant" else None,
                warmup_steps=self.args.warmup_steps,
            ),
        )

    def _make_config(self) -> TrainConfig:
        runtime_name = self.runtime.device.type
        if runtime_name not in {"cpu", "cuda", "xla"}:
            runtime_name = "xla" if self.runtime.name == "xla" else "cpu"
        precision = self.runtime.precision
        source = self._model_source or ModelSourceConfig()
        return TrainConfig(
            model=source,
            dataset=DatasetConfig(
                sequence_length=self.args.sequence_length,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            ),
            loss=LossConfig(implementation="causal_lm"),
            runtime=RuntimeConfig(device=runtime_name, precision=precision),
            parallelism=ParallelismConfig(data=1),
            optimizations=OptimizationConfig(policy="auto", allow_fallbacks=True),
            optimizer=OptimizerConfig(
                learning_rate=self.args.learning_rate,
                betas=self.args.betas,
                eps=self.args.eps,
                weight_decay=self.args.weight_decay,
                fused=False,
            ),
            scheduler=SchedulerConfig(
                name=self.args.lr_scheduler_type,
                horizon_steps=(self.args.max_steps or 1)
                if self.args.lr_scheduler_type != "constant" else None,
                warmup_steps=self.args.warmup_steps,
            ),
            trainer=TrainerConfig(
                max_steps=self.args.max_steps,
                max_tokens=self.args.max_tokens,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                materialize_loss_every_steps=self.args.logging_steps,
                max_grad_norm=1.0,
                seed=self.args.seed,
            ),
            checkpoint=CheckpointConfig(output_dir=Path(self.args.output_dir)),
            logging=LoggingConfig(
                log_every_steps=self.args.logging_steps,
                output_dir=Path(self.args.output_dir),
            ),
            monitoring=MonitoringConfig(enabled=False),
        )

    def train(self, *, resume_from_checkpoint: str | Path | None = None):
        """Run training and return the TrainLM trainer state."""

        if resume_from_checkpoint is not None:
            raise NotImplementedError(
                "Exact resume wiring is provided by the checkpoint service."
            )
        if self._tpu_coordinator is not None:
            return self._tpu_coordinator.run(self._make_tpu_request())
        if self.engine is None:
            raise RuntimeError("Trainer engine was not initialized.")
        return self.engine.train()

    def _make_tpu_request(self):
        from trainlm._tpu_coordinator import _TPURunRequest

        if not isinstance(self.train_dataset, (str, Path)):
            raise TypeError(
                "TPU training currently requires train_dataset to be a local manifest "
                "directory; PackedBinDataset support is the next public data story."
            )
        if self._model_source is None or self.args.max_steps is None:
            raise RuntimeError("TPU request prerequisites were not initialized.")
        precision = (
            "bf16" if self.args.bf16 else "fp16" if self.args.fp16 else "fp32"
        )
        return _TPURunRequest(
            model=self._model_source,
            manifest_dir=Path(self.train_dataset),
            output_dir=Path(self.args.output_dir),
            max_steps=self.args.max_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            micro_batch_per_device=self.args.per_device_train_batch_size,
            sequence_length=self.args.sequence_length,
            seed=self.args.seed,
            log_every_steps=self.args.logging_steps,
            learning_rate=self.args.learning_rate,
            betas=self.args.betas,
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
            scheduler=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            precision=precision,
        )

    def evaluate(self) -> dict[str, float]:
        if self.engine is None:
            raise NotImplementedError(
                "TPU evaluation will be added with lifecycle parity."
            )
        return self.engine.evaluate()

    def log_metrics(self, split: str, metrics: Mapping[str, Any]) -> None:
        """Expose a familiar metric hook without leaking engine internals."""

        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty string.")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping.")
        self._last_metrics = {
            f"{split}_{key}": value for key, value in metrics.items()
        }

    def save_model(self, output_dir: str | Path | None = None) -> Path:
        if self.model is None:
            raise NotImplementedError(
                "TPU model export will be added with lifecycle parity."
            )
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(destination, safe_serialization=True)
        else:
            torch.save(self.model.state_dict(), destination / "pytorch_model.bin")
        return destination

    def save_state(self, output_dir: str | Path | None = None) -> Path:
        if self.engine is None or self.optimizer is None or self.scheduler is None:
            raise NotImplementedError(
                "TPU state saving will be added with lifecycle parity."
            )
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        state = {
            "trainer": asdict(self.engine.state),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        path = destination / "trainer_state.pt"
        torch.save(state, path)
        return path

    def explain(self) -> dict[str, Any]:
        if self._tpu_coordinator is not None:
            return {
                "support_level": "compatible",
                "selected_path": "tpu_coordinator",
                "model": asdict(self._model_source),
                "backend": "xla",
                "precision": (
                    "bf16"
                    if self.args.bf16
                    else "fp16"
                    if self.args.fp16
                    else "fp32"
                ),
            }
        if self.loaded is None:
            return {
                "support_level": "compatible",
                "selected_path": "external_model",
                "model_class": type(self.model).__name__,
                "backend": self.runtime.name,
            }
        explanation = self.loaded
        from trainlm.model import explain_huggingface_compatibility

        return {
            "model": explanation.metadata.to_dict(),
            "compatibility": explain_huggingface_compatibility(explanation).to_dict(),
            "backend": self.runtime.name,
            "precision": self.runtime.precision,
        }


__all__ = ["TrainLMTrainer", "TrainLMTrainingArguments"]
