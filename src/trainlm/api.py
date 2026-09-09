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
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler

from trainlm.config import (
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
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
from trainlm.training import (
    TrainerCallback,
    TrainerControl,
    TrainerPhase,
    TrainerState,
    create_scheduler,
)

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
        if self.args.eval_steps is not None and eval_dataset is None:
            raise ValueError("eval_steps requires eval_dataset.")
        self._model_source: ModelSourceConfig | None = None
        self.loaded: LoadedCausalLM | None = None
        self._last_metrics: dict[str, Any] = {}
        self._train_loader_generator: torch.Generator | None = None
        self._train_loader_generator_initial_state: torch.Tensor | None = None
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
            train_dataloader=self._make_loader(
                train_dataset, self.args.per_device_train_batch_size, is_train=True
            ),
            eval_dataloader=(
                self._make_loader(
                    eval_dataset,
                    self.args.per_device_eval_batch_size,
                    is_train=False,
                )
                if eval_dataset is not None
                else None
            ),
            callbacks=self.callbacks,
            checkpoint_saver=self._save_training_checkpoint,
            checkpoint_loader=self._load_training_checkpoint,
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

    def _make_loader(
        self,
        source: Dataset | DataLoader | Any,
        batch_size: int,
        *,
        is_train: bool,
    ) -> DataLoader:
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
        generator = None
        if is_train and not is_iterable:
            generator = torch.Generator()
            generator.manual_seed(self.args.seed)
            self._train_loader_generator = generator
            self._train_loader_generator_initial_state = generator.get_state().clone()
        return DataLoader(
            source,
            batch_size=batch_size,
            shuffle=not is_iterable,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=self.data_collator,
            generator=generator,
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
            checkpoint=CheckpointConfig(
                output_dir=Path(self.args.output_dir),
                save_training_every_steps=self.args.save_steps,
            ),
            logging=LoggingConfig(
                log_every_steps=self.args.logging_steps,
                output_dir=Path(self.args.output_dir),
            ),
            monitoring=MonitoringConfig(enabled=False),
            evaluation=EvaluationConfig(
                enabled=self.eval_dataset is not None,
                eval_every_steps=self.args.eval_steps,
            ),
        )

    def train(self, *, resume_from_checkpoint: str | Path | None = None):
        """Run training and return the TrainLM trainer state."""

        if self._tpu_coordinator is not None:
            if resume_from_checkpoint is not None:
                raise NotImplementedError(
                    "TPU checkpoint resume will be added after worker checkpoint wiring."
                )
            return self._consume_tpu_result(
                self._tpu_coordinator.run(
                    self._make_tpu_request(resume_from_checkpoint=resume_from_checkpoint)
                )
            )
        if self.engine is None:
            raise RuntimeError("Trainer engine was not initialized.")
        if resume_from_checkpoint is not None:
            self.engine.load_checkpoint(Path(resume_from_checkpoint))
        return self.engine.train()

    def _consume_tpu_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Deliver worker artifacts through public state and callback contracts."""

        worker = result.get("worker_summary") or {}
        state = TrainerState(
            step=int(worker.get("steps", 0)),
            micro_step=int(worker.get("micro_steps", 0)),
            tokens_seen=int(worker.get("tokens_seen_rank0", 0)),
            samples_seen=int(worker.get("samples_seen_rank0", 0)),
            learning_rate=float(worker.get("learning_rate", 0.0)),
            loss=worker.get("last_loss_rank0"),
            phase=TrainerPhase.FINALIZED,
        )
        control = TrainerControl()
        for metrics in result.get("metrics", ()):
            self._last_metrics = dict(metrics)
            for callback in self.callbacks:
                callback.on_metrics(state, control, metrics)
        return {**result, "trainer_state": asdict(state)}

    def _save_training_checkpoint(self, engine: EngineTrainer, destination: object | None):
        root = Path(self.args.output_dir)
        path = root / str(destination or f"checkpoint-{engine.state.step}")
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "model": engine.model.state_dict(),
            "optimizer": engine.optimizer.state_dict(),
            "scheduler": engine.scheduler.state_dict(),
            "runtime": dict(engine.runtime.state_dict()),
            "trainer": {
                key: value
                for key, value in asdict(engine.state).items()
                if key not in {"phase", "is_training", "should_stop", "failure"}
            },
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generator_initial_state": self._train_loader_generator_initial_state,
        }
        temporary = path / "trainer_state.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(path / "trainer_state.pt")
        return path

    def _load_training_checkpoint(self, engine: EngineTrainer, source: object):
        path = Path(source)
        if path.is_dir():
            path = path / "trainer_state.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Training checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported TrainLM training checkpoint schema.")
        engine.model.load_state_dict(payload["model"])
        engine.optimizer.load_state_dict(payload["optimizer"])
        engine.scheduler.load_state_dict(payload["scheduler"])
        engine.runtime.load_state_dict(payload["runtime"])
        for key, value in payload["trainer"].items():
            setattr(engine.state, key, value)
        initial_state = payload.get("loader_generator_initial_state")
        sampler = getattr(engine.train_dataloader, "sampler", None)
        if isinstance(sampler, RandomSampler) and initial_state is None:
            raise ValueError(
                "Exact resume requires a TrainLM-owned deterministic train loader."
            )
        if self._train_loader_generator is not None and initial_state is not None:
            self._train_loader_generator.set_state(initial_state)
        iterator = iter(engine.train_dataloader)
        remaining = engine.state.micro_step
        while remaining:
            try:
                next(iterator)
                remaining -= 1
            except StopIteration:
                iterator = iter(engine.train_dataloader)
        engine._train_iterator = iterator
        # Iterator reconstruction may execute dataset code. Restore training
        # RNG only after positioning so the next model operation sees the
        # exact checkpoint state.
        torch.set_rng_state(payload["cpu_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return path

    def _make_tpu_request(
        self, *, resume_from_checkpoint: str | Path | None = None
    ):
        from trainlm._tpu_coordinator import _TPURunRequest
        from trainlm.data import PackedBinDataset

        if isinstance(self.train_dataset, PackedBinDataset):
            manifest_dir = self.train_dataset.coordinator_manifest_dir(
                self.args.output_dir
            )
        elif isinstance(self.train_dataset, (str, Path)):
            manifest_dir = Path(self.train_dataset)
        else:
            raise TypeError(
                "TPU training requires a PackedBinDataset or local manifest directory."
            )
        if self._model_source is None or self.args.max_steps is None:
            raise RuntimeError("TPU request prerequisites were not initialized.")
        precision = (
            "bf16" if self.args.bf16 else "fp16" if self.args.fp16 else "fp32"
        )
        return _TPURunRequest(
            model=self._model_source,
            manifest_dir=manifest_dir,
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
            save_every_steps=self.args.save_steps,
            resume_from_checkpoint=(
                Path(resume_from_checkpoint)
                if resume_from_checkpoint is not None
                else None
            ),
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

    def explain(
        self, *, format: Literal["dict", "json", "text"] = "dict", strict: bool = False
    ) -> dict[str, Any] | str:
        """Explain capabilities and execution selection without launching work."""

        from trainlm.optimization import OptimizationExplanation, inspect_dense_causal_lm

        if self._tpu_coordinator is not None:
            report = OptimizationExplanation(
                backend="xla",
                selected_path="tpu_coordinator",
                precision=(
                    "bf16"
                    if self.args.bf16
                    else "fp16"
                    if self.args.fp16
                    else "fp32"
                ),
                limitations=(
                    "Capabilities are inspected inside TPU workers after model loading.",
                    "TPU lifecycle parity and target-hardware certification remain pending.",
                ),
            )
        else:
            report = OptimizationExplanation(
                backend=self.runtime.name,
                precision=self.runtime.precision,
                selected_path=("external_model" if self.loaded is None else "huggingface_model"),
                certification="compatible",
                capabilities=inspect_dense_causal_lm(
                    self.model,
                    source_provider=("unknown" if self.loaded is None else "huggingface"),
                ),
                limitations=("No optimized provider execution plan has been selected.",),
            )
        if strict:
            report.require_supported()
        if format == "dict":
            return report.to_dict()
        if format == "json":
            return report.to_json()
        if format == "text":
            return report.to_text()
        raise ValueError("format must be 'dict', 'json', or 'text'.")


__all__ = ["TrainLMTrainer", "TrainLMTrainingArguments"]
