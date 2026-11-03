"""Public Hugging Face-like TrainLM training surface.

The classes in this module deliberately hide backend construction and the
training engine's internal lifecycle. TPU worker/coordinator integration can
be added behind the same API without changing user code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Literal
import warnings

import yaml

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)

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
from trainlm.runtime.torchtpu_discovery import (
    discover_tpu, normalize_worker_report, select_auto_backend,
)
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


PUBLIC_API_VERSION = "1"
DEPRECATED_CONFIG_KEYS = {"args": "training_args"}
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_PARAMETER_SHAPED_OPTIMIZER_SLOTS = frozenset(
    {
        "ax",
        "exp_avg",
        "exp_avg_sq",
        "exp_inf",
        "grad_avg",
        "master_param",
        "max_exp_avg_sq",
        "momentum_buffer",
        "prev",
        "square_avg",
        "step_size",
        "sum",
    }
)


def _checkpoint_component_type(value: Any) -> str:
    component_type = type(value)
    return f"{component_type.__module__}.{component_type.__qualname__}"


def _is_finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _is_scheduler_scalar(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return (
            value.ndim == 0
            and not value.is_complex()
            and value.dtype != torch.bool
            and bool(torch.isfinite(value).item())
        )
    return _is_finite_real(value)


def _resolve_hugging_face_model_revision(
    repo_id: str, revision: str | None
) -> str:
    """Resolve a user-friendly model revision before private TPU launch."""

    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id=repo_id, revision=revision or "main")
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or _COMMIT_SHA.fullmatch(resolved) is None:
        raise ValueError(
            "Hugging Face did not resolve the model to a lowercase "
            "40-character commit SHA."
        )
    return resolved


def _load_public_config(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Trainer configuration does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("Trainer configuration root must be a mapping.")
    unknown = set(data) - {"api_version", "model", "training_args", "args"}
    if unknown:
        raise ValueError(f"Unknown trainer configuration keys: {sorted(unknown)!r}")
    version = str(data.get("api_version", PUBLIC_API_VERSION))
    if version != PUBLIC_API_VERSION:
        raise ValueError(f"Unsupported public API version: {version!r}.")
    if "args" in data:
        if "training_args" in data:
            raise ValueError("Set only 'training_args'; deprecated 'args' is also present.")
        warnings.warn(
            "Configuration key 'args' is deprecated; use 'training_args'.",
            DeprecationWarning,
            stacklevel=2,
        )
        data["training_args"] = data.pop("args")
    return data


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
    lr_scheduler_type: Literal["constant", "linear", "cosine", "wsd"] = "cosine"
    warmup_steps: int = 0
    bf16: bool = False
    fp16: bool = False
    accelerator: Literal["auto", "cpu", "cuda", "tpu", "native_tpu"] = "auto"
    torch_compile: bool = False
    logging_steps: int = 10
    save_steps: int | None = None
    eval_steps: int | None = None
    max_eval_batches: int | None = None
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    seed: int = 42
    report_to: str | tuple[str, ...] = "none"
    logging_verbosity: Literal["quiet", "normal", "verbose"] = "normal"
    loss_implementation: Literal[
        "auto", "causal_lm", "model", "chunked_linear"
    ] = "auto"
    logits_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is None and self.max_tokens is None:
            raise ValueError("Set max_steps or max_tokens.")
        for name in (
            "max_steps",
            "max_tokens",
            "save_steps",
            "eval_steps",
            "max_eval_batches",
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
        if not isinstance(self.torch_compile, bool):
            raise ValueError("torch_compile must be boolean.")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled.")
        if self.accelerator not in {"auto", "cpu", "cuda", "tpu", "native_tpu"}:
            raise ValueError(f"Unsupported accelerator: {self.accelerator}")
        if self.torch_compile and self.accelerator != "native_tpu":
            raise ValueError(
                "torch_compile is currently supported only with "
                "accelerator='native_tpu'."
            )
        if self.lr_scheduler_type not in {"constant", "linear", "cosine", "wsd"}:
            raise ValueError(f"Unsupported scheduler: {self.lr_scheduler_type}")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if (
            not isinstance(self.logging_verbosity, str)
            or self.logging_verbosity not in {"quiet", "normal", "verbose"}
        ):
            raise ValueError(
                "logging_verbosity must be 'quiet', 'normal', or 'verbose'."
            )
        if self.loss_implementation not in {
            "auto", "causal_lm", "model", "chunked_linear"
        }:
            raise ValueError("Unsupported loss_implementation.")
        if self.logits_chunk_size is not None and (
            isinstance(self.logits_chunk_size, bool)
            or not isinstance(self.logits_chunk_size, int)
            or self.logits_chunk_size < 1
        ):
            raise ValueError("logits_chunk_size must be positive when configured.")


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

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainLMTrainer":
        """Construct the trainer with an explicit HF pretrained model source."""

        source = ModelSourceConfig(
            provider="huggingface",
            initialization="pretrained",
            name_or_path=str(model_name_or_path),
            revision=revision,
        )
        return cls(model=source, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path,
        *,
        train_dataset: Dataset | DataLoader | Any,
        eval_dataset: Dataset | DataLoader | Any | None = None,
        **kwargs: Any,
    ) -> "TrainLMTrainer":
        """Construct from a versioned mapping or YAML file plus dataset objects."""

        data = _load_public_config(config)
        if "model" not in data:
            raise ValueError("Trainer configuration requires 'model'.")
        model_data = data["model"]
        if isinstance(model_data, (str, Path)):
            model: str | Path | ModelSourceConfig = ModelSourceConfig(
                provider="huggingface",
                initialization="pretrained",
                name_or_path=str(model_data),
            )
        elif isinstance(model_data, Mapping):
            values = dict(model_data)
            values.setdefault("provider", "huggingface")
            model = ModelSourceConfig(**values)
        else:
            raise TypeError("'model' must be a model ID/path or mapping.")
        argument_data = data.get("training_args", {})
        if not isinstance(argument_data, Mapping):
            raise TypeError("'training_args' must be a mapping.")
        return cls(
            model=model,
            args=TrainLMTrainingArguments(**dict(argument_data)),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

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
        if self.args.max_eval_batches is not None and eval_dataset is None:
            raise ValueError("max_eval_batches requires eval_dataset.")
        self._model_source: ModelSourceConfig | None = None
        self.loaded: LoadedCausalLM | None = None
        self._last_metrics: dict[str, Any] = {}
        self._native_tpu_worker_report: dict[str, Any] | None = None
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
            from trainlm._tpu_coordinator import _TPUCoordinator

            self._tpu_coordinator = _TPUCoordinator()
            self.model = None
            self.runtime = None
            self.optimizer = None
            self.scheduler = None
            self.engine = None
            return
        if self.args.loss_implementation == "chunked_linear":
            raise NotImplementedError(
                "chunked_linear loss is currently available through the TPU worker; "
                "use the TPU accelerator or the lower-level training view on CPU/CUDA."
            )
        if self.args.torch_compile and runtime is not None:
            raise ValueError(
                "torch_compile cannot be combined with an injected runtime; "
                "select accelerator='native_tpu' so TrainLM can configure it."
            )
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
        if self.runtime.name == "torchtpu" and self.args.save_steps is not None:
            self._validate_native_resume_data(self.engine)

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
            report = discover_tpu(
                torch, verify_allocation=False, include_properties=False
            )
            accelerator = select_auto_backend(
                report, cuda_available=torch.cuda.is_available()
            )
        if accelerator == "native_tpu":
            from trainlm.runtime.torchtpu import TorchTPURuntime

            precision = "bf16" if self.args.bf16 else "fp16" if self.args.fp16 else "fp32"
            return TorchTPURuntime(
                precision=precision,
                seed=self.args.seed,
                compile_enabled=self.args.torch_compile,
            )
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
        if runtime_name not in {"cpu", "cuda", "xla", "tpu"}:
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
            runtime=RuntimeConfig(
                device="native_tpu" if runtime_name == "tpu" else runtime_name,
                precision=precision,
            ),
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
                max_batches=self.args.max_eval_batches,
            ),
        )

    def train(self, *, resume_from_checkpoint: str | Path | None = None):
        """Run training and return the TrainLM trainer state."""

        if self._tpu_coordinator is not None:
            return self._consume_tpu_result(
                self._tpu_coordinator.run(
                    self._make_tpu_request(resume_from_checkpoint=resume_from_checkpoint)
                )
            )
        if self.engine is None:
            raise RuntimeError("Trainer engine was not initialized.")
        if resume_from_checkpoint is not None:
            self._validate_native_resume_data(self.engine)
            self.engine.load_checkpoint(Path(resume_from_checkpoint))
        return self.engine.train()

    def _consume_tpu_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Deliver worker artifacts through public state and callback contracts."""

        worker = result.get("worker_summary") or {}
        self._native_tpu_worker_report = None
        if "native_tpu_report" in worker:
            self._native_tpu_worker_report = normalize_worker_report(
                worker["native_tpu_report"]
            )
            worker = {**worker, "native_tpu_report": self._native_tpu_worker_report}
            result = {**result, "worker_summary": worker}
        state = TrainerState(
            step=int(worker.get("steps", 0)),
            micro_step=int(worker.get("micro_steps", 0)),
            tokens_seen=int(worker.get("tokens_seen_rank0", 0)),
            samples_seen=int(worker.get("samples_seen_rank0", 0)),
            learning_rate=float(worker.get("learning_rate", 0.0)),
            loss=worker.get("last_loss_rank0"),
            global_batch_size=int(
                worker.get("global_batch_size", worker.get("samples_per_update", 0))
            ),
            phase=TrainerPhase.FINALIZED,
        )
        control = TrainerControl()
        for metrics in result.get("metrics", ()):
            self._last_metrics = dict(metrics)
            for callback in self.callbacks:
                callback.on_metrics(state, control, metrics)
        return {**result, "trainer_state": asdict(state)}

    def _training_checkpoint_payload(self, engine: EngineTrainer) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "model": engine.model.state_dict(),
            "optimizer": engine.optimizer.state_dict(),
            "optimizer_type": _checkpoint_component_type(engine.optimizer),
            "scheduler": engine.scheduler.state_dict(),
            "scheduler_type": _checkpoint_component_type(engine.scheduler),
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
        loader = engine.train_dataloader
        dataset = getattr(loader, "dataset", None)
        dataset_state_dict = getattr(dataset, "state_dict", None)
        if (
            getattr(loader, "num_workers", 0) == 0
            and callable(dataset_state_dict)
            and callable(getattr(dataset, "load_state_dict", None))
        ):
            payload["train_dataset_state"] = dataset_state_dict()
        return payload

    def _validate_native_resume_data(self, engine: EngineTrainer) -> None:
        """Reject native checkpoints whose data position cannot be replayed."""
        if getattr(self.runtime, "name", None) != "torchtpu":
            return
        loader = engine.train_dataloader
        dataset = getattr(loader, "dataset", None)
        sampler = getattr(loader, "sampler", None)
        if getattr(loader, "num_workers", 0) != 0:
            raise ValueError(
                "Native TPU checkpoint/resume currently requires "
                "dataloader_num_workers=0 so worker RNG and prefetch state are "
                "not lost."
            )
        if isinstance(dataset, IterableDataset):
            if not (
                callable(getattr(dataset, "state_dict", None))
                and callable(getattr(dataset, "load_state_dict", None))
            ):
                raise ValueError(
                    "Native TPU checkpoint/resume requires a deterministic "
                    "map-style dataset or an iterable dataset with an exact "
                    "state_dict/load_state_dict cursor."
                )
            return
        if not (hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")):
            raise ValueError(
                "Native TPU checkpoint/resume requires a deterministic "
                "map-style dataset or a stateful packed-data cursor."
            )
        if isinstance(sampler, RandomSampler):
            if self._train_loader_generator is None:
                raise ValueError(
                    "Native TPU resume with random sampling requires a "
                    "TrainLM-owned deterministic data-loader generator."
                )
        elif not isinstance(sampler, SequentialSampler):
            raise ValueError(
                "Native TPU checkpoint/resume supports only TrainLM random "
                "sampling or sequential sampling until custom sampler state is "
                "serializable."
            )

    def _save_training_checkpoint(self, engine: EngineTrainer, destination: object | None):
        self._validate_native_resume_data(engine)
        root = Path(self.args.output_dir)
        path = root / str(destination or f"checkpoint-{engine.state.step}")
        path.mkdir(parents=True, exist_ok=True)
        payload = self._training_checkpoint_payload(engine)
        temporary = path / "trainer_state.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(path / "trainer_state.pt")
        return path

    def _preflight_training_checkpoint(
        self, engine: EngineTrainer, payload: object
    ) -> tuple[Any, Any]:
        """Validate checkpoint compatibility before mutating trainer state."""
        if not isinstance(payload, Mapping):
            raise ValueError("TrainLM training checkpoint must be a mapping.")
        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise ValueError("Unsupported TrainLM training checkpoint schema.")

        required = {
            "model",
            "optimizer",
            "scheduler",
            "runtime",
            "trainer",
            "cpu_rng_state",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(
                "TrainLM training checkpoint is missing required state: "
                f"{sorted(missing)}."
            )

        model_payload = payload["model"]
        if not isinstance(model_payload, Mapping):
            raise ValueError("Training checkpoint model state must be a mapping.")
        current_model_state = engine.model.state_dict()
        if set(model_payload) != set(current_model_state):
            raise ValueError(
                "Training checkpoint model keys do not match the active model."
            )
        for key, current_value in current_model_state.items():
            saved_value = model_payload[key]
            if isinstance(current_value, torch.Tensor):
                if not isinstance(saved_value, torch.Tensor):
                    raise ValueError(
                        f"Training checkpoint model value for {key!r} is not a tensor."
                    )
                if saved_value.shape != current_value.shape:
                    raise ValueError(
                        f"Training checkpoint model tensor shape mismatch for {key!r}."
                    )

        for name in ("optimizer", "scheduler", "runtime", "trainer"):
            if not isinstance(payload[name], Mapping):
                raise ValueError(f"Training checkpoint {name} state must be a mapping.")
        for name, component in (
            ("optimizer_type", engine.optimizer),
            ("scheduler_type", engine.scheduler),
        ):
            if name not in payload:
                # Schema-v1 checkpoints written before class metadata remain valid.
                continue
            checkpoint_type = payload[name]
            if (
                not isinstance(checkpoint_type, str)
                or checkpoint_type != _checkpoint_component_type(component)
            ):
                component_name = name.removesuffix("_type")
                raise ValueError(
                    f"Training checkpoint {component_name} type does not match "
                    "the active trainer."
                )
        self._preflight_optimizer_state(engine.optimizer, payload["optimizer"])
        self._preflight_scheduler_state(
            engine.scheduler,
            payload["scheduler"],
            optimizer_group_count=len(engine.optimizer.param_groups),
        )
        validate_runtime_state = getattr(engine.runtime, "validate_state_dict", None)
        if callable(validate_runtime_state):
            validate_runtime_state(payload["runtime"])
        trainer_payload = payload["trainer"]
        for key in trainer_payload:
            if not isinstance(key, str) or not hasattr(engine.state, key):
                raise ValueError(
                    f"Unknown trainer-state field in checkpoint: {key!r}."
                )
        self._preflight_trainer_state(trainer_payload)

        cpu_rng_state = payload["cpu_rng_state"]
        if (
            not isinstance(cpu_rng_state, torch.Tensor)
            or cpu_rng_state.dtype != torch.uint8
            or cpu_rng_state.ndim != 1
        ):
            raise ValueError("Training checkpoint CPU RNG state is invalid.")
        try:
            torch.Generator().set_state(cpu_rng_state.cpu())
        except RuntimeError as exc:
            raise ValueError("Training checkpoint CPU RNG state is invalid.") from exc

        cuda_rng_state = payload.get("cuda_rng_state")
        if cuda_rng_state is not None and (
            not isinstance(cuda_rng_state, (list, tuple))
            or any(
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.uint8
                or state.ndim != 1
                for state in cuda_rng_state
            )
        ):
            raise ValueError("Training checkpoint accelerator RNG state is invalid.")

        initial_state = payload.get("loader_generator_initial_state")
        if initial_state is not None:
            if (
                not isinstance(initial_state, torch.Tensor)
                or initial_state.dtype != torch.uint8
                or initial_state.ndim != 1
            ):
                raise ValueError(
                    "Training checkpoint data-loader RNG state is invalid."
                )
            try:
                torch.Generator().set_state(initial_state.cpu())
            except RuntimeError as exc:
                raise ValueError(
                    "Training checkpoint data-loader RNG state is invalid."
                ) from exc
        sampler = getattr(engine.train_dataloader, "sampler", None)
        if isinstance(sampler, RandomSampler) and initial_state is None:
            raise ValueError(
                "Exact resume requires a TrainLM-owned deterministic train loader."
            )

        dataset_state = payload.get("train_dataset_state")
        dataset = getattr(engine.train_dataloader, "dataset", None)
        if dataset_state is not None:
            if not callable(getattr(dataset, "load_state_dict", None)):
                raise ValueError(
                    "Checkpoint contains train dataset state but the current "
                    "dataset cannot restore it."
                )
            if getattr(engine.train_dataloader, "num_workers", 0) != 0:
                raise ValueError(
                    "Exact iterable dataset resume requires "
                    "dataloader_num_workers=0."
                )
        return initial_state, dataset_state

    @staticmethod
    def _preflight_trainer_state(state: Mapping[str, Any]) -> None:
        """Validate persisted progress and metrics before applying the state."""
        integer_fields = (
            "step",
            "micro_step",
            "epoch",
            "tokens_seen",
            "samples_seen",
            "global_batch_size",
        )
        for name in integer_fields:
            if name not in state and name not in {"step", "micro_step"}:
                continue
            value = state.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"Training checkpoint trainer field {name!r} must be "
                    "a non-negative integer."
                )

        for name in ("learning_rate", "loss", "grad_norm"):
            if name not in state:
                continue
            value = state[name]
            if name in {"loss", "grad_norm"} and value is None:
                continue
            if not _is_finite_real(value) or (
                name in {"learning_rate", "grad_norm"} and value < 0
            ):
                requirement = (
                    "a finite non-negative scalar"
                    if name in {"learning_rate", "grad_norm"}
                    else "a finite scalar"
                )
                raise ValueError(
                    f"Training checkpoint trainer field {name!r} must be "
                    f"{requirement}."
                )

    @staticmethod
    def _preflight_optimizer_state(optimizer: Optimizer, state: Mapping[str, Any]) -> None:
        """Check optimizer groups and common parameter slots without loading."""
        optimizer_state = state.get("state")
        saved_groups = state.get("param_groups")
        if not isinstance(optimizer_state, Mapping) or not isinstance(
            saved_groups, (list, tuple)
        ):
            raise ValueError("Training checkpoint optimizer state is malformed.")

        active_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(active_groups, (list, tuple)):
            raise ValueError("Active optimizer parameter groups are invalid.")
        if len(saved_groups) != len(active_groups):
            raise ValueError(
                "Training checkpoint optimizer parameter-group count does not "
                "match the active optimizer."
            )

        seen_parameter_ids = set()
        parameter_by_saved_id = {}
        for index, (saved_group, active_group) in enumerate(
            zip(saved_groups, active_groups)
        ):
            if not isinstance(saved_group, Mapping) or not isinstance(
                active_group, Mapping
            ):
                raise ValueError(
                    f"Training checkpoint optimizer group {index} is malformed."
                )
            saved_parameters = saved_group.get("params")
            active_parameters = active_group.get("params")
            if not isinstance(saved_parameters, (list, tuple)) or not isinstance(
                active_parameters, (list, tuple)
            ):
                raise ValueError(
                    f"Training checkpoint optimizer group {index} has invalid params."
                )
            if len(saved_parameters) != len(active_parameters):
                raise ValueError(
                    f"Training checkpoint optimizer group {index} parameter count "
                    "does not match the active optimizer."
                )
            for parameter_id, parameter in zip(
                saved_parameters, active_parameters
            ):
                try:
                    if parameter_id in seen_parameter_ids:
                        raise ValueError(
                            "Training checkpoint optimizer parameter IDs are not unique."
                        )
                    seen_parameter_ids.add(parameter_id)
                except TypeError as exc:
                    raise ValueError(
                        "Training checkpoint optimizer parameter IDs must be hashable."
                    ) from exc
                parameter_by_saved_id[parameter_id] = parameter

            saved_names = saved_group.get("param_names")
            active_names = active_group.get("param_names")
            if (
                saved_names is not None
                and active_names is not None
                and saved_names != active_names
            ):
                raise ValueError(
                    f"Training checkpoint optimizer group {index} parameter names "
                    "do not match the active optimizer."
                )

        for parameter_id, saved_state in optimizer_state.items():
            parameter = parameter_by_saved_id.get(parameter_id)
            if parameter is None or not isinstance(saved_state, Mapping):
                continue
            step = saved_state.get("step")
            if step is not None:
                if isinstance(step, torch.Tensor):
                    valid_step = (
                        step.ndim == 0
                        and step.dtype != torch.bool
                        and not step.is_complex()
                        and bool(torch.isfinite(step).item())
                        and bool((step >= 0).item())
                    )
                else:
                    valid_step = _is_finite_real(step) and step >= 0
                if not valid_step:
                    raise ValueError(
                        "Training checkpoint optimizer state 'step' must be a "
                        "non-negative scalar."
                    )
            for name in _PARAMETER_SHAPED_OPTIMIZER_SLOTS:
                if name not in saved_state:
                    continue
                value = saved_state[name]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError(
                        f"Training checkpoint optimizer slot {name!r} shape "
                        "does not match its parameter."
                    )

    @staticmethod
    def _preflight_scheduler_state(
        scheduler: LRScheduler,
        state: Mapping[str, Any],
        *,
        optimizer_group_count: int,
    ) -> None:
        """Validate scheduler fields before loading can change trainer state."""
        current_state = scheduler.state_dict()
        missing = set(current_state) - state.keys()
        if missing:
            missing_names = ", ".join(sorted(str(name) for name in missing))
            raise ValueError(
                "Training checkpoint scheduler state is missing required fields: "
                f"{missing_names}."
            )

        for name, current_value in current_state.items():
            saved_value = state[name]
            if isinstance(current_value, bool):
                valid = isinstance(saved_value, bool)
            elif isinstance(current_value, int):
                valid = isinstance(saved_value, int) and not isinstance(
                    saved_value, bool
                )
            elif isinstance(current_value, float):
                valid = _is_finite_real(saved_value)
            elif isinstance(current_value, torch.Tensor):
                valid = (
                    isinstance(saved_value, torch.Tensor)
                    and saved_value.shape == current_value.shape
                )
            else:
                continue
            if not valid:
                raise ValueError(
                    f"Training checkpoint scheduler field {name!r} has an "
                    "incompatible value."
                )
            if name == "_step_count" and saved_value < 0:
                raise ValueError(
                    "Training checkpoint scheduler '_step_count' must be "
                    "non-negative."
                )
            if name == "last_epoch" and saved_value < -1:
                raise ValueError(
                    "Training checkpoint scheduler 'last_epoch' is invalid."
                )
            if name in {"T_max", "total_iters"} and saved_value < 1:
                raise ValueError(
                    f"Training checkpoint scheduler field {name!r} must be positive."
                )
            if name == "eta_min" and saved_value < 0:
                raise ValueError(
                    "Training checkpoint scheduler 'eta_min' must be non-negative."
                )
            if name == "start_factor" and not 0 < saved_value <= 1:
                raise ValueError(
                    "Training checkpoint scheduler 'start_factor' is invalid."
                )
            if name in {"end_factor", "min_lr_ratio"} and not 0 <= saved_value <= 1:
                raise ValueError(
                    f"Training checkpoint scheduler field {name!r} is invalid."
                )

        for name in ("base_lrs", "_last_lr"):
            if name not in state:
                continue
            values = state[name]
            if not isinstance(values, (list, tuple)) or len(values) != optimizer_group_count:
                raise ValueError(
                    f"Training checkpoint scheduler field {name!r} must contain "
                    "one value per optimizer group."
                )
            if any(not _is_scheduler_scalar(value) for value in values):
                raise ValueError(
                    f"Training checkpoint scheduler field {name!r} contains "
                    "an invalid learning rate."
                )

        if "lr_lambdas" in state:
            lambdas = state["lr_lambdas"]
            current_lambdas = current_state.get("lr_lambdas")
            if (
                not isinstance(lambdas, (list, tuple))
                or not isinstance(current_lambdas, (list, tuple))
                or len(lambdas) != len(current_lambdas)
                or any(value is not None and not isinstance(value, Mapping) for value in lambdas)
            ):
                raise ValueError(
                    "Training checkpoint scheduler 'lr_lambdas' state is invalid."
                )

        if "horizon_tokens" in state:
            integer_fields = (
                "horizon_tokens",
                "warmup_tokens",
                "stable_tokens",
                "stable_end_tokens",
                "last_tokens",
            )
            values = {name: state.get(name) for name in integer_fields}
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values.values()
            ) or values["horizon_tokens"] == 0:
                raise ValueError("Training checkpoint token scheduler state is invalid.")
            if (
                values["warmup_tokens"] + values["stable_tokens"]
                != values["stable_end_tokens"]
                or values["stable_end_tokens"] > values["horizon_tokens"]
            ):
                raise ValueError("Training checkpoint token scheduler state is inconsistent.")
            ratio = state.get("min_lr_ratio")
            if not _is_finite_real(ratio) or not 0 <= ratio <= 1:
                raise ValueError("Training checkpoint token scheduler LR ratio is invalid.")

    def _load_training_checkpoint(self, engine: EngineTrainer, source: object):
        self._validate_native_resume_data(engine)
        path = Path(source)
        if path.is_dir():
            path = path / "trainer_state.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Training checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        initial_state, dataset_state = self._preflight_training_checkpoint(
            engine, payload
        )
        engine.model.load_state_dict(payload["model"])
        engine.optimizer.load_state_dict(payload["optimizer"])
        engine.scheduler.load_state_dict(payload["scheduler"])
        for key, value in payload["trainer"].items():
            setattr(engine.state, key, value)
        if self._train_loader_generator is not None and initial_state is not None:
            self._train_loader_generator.set_state(initial_state)
        dataset = getattr(engine.train_dataloader, "dataset", None)
        load_dataset_state = getattr(dataset, "load_state_dict", None)
        if dataset_state is not None:
            load_dataset_state(dataset_state)
            iterator = iter(engine.train_dataloader)
        else:
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
        # Restore accelerator state after rebuilding the data iterator so any
        # host-side iterator setup cannot advance the checkpointed device RNG.
        engine.runtime.load_state_dict(payload["runtime"])
        return path

    def _make_tpu_request(
        self, *, resume_from_checkpoint: str | Path | None = None
    ):
        from trainlm._tpu_coordinator import _TPURunRequest
        from trainlm.data import PackedBinDataset

        if self._model_source is None or self.args.max_steps is None:
            raise RuntimeError("TPU request prerequisites were not initialized.")
        model_path = self._model_source.name_or_path
        is_local_model = model_path is not None and Path(model_path).exists()
        if (
            self._model_source.provider == "huggingface"
            and self._model_source.initialization == "pretrained"
            and not is_local_model
        ):
            revision = self._model_source.revision
            if revision is None or _COMMIT_SHA.fullmatch(revision) is None:
                self._model_source = replace(
                    self._model_source,
                    revision=_resolve_hugging_face_model_revision(
                        self._model_source.name_or_path, revision
                    ),
                )
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
        eval_manifest_dir = None
        if self.eval_dataset is not None:
            if isinstance(self.eval_dataset, PackedBinDataset):
                eval_manifest_dir = self.eval_dataset.coordinator_manifest_dir(
                    Path(self.args.output_dir) / "eval_data"
                )
            elif isinstance(self.eval_dataset, (str, Path)):
                eval_manifest_dir = Path(self.eval_dataset)
            else:
                raise TypeError(
                    "TPU evaluation requires a PackedBinDataset or local manifest directory."
                )
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
            logging_verbosity=self.args.logging_verbosity,
            loss_implementation=self.args.loss_implementation,
            logits_chunk_size=self.args.logits_chunk_size,
            save_every_steps=self.args.save_steps,
            resume_from_checkpoint=(
                Path(resume_from_checkpoint)
                if resume_from_checkpoint is not None
                else None
            ),
            eval_manifest_dir=eval_manifest_dir,
            eval_every_steps=self.args.eval_steps,
            max_eval_batches=self.args.max_eval_batches,
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
        save_processing = getattr(self.processing_class, "save_pretrained", None)
        if callable(save_processing):
            save_processing(destination)
        return destination

    def save_state(self, output_dir: str | Path | None = None) -> Path:
        if self.engine is None or self.optimizer is None or self.scheduler is None:
            raise NotImplementedError(
                "TPU state saving will be added with lifecycle parity."
            )
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._validate_native_resume_data(self.engine)
        path = destination / "trainer_state.pt"
        temporary = path.with_suffix(".pt.tmp")
        self.runtime.before_checkpoint("manual-state")
        success = False
        try:
            torch.save(self._training_checkpoint_payload(self.engine), temporary)
            temporary.replace(path)
            success = True
            return path
        finally:
            self.runtime.after_checkpoint("manual-state", success=success)

    def preflight(self) -> dict[str, Any]:
        """Inspect native TPU availability without allocating or launching workers.

        This parent-process snapshot cannot certify hardware, compilation or
        native training. Device properties and allocation belong to a worker.
        """
        report = discover_tpu(torch, verify_allocation=False, include_properties=False)
        return {
            "schema_version": 1,
            "requested_accelerator": self.args.accelerator,
            "selected_backend": (
                "xla" if self._tpu_coordinator is not None else self.runtime.name
            ),
            "native_tpu_training": (
                "experimental_single_device"
                if self.runtime is not None and self.runtime.name == "torchtpu"
                else "not_integrated"
            ),
            "native_tpu": report,
            "native_tpu_worker": deepcopy(self._native_tpu_worker_report),
        }

    def explain(
        self, *, format: Literal["dict", "json", "text"] = "dict",
        strict: bool = False, include_hardware: bool = False
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
            native_tpu = self.runtime.name == "torchtpu"
            report = OptimizationExplanation(
                backend=self.runtime.name,
                precision=self.runtime.precision,
                selected_path=("external_model" if self.loaded is None else "huggingface_model"),
                certification="unverified" if native_tpu else "compatible",
                capabilities=inspect_dense_causal_lm(
                    self.model,
                    source_provider=("unknown" if self.loaded is None else "huggingface"),
                ),
                limitations=(
                    (
                        "Native TorchTPU training is experimental and limited to "
                        "one visible device; distributed training and exact TPU "
                        "resume are not integrated. "
                        + (
                            "torch.compile was requested but remains unvalidated "
                            "on target hardware."
                            if self.args.torch_compile
                            else "torch.compile is disabled."
                        )
                    )
                    if native_tpu
                    else "No optimized provider execution plan has been selected.",
                ),
            )
        if strict:
            report.require_supported()
        if include_hardware:
            preflight = self.preflight()
            if format == "dict":
                return {**report.to_dict(), "preflight": preflight}
            if format == "json":
                return json.dumps(
                    {**report.to_dict(), "preflight": preflight},
                    indent=2, sort_keys=True,
                )
            if format == "text":
                native = preflight["native_tpu"]
                worker = preflight["native_tpu_worker"]
                return (
                    report.to_text()
                    + "\nNative TPU preflight: " + native["status"]
                    + " (training route not integrated; no allocation performed)"
                    + "\nNative TPU worker report: "
                    + (worker["status"] if worker is not None else "not available")
                )
        if format == "dict":
            return report.to_dict()
        if format == "json":
            return report.to_json()
        if format == "text":
            return report.to_text()
        raise ValueError("format must be 'dict', 'json', or 'text'.")


__all__ = [
    "DEPRECATED_CONFIG_KEYS",
    "PUBLIC_API_VERSION",
    "TrainLMTrainer",
    "TrainLMTrainingArguments",
]
