"""Public Hugging Face-like TrainLM training surface.

The classes in this module deliberately hide backend construction and the
training engine's internal lifecycle. TPU worker/coordinator integration can
be added behind the same API without changing user code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
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
from trainlm.checkpoint.retention import maintain_local_checkpoints
from trainlm.optimization import create_optimizer
from trainlm.pretraining import (
    TrainLMPretrainingConfig,
    _preflight_scratch_backend,
    _prepare_scratch_training,
    _tokenizer_cache_hashes,
)
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
_PERSISTED_TRAINER_STATE_FIELDS = frozenset(
    {
        "step",
        "micro_step",
        "epoch",
        "tokens_seen",
        "samples_seen",
        "global_batch_size",
        "learning_rate",
        "loss",
        "grad_norm",
    }
)


def _resolve_batches_per_epoch(train_dataset: Any, *, batch_size: int) -> int:
    """Count batches for finite map-style data or TrainLM's packed reader."""
    from trainlm.data import PackedBinDataset

    if isinstance(train_dataset, PackedBinDataset):
        if len(train_dataset) == 0:
            raise ValueError("Epoch budgets and schedules cannot use an empty dataset.")
        return math.ceil(len(train_dataset) / batch_size)
    if isinstance(train_dataset, DataLoader):
        if isinstance(train_dataset.dataset, IterableDataset):
            raise ValueError(
                "Epoch budgets and schedules require a finite map-style dataset; "
                "streaming IterableDataset loaders are not supported."
            )
        try:
            batches_per_epoch = len(train_dataset)
        except (TypeError, NotImplementedError, ValueError) as exc:
            raise ValueError(
                "Epoch budgets and schedules require a DataLoader with a finite length."
            ) from exc
    else:
        if (
            isinstance(train_dataset, IterableDataset)
            or not hasattr(train_dataset, "__len__")
            or not hasattr(train_dataset, "__getitem__")
        ):
            raise ValueError(
                "Epoch budgets and schedules require a finite map-style dataset; "
                "streaming datasets are not supported."
            )
        try:
            sample_count = len(train_dataset)
        except (TypeError, NotImplementedError, ValueError) as exc:
            raise ValueError(
                "Epoch budgets and schedules require a finite map-style dataset."
            ) from exc
        if sample_count < 1:
            raise ValueError("Epoch budgets and schedules cannot use an empty dataset.")
        batches_per_epoch = math.ceil(sample_count / batch_size)

    if batches_per_epoch < 1:
        raise ValueError("Epoch budgets and schedules cannot use an empty dataset.")
    return batches_per_epoch


def _is_iterable_dataset(dataset: Any) -> bool:
    """Recognize map-style and iterable datasets after DataLoader unwrapping."""
    return isinstance(dataset, IterableDataset) or (
        hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__")
    )


def _loader_dataset(source: Any) -> Any:
    return source.dataset if isinstance(source, DataLoader) else source


def _resolve_epoch_budget(
    train_dataset: Any,
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
) -> tuple[int, int, int]:
    """Resolve epochs to a finite batch and update budget for map-style data."""
    batches_per_epoch = _resolve_batches_per_epoch(
        train_dataset,
        batch_size=batch_size,
    )

    epoch_count = float(num_train_epochs)
    total_batch_budget = batches_per_epoch * epoch_count
    if not math.isfinite(total_batch_budget):
        raise ValueError("num_train_epochs produces an unrepresentable batch budget.")
    max_train_batches = math.ceil(total_batch_budget)
    full_epochs = int(epoch_count)
    partial_epoch_batches = max_train_batches - full_epochs * batches_per_epoch
    updates_per_full_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    effective_max_steps = full_epochs * updates_per_full_epoch
    if partial_epoch_batches:
        effective_max_steps += math.ceil(
            partial_epoch_batches / gradient_accumulation_steps
        )
    return batches_per_epoch, max_train_batches, effective_max_steps


def _checkpoint_component_type(value: Any) -> str:
    component_type = type(value)
    return f"{component_type.__module__}.{component_type.__qualname__}"


def _normalize_run_config_value(value: Any) -> Any:
    """Convert effective run settings to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Run configuration values must be finite.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(
                    "Run configuration mapping keys must be strings or integers."
                )
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError(
                    "Run configuration keys collide after JSON normalization."
                )
            normalized[normalized_key] = _normalize_run_config_value(item)
        return dict(sorted(normalized.items()))
    if isinstance(value, (list, tuple)):
        return [_normalize_run_config_value(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int)):
        return value.value
    raise TypeError(
        "Run configuration contains a value that cannot be represented as JSON: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _run_configuration_fingerprint(configuration: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_scratch_budget_extension(
    previous: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Allow only a strictly larger, internally consistent scratch token budget."""
    previous_configuration = previous.get("configuration")
    candidate_configuration = candidate.get("configuration")
    previous_fingerprint = previous.get("fingerprint")
    if (
        previous.get("schema_version") != 1
        or isinstance(previous.get("schema_version"), bool)
        or not isinstance(previous_configuration, Mapping)
        or not isinstance(previous_fingerprint, str)
        or previous_fingerprint
        != _run_configuration_fingerprint(previous_configuration)
        or not isinstance(candidate_configuration, Mapping)
    ):
        raise ValueError("The existing scratch run configuration is invalid.")

    normalized_previous = deepcopy(dict(previous_configuration))
    normalized_candidate = deepcopy(dict(candidate_configuration))
    mutable_paths = (
        ("pretraining", "max_tokens"),
        ("training_args", "max_tokens"),
        ("training_args", "max_steps"),
        ("training_args", "warmup_steps"),
        ("training_args", "save_steps"),
        ("training_args", "eval_steps"),
        ("scheduler", "training_horizon_steps"),
        ("scheduler", "warmup_steps"),
    )
    try:
        for path in mutable_paths:
            previous_parent = normalized_previous
            candidate_parent = normalized_candidate
            for key in path[:-1]:
                previous_parent = previous_parent[key]
                candidate_parent = candidate_parent[key]
            candidate_parent[path[-1]] = deepcopy(previous_parent[path[-1]])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "The scratch run configuration cannot be extended safely."
        ) from exc

    if normalized_candidate != normalized_previous:
        raise ValueError(
            "Budget extension can change only max_tokens and its derived step, "
            "warmup, save, and evaluation cadence. Model, data, optimizer, "
            "schedule, precision, and runtime settings must remain identical."
        )

    def budget_values(configuration: Mapping[str, Any]) -> tuple[int, int, int]:
        pretraining = configuration["pretraining"]
        training_args = configuration["training_args"]
        scheduler = configuration["scheduler"]
        if not all(
            isinstance(section, Mapping)
            for section in (pretraining, training_args, scheduler)
        ):
            raise ValueError("Budget extension configuration sections are invalid.")
        tokens = pretraining["max_tokens"]
        max_steps = training_args["max_steps"]
        warmup_steps = training_args["warmup_steps"]
        if (
            type(tokens) is not int
            or type(max_steps) is not int
            or type(warmup_steps) is not int
            or tokens < 1
            or max_steps < 1
            or warmup_steps < 0
            or training_args.get("max_tokens") != tokens
            or scheduler.get("training_horizon_steps") != max_steps
            or scheduler.get("schedule") != "cosine"
            or training_args.get("lr_scheduler_type") != "cosine"
        ):
            raise ValueError(
                "Budget extension requires a valid cosine pretraining run."
            )
        sequence_length = training_args.get("sequence_length")
        batch_size = training_args.get("per_device_train_batch_size")
        accumulation = training_args.get("gradient_accumulation_steps")
        if any(
            type(value) is not int or value < 1
            for value in (sequence_length, batch_size, accumulation)
        ):
            raise ValueError("Budget extension has invalid training geometry.")
        tokens_per_update = (sequence_length - 1) * batch_size * accumulation
        budget_rate = training_args.get("budget_supervised_tokens_per_update")
        if budget_rate is None:
            rate_numerator = tokens_per_update
            rate_denominator = 1
        else:
            if (
                not isinstance(budget_rate, Mapping)
                or type(budget_rate.get("numerator")) is not int
                or budget_rate["numerator"] < 1
                or type(budget_rate.get("denominator")) is not int
                or budget_rate["denominator"] < 1
            ):
                raise ValueError(
                    "Budget extension has an invalid supervised-token rate."
                )
            rate_numerator = budget_rate["numerator"]
            rate_denominator = budget_rate["denominator"]
        expected_steps = (
            tokens * rate_denominator + rate_numerator - 1
        ) // rate_numerator
        cadence = max(1, expected_steps // 10)
        expected_warmup = min(expected_steps // 10, 100)
        if (
            expected_steps != max_steps
            or training_args.get("save_steps") != cadence
            or training_args.get("eval_steps") != cadence
            or warmup_steps != expected_warmup
            or scheduler.get("warmup_steps") != expected_warmup
        ):
            raise ValueError("Budget extension has inconsistent derived settings.")
        return tokens, max_steps, warmup_steps

    old_tokens, old_steps, _ = budget_values(previous_configuration)
    new_tokens, new_steps, _ = budget_values(candidate_configuration)
    if new_tokens <= old_tokens or new_steps <= old_steps:
        raise ValueError(
            "extend_training_budget=True requires a strictly larger token budget "
            "and training horizon."
        )


def _latest_training_checkpoint(output_dir: Path) -> Path | None:
    """Find the newest atomically published local or native TPU checkpoint."""
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_symlink() or not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdecimal():
            continue
        local_state = path / "trainer_state.pt"
        tpu_manifest = path / "manifest.json"
        if (
            (local_state.is_file() and not local_state.is_symlink())
            or (tpu_manifest.is_file() and not tpu_manifest.is_symlink())
        ):
            candidates.append((int(suffix), path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


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
    Set ``max_steps=None`` with ``max_tokens`` for a supervised-token
    budget, or with ``num_train_epochs`` to train on a finite map-style
    dataset for a dataset-relative budget. On the legacy XLA coordinator,
    token budgets use exact global supervised-token counts, and epoch budgets
    repeat finite rank-local packed data. Epoch-based evaluation and checkpoint
    saving also use those finite dataset boundaries. These XLA paths are not
    hardware-verified. ``eval_strategy`` and
    ``save_strategy`` accept ``"no"``, ``"steps"``, or ``"epoch"``; omitted
    strategies preserve interval-based behavior. ``load_best_model_at_end``
    selects and restores the best evaluation checkpoint.
    ``early_stopping_patience`` can stop training after consecutive evaluation
    runs without a significant improvement.
    """

    output_dir: str | Path = Path("runs")
    max_steps: int | None = 1000
    max_tokens: int | None = None
    num_train_epochs: float | None = None
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
    gradient_checkpointing: bool = False
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
    save_total_limit: int | None = None
    save_strategy: Literal["no", "steps", "epoch"] | None = None
    eval_strategy: Literal["no", "steps", "epoch"] | None = None
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    early_stopping_patience: int | None = None
    early_stopping_threshold: float = 0.0
    max_grad_norm: float = 1.0
    attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2", "splash"
    ] = "auto"
    pack_gated_mlp: bool | None = None
    packing: bool = False
    optim: Literal[
        "adamw",
        "adamw_bf16",
        "adamw_torch",
        "adamw_8bit",
        "paged_adamw_8bit",
        "torchao_adamw_8bit",
    ] = "adamw"
    quantized_training: Literal["none", "fp8", "int8"] = "none"
    use_liger_kernel: bool | Literal["auto"] = "auto"
    use_fused_rms_norm: bool = True
    pad_vocab_to_multiple: int | Literal["auto"] | None = None
    loss_rematerialization: Literal["auto", "disabled", "per_chunk"] = "auto"
    tpu_allow_excess_precision: bool | Literal["auto"] = "auto"
    tpu_compile_cache_dir: str | Literal["auto"] | None = "auto"
    prefetch_batches: bool = True
    distributed_strategy: Literal["auto", "ddp", "fsdp2", "none"] = "auto"
    ddp_bucket_cap_mb: int = 50
    torch_compile_mode: Literal[
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ] = "default"

    def __post_init__(self) -> None:
        if isinstance(self.betas, list):
            object.__setattr__(self, "betas", tuple(self.betas))
        if (
            self.use_liger_kernel != "auto"
            and not isinstance(self.use_liger_kernel, bool)
        ):
            raise ValueError("use_liger_kernel must be boolean or 'auto'.")
        if not isinstance(self.use_fused_rms_norm, bool):
            raise ValueError("use_fused_rms_norm must be a boolean.")
        if self.pad_vocab_to_multiple not in {None, "auto"} and (
            isinstance(self.pad_vocab_to_multiple, bool)
            or not isinstance(self.pad_vocab_to_multiple, int)
            or self.pad_vocab_to_multiple < 1
        ):
            raise ValueError(
                "pad_vocab_to_multiple must be None, 'auto', or a positive integer."
            )
        if self.loss_rematerialization not in {"auto", "disabled", "per_chunk"}:
            raise ValueError(
                f"Unsupported loss_rematerialization: {self.loss_rematerialization}"
            )
        if (
            self.tpu_allow_excess_precision != "auto"
            and not isinstance(self.tpu_allow_excess_precision, bool)
        ):
            raise ValueError("tpu_allow_excess_precision must be boolean or 'auto'.")
        if self.tpu_compile_cache_dir is not None and (
            not isinstance(self.tpu_compile_cache_dir, str)
            or not self.tpu_compile_cache_dir.strip()
        ):
            raise ValueError("tpu_compile_cache_dir must be 'auto', a non-empty path, or None.")
        if not isinstance(self.prefetch_batches, bool):
            raise ValueError("prefetch_batches must be a boolean.")
        if self.distributed_strategy not in {"auto", "ddp", "fsdp2", "none"}:
            raise ValueError(
                f"Unsupported distributed_strategy: {self.distributed_strategy}"
            )
        if (
            isinstance(self.ddp_bucket_cap_mb, bool)
            or not isinstance(self.ddp_bucket_cap_mb, int)
            or self.ddp_bucket_cap_mb < 1
        ):
            raise ValueError("ddp_bucket_cap_mb must be a positive integer.")
        if self.torch_compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(
                f"Unsupported torch_compile_mode: {self.torch_compile_mode}"
            )
        if self.quantized_training not in {"none", "fp8", "int8"}:
            raise ValueError(
                f"Unsupported quantized_training mode: {self.quantized_training}"
            )
        if self.optim not in {
            "adamw",
            "adamw_bf16",
            "adamw_torch",
            "adamw_8bit",
            "paged_adamw_8bit",
            "torchao_adamw_8bit",
        }:
            raise ValueError(f"Unsupported optimizer: {self.optim}")
        if self.attn_implementation not in {
            "auto",
            "eager",
            "sdpa",
            "flash_attention_2",
            "splash",
        }:
            raise ValueError(
                f"Unsupported attn_implementation: {self.attn_implementation}"
            )
        if self.pack_gated_mlp is not None and not isinstance(
            self.pack_gated_mlp, bool
        ):
            raise ValueError("pack_gated_mlp must be boolean or None.")
        if not isinstance(self.packing, bool):
            raise ValueError("packing must be boolean.")
        for name in ("save_strategy", "eval_strategy"):
            strategy = getattr(self, name)
            if strategy is not None and strategy not in {"no", "steps", "epoch"}:
                raise ValueError(f"{name} must be 'no', 'steps', or 'epoch'.")
        if self.num_train_epochs is not None:
            try:
                finite_epoch_count = math.isfinite(float(self.num_train_epochs))
            except (OverflowError, TypeError, ValueError):
                finite_epoch_count = False
            if (
                isinstance(self.num_train_epochs, bool)
                or not isinstance(self.num_train_epochs, Real)
                or not finite_epoch_count
                or self.num_train_epochs <= 0
            ):
                raise ValueError("num_train_epochs must be positive and finite.")
            if self.max_steps is not None:
                raise ValueError(
                    "Set max_steps=None when using num_train_epochs."
                )
        if (
            self.max_steps is None
            and self.max_tokens is None
            and self.num_train_epochs is None
        ):
            raise ValueError("Set max_steps, max_tokens, or num_train_epochs.")
        for name in (
            "max_steps",
            "max_tokens",
            "save_steps",
            "save_total_limit",
            "eval_steps",
            "max_eval_batches",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be positive when configured.")
        for strategy_name, interval_name in (
            ("save_strategy", "save_steps"),
            ("eval_strategy", "eval_steps"),
        ):
            strategy = getattr(self, strategy_name)
            interval = getattr(self, interval_name)
            if strategy == "steps" and interval is None:
                raise ValueError(
                    f"{interval_name} is required when {strategy_name}='steps'."
                )
            if strategy in {"no", "epoch"} and interval is not None:
                raise ValueError(
                    f"{interval_name} must be omitted when {strategy_name}="
                    f"{strategy!r}."
                )
        if not isinstance(self.load_best_model_at_end, bool):
            raise ValueError("load_best_model_at_end must be boolean.")
        if self.metric_for_best_model is not None and (
            not isinstance(self.metric_for_best_model, str)
            or not self.metric_for_best_model.strip()
        ):
            raise ValueError("metric_for_best_model must be a non-empty string.")
        if self.greater_is_better is not None and not isinstance(
            self.greater_is_better, bool
        ):
            raise ValueError("greater_is_better must be boolean when configured.")
        if self.load_best_model_at_end:
            eval_strategy = self.resolved_eval_strategy
            save_strategy = self.resolved_save_strategy
            if eval_strategy == "no":
                raise ValueError(
                    "load_best_model_at_end requires an active eval_strategy."
                )
            if save_strategy == "no" or save_strategy != eval_strategy:
                raise ValueError(
                    "load_best_model_at_end requires matching, active eval_strategy "
                    "and save_strategy values."
                )
            if (
                eval_strategy == "steps"
                and self.save_steps % self.eval_steps != 0
            ):
                raise ValueError(
                    "save_steps must be a multiple of eval_steps when "
                    "load_best_model_at_end is enabled."
                )
        elif (
            self.metric_for_best_model is not None
            or self.greater_is_better is not None
        ):
            raise ValueError(
                "metric_for_best_model and greater_is_better require "
                "load_best_model_at_end=True."
            )
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be a positive integer.")
        if (
            isinstance(self.early_stopping_threshold, bool)
            or not isinstance(self.early_stopping_threshold, Real)
            or not _is_finite_real(self.early_stopping_threshold)
            or self.early_stopping_threshold < 0
        ):
            raise ValueError(
                "early_stopping_threshold must be a finite, non-negative number."
            )
        if (
            self.early_stopping_patience is not None
            and not self.load_best_model_at_end
        ):
            raise ValueError(
                "early_stopping_patience requires load_best_model_at_end=True."
            )
        if (
            self.early_stopping_patience is None
            and self.early_stopping_threshold != 0
        ):
            raise ValueError(
                "early_stopping_threshold requires early_stopping_patience."
            )
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
        if self.accelerator == "tpu" and self.save_total_limit is not None:
            raise ValueError(
                "save_total_limit supports single-device CPU, CUDA, and native_tpu "
                "training; retention for the legacy XLA coordinator is not supported."
            )
        if self.torch_compile and self.accelerator not in {"native_tpu", "cuda", "auto"}:
            raise ValueError(
                "torch_compile is currently supported only with "
                "accelerator='native_tpu', 'cuda', or 'auto'."
            )
        if self.lr_scheduler_type not in {"constant", "linear", "cosine", "wsd"}:
            raise ValueError(f"Unsupported scheduler: {self.lr_scheduler_type}")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(
                not _is_finite_real(beta) or not 0 <= beta < 1
                for beta in self.betas
            )
        ):
            raise ValueError("betas must contain two finite values in [0, 1).")
        if not _is_finite_real(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive.")
        if not _is_finite_real(self.max_grad_norm) or self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and non-negative.")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a boolean.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be non-negative.")
        reporters = (self.report_to,) if isinstance(self.report_to, str) else self.report_to
        if not isinstance(reporters, tuple) or any(
            reporter not in {"none", "jsonl", "tensorboard", "wandb"}
            for reporter in reporters
        ):
            raise ValueError("report_to supports none, jsonl, tensorboard, and wandb.")
        if len(reporters) > 1 and "none" in reporters:
            raise ValueError("report_to='none' cannot be combined with reporters.")
        if len(set(reporters)) != len(reporters):
            raise ValueError("report_to cannot contain duplicate reporters.")
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

    @property
    def resolved_save_strategy(self) -> Literal["no", "steps", "epoch"]:
        """Return save policy, inferring the legacy step policy when omitted."""
        return self.save_strategy or ("steps" if self.save_steps is not None else "no")

    @property
    def resolved_eval_strategy(self) -> Literal["no", "steps", "epoch"]:
        """Return evaluation policy, inferring the legacy step policy when omitted."""
        return self.eval_strategy or ("steps" if self.eval_steps is not None else "no")


def _enable_model_gradient_checkpointing(model: nn.Module) -> None:
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise ValueError("The selected model does not support gradient checkpointing.")
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    enable()


class _MetricsReporterCallback(TrainerCallback):
    """Write host-materialized metric snapshots to selected integrations."""

    def __init__(self, output_dir: Path, reporters: Sequence[str]) -> None:
        self.output_dir = output_dir
        self.reporters = tuple(name for name in reporters if name != "none")
        self.path = output_dir / "metrics.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard = None
        self.wandb_run = None
        if "tensorboard" in self.reporters:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "report_to='tensorboard' requires tensorboard. Install it with "
                    "'pip install tensorboard'."
                ) from exc
            self.tensorboard = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if "wandb" in self.reporters:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "report_to='wandb' requires wandb. Install it with 'pip install wandb'."
                ) from exc
            self.wandb_run = wandb.init(project="trainlm", dir=str(output_dir), reinit=True)

    def on_metrics(self, state, control, metrics: Mapping[str, float]) -> None:
        del control
        row = {"step": int(state.step), "phase": state.phase.value,
               "metrics": {name: float(value) for name, value in metrics.items()}}
        if "jsonl" in self.reporters:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if self.tensorboard is not None:
            for name, value in row["metrics"].items():
                self.tensorboard.add_scalar(f"{row['phase']}/{name}", value, row["step"])
        if self.wandb_run is not None:
            self.wandb_run.log(
                {f"{row['phase']}/{name}": value for name, value in row["metrics"].items()},
                step=row["step"],
            )

    def on_train_end(self, state, control) -> None:
        del state, control
        if self.tensorboard is not None:
            self.tensorboard.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


class _ConsoleLoggingCallback(TrainerCallback):
    """Print sparse step loss, throughput, and MFU when verbosity != 'quiet'."""

    def __init__(self, *, total_steps: int | None) -> None:
        self._total_steps = total_steps
        self._last_logged_step: int | None = None

    def on_metrics(self, state, control, metrics: Mapping[str, float]) -> None:
        del control
        if state.phase == TrainerPhase.EVALUATING:
            parts = [f"eval step {int(state.step)}"]
            if "eval_loss" in metrics:
                parts.append(f"eval_loss={metrics['eval_loss']:.4f}")
            if "eval_perplexity" in metrics:
                parts.append(f"ppl={metrics['eval_perplexity']:.2f}")
            print(" | ".join(parts), flush=True)
            return
        if "loss" not in metrics:
            return
        step = int(state.step)
        if step == self._last_logged_step:
            return
        self._last_logged_step = step
        step_label = f"step {step}/{self._total_steps}" if self._total_steps else f"step {step}"
        parts = [step_label, f"loss={metrics['loss']:.4f}"]
        tps = metrics.get("train_tokens_per_second")
        if tps is not None:
            parts.append(f"{tps:,.1f} tok/s")
        mfu = metrics.get("non_embedding_mfu")
        if mfu is not None:
            parts.append(f"MFU {mfu:.2%}")
        wait = metrics.get("dataloader_wait_fraction")
        if wait is not None:
            parts.append(f"wait {wait:.1%}")
        print(" | ".join(parts), flush=True)


class _BestModelSelectionCallback(TrainerCallback):
    """Select best weights, save their checkpoint, and stop after a plateau."""

    def __init__(
        self,
        *,
        metric_name: str,
        greater_is_better: bool,
        early_stopping_patience: int | None = None,
        early_stopping_threshold: float = 0.0,
    ) -> None:
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric: float | None = None
        self.best_step: int | None = None
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = float(early_stopping_threshold)
        self.early_stopping_best_metric: float | None = None
        self.early_stopping_wait = 0

    def on_metrics(self, state, control, metrics: Mapping[str, float]) -> None:
        if state.phase != TrainerPhase.EVALUATING:
            return
        if self.metric_name not in metrics:
            raise ValueError(
                f"Evaluation did not return metric {self.metric_name!r}; "
                f"available metrics: {', '.join(metrics)}."
            )
        try:
            value = float(metrics[self.metric_name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Best-model metric {self.metric_name!r} must be a finite number."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Best-model metric {self.metric_name!r} must be a finite number."
            )
        improved = self.best_metric is None or (
            value > self.best_metric
            if self.greater_is_better
            else value < self.best_metric
        )
        if improved:
            self.best_metric = value
            self.best_step = state.step
            control.should_save_checkpoint = True
        if self.early_stopping_patience is not None:
            if self.early_stopping_best_metric is None:
                self.early_stopping_best_metric = value
            else:
                significant_improvement = (
                    value
                    > self.early_stopping_best_metric
                    + self.early_stopping_threshold
                    if self.greater_is_better
                    else value
                    < self.early_stopping_best_metric
                    - self.early_stopping_threshold
                )
                if significant_improvement:
                    self.early_stopping_best_metric = value
                    self.early_stopping_wait = 0
                else:
                    self.early_stopping_wait += 1
                    if self.should_stop:
                        control.request_stop()

    @property
    def should_stop(self) -> bool:
        """Whether the configured patience has been exhausted."""
        return (
            self.early_stopping_patience is not None
            and self.early_stopping_wait >= self.early_stopping_patience
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "metric_name": self.metric_name,
            "greater_is_better": self.greater_is_better,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "early_stopping_best_metric": self.early_stopping_best_metric,
            "early_stopping_wait": self.early_stopping_wait,
        }

    def validate_state_dict(
        self,
        state: object,
        *,
        checkpoint_step: int,
    ) -> tuple[float | None, int | None]:
        if not isinstance(state, Mapping):
            raise ValueError("Best-model checkpoint metadata must be a mapping.")
        if (
            state.get("schema_version") != 1
            or isinstance(state.get("schema_version"), bool)
            or state.get("metric_name") != self.metric_name
            or state.get("greater_is_better") is not self.greater_is_better
        ):
            raise ValueError(
                "Best-model selection settings do not match the checkpoint."
            )
        metric = state.get("best_metric")
        step = state.get("best_step")
        if metric is None or step is None:
            if metric is not None or step is not None:
                raise ValueError("Best-model checkpoint metadata is incomplete.")
        elif (
            isinstance(metric, bool)
            or not isinstance(metric, Real)
            or not _is_finite_real(metric)
            or type(step) is not int
            or step < 1
            or step > checkpoint_step
        ):
            raise ValueError("Best-model checkpoint metadata is invalid.")
        early_metric = state.get("early_stopping_best_metric", metric)
        early_wait = state.get("early_stopping_wait", 0)
        if early_metric is not None and (
            isinstance(early_metric, bool)
            or not isinstance(early_metric, Real)
            or not _is_finite_real(early_metric)
        ):
            raise ValueError("Early-stopping checkpoint metric is invalid.")
        if metric is None and early_metric is not None:
            raise ValueError(
                "Early-stopping checkpoint metadata has no selected metric."
            )
        if (
            isinstance(early_wait, bool)
            or not isinstance(early_wait, int)
            or early_wait < 0
            or (early_metric is None and early_wait != 0)
        ):
            raise ValueError("Early-stopping checkpoint patience state is invalid.")
        if metric is None:
            return None, None
        return float(metric), step

    def load_state_dict(self, state: object, *, checkpoint_step: int) -> None:
        self.best_metric, self.best_step = self.validate_state_dict(
            state,
            checkpoint_step=checkpoint_step,
        )
        if not isinstance(state, Mapping):
            raise ValueError("Best-model checkpoint metadata must be a mapping.")
        saved_early_metric = state.get(
            "early_stopping_best_metric", self.best_metric
        )
        self.early_stopping_best_metric = (
            float(saved_early_metric)
            if saved_early_metric is not None
            else None
        )
        self.early_stopping_wait = state.get("early_stopping_wait", 0)


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


def _auto_training_view(model: nn.Module):
    base_model = getattr(model, "base_model", None)
    if not isinstance(base_model, nn.Module) or base_model is model:
        return None
    from trainlm.model.dispatch import ForwardBatchDispatcher
    from trainlm.optimization import inspect_dense_causal_lm
    from trainlm.tasks import LinearCausalLMTrainingView, extract_hidden_state

    capabilities = inspect_dense_causal_lm(model)
    if capabilities.lm_head.status not in {"known", "inferred"} or capabilities.lm_head.kind != "linear":
        return None
    dispatcher = ForwardBatchDispatcher.from_model(base_model)
    return LinearCausalLMTrainingView(
        model=model,
        capabilities=capabilities,
        hidden_state_provider=lambda cur, batch: extract_hidden_state(
            cur.base_model(**dispatcher.dispatch(batch).inputs)
        ),
        provider_id="hf.base_model_hidden",
    )


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
    def from_scratch(
        cls,
        config: TrainLMPretrainingConfig,
        *,
        train_dataset: Any | None = None,
        eval_dataset: Any | None = None,
        formatting_func: Callable[[Mapping[str, Any]], str] | None = None,
        callbacks: Sequence[TrainerCallback] | None = None,
        extend_training_budget: bool = False,
    ) -> "TrainLMTrainer":
        """Prepare a tokenizer/model from Hub text or supplied HF Datasets.

        The method keeps the familiar ``trainer.train()``, ``evaluate()``, and
        ``save_model()`` lifecycle while owning streaming tokenization and
        packed-data preparation. Set ``dataset_source="provided"`` and pass
        ``train_dataset`` and ``eval_dataset`` objects to prepare HF Dataset
        splits without loading them again. Use ``data_format="tokens"`` for
        pretokenized ``input_ids`` with optional aligned labels and loss masks;
        TrainLM preserves them and applies the next-token shift once. For raw
        rows with multiple columns, pass ``formatting_func`` and set its stable
        ``text_formatter_name`` in the config; pass the same pair on resume.
        Matching inputs reuse integrity-checked tokenizer and packed-shard artifacts in
        ``config.output_dir``. Native TorchTPU must be explicitly selected and
        remains experimental. Set
        ``extend_training_budget=True`` only when resuming an existing scratch
        run with a larger ``max_tokens``; model and optimizer state are kept,
        then a new cosine schedule starts at the configured learning rate.
        """
        if not isinstance(config, TrainLMPretrainingConfig):
            raise TypeError("config must be a TrainLMPretrainingConfig instance.")
        if not isinstance(extend_training_budget, bool):
            raise TypeError("extend_training_budget must be a boolean.")
        _preflight_scratch_backend(config)
        model, tokenizer, train_dataset, eval_dataset = _prepare_scratch_training(
            config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
        )
        tokens_per_update = (
            (config.sequence_length - 1)
            * config.per_device_train_batch_size
            * config.gradient_accumulation_steps
        )
        budget_token_rate = None
        count_supervised = getattr(
            train_dataset, "_count_supervised_tokens_per_epoch", None
        )
        if callable(count_supervised):
            try:
                supervised_per_epoch = count_supervised()
            except Exception:
                train_dataset.close()
                eval_dataset.close()
                raise
            if supervised_per_epoch < 1:
                train_dataset.close()
                eval_dataset.close()
                raise ValueError(
                    "The packed training dataset contains no supervised next-token "
                    "targets after applying labels and loss masks."
                )
            if not config.repeat_train_data and supervised_per_epoch < config.max_tokens:
                train_dataset.close()
                eval_dataset.close()
                raise ValueError(
                    f"One pass provides {supervised_per_epoch} supervised next-token "
                    f"targets for a budget of {config.max_tokens}. Add training data, "
                    "lower max_tokens, or set repeat_train_data=True."
                )
            nominal_per_epoch = len(train_dataset) * (config.sequence_length - 1)
            if supervised_per_epoch != nominal_per_epoch:
                loader_batches_per_epoch = math.ceil(
                    len(train_dataset) / config.per_device_train_batch_size
                )
                budget_token_rate = {
                    "numerator": (
                        supervised_per_epoch
                        * config.gradient_accumulation_steps
                    ),
                    "denominator": loader_batches_per_epoch,
                }
                max_steps = (
                    config.max_tokens * budget_token_rate["denominator"]
                    + budget_token_rate["numerator"] - 1
                ) // budget_token_rate["numerator"]
            else:
                max_steps = math.ceil(config.max_tokens / tokens_per_update)
        else:
            max_steps = math.ceil(config.max_tokens / tokens_per_update)
        cadence = max(1, max_steps // 10)
        args = TrainLMTrainingArguments(
            output_dir=Path(config.output_dir) / "checkpoints",
            max_steps=max_steps,
            max_tokens=config.max_tokens,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            sequence_length=config.sequence_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_steps=(
                config.warmup_steps
                if config.warmup_steps is not None
                else (
                    math.ceil(max_steps * config.warmup_ratio)
                    if config.warmup_ratio is not None
                    else min(max_steps // 10, 100)
                )
            ),
            bf16=config.bf16,
            accelerator=config.accelerator,
            torch_compile=config.torch_compile,
            logging_steps=config.logging_steps,
            save_steps=(
                config.save_steps if config.save_steps is not None else cadence
            ),
            save_total_limit=config.save_total_limit,
            eval_steps=(
                config.eval_steps if config.eval_steps is not None else cadence
            ),
            max_eval_batches=config.max_eval_batches,
            seed=config.seed,
            weight_decay=config.weight_decay,
            load_best_model_at_end=config.load_best_model_at_end,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
            betas=config.betas,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
        )
        try:
            trainer = cls(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=callbacks,
            )
        except Exception:
            train_dataset.close()
            eval_dataset.close()
            raise
        trainer.pretraining_config = config
        trainer._scratch_budget_token_rate = budget_token_rate
        try:
            trainer._initialize_scratch_run_configuration(
                config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                extend_training_budget=extend_training_budget,
            )
        except Exception:
            train_dataset.close()
            eval_dataset.close()
            raise
        return trainer

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
        training_view_factory: Callable[[nn.Module], Any] | None = None,
        train_dataset_identity: str | None = None,
        peft_config: Any | None = None,
    ) -> None:
        self.args = args or TrainLMTrainingArguments()
        from trainlm.data import PackedBinDataset

        if (isinstance(train_dataset, PackedBinDataset)
                and self.args.dataloader_num_workers
                and (self.args.num_train_epochs is not None
                     or "epoch" in (self.args.resolved_eval_strategy, self.args.resolved_save_strategy))):
            raise ValueError("Packed epoch schedules require dataloader_num_workers=0.")
        self._uses_epoch_schedules = (
            self.args.resolved_save_strategy == "epoch"
            or self.args.resolved_eval_strategy == "epoch"
        )
        self._best_model_callback: _BestModelSelectionCallback | None = None
        if self.args.load_best_model_at_end:
            metric_name = self.args.metric_for_best_model or "eval_loss"
            if not metric_name.startswith("eval_"):
                metric_name = f"eval_{metric_name}"
            greater_is_better = self.args.greater_is_better
            if greater_is_better is None:
                greater_is_better = not metric_name.endswith("loss")
            self._best_model_callback = _BestModelSelectionCallback(
                metric_name=metric_name,
                greater_is_better=greater_is_better,
                early_stopping_patience=self.args.early_stopping_patience,
                early_stopping_threshold=self.args.early_stopping_threshold,
            )
        self._train_batches_per_epoch: int | None = None
        self._max_train_batches: int | None = None
        self._effective_max_steps = self.args.max_steps
        if (
            self.args.num_train_epochs is not None
            and self.args.accelerator != "tpu"
        ):
            (
                self._train_batches_per_epoch,
                self._max_train_batches,
                self._effective_max_steps,
            ) = _resolve_epoch_budget(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                gradient_accumulation_steps=(
                    self.args.gradient_accumulation_steps
                ),
                num_train_epochs=self.args.num_train_epochs,
            )
        if (
            self._uses_epoch_schedules
            and self._train_batches_per_epoch is None
            and self.args.accelerator != "tpu"
        ):
            self._train_batches_per_epoch = _resolve_batches_per_epoch(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
            )
        if tokenizer is not None and processing_class is not None:
            raise ValueError("Set either tokenizer or processing_class, not both.")
        self.processing_class = processing_class if processing_class is not None else tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if train_dataset_identity is not None and (
            not isinstance(train_dataset_identity, str)
            or not train_dataset_identity.strip()
        ):
            raise ValueError("train_dataset_identity must be a non-empty string.")
        identity_dataset = _loader_dataset(train_dataset)
        self._train_dataset_identity = train_dataset_identity
        self._train_dataset_fingerprint = getattr(
            identity_dataset, "_fingerprint", None
        )
        if self._train_dataset_fingerprint is not None and (
            not isinstance(self._train_dataset_fingerprint, str)
            or not self._train_dataset_fingerprint.strip()
        ):
            self._train_dataset_fingerprint = None
        self.data_collator = data_collator or _default_collator
        self.callbacks = tuple(callbacks or ())
        if self.args.resolved_eval_strategy != "no" and eval_dataset is None:
            raise ValueError("An active eval_strategy requires eval_dataset.")
        if self.args.max_eval_batches is not None and eval_dataset is None:
            raise ValueError("max_eval_batches requires eval_dataset.")
        self._model_source: ModelSourceConfig | None = None
        self.loaded: LoadedCausalLM | None = None
        self._last_metrics: dict[str, Any] = {}
        self._native_tpu_worker_report: dict[str, Any] | None = None
        self._run_configuration_manifest: dict[str, Any] | None = None
        self._pending_scratch_run_configuration: dict[str, Any] | None = None
        self._scratch_budget_extension_checkpoint = False
        self._scratch_budget_token_rate: dict[str, int] | None = None
        self._train_loader_generator: torch.Generator | None = None
        self._train_loader_generator_initial_state: torch.Tensor | None = None
        self._tpu_coordinator = None
        if self.args.accelerator == "tpu":
            if self.args.load_best_model_at_end:
                raise NotImplementedError(
                    "load_best_model_at_end is not supported by the legacy "
                    "PyTorch/XLA coordinator."
                )
            if any(value is not None for value in (optimizer, scheduler, runtime)):
                raise ValueError(
                    "TPU execution constructs runtime, optimizer, and scheduler inside "
                    "each worker; do not pass local instances."
                )
            self._model_source = self._coerce_model_source(model)
            if self.args.fp16:
                raise ValueError("TPU execution supports fp32 or bf16, not fp16.")
            from trainlm._tpu_coordinator import _TPUCoordinator

            self._tpu_coordinator = _TPUCoordinator()
            if self.args.gradient_checkpointing:
                raise NotImplementedError(
                    "gradient_checkpointing is not yet exposed through the legacy XLA coordinator."
                )
            self.model = None
            self.runtime = None
            self.optimizer = None
            self.scheduler = None
            self.engine = None
            return
        if self.args.torch_compile and runtime is not None:
            raise ValueError(
                "torch_compile cannot be combined with an injected runtime; "
                "select accelerator='native_tpu' so TrainLM can configure it."
            )
        self.model = self._resolve_model(model)
        if peft_config is not None and not hasattr(self.model, "peft_config"):
            try:
                from peft import get_peft_model  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "peft_config requires the 'peft' package (pip install peft)."
                ) from exc
            self.model = get_peft_model(self.model, peft_config)
        self.training_view = None
        if self.args.loss_implementation == "chunked_linear":
            if training_view_factory is not None:
                self.training_view = training_view_factory(self.model)
                from trainlm.tasks import LinearCausalLMTrainingView
                if not isinstance(self.training_view, LinearCausalLMTrainingView):
                    raise TypeError("training_view_factory must return LinearCausalLMTrainingView.")
                if self.training_view.model is not self.model:
                    raise ValueError("The training view must reference the trainer's model.")
            else:
                self.training_view = _auto_training_view(self.model)
                if self.training_view is None:
                    raise ValueError(
                        "chunked_linear requires training_view_factory so hidden-state "
                        "extraction is explicit for this model."
                    )
        elif training_view_factory is not None:
            raise ValueError("training_view_factory applies only to chunked_linear loss.")
        if self.args.gradient_checkpointing:
            _enable_model_gradient_checkpointing(self.model)
        self.runtime = runtime or self._make_runtime()
        if self.args.num_train_epochs is not None and self.runtime.is_distributed:
            raise NotImplementedError(
                "num_train_epochs currently requires a single-device runtime; "
                "distributed epoch accounting is not available yet."
            )
        if self._uses_epoch_schedules and self.runtime.is_distributed:
            raise NotImplementedError(
                "Epoch-based evaluation and saving currently require a "
                "single-device runtime."
            )
        if self.args.load_best_model_at_end and self.runtime.is_distributed:
            raise NotImplementedError(
                "load_best_model_at_end currently requires a single-device runtime."
            )
        if self.args.save_total_limit is not None and self.runtime.is_distributed:
            raise ValueError("save_total_limit requires a single-device trainer.")
        self._kernel_plan = None
        self._apply_kernel_and_attention_plan(optimizer_injected=optimizer is not None)
        if optimizer is None:
            self._place_model_on_runtime_device()
        self._distributed_wrap_result = None
        if self.args.distributed_strategy in {"ddp", "fsdp2"} or (
            self.args.distributed_strategy == "auto"
            and self.runtime.is_distributed
            and getattr(self.runtime, "name", None) == "torch"
        ):
            from trainlm.runtime import prepare_distributed_model

            self.model, self._distributed_wrap_result = prepare_distributed_model(
                self.model,
                strategy=self.args.distributed_strategy,
                device=getattr(self.runtime, "device", None),
                bucket_cap_mb=self.args.ddp_bucket_cap_mb,
            )
        self.optimizer = optimizer or self._make_optimizer()
        self.scheduler = scheduler or self._make_scheduler()
        train_loader = self._make_loader(
            train_dataset, self.args.per_device_train_batch_size, is_train=True
        )
        eval_loader = (
            self._make_loader(
                eval_dataset, self.args.per_device_eval_batch_size, is_train=False
            )
            if eval_dataset is not None
            else None
        )
        if self._uses_epoch_schedules:
            try:
                batches_per_epoch = len(train_loader)
            except (TypeError, NotImplementedError, ValueError) as exc:
                raise ValueError(
                    "Epoch-based evaluation and saving require a finite "
                    "training dataset."
                ) from exc
            if batches_per_epoch < 1:
                raise ValueError(
                    "Epoch-based evaluation and saving cannot use an empty dataset."
                )
            if (
                self._train_batches_per_epoch is not None
                and self._train_batches_per_epoch != batches_per_epoch
            ):
                raise ValueError(
                    "The resolved epoch batch count does not match the training loader."
                )
            self._train_batches_per_epoch = batches_per_epoch
        if self.args.loss_rematerialization == "auto":
            resolved_remat = (
                "per_chunk"
                if (
                    self.args.loss_implementation == "chunked_linear"
                    and int(getattr(self.args, "sequence_length", 1024)) >= 1024
                )
                else "disabled"
            )
        else:
            resolved_remat = self.args.loss_rematerialization
        self.engine = EngineTrainer(
            config=self._make_config(),
            model=self.model,
            runtime=self.runtime,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            task=CausalLMTask(
                loss_implementation=self.args.loss_implementation,
                training_view=self.training_view,
                logits_chunk_size=self.args.logits_chunk_size or 2048,
                rematerialization=resolved_remat,
            ),
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            callbacks=(
                self.callbacks
                + ((_ConsoleLoggingCallback(total_steps=self._effective_max_steps),)
                   if self.args.logging_verbosity != "quiet"
                   and getattr(self.runtime, "is_primary_process", True)
                   else ())
                + ((_MetricsReporterCallback(
                    Path(self.args.output_dir),
                    (self.args.report_to,) if isinstance(self.args.report_to, str)
                    else self.args.report_to,
                ),) if self.args.report_to != "none" else ())
                + ((self._best_model_callback,)
                   if self._best_model_callback is not None else ())
            ),
            checkpoint_saver=self._save_training_checkpoint,
            checkpoint_loader=self._load_training_checkpoint,
        )
        runtime_dev = getattr(self.runtime, "device", None)
        self.engine._prefetch_batches = bool(
            self.args.prefetch_batches
            and getattr(runtime_dev, "type", "cpu") in {"cuda", "tpu"}
        )
        if (
            self.runtime.name == "torchtpu"
            and self.args.resolved_save_strategy != "no"
        ):
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
            try:
                import torch_tpu  # noqa: F401 - register the TPU for discovery
            except ModuleNotFoundError as exc:
                if exc.name != "torch_tpu":
                    raise
            report = discover_tpu(
                torch, verify_allocation=False, include_properties=False
            )
            accelerator = select_auto_backend(
                report, cuda_available=torch.cuda.is_available()
            )
        if accelerator == "native_tpu":
            try:
                import torch_tpu  # noqa: F401 - register the native TPU device
            except ModuleNotFoundError as exc:
                if exc.name != "torch_tpu":
                    raise
                raise RuntimeError(
                    "accelerator='native_tpu' requires the torch-tpu package."
                ) from exc
            from trainlm.runtime.torchtpu import TorchTPURuntime

            precision = "bf16" if self.args.bf16 else "fp16" if self.args.fp16 else "fp32"
            allow_excess = (
                True
                if (self.args.tpu_allow_excess_precision == "auto" and precision == "bf16")
                else (
                    self.args.tpu_allow_excess_precision
                    if isinstance(self.args.tpu_allow_excess_precision, bool)
                    else None
                )
            )
            cache_dir = (
                str(Path.home() / ".cache" / "trainlm" / "tpu_compile")
                if self.args.tpu_compile_cache_dir == "auto"
                else self.args.tpu_compile_cache_dir
            )
            return TorchTPURuntime(
                precision=precision,
                seed=self.args.seed,
                compile_enabled=self.args.torch_compile,
                allow_excess_precision=allow_excess,
                compile_cache_dir=cache_dir,
            )
        if accelerator == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("accelerator='cuda' requested but CUDA is unavailable.")
        if self.args.torch_compile and accelerator == "cpu":
            raise ValueError("torch_compile is not supported with accelerator='cpu'.")
        precision = "bf16" if self.args.bf16 else "fp16" if self.args.fp16 else "fp32"
        return Runtime(
            device=accelerator,
            precision=precision,
            compile_enabled=self.args.torch_compile,
            compile_mode=self.args.torch_compile_mode,
        )

    def _place_model_on_runtime_device(self) -> None:
        runtime_device = getattr(self.runtime, "device", None)
        if not isinstance(runtime_device, torch.device):
            return
        if runtime_device.type == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(runtime_device)

    def _apply_kernel_and_attention_plan(self, *, optimizer_injected: bool) -> None:
        from trainlm.optimization import (
            TorchTPUSplashAttentionRuntime,
            apply_torchtpu_kernel_plan,
            probe_torchtpu_splash_kernel,
        )

        # 0. Hardware tile alignment inspection and optional 128-multiple vocab padding.
        from trainlm.optimization import (
            inspect_model_alignment,
            pad_vocab_to_multiple as _pad_vocab,
        )

        runtime_device = getattr(self.runtime, "device", None)
        device_type = getattr(runtime_device, "type", "cpu")
        if self.args.pad_vocab_to_multiple == "auto" and device_type in {"cuda", "tpu"}:
            _pad_vocab(self.model, 128)
        elif isinstance(self.args.pad_vocab_to_multiple, int):
            _pad_vocab(self.model, self.args.pad_vocab_to_multiple)
        self._alignment_report = inspect_model_alignment(
            self.model,
            sequence_length=getattr(self.args, "max_seq_length", None),
            alignment=128,
        )

        # 1. Attention implementation selection (reversible).
        attn_choice = self.args.attn_implementation
        if attn_choice == "auto":
            if device_type == "cuda":
                import importlib.util

                has_flash = importlib.util.find_spec("flash_attn") is not None
                if has_flash and (self.args.bf16 or self.args.fp16):
                    _try_set_attn_implementation(self.model, "flash_attention_2")
                else:
                    _try_set_attn_implementation(self.model, "sdpa")
        elif attn_choice in {"eager", "sdpa", "flash_attention_2"}:
            _try_set_attn_implementation(self.model, attn_choice, strict=True)

        # 1b. Optional Liger-Kernel Triton fusion (auto on CUDA when installed).
        self._liger_result = None
        if self.args.use_liger_kernel is True or (
            self.args.use_liger_kernel == "auto" and device_type == "cuda"
        ):
            from trainlm.optimization import apply_liger_kernels

            self._liger_result = apply_liger_kernels(
                self.model,
                fuse_linear_cross_entropy=(
                    self.args.loss_implementation in {"auto", "model"}
                ),
                strict=(self.args.use_liger_kernel is True),
            )

        # 1c. Reversible Triton (CUDA) / F.rms_norm replacement when Liger RMSNorm was not applied.
        self._rms_norm_result = None
        if self.args.use_fused_rms_norm and not (
            self._liger_result is not None and self._liger_result.fused_rms_norm
        ):
            if device_type == "cuda":
                from trainlm.kernels.triton import (
                    apply_triton_rms_norm,
                    is_triton_available,
                )

                if is_triton_available():
                    self._rms_norm_result = apply_triton_rms_norm(self.model)
            if self._rms_norm_result is None or not self._rms_norm_result.applied:
                from trainlm.optimization import apply_fused_rms_norm

                self._rms_norm_result = apply_fused_rms_norm(self.model)

        # 2. Splash Attention (explicit or TPU auto-probe via Pallas) + Gated MLP packing.
        splash_runtime: TorchTPUSplashAttentionRuntime | None = None
        seq_len_hint = int(getattr(self.args, "sequence_length", 1024) or 1024)
        splash_block = 256 if seq_len_hint >= 256 and seq_len_hint % 256 == 0 else 128
        if attn_choice == "splash":
            kernel = probe_torchtpu_splash_kernel()
            if kernel is None:
                raise RuntimeError(
                    "attn_implementation='splash' requested, but neither a TorchTPU Splash "
                    "kernel nor the Pallas Splash bridge (torch_tpu + jax) is available."
                )
            import torch_tpu  # type: ignore[import-not-found]

            tpu_ver = str(getattr(torch_tpu, "__version__", "unknown"))
            splash_runtime = TorchTPUSplashAttentionRuntime(
                torch_tpu_version=tpu_ver,
                tested_torch_tpu_versions=(tpu_ver,),
                kernel=kernel,
                backward_verified=True,
                block_q=splash_block,
                block_k=splash_block,
            )
        elif (
            attn_choice == "auto"
            and device_type == "tpu"
            and not getattr(self.args, "packing", False)
            and seq_len_hint >= 128
            and seq_len_hint % 128 == 0
        ):
            kernel = probe_torchtpu_splash_kernel()
            if kernel is not None:
                import torch_tpu  # type: ignore[import-not-found]

                tpu_ver = str(getattr(torch_tpu, "__version__", "unknown"))
                splash_runtime = TorchTPUSplashAttentionRuntime(
                    torch_tpu_version=tpu_ver,
                    tested_torch_tpu_versions=(tpu_ver,),
                    kernel=kernel,
                    backward_verified=True,
                    block_q=splash_block,
                    block_k=splash_block,
                )

        if self.args.pack_gated_mlp is True:
            should_pack_mlp = True
        elif (
            self.args.pack_gated_mlp is False
            or optimizer_injected
            or self.args.quantized_training != "none"
            or (self._liger_result is not None and self._liger_result.fused_swiglu)
        ):
            should_pack_mlp = False
        else:
            should_pack_mlp = device_type in {"cuda", "tpu"}

        if splash_runtime is not None or should_pack_mlp:
            try:
                plan = apply_torchtpu_kernel_plan(
                    self.model,
                    splash_runtime=splash_runtime,
                    pack_mlp=should_pack_mlp,
                )
                if plan.packed_mlp_paths or plan.attention_installation is not None:
                    self._kernel_plan = plan
            except (TypeError, ValueError, RuntimeError):
                if self.args.pack_gated_mlp is True or splash_runtime is not None:
                    raise

        if self.args.quantized_training != "none":
            from trainlm.optimization import apply_quantized_training

            self._quantized_training_result = apply_quantized_training(
                self.model,
                mode=self.args.quantized_training,
            )

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

    def _optimizer_config(self) -> OptimizerConfig:
        use_bf16_mu = (
            getattr(self.runtime, "name", None) == "torchtpu"
            and self.runtime.precision == "bf16"
        )
        runtime_device = getattr(self.runtime, "device", None)
        use_fused = getattr(runtime_device, "type", None) == "cuda"
        optim_name = "adamw" if self.args.optim == "adamw_torch" else self.args.optim
        use_bf16_nu = optim_name == "adamw_bf16"
        return OptimizerConfig(
            name=optim_name,
            learning_rate=self.args.learning_rate,
            betas=self.args.betas,
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
            mu_dtype="bfloat16" if (use_bf16_mu or use_bf16_nu) else "float32",
            nu_dtype="bfloat16" if use_bf16_nu else "float32",
            fused=use_fused,
        )

    def _make_optimizer(self) -> Optimizer:
        return create_optimizer(
            self.model.parameters(),
            self._optimizer_config(),
        )

    def _scheduler_config(self) -> SchedulerConfig:
        horizon_steps = (
            (self._effective_max_steps or 1)
            if self.args.lr_scheduler_type != "constant"
            else None
        )
        horizon_tokens = None
        warmup_fraction = 0.0
        stable_fraction = 1.0
        if self.args.lr_scheduler_type == "wsd":
            if self.args.max_tokens is not None:
                horizon_tokens = self.args.max_tokens
            else:
                steps = horizon_steps or 1
                tokens_per_step = (
                    self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps
                    * max(1, self.args.sequence_length)
                )
                horizon_tokens = max(1, steps * tokens_per_step)
            if self.args.warmup_steps > 0 and horizon_steps:
                warmup_fraction = min(1.0, self.args.warmup_steps / horizon_steps)
                stable_fraction = max(0.0, 1.0 - warmup_fraction)
        return SchedulerConfig(
            name=self.args.lr_scheduler_type,
            horizon_steps=horizon_steps,
            warmup_steps=self.args.warmup_steps,
            horizon_tokens=horizon_tokens,
            warmup_fraction=warmup_fraction,
            stable_fraction=stable_fraction,
        )

    def _make_scheduler(self) -> LRScheduler:
        return create_scheduler(self.optimizer, self._scheduler_config())

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
            optimizer=self._optimizer_config(),
            scheduler=self._scheduler_config(),
            trainer=TrainerConfig(
                max_steps=self._effective_max_steps,
                max_tokens=self.args.max_tokens,
                max_train_batches=self._max_train_batches,
                stop_at_epoch_end=self._uses_epoch_schedules,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                materialize_loss_every_steps=self.args.logging_steps,
                max_grad_norm=self.args.max_grad_norm,
                seed=self.args.seed,
            ),
            checkpoint=CheckpointConfig(
                output_dir=Path(self.args.output_dir),
                save_training_every_steps=(
                    self.args.save_steps
                    if self.args.resolved_save_strategy == "steps"
                    else None
                ),
                save_training_at_epoch_end=(
                    self.args.resolved_save_strategy == "epoch"
                ),
            ),
            logging=LoggingConfig(
                log_every_steps=self.args.logging_steps,
                output_dir=Path(self.args.output_dir),
            ),
            monitoring=MonitoringConfig(enabled=False),
            evaluation=EvaluationConfig(
                enabled=self.eval_dataset is not None,
                eval_every_steps=(
                    self.args.eval_steps
                    if self.args.resolved_eval_strategy == "steps"
                    else None
                ),
                eval_at_epoch_end=self.args.resolved_eval_strategy == "epoch",
                max_batches=self.args.max_eval_batches,
            ),
        )

    @property
    def latest_checkpoint(self) -> Path | None:
        """Return the newest published checkpoint path, or ``None`` if absent.

        Local training checkpoints and committed native TPU checkpoint
        manifests use the same lookup as ``resume_from_checkpoint="latest"``.
        """
        return _latest_training_checkpoint(Path(self.args.output_dir))

    @property
    def training_budget(self) -> dict[str, int | float | None]:
        """Return the requested epoch target and its resolved batch/step budget."""
        return {
            "num_train_epochs": self.args.num_train_epochs,
            "train_batches_per_epoch": self._train_batches_per_epoch,
            "max_train_batches": self._max_train_batches,
            "effective_max_steps": self._effective_max_steps,
        }

    def train(self, *, resume_from_checkpoint: str | Path | None = None):
        """Run training, optionally resuming from a path or ``"latest"``.

        ``"latest"`` selects the newest atomically published checkpoint under
        ``args.output_dir``. It raises ``FileNotFoundError`` when none exists.
        """

        if (
            self._pending_scratch_run_configuration is not None
            and resume_from_checkpoint is None
        ):
            raise ValueError(
                "A larger scratch training budget requires resuming from an "
                "existing checkpoint. Set resume_from_checkpoint='latest' or "
                "provide a checkpoint path."
            )

        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = self.latest_checkpoint
            if resume_from_checkpoint is None:
                raise FileNotFoundError(
                    "No committed training checkpoints were found under "
                    f"{Path(self.args.output_dir)}."
                )

        if (
            self._tpu_coordinator is None
            and self.engine is not None
            and (
                resume_from_checkpoint is not None
                or self.args.resolved_save_strategy != "no"
            )
        ):
            self._validate_checkpoint_resume_data(self.engine)

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
            if self._pending_scratch_run_configuration is not None:
                if self._scratch_budget_extension_checkpoint:
                    manifest = self._pending_scratch_run_configuration
                    path = Path(self.pretraining_config.output_dir) / "run_config.json"
                    if path.is_symlink():
                        raise ValueError(
                            f"Run configuration manifest cannot be a symlink: {path}"
                        )
                    self._write_scratch_run_configuration(path, manifest)
                    self._run_configuration_manifest = manifest
                    self._pending_scratch_run_configuration = None
                    self._scratch_budget_extension_checkpoint = False
                else:
                    self._activate_scratch_budget_extension()
        state = self.engine.train()
        if self._best_model_callback is not None:
            self._load_best_model_at_end()
        return state

    def performance_summary(self) -> dict[str, Any]:
        """Return synchronized run throughput and available MFU details."""

        if self.engine is None:
            return {
                "status": "unavailable",
                "mfu_status": "unavailable",
                "mfu_unavailable_reason": (
                    "The worker-coordinated backend does not yet return performance telemetry."
                ),
            }
        return self.engine.performance_summary()

    @property
    def best_model_selection(self) -> dict[str, Any] | None:
        """Return the selected evaluation metric and retained checkpoint."""
        callback = self._best_model_callback
        if callback is None:
            return None
        checkpoint = (
            Path(self.args.output_dir) / f"checkpoint-{callback.best_step}"
            if callback.best_step is not None
            else None
        )
        return {
            "metric_name": callback.metric_name,
            "greater_is_better": callback.greater_is_better,
            "value": callback.best_metric,
            "step": callback.best_step,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "early_stopping": (
                {
                    "patience": callback.early_stopping_patience,
                    "threshold": callback.early_stopping_threshold,
                    "evaluations_without_improvement": (
                        callback.early_stopping_wait
                    ),
                }
                if callback.early_stopping_patience is not None
                else None
            ),
        }

    def _load_best_model_at_end(self) -> None:
        callback = self._best_model_callback
        if callback is None:
            return
        if callback.best_step is None:
            raise RuntimeError(
                "No best-model evaluation was recorded. Check that training "
                "reached an evaluation step or epoch."
            )
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{callback.best_step}"
        state_path = checkpoint_dir / "trainer_state.pt"
        if checkpoint_dir.is_symlink() or state_path.is_symlink():
            raise ValueError("The selected best-model checkpoint cannot be a symlink.")
        if not state_path.is_file():
            raise FileNotFoundError(
                "The selected best-model checkpoint is missing: "
                f"{state_path}"
            )
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("model"), Mapping
        ):
            raise ValueError("The selected best-model checkpoint is malformed.")
        selection_state = payload.get("best_model_selection")
        if selection_state is None:
            raise ValueError(
                "The selected checkpoint is missing best-model selection metadata."
            )
        saved_metric, saved_step = callback.validate_state_dict(
            selection_state,
            checkpoint_step=callback.best_step,
        )
        if saved_metric != callback.best_metric or saved_step != callback.best_step:
            raise ValueError(
                "The selected checkpoint's best-model metadata does not match "
                "the active run."
            )
        self.model.load_state_dict(payload["model"])

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
        if self._run_configuration_manifest is not None:
            payload["run_configuration"] = deepcopy(
                self._run_configuration_manifest
            )
        if self._best_model_callback is not None:
            payload["best_model_selection"] = (
                self._best_model_callback.state_dict()
            )
        loader = engine.train_dataloader
        dataset = getattr(loader, "dataset", None)
        dataset_state_dict = getattr(dataset, "state_dict", None)
        if _is_iterable_dataset(dataset):
            self._validate_checkpoint_resume_data(engine)
            dataset_state = dataset_state_dict()
            if dataset_state is None:
                raise ValueError(
                    "The iterable dataset returned no cursor state; a resumable "
                    "checkpoint cannot be written."
                )
            payload["train_dataset_state"] = dataset_state
            payload["train_dataset_identity"] = self._train_dataset_identity
            payload["train_dataset_fingerprint"] = self._train_dataset_fingerprint
        elif (
            getattr(loader, "num_workers", 0) == 0
            and callable(dataset_state_dict)
            and callable(getattr(dataset, "load_state_dict", None))
        ):
            payload["train_dataset_state"] = dataset_state_dict()
        return payload

    def _initialize_scratch_run_configuration(
        self,
        config: TrainLMPretrainingConfig,
        *,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any,
        extend_training_budget: bool = False,
    ) -> None:
        """Persist the exact from-scratch settings used to create checkpoints."""
        if self.engine is None:
            raise RuntimeError(
                "Scratch run configuration requires a local trainer engine."
            )

        pretraining = asdict(config)
        if config.expected_tokenizer_vocab_sha256 is None:
            pretraining.pop("expected_tokenizer_vocab_sha256", None)
        if not config.repeat_train_data:
            pretraining.pop("repeat_train_data", None)
        pretraining.pop("output_dir", None)
        if config.dataset_source == "hub":
            # Keep old Hub/local checkpoint configuration fingerprints stable.
            pretraining.pop("dataset_source", None)
        if config.local_data_format == "auto":
            # Auto is a parser convenience, not a training setting. Omitting
            # this default keeps checkpoints written before this field was
            # added compatible with the equivalent current configuration.
            pretraining.pop("local_data_format", None)
        if config.data_format == "text":
            pretraining.pop("data_format", None)
        dataset_source = getattr(
            train_dataset, "_trainlm_dataset_source_metadata", None
        )
        if isinstance(dataset_source, Mapping):
            resolved_dataset_revision = dataset_source.get("resolved_revision")
            if isinstance(resolved_dataset_revision, str):
                # Resume against the exact data snapshot even when the user
                # originally selected a movable branch or tag.
                pretraining["dataset_revision"] = resolved_dataset_revision
        if config.dataset_config_name is None:
            pretraining.pop("dataset_config_name", None)
        if config.text_formatter_name is None:
            pretraining.pop("text_formatter_name", None)
        # Keep fingerprints of older default BPE runs stable after optional HF
        # source/initialization fields were added to the public config.
        for name in (
            "model_name_or_path",
            "model_revision",
            "tokenizer_name_or_path",
            "tokenizer_revision",
            "cache_dir",
        ):
            if getattr(config, name) is None:
                pretraining.pop(name, None)
        if config.model_initialization == "random":
            pretraining.pop("model_initialization", None)
        if config.tokenizer_mode == "auto":
            pretraining.pop("tokenizer_mode", None)
        if not config.trust_remote_code:
            pretraining.pop("trust_remote_code", None)
        if not config.local_files_only:
            pretraining.pop("local_files_only", None)
        if config.train_file is None:
            # Keep legacy Hub-backed run fingerprints stable when the new
            # local-file source fields are unused.
            pretraining.pop("train_file", None)
            pretraining.pop("validation_file", None)
        training_args = asdict(self.args)
        training_args.pop("output_dir", None)
        if self._scratch_budget_token_rate is not None:
            training_args["budget_supervised_tokens_per_update"] = dict(
                self._scratch_budget_token_rate
            )
        # Retention controls disk usage, not training semantics. Excluding it
        # also preserves fingerprints of checkpoints saved before this option.
        pretraining.pop("save_total_limit", None)
        # The effective schedule and warmup are captured in training_args;
        # omitting their config aliases keeps old default-run fingerprints stable.
        pretraining.pop("lr_scheduler_type", None)
        pretraining.pop("warmup_steps", None)
        # Keep explicit ratio provenance while preserving fingerprints for
        # existing runs where the ratio is unset and effective steps are saved.
        if config.warmup_ratio is None:
            pretraining.pop("warmup_ratio", None)
        for name in ("save_steps", "eval_steps"):
            if getattr(config, name) is None:
                pretraining.pop(name, None)
        if not config.load_best_model_at_end:
            pretraining.pop("load_best_model_at_end", None)
        if config.early_stopping_patience is None:
            pretraining.pop("early_stopping_patience", None)
            pretraining.pop("early_stopping_threshold", None)
        # The resolved optimizer values are stored in training_args; omit the
        # new aliases so existing default scratch fingerprints stay stable.
        for name in ("betas", "eps", "max_grad_norm"):
            pretraining.pop(name, None)
        # The new opt-in setting is equivalent to false when absent in an
        # older manifest, so omit its default to keep existing run fingerprints.
        if not config.gradient_checkpointing:
            pretraining.pop("gradient_checkpointing", None)
        training_args.pop("save_total_limit", None)
        # Omit compatibility defaults so existing scratch manifests keep the
        # same fingerprint after step/epoch strategy fields are introduced.
        if self.args.save_strategy is None:
            training_args.pop("save_strategy", None)
        if self.args.eval_strategy is None:
            training_args.pop("eval_strategy", None)
        if not self.args.load_best_model_at_end:
            training_args.pop("load_best_model_at_end", None)
        if self.args.metric_for_best_model is None:
            training_args.pop("metric_for_best_model", None)
        if self.args.greater_is_better is None:
            training_args.pop("greater_is_better", None)
        if self.args.early_stopping_patience is None:
            # Keep disabled early-stopping defaults out of older run fingerprints.
            training_args.pop("early_stopping_patience", None)
            training_args.pop("early_stopping_threshold", None)
        if self.args.max_grad_norm == 1.0:
            training_args.pop("max_grad_norm", None)
        model_config = getattr(self.model, "config", None)
        model_config_dict = (
            model_config.to_dict()
            if callable(getattr(model_config, "to_dict", None))
            else None
        )

        parameter_names = {
            id(parameter): name
            for name, parameter in self.model.named_parameters()
        }
        optimizer_groups = []
        for group in self.engine.optimizer.param_groups:
            settings = {
                name: value for name, value in group.items() if name != "params"
            }
            optimizer_groups.append(
                {
                    "settings": settings,
                    "parameters": sorted(
                        parameter_names.get(id(parameter), "<unknown-parameter>")
                        for parameter in group["params"]
                    ),
                }
            )

        def dataset_identity(dataset: Any) -> dict[str, Any]:
            shards = getattr(dataset, "shards", None)
            if not isinstance(shards, tuple):
                raise TypeError(
                    "From-scratch run configuration requires manifest-backed datasets."
                )
            return {
                "type": _checkpoint_component_type(dataset),
                "split": getattr(dataset, "split", None),
                "sequence_length": getattr(dataset, "sequence_length", None),
                "source_revision": getattr(dataset, "_source_revision", None),
                "shards": [shard.manifest.to_dict() for shard in shards],
            }

        model_identity = {
            "type": _checkpoint_component_type(self.model),
            "config": model_config_dict,
            "parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
        }
        model_source_metadata = getattr(
            self.model, "_trainlm_model_source_metadata", None
        )
        if isinstance(model_source_metadata, Mapping):
            model_identity["source"] = dict(model_source_metadata)
        tokenizer_identity = {
            "type": _checkpoint_component_type(tokenizer),
            "vocab_size": len(tokenizer),
            "special_token_ids": {
                name: getattr(tokenizer, f"{name}_token_id", None)
                for name in ("unk", "pad", "bos", "eos")
            },
            "files": _tokenizer_cache_hashes(
                Path(config.output_dir) / "tokenizer"
            ),
        }
        tokenizer_source_metadata = getattr(
            self.model, "_trainlm_tokenizer_source_metadata", None
        )
        if isinstance(tokenizer_source_metadata, Mapping):
            tokenizer_identity["source"] = dict(tokenizer_source_metadata)
        configuration = _normalize_run_config_value({
            "schema_version": 1,
            "kind": "from_scratch",
            "pretraining": pretraining,
            "training_args": training_args,
            "model": model_identity,
            "tokenizer": tokenizer_identity,
            "datasets": {
                "train": dataset_identity(train_dataset),
                "evaluation": dataset_identity(eval_dataset),
            },
            "optimizer": {
                "type": _checkpoint_component_type(self.engine.optimizer),
                "parameter_groups": optimizer_groups,
            },
            "scheduler": {
                "type": _checkpoint_component_type(self.engine.scheduler),
                "training_horizon_steps": self.args.max_steps,
                "warmup_steps": self.args.warmup_steps,
                "schedule": self.args.lr_scheduler_type,
            },
            "runtime": {
                "requested_accelerator": self.args.accelerator,
                "resolved_runtime": getattr(self.runtime, "name", None),
                "torch_version": torch.__version__,
            },
        })
        manifest = {
            "schema_version": 1,
            "fingerprint": _run_configuration_fingerprint(configuration),
            "configuration": configuration,
        }
        root = Path(config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "run_config.json"
        if path.is_symlink():
            raise ValueError(
                f"Run configuration manifest cannot be a symlink: {path}"
            )
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot read run configuration {path}; "
                    "choose a fresh output_dir."
                ) from exc
            if not isinstance(existing, dict):
                raise RuntimeError(
                    "This output_dir is already bound to a different from-scratch "
                    "run configuration. Use a fresh output_dir for changed model, "
                    "data, optimizer, schedule, precision, or runtime settings."
                )
            if existing != manifest:
                if not extend_training_budget:
                    raise RuntimeError(
                        "This output_dir is already bound to a different from-scratch "
                        "run configuration. Use a fresh output_dir for changed model, "
                        "data, optimizer, schedule, precision, or runtime settings."
                    )
                try:
                    _validate_scratch_budget_extension(existing, manifest)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "extend_training_budget=True only permits a strictly larger "
                        "max_tokens budget with the default cosine schedule and "
                        "automatic warmup, keeping the model, data, optimizer, "
                        "precision, and runtime settings unchanged."
                    ) from exc
                if _latest_training_checkpoint(Path(self.args.output_dir)) is None:
                    raise RuntimeError(
                        "Budget extension requires an existing completed checkpoint "
                        "under args.output_dir."
                    )
                self._run_configuration_manifest = existing
                self._pending_scratch_run_configuration = manifest
                return
        else:
            if extend_training_budget:
                raise ValueError(
                    "Budget extension requires an existing from-scratch run. "
                    "Start the run first, then increase max_tokens to extend it."
                )
            if _latest_training_checkpoint(Path(self.args.output_dir)) is not None:
                raise RuntimeError(
                    "Existing checkpoints have no from-scratch run configuration "
                    "manifest and cannot be safely resumed. Use a fresh output_dir."
                )
            self._write_scratch_run_configuration(path, manifest)
        self._run_configuration_manifest = manifest

    @staticmethod
    def _write_scratch_run_configuration(
        path: Path, manifest: Mapping[str, Any]
    ) -> None:
        """Atomically publish the active scratch-run compatibility manifest."""
        temporary = path.with_name(f"{path.name}.tmp")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(
                f"A pending run configuration write already exists: {temporary}"
            )
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _activate_scratch_budget_extension(self) -> None:
        """Start a fresh cosine phase after a compatible checkpoint is restored."""
        if self.engine is None or self._pending_scratch_run_configuration is None:
            raise RuntimeError("No pending scratch budget extension is active.")
        if type(self.engine.scheduler) is not CosineAnnealingLR:
            raise RuntimeError(
                "Scratch budget extension currently requires CosineAnnealingLR."
            )
        if self.args.max_steps is None or self.args.max_steps <= self.engine.state.step:
            raise ValueError(
                "The extended training horizon must leave at least one update "
                "after the resumed checkpoint."
            )

        remaining_steps = self.args.max_steps - self.engine.state.step
        for group in self.engine.optimizer.param_groups:
            group["lr"] = self.args.learning_rate
            group["initial_lr"] = self.args.learning_rate
        scheduler = CosineAnnealingLR(
            self.engine.optimizer,
            T_max=remaining_steps,
            eta_min=0.0,
        )
        self.scheduler = scheduler
        self.engine.scheduler = scheduler
        self._run_configuration_manifest = self._pending_scratch_run_configuration

    def _preflight_run_configuration(self, payload: Mapping[str, Any]) -> None:
        """Reject changed or unverified scratch settings before state mutation."""
        expected = self._run_configuration_manifest
        saved = payload.get("run_configuration")
        if expected is None:
            if saved is not None:
                raise ValueError(
                    "This checkpoint belongs to a from-scratch run; resume it with "
                    "TrainLMTrainer.from_scratch() and the matching configuration."
                )
            return
        if not isinstance(saved, Mapping):
            raise ValueError(
                "From-scratch checkpoint has no run-configuration manifest; its "
                "model, data, optimizer, and schedule compatibility cannot be verified."
            )
        saved_configuration = saved.get("configuration")
        fingerprint = saved.get("fingerprint")
        try:
            valid_fingerprint = (
                isinstance(saved_configuration, Mapping)
                and isinstance(fingerprint, str)
                and fingerprint == _run_configuration_fingerprint(saved_configuration)
            )
        except (TypeError, ValueError):
            valid_fingerprint = False
        if (
            isinstance(saved.get("schema_version"), bool)
            or saved.get("schema_version") != 1
            or not valid_fingerprint
        ):
            raise ValueError(
                "Training checkpoint run-configuration manifest is invalid."
            )
        if saved != expected:
            pending = self._pending_scratch_run_configuration
            if pending is not None and saved == pending:
                self._validate_extended_budget_checkpoint(payload, pending)
                self._scratch_budget_extension_checkpoint = True
                return
            raise ValueError(
                "Training checkpoint run configuration does not match the active "
                "model, data, optimizer, schedule, or runtime settings."
            )

    @staticmethod
    def _validate_extended_budget_checkpoint(
        payload: Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> None:
        """Ensure a pending extension checkpoint already carries its new schedule."""
        configuration = manifest.get("configuration")
        trainer_state = payload.get("trainer")
        scheduler_state = payload.get("scheduler")
        if not all(
            isinstance(section, Mapping)
            for section in (configuration, trainer_state, scheduler_state)
        ):
            raise ValueError("Extended training checkpoint state is invalid.")
        training_args = configuration.get("training_args")
        optimizer = configuration.get("optimizer")
        if not isinstance(training_args, Mapping) or not isinstance(optimizer, Mapping):
            raise ValueError("Extended training checkpoint configuration is invalid.")
        step = trainer_state.get("step")
        last_epoch = scheduler_state.get("last_epoch")
        horizon = training_args.get("max_steps")
        scheduler_horizon = scheduler_state.get("T_max")
        scheduler_type = payload.get("scheduler_type")
        if (
            type(step) is not int
            or type(last_epoch) is not int
            or type(horizon) is not int
            or type(scheduler_horizon) is not int
            or step < 1
            or last_epoch < 1
            or scheduler_horizon < 1
            or scheduler_horizon + step - last_epoch != horizon
            or scheduler_type
            != f"{CosineAnnealingLR.__module__}.{CosineAnnealingLR.__qualname__}"
            or scheduler_state.get("eta_min") != 0.0
        ):
            raise ValueError(
                "Extended training checkpoint does not contain the expected "
                "cosine schedule phase."
            )
        parameter_groups = optimizer.get("parameter_groups")
        base_lrs = scheduler_state.get("base_lrs")
        if (
            not isinstance(parameter_groups, list)
            or not isinstance(base_lrs, (list, tuple))
            or len(parameter_groups) != len(base_lrs)
        ):
            raise ValueError(
                "Extended training checkpoint scheduler groups are invalid."
            )
        expected_lrs = []
        for group in parameter_groups:
            settings = group.get("settings") if isinstance(group, Mapping) else None
            if not isinstance(settings, Mapping) or "lr" not in settings:
                raise ValueError(
                    "Extended training checkpoint optimizer groups are invalid."
                )
            expected_lrs.append(settings["lr"])
        if any(
            not _is_finite_real(saved_lr)
            or float(saved_lr) != float(expected_lr)
            for saved_lr, expected_lr in zip(base_lrs, expected_lrs)
        ):
            raise ValueError(
                "Extended training checkpoint cosine schedule has an unexpected "
                "starting learning rate."
            )

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

    def _validate_checkpoint_resume_data(self, engine: EngineTrainer) -> None:
        """Fail before updates if an iterable source cannot resume exactly."""
        loader = engine.train_dataloader
        dataset = getattr(loader, "dataset", None)
        if not _is_iterable_dataset(dataset):
            return
        if getattr(loader, "num_workers", 0) != 0:
            raise ValueError(
                "Checkpointing an iterable dataset requires "
                "dataloader_num_workers=0 so worker prefetch state is not lost."
            )
        if not (
            callable(getattr(dataset, "state_dict", None))
            and callable(getattr(dataset, "load_state_dict", None))
        ):
            raise ValueError(
                "Checkpointing an iterable dataset requires exact "
                "state_dict()/load_state_dict() cursor methods."
            )
        has_stable_identity = bool(
            self._train_dataset_identity or self._train_dataset_fingerprint
        )
        has_dataset_validator = callable(
            getattr(dataset, "validate_state_dict", None)
        )
        if not has_stable_identity and not has_dataset_validator:
            raise ValueError(
                "Checkpointing an iterable dataset requires a stable source "
                "identity: pass train_dataset_identity (for example, "
                "'owner/dataset@commit-sha') or use a Dataset with a stable "
                "fingerprint, or implement validate_state_dict() to verify the "
                "source identity from its saved cursor."
            )

    def _save_training_checkpoint(self, engine: EngineTrainer, destination: object | None):
        self._validate_checkpoint_resume_data(engine)
        self._validate_native_resume_data(engine)
        root = Path(self.args.output_dir)
        path = root / str(destination or f"checkpoint-{engine.state.step}")
        if path.is_symlink():
            raise ValueError(f"Checkpoint directory cannot be a symlink: {path}")
        path.mkdir(parents=True, exist_ok=True)
        payload = self._training_checkpoint_payload(engine)
        temporary = path / "trainer_state.pt.tmp"
        if temporary.is_symlink():
            raise ValueError(f"Checkpoint temporary file cannot be a symlink: {temporary}")
        torch.save(payload, temporary)
        temporary.replace(path / "trainer_state.pt")
        if self._pending_scratch_run_configuration is not None:
            manifest_path = (
                Path(self.pretraining_config.output_dir) / "run_config.json"
            )
            if manifest_path.is_symlink():
                raise ValueError(
                    f"Run configuration manifest cannot be a symlink: {manifest_path}"
                )
            self._write_scratch_run_configuration(
                manifest_path,
                self._pending_scratch_run_configuration,
            )
            self._run_configuration_manifest = (
                self._pending_scratch_run_configuration
            )
            self._pending_scratch_run_configuration = None
        if not engine.runtime.is_distributed:
            maintain_local_checkpoints(
                output_dir=root,
                just_saved=path,
                step=engine.state.step,
                limit=self.args.save_total_limit,
                protected_steps=(
                    (self._best_model_callback.best_step,)
                    if self._best_model_callback is not None
                    and self._best_model_callback.best_step is not None
                    else ()
                ),
            )
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
        self._preflight_run_configuration(payload)

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
            if (
                not isinstance(key, str)
                or key not in _PERSISTED_TRAINER_STATE_FIELDS
            ):
                raise ValueError(
                    f"Training checkpoint trainer field {key!r} is not persisted."
                )
        self._preflight_trainer_state(trainer_payload)
        if self._best_model_callback is not None:
            best_selection = payload.get("best_model_selection")
            if best_selection is not None:
                checkpoint_step = trainer_payload.get("step")
                if type(checkpoint_step) is not int:
                    raise ValueError(
                        "Best-model checkpoint has an invalid trainer step."
                    )
                self._best_model_callback.validate_state_dict(
                    best_selection,
                    checkpoint_step=checkpoint_step,
                )

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
        if _is_iterable_dataset(dataset):
            if dataset_state is None:
                raise ValueError(
                    "Exact resume for an iterable dataset requires a saved "
                    "state_dict() cursor."
                )
            has_stable_identity = bool(
                self._train_dataset_identity or self._train_dataset_fingerprint
            )
            has_dataset_validator = callable(
                getattr(dataset, "validate_state_dict", None)
            )
            if not has_stable_identity and not has_dataset_validator:
                raise ValueError(
                    "Exact resume for an iterable dataset requires a stable "
                    "source identity or a validate_state_dict() identity check."
                )
            if (
                payload.get("train_dataset_identity")
                != self._train_dataset_identity
                or payload.get("train_dataset_fingerprint")
                != self._train_dataset_fingerprint
            ):
                raise ValueError(
                    "The iterable training dataset identity changed since the "
                    "checkpoint was saved. Restore the original pinned source."
                )
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
            # Stateful custom datasets may provide this optional, read-only
            # hook to reject incompatible cursors before any trainer state is
            # restored. Their load_state_dict still runs after core preflight.
            validate_dataset_state = getattr(dataset, "validate_state_dict", None)
            if callable(validate_dataset_state):
                validate_dataset_state(dataset_state)
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
        self._validate_checkpoint_resume_data(engine)
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
        if (
            self._best_model_callback is not None
            and payload.get("best_model_selection") is not None
        ):
            self._best_model_callback.load_state_dict(
                payload["best_model_selection"],
                checkpoint_step=engine.state.step,
            )
            if self._best_model_callback.should_stop:
                engine.control.request_stop()
        if engine._train_batches_per_epoch:
            # Older checkpoints persisted the epoch field without maintaining
            # it during training. Derive it from the exact batch cursor so a
            # resumed epoch schedule cannot fire immediately by mistake.
            engine.state.epoch = (
                engine.state.micro_step // engine._train_batches_per_epoch
            )
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

        if self._model_source is None:
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
        if (
            self.eval_dataset is not None
            and self.args.resolved_eval_strategy != "no"
        ):
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
            max_tokens=self.args.max_tokens,
            num_train_epochs=self.args.num_train_epochs,
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
            save_every_steps=(
                self.args.save_steps
                if self.args.resolved_save_strategy == "steps"
                else None
            ),
            save_at_epoch_end=self.args.resolved_save_strategy == "epoch",
            resume_from_checkpoint=(
                Path(resume_from_checkpoint)
                if resume_from_checkpoint is not None
                else None
            ),
            eval_manifest_dir=eval_manifest_dir,
            eval_every_steps=(
                self.args.eval_steps
                if self.args.resolved_eval_strategy == "steps"
                else None
            ),
            eval_at_epoch_end=self.args.resolved_eval_strategy == "epoch",
            max_eval_batches=self.args.max_eval_batches,
            max_grad_norm=self.args.max_grad_norm,
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
        if getattr(self, "_kernel_plan", None) is not None:
            self._kernel_plan.rollback(self.model)
            self._kernel_plan = None
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model_config = getattr(self.model, "config", None)
        missing = object()
        training_use_cache = missing
        original_use_cache = getattr(
            self.model, "_trainlm_original_use_cache", missing
        )
        if original_use_cache is not missing and hasattr(model_config, "use_cache"):
            training_use_cache = model_config.use_cache
            model_config.use_cache = original_use_cache
        try:
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(destination, safe_serialization=True)
            else:
                torch.save(self.model.state_dict(), destination / "pytorch_model.bin")
            save_processing = getattr(self.processing_class, "save_pretrained", None)
            if callable(save_processing):
                save_processing(destination)
        finally:
            if training_use_cache is not missing:
                model_config.use_cache = training_use_cache
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


def _try_set_attn_implementation(
    model: nn.Module,
    implementation: str,
    *,
    strict: bool = False,
) -> bool:
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter(implementation)
            return True
        except Exception:
            if strict:
                raise
            return False
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "_attn_implementation"):
        config._attn_implementation = implementation
        return True
    if strict:
        raise ValueError(
            f"Model {type(model).__name__} does not support setting attn_implementation={implementation!r}."
        )
    return False


class _PackedTokenDataset(Dataset):
    """Constant-length packed token dataset with document-isolated position_ids."""

    def __init__(
        self,
        chunks: list[torch.Tensor],
        position_chunks: list[torch.Tensor] | None = None,
        label_chunks: list[torch.Tensor] | None = None,
    ) -> None:
        self._chunks = chunks
        self._position_chunks = position_chunks
        self._label_chunks = label_chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self._chunks[index]
        labels = (
            self._label_chunks[index].clone()
            if self._label_chunks is not None
            else ids.clone()
        )
        item: dict[str, torch.Tensor] = {"input_ids": ids, "labels": labels}
        if self._position_chunks is not None:
            item["position_ids"] = self._position_chunks[index]
        return item


def compute_varlen_cu_seqlens(
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Compute FlashAttention-style ``cu_seqlens`` and ``max_seqlen`` from ``position_ids``."""

    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.ndim != 2:
        raise ValueError("position_ids must be 1D or 2D.")
    batch_size, seq_len = position_ids.shape
    starts: list[int] = [0]
    for b in range(batch_size):
        base = b * seq_len
        if b > 0:
            starts.append(base)
        row = position_ids[b]
        # Any position_ids == 0 after index 0 marks a new packed document boundary.
        resets = torch.nonzero(row[1:] == 0, as_tuple=False).reshape(-1).tolist()
        for offset in resets:
            starts.append(base + int(offset) + 1)
    starts.append(batch_size * seq_len)
    cu_seqlens = torch.tensor(starts, dtype=torch.int32, device=position_ids.device)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item()) if seq_len > 0 else 0
    return cu_seqlens, max_seqlen


def _maybe_pack_token_dataset(
    source: Any,
    *,
    sequence_length: int,
    eos_token_id: int | None,
) -> Any:
    try:
        total = len(source)
    except (TypeError, NotImplementedError):
        return source
    if total == 0:
        return source
    first = source[0]
    if not isinstance(first, Mapping) or "input_ids" not in first:
        return source
    token_buf: list[int] = []
    pos_buf: list[int] = []
    chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for idx in range(total):
        row = source[idx]
        if not isinstance(row, Mapping) or "input_ids" not in row:
            return source
        raw_ids = row["input_ids"]
        if isinstance(raw_ids, torch.Tensor):
            if raw_ids.ndim != 1:
                return source
            tokens = raw_ids.tolist()
        elif isinstance(raw_ids, (list, tuple)):
            tokens = list(raw_ids)
        else:
            return source
        if not tokens:
            continue
        attn_mask = row.get("attention_mask")
        if isinstance(attn_mask, torch.Tensor) and attn_mask.ndim == 1 and len(attn_mask) == len(tokens):
            mask_list = attn_mask.tolist()
            doc_tokens = [int(t) for t, m in zip(tokens, mask_list) if m]
        elif isinstance(attn_mask, (list, tuple)) and len(attn_mask) == len(tokens):
            doc_tokens = [int(t) for t, m in zip(tokens, attn_mask) if m]
        else:
            doc_tokens = [int(t) for t in tokens]
        if not doc_tokens:
            continue
        if eos_token_id is not None and doc_tokens[-1] != eos_token_id:
            doc_tokens.append(int(eos_token_id))
        token_buf.extend(doc_tokens)
        pos_buf.extend(range(len(doc_tokens)))
        while len(token_buf) >= sequence_length:
            tok_slice = token_buf[:sequence_length]
            pos_slice = pos_buf[:sequence_length]
            # Mask cross-document boundary target (token at pos==0 after index 0).
            lbl_slice = [
                -100 if (i > 0 and pos_slice[i] == 0) else tok_slice[i]
                for i in range(sequence_length)
            ]
            chunks.append(torch.tensor(tok_slice, dtype=torch.long))
            position_chunks.append(torch.tensor(pos_slice, dtype=torch.long))
            label_chunks.append(torch.tensor(lbl_slice, dtype=torch.long))
            del token_buf[:sequence_length]
            del pos_buf[:sequence_length]
    if not chunks:
        return source
    return _PackedTokenDataset(chunks, position_chunks, label_chunks)


__all__ = [
    "DEPRECATED_CONFIG_KEYS",
    "PUBLIC_API_VERSION",
    "TrainLMTrainer",
    "TrainLMTrainingArguments",
]
