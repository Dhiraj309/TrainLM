"""Task-focused fine-tuning of a pretrained causal language model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
import json
import math
from numbers import Integral, Real
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from .api import (
    TrainLMTrainer,
    TrainLMTrainingArguments,
    _resolve_epoch_budget,
    _latest_training_checkpoint,
)
from .hub import HubPublishResult, _publish_model_export
from .training import TrainerCallback


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_FIELD_NAME = re.compile(r"[A-Za-z0-9._+-]{1,96}\Z")
_SNAPSHOT_COMMIT = re.compile(
    r"(?:^|[/\\])snapshots[/\\]([0-9a-f]{40})(?:[/\\]|\Z)"
)


@dataclass(frozen=True, slots=True)
class TrainLMFineTuningConfig:
    """Inputs and data provenance for pretrained causal-LM fine-tuning.

    The train and validation splits stay separate. The default path loads the
    pinned Hub dataset and tokenizes ``text_field``. Set ``data_format="tokens"``
    when dataset rows already contain aligned ``input_ids`` and optional
    causal-LM labels or masks. ``run_finetuning`` can also receive dataset
    objects and a row formatter for text input. Training and evaluation
    settings live in :class:`TrainLMTrainingArguments`. The
    tokenizer defaults to ``model_name_or_path`` and ``model_revision``;
    configure ``tokenizer_name_or_path`` and ``tokenizer_revision`` to select
    an independent tokenizer source.
    """

    model_name_or_path: str
    dataset_id: str | None = None
    dataset_revision: str | None = None
    args: TrainLMTrainingArguments = field(
        default_factory=TrainLMTrainingArguments
    )
    text_field: str = "text"
    train_split: str = "train"
    validation_split: str = "validation"
    model_revision: str | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    data_format: Literal["text", "tokens"] = "text"
    _tokenizer_object_required: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _expected_tokenizer_revision: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_name_or_path, str)
            or not self.model_name_or_path.strip()
        ):
            raise ValueError("model_name_or_path must be a non-empty model ID or path.")
        if self.dataset_id is not None and (
            not isinstance(self.dataset_id, str) or not self.dataset_id.strip()
        ):
            raise ValueError("dataset_id must be a non-empty string when provided.")
        if self.dataset_revision is not None and (
            not isinstance(self.dataset_revision, str)
            or not _COMMIT_SHA.fullmatch(self.dataset_revision)
        ):
            raise ValueError(
                "dataset_revision must be an immutable 40-character commit SHA "
                "when provided."
            )
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ValueError("model_revision must be non-empty when configured.")
        if self.tokenizer_name_or_path is not None and (
            not isinstance(self.tokenizer_name_or_path, str)
            or not self.tokenizer_name_or_path.strip()
        ):
            raise ValueError(
                "tokenizer_name_or_path must be non-empty when configured."
            )
        if self.tokenizer_revision is not None and (
            not isinstance(self.tokenizer_revision, str)
            or not self.tokenizer_revision.strip()
        ):
            raise ValueError("tokenizer_revision must be non-empty when configured.")
        if not isinstance(self.text_field, str) or not _FIELD_NAME.fullmatch(
            self.text_field
        ):
            raise ValueError(
                "text_field must be a simple non-empty field name."
            )
        if any(
            not isinstance(split, str)
            or not split.strip()
            or any(ord(char) < 32 for char in split)
            for split in (self.train_split, self.validation_split)
        ):
            raise ValueError("Dataset split selectors must be non-empty strings.")
        if self.train_split == self.validation_split:
            raise ValueError("Training and validation splits must be distinct.")
        if not isinstance(self.data_format, str) or self.data_format not in {
            "text",
            "tokens",
        }:
            raise ValueError("data_format must be 'text' or 'tokens'.")
        if not isinstance(self.args, TrainLMTrainingArguments):
            raise TypeError("args must be a TrainLMTrainingArguments instance.")

    def training_plan(self, train_dataset: Any | None = None) -> dict[str, Any]:
        """Describe the configured fine-tuning budget before starting a run.

        Pass an already-loaded finite map-style dataset to resolve an epoch
        budget to batches and optimizer steps. This method never loads models,
        datasets, or accelerator runtimes.
        """
        args = self.args
        stop_conditions: list[dict[str, Any]] = []
        if args.max_steps is not None:
            stop_conditions.append({
                "unit": "optimizer_steps",
                "target": args.max_steps,
            })
        if args.max_tokens is not None:
            tokens_per_update = (
                (args.sequence_length - 1)
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            )
            stop_conditions.append({
                "unit": "supervised_tokens",
                "target": args.max_tokens,
                "estimated_tokens_per_optimizer_update": tokens_per_update,
                "estimated_optimizer_steps": math.ceil(
                    args.max_tokens / tokens_per_update
                ),
                "estimate_note": (
                    "Assumes one device and dense targets; world size, labels, "
                    "masks, or padding can change the steps needed."
                ),
            })
        if args.num_train_epochs is not None:
            stop_conditions.append({
                "unit": "epochs",
                "target": args.num_train_epochs,
            })
        budget: dict[str, Any]
        if args.max_steps is not None:
            budget = {
                "unit": "optimizer_steps",
                "target": args.max_steps,
                "estimated_optimizer_steps": args.max_steps,
            }
        elif args.max_tokens is not None:
            budget = {
                "unit": "supervised_tokens",
                "target": args.max_tokens,
                "estimated_optimizer_steps": None,
            }
        else:
            budget = {
                "unit": "epochs",
                "target": args.num_train_epochs,
                "estimated_optimizer_steps": None,
            }
            if train_dataset is not None:
                batches, max_batches, steps = _resolve_epoch_budget(
                    train_dataset,
                    batch_size=args.per_device_train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    num_train_epochs=args.num_train_epochs,
                )
                budget.update({
                    "batches_per_epoch": batches,
                    "estimated_train_batches": max_batches,
                    "estimated_optimizer_steps": steps,
                })
        if args.accelerator == "tpu" and args.num_train_epochs is not None:
            budget_support = {
                "status": "unsupported_by_requested_backend",
                "reason": (
                    "The legacy PyTorch/XLA coordinator supports optimizer-step "
                    "and supervised-token budgets, but not num_train_epochs."
                ),
            }
        elif args.accelerator == "tpu":
            budget_support = {
                "status": "implemented_unverified",
                "reason": (
                    "The TPU coordinator implements optimizer-step and global "
                    "supervised-token limits; this plan does not verify hardware execution."
                ),
            }
        else:
            budget_support = {
                "status": "not_preflighted",
                "reason": None,
            }
        return {
            "workflow": "fine_tuning",
            "model": {
                "source": self.model_name_or_path,
                "revision": self.model_revision,
                "parameter_count": "resolved_during_model_loading",
            },
            "data": {
                "source": self.dataset_id or "provided_dataset",
                "format": self.data_format,
                "train_split": self.train_split,
                "validation_split": self.validation_split,
                "sequence_length": args.sequence_length,
            },
            "budget": budget,
            "stop_conditions": stop_conditions,
            "stop_rule": (
                "Training stops when the first configured limit is reached."
            ),
            "training": {
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_steps": args.warmup_steps,
                "logging_steps": args.logging_steps,
                "save_strategy": args.resolved_save_strategy,
                "save_steps": args.save_steps,
                "eval_strategy": args.resolved_eval_strategy,
                "eval_steps": args.eval_steps,
                "load_best_model_at_end": args.load_best_model_at_end,
                "metric_for_best_model": args.metric_for_best_model,
                "greater_is_better": args.greater_is_better,
                "early_stopping_patience": args.early_stopping_patience,
                "bf16": args.bf16,
                "fp16": args.fp16,
            },
            "runtime": {
                "requested": args.accelerator,
                "selected": "resolved_during_runtime_initialization",
                "budget_support": budget_support,
            },
            "estimate_scope": "no wall time or memory estimate",
        }


@dataclass(frozen=True, slots=True)
class TrainLMFineTuningResult:
    """Summary of a completed fine-tuning, evaluation, and export run."""

    model_name_or_path: str
    model_revision: str | None
    dataset_id: str | None
    dataset_revision: str | None
    model_path: Path
    step: int
    train_loss: float | None
    evaluation: Mapping[str, float]
    resolved_model_revision: str | None = None
    resolved_tokenizer_revision: str | None = None
    model_tokenizer_revision_match: bool | None = None
    summary_path: Path | None = None
    training_budget: Mapping[str, Any] = field(default_factory=dict)
    best_model_selection: Mapping[str, Any] | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_compatibility: Mapping[str, Any] | None = None
    tokenizer_object_required: bool = False

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = True,
        commit_message: str | None = None,
    ) -> HubPublishResult:
        """Explicitly publish this fine-tuned model export to the Hub.

        New repositories are private by default. Existing repository visibility
        must match ``private`` and is never changed by this method. Authentication
        uses the standard Hugging Face login or ``HF_TOKEN`` configuration.
        Only model/tokenizer artifacts and a model card are uploaded; run reports,
        datasets, and resumable checkpoints are excluded.
        """
        base_model: str | None = self.model_name_or_path
        if Path(base_model).expanduser().exists():
            base_model = None
        return _publish_model_export(
            self.model_path,
            repo_id,
            private=private,
            commit_message=commit_message,
            metadata={
                "workflow": "fine-tuning",
                "base_model": base_model,
                "base_model_revision": self.resolved_model_revision,
            },
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact, JSON-compatible run summary."""
        evaluation = {
            str(name): (
                float(value) if math.isfinite(float(value)) else None
            )
            for name, value in self.evaluation.items()
        }
        return {
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "resolved_model_revision": self.resolved_model_revision,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "model_tokenizer_revision_match": self.model_tokenizer_revision_match,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_compatibility": _json_compatible(
                self.tokenizer_compatibility
            ),
            "tokenizer_object_required": self.tokenizer_object_required,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "model_path": str(self.model_path),
            "step": self.step,
            "train_loss": self.train_loss,
            "evaluation": evaluation,
            "training_budget": _json_compatible(self.training_budget),
            "best_model_selection": _json_compatible(
                self.best_model_selection
            ),
        }


@dataclass(frozen=True, slots=True)
class _CausalLMCollator:
    tokenizer: Any

    def __call__(
        self, features: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            raise ValueError("The tokenizer must return an attention_mask for padding.")
        labels = batch["input_ids"].clone()
        labels[attention_mask == 0] = -100
        batch["labels"] = labels
        return batch


@dataclass(frozen=True, slots=True)
class _TokenizedCausalLMCollator:
    """Pad prepared causal-LM token rows without changing their targets."""

    tokenizer: Any
    sequence_length: int

    def __call__(
        self, features: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty tokenized batch.")
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, Integral):
            raise ValueError(
                "Tokenized causal-LM batches need an integer tokenizer pad_token_id."
            )
        pad_token_id = int(pad_token_id)
        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths)
        if max_length > self.sequence_length:
            raise ValueError(
                f"Tokenized sequence length {max_length} exceeds configured "
                f"sequence_length={self.sequence_length}; TrainLM will not "
                "silently truncate prepared tokens."
            )

        batch: dict[str, Any] = {}
        batch["input_ids"] = torch.tensor(
            [
                list(feature["input_ids"])
                + [pad_token_id] * (max_length - len(feature["input_ids"]))
                for feature in features
            ],
            dtype=torch.long,
        )
        batch["attention_mask"] = torch.tensor(
            [
                list(feature.get("attention_mask", [1] * length))
                + [0] * (max_length - length)
                for feature, length in zip(features, lengths)
            ],
            dtype=torch.long,
        )
        batch["labels"] = torch.tensor(
            [
                list(feature.get("labels", feature["input_ids"]))
                + [-100] * (max_length - len(feature["input_ids"]))
                for feature in features
            ],
            dtype=torch.long,
        )

        if any("loss_mask" in feature for feature in features):
            batch["loss_mask"] = torch.tensor(
                [
                    list(feature.get("loss_mask", [1] * length))
                    + [0] * (max_length - length)
                    for feature, length in zip(features, lengths)
                ],
                dtype=torch.long,
            )
        if any("position_ids" in feature for feature in features):
            batch["position_ids"] = torch.tensor(
                [
                    list(feature.get("position_ids", range(length)))
                    + [0] * (max_length - length)
                    for feature, length in zip(features, lengths)
                ],
                dtype=torch.long,
            )
        if any("token_type_ids" in feature for feature in features):
            batch["token_type_ids"] = torch.tensor(
                [
                    list(feature.get("token_type_ids", [0] * length))
                    + [0] * (max_length - length)
                    for feature, length in zip(features, lengths)
                ],
                dtype=torch.long,
            )
        return batch


_TOKENIZED_MODEL_FIELDS = (
    "input_ids",
    "attention_mask",
    "labels",
    "loss_mask",
    "position_ids",
    "token_type_ids",
)


def _tokenized_sequence(
    value: Any,
    *,
    field_name: str,
    allow_ignore_index: bool = False,
) -> list[int]:
    """Validate one token or mask sequence and normalize it to Python ints."""
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"Tokenized {field_name} values must be non-empty sequences.")
    result: list[int] = []
    for item in value:
        is_mask = field_name in {"attention_mask", "loss_mask"}
        if (isinstance(item, bool) and not is_mask) or not isinstance(item, Integral):
            raise ValueError(f"Tokenized {field_name} values must be integers.")
        item = int(item)
        if allow_ignore_index:
            if item < 0 and item != -100:
                raise ValueError("Tokenized labels may use only -100 as a negative ID.")
        elif item < 0:
            raise ValueError(f"Tokenized {field_name} values must be non-negative.")
        result.append(item)
    return result


def _validate_tokenized_batch(
    batch: Mapping[str, Any],
    *,
    sequence_length: int,
) -> tuple[dict[str, list[list[int]]], int]:
    """Validate HF token columns without decoding, shifting, or truncating."""
    if "input_ids" not in batch:
        raise ValueError("Tokenized datasets must contain an 'input_ids' column.")
    input_rows = batch["input_ids"]
    if hasattr(input_rows, "tolist") and callable(input_rows.tolist):
        input_rows = input_rows.tolist()
    if not isinstance(input_rows, Sequence) or isinstance(input_rows, (str, bytes)):
        raise ValueError("Tokenized 'input_ids' batches must be sequences.")

    columns: dict[str, Any] = {}
    for name in _TOKENIZED_MODEL_FIELDS:
        if name not in batch:
            continue
        value = batch[name]
        if hasattr(value, "tolist") and callable(value.tolist):
            value = value.tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"Tokenized {name!r} batches must be sequences.")
        if len(value) != len(input_rows):
            raise ValueError(
                f"Tokenized {name!r} batch size must match 'input_ids'."
            )
        columns[name] = value

    result: dict[str, list[list[int]]] = {name: [] for name in columns}
    maximum_id = -1
    for index, raw_input_ids in enumerate(input_rows):
        input_ids = _tokenized_sequence(
            raw_input_ids, field_name="input_ids"
        )
        if len(input_ids) < 2:
            raise ValueError("Causal-LM token rows need at least two input IDs.")
        if len(input_ids) > sequence_length:
            raise ValueError(
                f"Tokenized sequence length {len(input_ids)} exceeds configured "
                f"sequence_length={sequence_length}; TrainLM will not "
                "silently truncate prepared tokens."
            )
        maximum_id = max(maximum_id, max(input_ids))
        row: dict[str, list[int]] = {"input_ids": input_ids}
        for name in columns:
            if name == "input_ids":
                continue
            values = columns[name]
            allow_ignore = name == "labels"
            sequence = _tokenized_sequence(
                values[index],
                field_name=name,
                allow_ignore_index=allow_ignore,
            )
            if len(sequence) != len(input_ids):
                raise ValueError(
                    f"Tokenized {name!r} rows must match the input_ids length."
                )
            if name in {"attention_mask", "loss_mask"} and any(
                value not in {0, 1} for value in sequence
            ):
                raise ValueError(
                    f"Tokenized {name!r} values must contain only 0 or 1."
                )
            if name == "labels":
                valid_ids = [value for value in sequence if value >= 0]
                if valid_ids:
                    maximum_id = max(maximum_id, max(valid_ids))
            row[name] = sequence
        for name, sequence in row.items():
            result.setdefault(name, []).append(sequence)
    return result, maximum_id


def _scan_tokenized_dataset(dataset: Any, *, sequence_length: int) -> int:
    """Validate mapped token rows even when Hugging Face reuses its map cache."""
    maximum_id = -1
    dataset_iter = getattr(dataset, "iter", None)
    if callable(dataset_iter):
        batches = dataset_iter(batch_size=1024)
    else:
        # Lightweight Dataset-compatible implementations can expose ``map``
        # without Hugging Face Dataset.iter. Force the validation callback to
        # run on the returned data instead of relying on its transform cache.
        mapper = getattr(dataset, "map", None)
        if not callable(mapper):
            raise TypeError(
                "Mapped tokenized data must expose Dataset.iter or Dataset.map."
            )

        def collect(batch: Mapping[str, Any]) -> Mapping[str, Any]:
            nonlocal maximum_id
            _, batch_maximum = _validate_tokenized_batch(
                batch, sequence_length=sequence_length
            )
            maximum_id = max(maximum_id, batch_maximum)
            return batch

        mapper(collect, batched=True, remove_columns=[])
        return maximum_id

    for batch in batches:
        _, batch_maximum = _validate_tokenized_batch(
            batch, sequence_length=sequence_length
        )
        maximum_id = max(maximum_id, batch_maximum)
    return maximum_id


def _resolved_commit(value: Any) -> str | None:
    if isinstance(value, str) and _COMMIT_SHA.fullmatch(value):
        return value
    return None


def _tokenizer_commit(tokenizer: Any) -> str | None:
    revision = getattr(tokenizer, "_commit_hash", None)
    resolved = _resolved_commit(revision)
    if resolved is not None:
        return resolved

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, Mapping):
        return None
    resolved = _resolved_commit(init_kwargs.get("_commit_hash"))
    if resolved is not None:
        return resolved

    snapshot_commits = set()
    for name, value in init_kwargs.items():
        if not name.endswith("_file") or not isinstance(value, (str, Path)):
            continue
        match = _SNAPSHOT_COMMIT.search(str(value))
        if match is not None:
            snapshot_commits.add(match.group(1))
    if len(snapshot_commits) == 1:
        return snapshot_commits.pop()
    if len(snapshot_commits) > 1:
        raise RuntimeError(
            "Tokenizer files resolved from multiple Hub commits; refusing to "
            "train with an inconsistent tokenizer."
        )
    return None


def _tokenizer_model_compatibility(
    model: Any,
    tokenizer: Any,
    *,
    sequence_length: int,
    same_source: bool,
    source_known: bool,
    revision_match: bool | None,
) -> dict[str, Any]:
    """Report structural checks without claiming token-semantic equivalence."""
    token_ids: list[int] = []
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        if isinstance(vocabulary, Mapping):
            token_ids.extend(
                value
                for value in vocabulary.values()
                if type(value) is int and value >= 0
            )
    all_special_ids = getattr(tokenizer, "all_special_ids", ())
    if isinstance(all_special_ids, Sequence) and not isinstance(
        all_special_ids, (str, bytes)
    ):
        token_ids.extend(
            value
            for value in all_special_ids
            if type(value) is int and value >= 0
        )
    try:
        tokenizer_size = len(tokenizer)
    except (TypeError, NotImplementedError):
        tokenizer_size = None
    if type(tokenizer_size) is int and tokenizer_size > 0:
        token_ids.append(tokenizer_size - 1)
    max_token_id = max(token_ids) if token_ids else None

    embedding_rows = None
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        rows = getattr(embeddings, "num_embeddings", None)
        if type(rows) is int and rows > 0:
            embedding_rows = rows
        else:
            weight = getattr(embeddings, "weight", None)
            shape = getattr(weight, "shape", None)
            if shape is not None and len(shape) > 0:
                rows = int(shape[0])
                if rows > 0:
                    embedding_rows = rows
    model_config = getattr(model, "config", None)
    if embedding_rows is None:
        rows = getattr(model_config, "vocab_size", None)
        if type(rows) is int and rows > 0:
            embedding_rows = rows

    if max_token_id is None or embedding_rows is None:
        token_id_bounds = "unknown"
    elif max_token_id < embedding_rows:
        token_id_bounds = "compatible"
    else:
        token_id_bounds = "incompatible"

    context_limit = None
    # Rope scaling can extend a base positional limit. Avoid rejecting those
    # models using a static config field that may no longer describe the limit.
    if not getattr(model_config, "rope_scaling", None):
        for attribute in ("max_position_embeddings", "n_positions", "max_seq_len"):
            value = getattr(model_config, attribute, None)
            if type(value) is int and value > 0:
                context_limit = value
                break
    if context_limit is None:
        context_status = "unknown"
    elif sequence_length <= context_limit:
        context_status = "compatible"
    else:
        context_status = "exceeds_model_limit"

    return {
        "source_relation": (
            "same_source"
            if same_source
            else "separate_source"
            if source_known
            else "unknown"
        ),
        "revision_match": revision_match,
        "token_id_bounds": token_id_bounds,
        "tokenizer_max_token_id": max_token_id,
        "model_embedding_rows": embedding_rows,
        "context_length": context_status,
        "configured_sequence_length": sequence_length,
        "model_context_limit": context_limit,
        "token_semantics": "not_verified_by_trainlm",
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Real) and not isinstance(value, (bool, int, float)):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported run-summary value: {type(value).__name__}.")


def _is_finite_epoch_count(value: Real) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_finetuning(
    config: TrainLMFineTuningConfig,
    *,
    train_dataset: Any | None = None,
    eval_dataset: Any | None = None,
    tokenizer: Any | None = None,
    formatting_func: Callable[[Mapping[str, Any]], str] | None = None,
    data_collator: Callable[[Sequence[Any]], Mapping[str, Any]] | None = None,
    callbacks: Sequence[TrainerCallback] | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> TrainLMFineTuningResult:
    """Run pretrained causal-LM fine-tuning and persist a run summary.

    By default, load the configured pinned Hub dataset. To use preloaded
    Hugging Face ``Dataset`` splits, pass both ``train_dataset`` and
    ``eval_dataset``. With the default ``data_format="text"``,
    ``formatting_func`` receives one row and returns its training text. With
    ``data_format="tokens"``, rows must already contain ``input_ids`` and are
    validated and passed through without decoding or tokenization. A custom
    ``data_collator`` receives the mapped token fields and original columns.
    Supply the same objects and callbacks again when resuming such a run.

    The JSON summary is written under ``args.output_dir`` before work begins,
    then finalized on success or updated with the failed phase and last known
    checkpoint before the original exception is re-raised.
    """
    if not isinstance(config, TrainLMFineTuningConfig):
        raise TypeError("config must be a TrainLMFineTuningConfig instance.")
    if (train_dataset is None) != (eval_dataset is None):
        raise ValueError("Set both train_dataset and eval_dataset, or neither.")
    if train_dataset is None and (
        config.dataset_id is None or config.dataset_revision is None
    ):
        raise ValueError(
            "dataset_id and immutable dataset_revision are required when "
            "run_finetuning loads splits from the Hugging Face Hub."
        )
    if tokenizer is not None and not callable(tokenizer):
        raise TypeError("tokenizer must be callable when provided.")
    if config._tokenizer_object_required and tokenizer is None:
        raise ValueError(
            "This restored run used a tokenizer object without a reproducible "
            "source ID. Pass that tokenizer object again to resume."
        )
    if config._tokenizer_object_required and tokenizer is not None:
        expected_revision = config._expected_tokenizer_revision
        actual_revision = _tokenizer_commit(tokenizer)
        if expected_revision is not None and actual_revision != expected_revision:
            raise ValueError(
                "The supplied tokenizer does not match the immutable revision "
                "recorded by the restored run."
            )
    if formatting_func is not None and not callable(formatting_func):
        raise TypeError("formatting_func must be callable when provided.")
    if config.data_format == "tokens" and formatting_func is not None:
        raise ValueError(
            "formatting_func is only used with data_format='text'; tokenized "
            "rows are passed through without decoding or retokenizing."
        )
    if data_collator is not None and not callable(data_collator):
        raise TypeError("data_collator must be callable when provided.")

    summary_path = Path(config.args.output_dir) / "finetuning-summary.json"
    dataset_source = (
        "provided_dataset" if train_dataset is not None else "huggingface_hub"
    )
    report = _initial_finetuning_summary(
        config,
        summary_path,
        resume_from_checkpoint=resume_from_checkpoint,
        dataset_source=dataset_source,
        formatting_func=formatting_func,
    )
    _write_summary(summary_path, report)
    run_state: dict[str, Any] = {
        "phase": "validation",
        "trainer": None,
        "state": None,
        "evaluation": {},
        "best_model_selection": None,
        "tokenizer_compatibility": None,
        "tokenizer_object_required": config._tokenizer_object_required,
        "dataset_fingerprints": None,
        "max_dataset_token_id": None,
    }
    try:
        result = _run_finetuning_impl(
            config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
            data_collator=data_collator,
            callbacks=callbacks,
            resume_from_checkpoint=resume_from_checkpoint,
            run_state=run_state,
        )
        report.update(result.summary())
        report["data"]["dataset_fingerprints"] = run_state.get(
            "dataset_fingerprints"
        )
        report["data"]["max_dataset_token_id"] = run_state.get(
            "max_dataset_token_id"
        )
        report.update(
            status="completed",
            phase="completed",
            error_type=None,
            last_known_state=_last_known_state(run_state),
        )
        _write_summary(summary_path, report)
        return result
    except Exception as exc:
        report["data"]["dataset_fingerprints"] = run_state.get(
            "dataset_fingerprints"
        )
        report["data"]["max_dataset_token_id"] = run_state.get(
            "max_dataset_token_id"
        )
        report.update(
            status="failed",
            phase=run_state["phase"],
            error_type=type(exc).__name__,
            cause_error_type=(
                type(exc.__cause__).__name__ if exc.__cause__ is not None else None
            ),
            resolved_model_revision=run_state.get("resolved_model_revision"),
            resolved_tokenizer_revision=run_state.get("resolved_tokenizer_revision"),
            model_tokenizer_revision_match=run_state.get(
                "model_tokenizer_revision_match"
            ),
            tokenizer_name_or_path=run_state.get("tokenizer_name_or_path"),
            tokenizer_revision=run_state.get("tokenizer_revision"),
            tokenizer_compatibility=run_state.get("tokenizer_compatibility"),
            tokenizer_object_required=run_state.get(
                "tokenizer_object_required", False
            ),
            training_budget=run_state.get("training_budget", report["training_budget"]),
            best_model_selection=run_state.get("best_model_selection"),
            last_known_state=_last_known_state(run_state),
        )
        known_state = report["last_known_state"]
        report.update(
            step=known_state["step"],
            train_loss=known_state["train_loss"],
            evaluation=known_state["evaluation"],
        )
        if run_state.get("model_path") is not None:
            report["model_path"] = run_state["model_path"]
        try:
            _write_summary(summary_path, report)
        except Exception:
            # Preserve the original run failure if the report location also
            # becomes unavailable.
            pass
        raise


def _initial_finetuning_summary(
    config: TrainLMFineTuningConfig,
    summary_path: Path,
    *,
    resume_from_checkpoint: str | Path | None,
    dataset_source: str,
    formatting_func: Callable[[Mapping[str, Any]], str] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "pretrained_causal_lm_finetuning",
        "status": "not_started",
        "phase": "initialization",
        "error_type": None,
        "cause_error_type": None,
        "summary_path": str(summary_path),
        "model_name_or_path": config.model_name_or_path,
        "model_revision": config.model_revision,
        "tokenizer_name_or_path": config.tokenizer_name_or_path,
        "tokenizer_revision": config.tokenizer_revision,
        "resolved_model_revision": None,
        "resolved_tokenizer_revision": None,
        "model_tokenizer_revision_match": None,
        "tokenizer_compatibility": None,
        "tokenizer_object_required": config._tokenizer_object_required,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "model_path": str(Path(config.args.output_dir) / "model"),
        "resume_from_checkpoint": _json_compatible(resume_from_checkpoint),
        "step": None,
        "train_loss": None,
        "evaluation": {},
        "best_model_selection": None,
        "data": {
            "source": dataset_source,
            "data_format": config.data_format,
            "text_field": config.text_field,
            "train_split": config.train_split,
            "validation_split": config.validation_split,
            "formatting_function": _callable_label(formatting_func),
            "dataset_fingerprints": None,
            "max_dataset_token_id": None,
        },
        "training_arguments": _json_compatible({
            item.name: getattr(config.args, item.name)
            for item in fields(config.args)
        }),
        "training_budget": {
            "num_train_epochs": config.args.num_train_epochs,
            "train_batches_per_epoch": None,
            "max_train_batches": None,
            "effective_max_steps": config.args.max_steps,
        },
        "last_known_state": {
            "step": None,
            "epoch": None,
            "train_loss": None,
            "evaluation": {},
            "latest_checkpoint": None,
        },
    }


def _callable_label(value: Callable[..., Any] | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def _last_known_state(run_state: Mapping[str, Any]) -> dict[str, Any]:
    trainer = run_state.get("trainer")
    state = run_state.get("state")
    if state is None:
        state = getattr(getattr(trainer, "engine", None), "state", None)

    step = getattr(state, "step", None)
    try:
        step = int(step) if step is not None else None
    except (TypeError, ValueError, OverflowError):
        step = None

    train_loss = getattr(state, "loss", None)
    try:
        train_loss = float(train_loss) if train_loss is not None else None
    except (TypeError, ValueError, OverflowError):
        train_loss = None
    if train_loss is not None and not math.isfinite(train_loss):
        train_loss = None

    latest_checkpoint = None
    if trainer is not None:
        try:
            checkpoint = trainer.latest_checkpoint
            latest_checkpoint = str(checkpoint) if checkpoint is not None else None
        except Exception:
            pass

    return {
        "step": step,
        "epoch": getattr(state, "epoch", None),
        "train_loss": train_loss,
        "evaluation": _json_compatible(run_state.get("evaluation", {})),
        "latest_checkpoint": latest_checkpoint,
    }


def _validate_resume_checkpoint(checkpoint: Path) -> None:
    """Check checkpoint publication metadata without loading saved Python data."""
    state_path = checkpoint / "trainer_state.pt"
    if state_path.is_file() and not state_path.is_symlink():
        try:
            with zipfile.ZipFile(state_path) as archive:
                infos = archive.infolist()
                data_infos = [
                    info
                    for info in infos
                    if info.filename.rsplit("/", 1)[-1] == "data.pkl"
                ]
                version_infos = [
                    info
                    for info in infos
                    if info.filename.rsplit("/", 1)[-1] == "version"
                ]
                if len(data_infos) != 1 or len(version_infos) != 1:
                    raise ValueError("missing Torch checkpoint metadata")
                data_info = data_infos[0]
                version_info = version_infos[0]
                if (
                    data_info.filename.rsplit("/", 1)[0]
                    != version_info.filename.rsplit("/", 1)[0]
                    or data_info.file_size == 0
                ):
                    raise ValueError("inconsistent Torch checkpoint metadata")
                archive.read(data_info)
                version = archive.read(version_info).decode("ascii").strip()
                if not version.isdecimal():
                    raise ValueError("invalid Torch checkpoint version")
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            raise ValueError(
                "The published local checkpoint is not a readable Torch "
                "archive. Restore the complete run directory or "
                "start a new run."
            ) from exc
        return

    manifest_path = checkpoint / "manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        from ._tpu_checkpoint import find_latest_committed_tpu_checkpoint

        committed = find_latest_committed_tpu_checkpoint(checkpoint.parent)
        if committed == checkpoint:
            return
        raise ValueError(
            "The published native TPU checkpoint manifest or shard set is "
            "incomplete or invalid. Restore the complete run directory or "
            "start a new run."
        )

    raise ValueError(
        "The published training checkpoint has no supported state file or "
        "native TPU manifest. Restore the complete run directory or start a new run."
    )


def restore_finetuning_config(
    summary_path: str | Path,
    *,
    max_steps: int | None = None,
    num_train_epochs: float | None = None,
) -> TrainLMFineTuningConfig:
    """Restore a fine-tuning config from its saved summary for checkpoint resume.

    All recorded model, data, and training settings are restored. Resolved
    immutable model/tokenizer commits take precedence over mutable revision
    names. If the original tokenizer was a caller-supplied object without a
    reproducible source ID, the same object is required again. Supply a larger
    `max_steps` or `num_train_epochs` for the resumed run. The output directory
    is anchored to the summary file's current location. A published checkpoint
    must exist before the config is returned.
    """
    if (max_steps is None) == (num_train_epochs is None):
        raise ValueError("Set exactly one of max_steps or num_train_epochs.")
    if max_steps is not None and (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps < 1
    ):
        raise ValueError("max_steps must be a positive integer.")
    if num_train_epochs is not None and (
        isinstance(num_train_epochs, bool)
        or not isinstance(num_train_epochs, Real)
        or not _is_finite_epoch_count(num_train_epochs)
        or num_train_epochs <= 0
    ):
        raise ValueError("num_train_epochs must be positive and finite.")

    path = Path(summary_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read a fine-tuning summary from {path}."
        ) from exc
    if (
        not isinstance(report, Mapping)
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("scope") != "pretrained_causal_lm_finetuning"
    ):
        raise ValueError("The file is not a supported TrainLM fine-tuning summary.")

    training_args = report.get("training_arguments")
    data = report.get("data")
    if not isinstance(training_args, Mapping) or not isinstance(data, Mapping):
        raise ValueError("The fine-tuning summary is missing its saved settings.")

    expected_argument_names = {item.name for item in fields(TrainLMTrainingArguments)}
    additive_arguments = {
        "num_train_epochs": None,
        "save_strategy": None,
        "eval_strategy": None,
        "load_best_model_at_end": False,
        "metric_for_best_model": None,
        "greater_is_better": None,
        "early_stopping_patience": None,
        "early_stopping_threshold": 0.0,
    }
    missing_additive = expected_argument_names - set(training_args)
    if (
        missing_additive
        and missing_additive <= additive_arguments.keys()
        and set(training_args) == expected_argument_names - missing_additive
    ):
        # Additive argument fields default to the behavior of older summaries.
        training_args = {
            **training_args,
            **{name: additive_arguments[name] for name in missing_additive},
        }
    if set(training_args) != expected_argument_names:
        raise ValueError(
            "The saved training arguments do not match this TrainLM version."
        )
    restored_args = dict(training_args)
    restored_args["output_dir"] = path.parent
    saved_num_train_epochs = training_args.get("num_train_epochs")
    if max_steps is not None:
        restored_args["max_steps"] = max_steps
        restored_args["num_train_epochs"] = None
    else:
        if saved_num_train_epochs is None:
            raise ValueError(
                "num_train_epochs can extend only a run that was configured "
                "with an epoch budget."
            )
        if (
            isinstance(saved_num_train_epochs, bool)
            or not isinstance(saved_num_train_epochs, Real)
            or not _is_finite_epoch_count(saved_num_train_epochs)
            or saved_num_train_epochs <= 0
        ):
            raise ValueError("The saved epoch budget is invalid.")
        if num_train_epochs <= saved_num_train_epochs:
            raise ValueError(
                "num_train_epochs must be greater than the saved epoch budget "
                f"({saved_num_train_epochs})."
            )
        restored_args["max_steps"] = None
        restored_args["num_train_epochs"] = num_train_epochs
    for name in ("betas", "report_to"):
        if isinstance(restored_args.get(name), list):
            restored_args[name] = tuple(restored_args[name])
    saved_step = report.get("step")
    if max_steps is not None and saved_step is not None:
        if type(saved_step) is not int or saved_step < 0:
            raise ValueError("The fine-tuning summary has an invalid saved step.")
        if max_steps <= saved_step:
            raise ValueError(
                "max_steps must be greater than the last recorded training step "
                f"({saved_step})."
            )

    try:
        args = TrainLMTrainingArguments(**restored_args)
        config = TrainLMFineTuningConfig(
            model_name_or_path=report["model_name_or_path"],
            model_revision=(
                _resolved_commit(report.get("resolved_model_revision"))
                or report.get("model_revision")
            ),
            tokenizer_name_or_path=report.get("tokenizer_name_or_path"),
            tokenizer_revision=(
                _resolved_commit(report.get("resolved_tokenizer_revision"))
                if report.get("tokenizer_name_or_path") is not None
                else report.get("tokenizer_revision")
            ),
            dataset_id=report.get("dataset_id"),
            dataset_revision=report.get("dataset_revision"),
            text_field=data["text_field"],
            train_split=data["train_split"],
            validation_split=data["validation_split"],
            data_format=data.get("data_format", "text"),
            args=args,
        )
        tokenizer_object_required = report.get(
            "tokenizer_object_required", False
        )
        if type(tokenizer_object_required) is not bool:
            raise ValueError("The saved tokenizer source metadata is invalid.")
        object.__setattr__(
            config, "_tokenizer_object_required", tokenizer_object_required
        )
        if tokenizer_object_required:
            object.__setattr__(
                config,
                "_expected_tokenizer_revision",
                _resolved_commit(report.get("resolved_tokenizer_revision")),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "The fine-tuning summary contains invalid saved settings."
        ) from exc
    checkpoint = _latest_training_checkpoint(path.parent)
    if checkpoint is None:
        raise ValueError(
            "No published training checkpoint was found under "
            f"{path.parent}; restore the full run directory or start a new run."
        )
    _validate_resume_checkpoint(checkpoint)
    return config


def _run_finetuning_impl(
    config: TrainLMFineTuningConfig,
    *,
    train_dataset: Any | None = None,
    eval_dataset: Any | None = None,
    tokenizer: Any | None = None,
    formatting_func: Callable[[Mapping[str, Any]], str] | None = None,
    data_collator: Callable[[Sequence[Any]], Mapping[str, Any]] | None = None,
    callbacks: Sequence[TrainerCallback] | None = None,
    resume_from_checkpoint: str | Path | None = None,
    run_state: dict[str, Any],
) -> TrainLMFineTuningResult:
    """Load data, fine-tune, evaluate, and export one pretrained causal LM.

    Hub sources are pinned to an immutable dataset revision. Caller-provided
    data must expose the Hugging Face Dataset ``column_names`` and ``map``
    interface. The result is saved locally and never uploaded.
    """
    if not isinstance(config, TrainLMFineTuningConfig):
        raise TypeError("config must be a TrainLMFineTuningConfig instance.")
    if config.args.accelerator == "tpu":
        raise ValueError(
            "run_finetuning does not support the legacy PyTorch/XLA coordinator "
            "export path. Use accelerator='cpu', 'cuda', or the experimental "
            "'native_tpu' path, or construct TrainLMTrainer directly."
        )
    if callbacks is not None and not all(
        isinstance(callback, TrainerCallback) for callback in callbacks
    ):
        raise TypeError("callbacks must contain TrainLM TrainerCallback instances.")
    if resume_from_checkpoint is not None and not isinstance(
        resume_from_checkpoint, (str, Path)
    ):
        raise TypeError("resume_from_checkpoint must be a path or None.")
    tokenizer_was_provided = tokenizer is not None

    if train_dataset is None:
        if eval_dataset is not None:
            raise ValueError("Set both train_dataset and eval_dataset, or neither.")
        run_state["phase"] = "dependency_loading"
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            if exc.name != "datasets":
                raise
            raise RuntimeError(
                "Hub text fine-tuning requires the optional datasets package; "
                "install TrainLM with `pip install 'trainlm[finetuning]'`."
            ) from exc
    elif eval_dataset is None:
        raise ValueError("Set both train_dataset and eval_dataset, or neither.")

    model_revision = config.model_revision
    run_state["phase"] = "tokenizer_loading"
    tokenizer_source = config.tokenizer_name_or_path
    tokenizer_source_is_declared = tokenizer_source is not None
    if tokenizer is None:
        from transformers import AutoTokenizer

        if tokenizer_source is None:
            tokenizer_source = config.model_name_or_path
        tokenizer_requested_revision = config.tokenizer_revision
        if tokenizer_requested_revision is None and (
            tokenizer_source == config.model_name_or_path
        ):
            tokenizer_requested_revision = model_revision
        run_state["tokenizer_name_or_path"] = tokenizer_source
        run_state["tokenizer_revision"] = tokenizer_requested_revision
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                **(
                    {"revision": tokenizer_requested_revision}
                    if tokenizer_requested_revision is not None
                    else {}
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tokenizer for {tokenizer_source!r}. "
                "Check the model ID, revision, and Hub access."
            ) from exc
    else:
        if tokenizer_source is None:
            candidate = getattr(tokenizer, "name_or_path", None)
            if isinstance(candidate, str) and candidate.strip():
                tokenizer_source = candidate
        tokenizer_requested_revision = config.tokenizer_revision
    if tokenizer_source is None and not tokenizer_source_is_declared:
        # A caller-supplied tokenizer object without an identifier has unknown
        # provenance. Its commit must not be treated as the model's revision.
        tokenizer_source_is_same_as_model = False
    else:
        tokenizer_source_is_same_as_model = (
            tokenizer_source == config.model_name_or_path
        )
    shared_revision_contract = (
        tokenizer_source_is_same_as_model
        and (
            not tokenizer_was_provided
            or config.tokenizer_revision is not None
        )
        and tokenizer_requested_revision == config.model_revision
    )
    run_state["tokenizer_name_or_path"] = tokenizer_source
    run_state["tokenizer_revision"] = tokenizer_requested_revision
    tokenizer_object_required = tokenizer_was_provided and tokenizer_source is None
    run_state["tokenizer_object_required"] = tokenizer_object_required
    resolved_tokenizer_revision = (
        _tokenizer_commit(tokenizer)
        or _resolved_commit(tokenizer_requested_revision)
    )
    run_state["resolved_tokenizer_revision"] = resolved_tokenizer_revision

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "The pretrained tokenizer has no padding or EOS token; configure "
                "a tokenizer with a usable padding token before fine-tuning."
            )
        tokenizer.pad_token = tokenizer.eos_token

    run_state["phase"] = "dataset_loading"
    if train_dataset is None:
        train_source = load_dataset(
            config.dataset_id,
            revision=config.dataset_revision,
            split=config.train_split,
        )
        validation_source = load_dataset(
            config.dataset_id,
            revision=config.dataset_revision,
            split=config.validation_split,
        )
    else:
        train_source = train_dataset
        validation_source = eval_dataset
    run_state["dataset_fingerprints"] = {
        "train": getattr(train_source, "_fingerprint", None),
        "validation": getattr(validation_source, "_fingerprint", None),
    }
    for split_name, dataset in (
        (config.train_split, train_source),
        (config.validation_split, validation_source),
    ):
        columns = getattr(dataset, "column_names", None)
        mapper = getattr(dataset, "map", None)
        if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
            raise TypeError(
                f"{split_name} dataset must expose the Hugging Face Dataset "
                "`column_names` interface."
            )
        if not callable(mapper):
            raise TypeError(
                f"{split_name} dataset must expose the Hugging Face Dataset "
                "`map` method."
            )
        if (
            config.data_format == "text"
            and formatting_func is None
            and config.text_field not in columns
        ):
            raise ValueError(
                f"Dataset split {split_name!r} has no {config.text_field!r} column."
            )
        try:
            if len(dataset) == 0:
                raise ValueError(f"Dataset split {split_name!r} is empty.")
        except (TypeError, NotImplementedError):
            pass

    run_state["phase"] = "tokenizing"
    if config.data_format == "text":
        def tokenize(batch: Mapping[str, Any]) -> Mapping[str, Any]:
            if formatting_func is None:
                texts = batch[config.text_field]
            else:
                values = tuple(batch.values())
                if not values:
                    raise ValueError("Cannot format a dataset batch with no columns.")
                size = len(values[0])
                if any(len(column) != size for column in values):
                    raise ValueError("Dataset batch columns have inconsistent lengths.")
                texts = [
                    formatting_func({name: batch[name][index] for name in batch})
                    for index in range(size)
                ]
            if not isinstance(texts, list) or any(
                not isinstance(text, str) for text in texts
            ):
                raise ValueError(
                    "The selected dataset text or formatting_func must produce "
                    "one string per example."
                )
            return tokenizer(
                texts,
                truncation=True,
                max_length=config.args.sequence_length,
            )

        train_mapper = tokenize
        validation_mapper = tokenize
    else:
        def validate_tokens(batch: Mapping[str, Any]) -> Mapping[str, Any]:
            token_columns, _ = _validate_tokenized_batch(
                batch,
                sequence_length=config.args.sequence_length,
            )
            return token_columns

        train_mapper = validate_tokens
        validation_mapper = validate_tokens

    train_dataset = train_source.map(
        train_mapper,
        batched=True,
        remove_columns=[] if data_collator is not None else train_source.column_names,
    )
    validation_dataset = validation_source.map(
        validation_mapper,
        batched=True,
        remove_columns=(
            [] if data_collator is not None else validation_source.column_names
        ),
    )
    if config.data_format == "tokens":
        # Dataset.map may return cached rows without invoking validate_tokens.
        # Scan the mapped output in bounded batches so embedding-bound and row
        # validation remain effective on both cache hits and cache misses.
        train_maximum = _scan_tokenized_dataset(
            train_dataset,
            sequence_length=config.args.sequence_length,
        )
        validation_maximum = _scan_tokenized_dataset(
            validation_dataset,
            sequence_length=config.args.sequence_length,
        )
        run_state["max_dataset_token_id"] = max(
            train_maximum, validation_maximum
        )

    model_load_revision = (
        (resolved_tokenizer_revision or model_revision)
        if shared_revision_contract
        else model_revision
    )
    run_state["phase"] = "model_loading"
    trainer = TrainLMTrainer.from_pretrained(
        config.model_name_or_path,
        # Pin to the tokenizer snapshot only when model and tokenizer use the
        # same source and requested revision. Independent repositories and
        # revisions must be resolved separately.
        revision=model_load_revision,
        args=config.args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=(
            data_collator
            if data_collator is not None
            else (
                _CausalLMCollator(tokenizer)
                if config.data_format == "text"
                else _TokenizedCausalLMCollator(
                    tokenizer, config.args.sequence_length
                )
            )
        ),
        callbacks=callbacks,
    )
    run_state["trainer"] = trainer
    resolved_training_budget = getattr(trainer, "training_budget", None)
    if isinstance(resolved_training_budget, Mapping):
        run_state["training_budget"] = dict(resolved_training_budget)
    loaded_metadata = getattr(getattr(trainer, "loaded", None), "metadata", None)
    resolved_model_revision = _resolved_commit(
        getattr(loaded_metadata, "resolved_revision", None)
    )
    if resolved_model_revision is None:
        model_config = getattr(getattr(trainer, "model", None), "config", None)
        resolved_model_revision = _resolved_commit(
            getattr(model_config, "_commit_hash", None)
        )
    if resolved_model_revision is None:
        # The model was loaded successfully at this immutable commit even if
        # the installed Transformers version does not return config metadata.
        resolved_model_revision = _resolved_commit(model_load_revision)
    run_state["resolved_model_revision"] = resolved_model_revision
    model_tokenizer_revision_match = None
    if (
        shared_revision_contract
        and resolved_model_revision
        and resolved_tokenizer_revision
    ):
        model_tokenizer_revision_match = (
            resolved_model_revision == resolved_tokenizer_revision
        )
        run_state["model_tokenizer_revision_match"] = (
            model_tokenizer_revision_match
        )
        if not model_tokenizer_revision_match:
            raise RuntimeError(
                "The pretrained model and tokenizer resolved to different Hub "
                f"commits ({resolved_model_revision} and "
                f"{resolved_tokenizer_revision}); training stopped before the "
                "first step."
            )
    run_state["model_tokenizer_revision_match"] = model_tokenizer_revision_match
    if getattr(trainer.model, "config", None) is not None:
        model_config = trainer.model.config
        if getattr(model_config, "pad_token_id", None) is None:
            model_config.pad_token_id = tokenizer.pad_token_id
    tokenizer_compatibility = _tokenizer_model_compatibility(
        trainer.model,
        tokenizer,
        sequence_length=config.args.sequence_length,
        same_source=tokenizer_source_is_same_as_model,
        source_known=tokenizer_source is not None,
        revision_match=model_tokenizer_revision_match,
    )
    run_state["tokenizer_compatibility"] = tokenizer_compatibility
    if tokenizer_compatibility["token_id_bounds"] == "incompatible":
        raise ValueError(
            "The tokenizer can emit token ID "
            f"{tokenizer_compatibility['tokenizer_max_token_id']}, but the model "
            "input embeddings contain only "
            f"{tokenizer_compatibility['model_embedding_rows']} rows. TrainLM "
            "will not resize embeddings or remap token IDs."
        )
    maximum_dataset_token_id = run_state.get("max_dataset_token_id")
    embedding_rows = tokenizer_compatibility.get("model_embedding_rows")
    if (
        isinstance(maximum_dataset_token_id, int)
        and isinstance(embedding_rows, int)
        and maximum_dataset_token_id >= embedding_rows
    ):
        raise ValueError(
            "The tokenized dataset contains token ID "
            f"{maximum_dataset_token_id}, but the model input embeddings contain "
            f"only {embedding_rows} rows. TrainLM will not resize embeddings or "
            "remap token IDs."
        )
    if tokenizer_compatibility["context_length"] == "exceeds_model_limit":
        raise ValueError(
            f"sequence_length={config.args.sequence_length} exceeds the model's "
            "configured context limit of "
            f"{tokenizer_compatibility['model_context_limit']}."
        )
    run_state["phase"] = "training"
    state = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    run_state["state"] = state
    best_model_selection = getattr(trainer, "best_model_selection", None)
    if isinstance(best_model_selection, Mapping):
        run_state["best_model_selection"] = dict(best_model_selection)
    run_state["phase"] = "evaluation"
    evaluation = trainer.evaluate()
    run_state["evaluation"] = evaluation
    run_state["phase"] = "export"
    model_path = trainer.save_model(Path(config.args.output_dir) / "model")
    run_state["model_path"] = str(model_path)
    raw_loss = getattr(state, "loss", None)
    train_loss = float(raw_loss) if raw_loss is not None else None
    if train_loss is not None and not math.isfinite(train_loss):
        train_loss = None
    summary_path = Path(config.args.output_dir) / "finetuning-summary.json"
    run_state["phase"] = "summary"
    result = TrainLMFineTuningResult(
        model_name_or_path=config.model_name_or_path,
        model_revision=config.model_revision,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        model_path=Path(model_path),
        step=int(getattr(state, "step", 0)),
        train_loss=train_loss,
        evaluation={str(name): float(value) for name, value in evaluation.items()},
        resolved_model_revision=resolved_model_revision,
        resolved_tokenizer_revision=resolved_tokenizer_revision,
        model_tokenizer_revision_match=model_tokenizer_revision_match,
        tokenizer_name_or_path=tokenizer_source,
        tokenizer_revision=tokenizer_requested_revision,
        tokenizer_compatibility=tokenizer_compatibility,
        tokenizer_object_required=tokenizer_object_required,
        summary_path=summary_path,
        training_budget=run_state.get(
            "training_budget",
            {
                "num_train_epochs": config.args.num_train_epochs,
                "train_batches_per_epoch": None,
                "max_train_batches": None,
                "effective_max_steps": config.args.max_steps,
            },
        ),
        best_model_selection=run_state.get("best_model_selection"),
    )
    return result
