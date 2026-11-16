"""Task-focused fine-tuning of a pretrained causal language model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
import json
import math
from numbers import Real
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .api import (
    TrainLMTrainer,
    TrainLMTrainingArguments,
    _latest_training_checkpoint,
)
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
    pinned Hub dataset and tokenizes ``text_field``. ``run_finetuning`` can
    instead receive dataset objects and a row formatter. Training and
    evaluation settings live in :class:`TrainLMTrainingArguments`.
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
        if not all(
            isinstance(value, str) and _FIELD_NAME.fullmatch(value)
            for value in (self.text_field, self.train_split, self.validation_split)
        ):
            raise ValueError(
                "text_field and split names must be simple non-empty names."
            )
        if self.train_split == self.validation_split:
            raise ValueError("Training and validation splits must be distinct.")
        if not isinstance(self.args, TrainLMTrainingArguments):
            raise TypeError("args must be a TrainLMTrainingArguments instance.")


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
    """Run pretrained text fine-tuning and persist a success or failure summary.

    By default, load the configured pinned Hub dataset. To use preloaded
    Hugging Face ``Dataset`` splits, pass both ``train_dataset`` and
    ``eval_dataset``. ``formatting_func`` receives one row and returns its
    training text; a custom ``data_collator`` receives the mapped token fields
    and original columns so it can build task-specific labels. Supply the same
    objects and callbacks again when resuming such a run.

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
    if formatting_func is not None and not callable(formatting_func):
        raise TypeError("formatting_func must be callable when provided.")
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
        report.update(
            status="completed",
            phase="completed",
            error_type=None,
            last_known_state=_last_known_state(run_state),
        )
        _write_summary(summary_path, report)
        return result
    except Exception as exc:
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
        "resolved_model_revision": None,
        "resolved_tokenizer_revision": None,
        "model_tokenizer_revision_match": None,
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
            "text_field": config.text_field,
            "train_split": config.train_split,
            "validation_split": config.validation_split,
            "formatting_function": _callable_label(formatting_func),
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

    All recorded model, data, and training settings are restored. Supply a
    larger `max_steps` or `num_train_epochs` for the resumed run. The output
    directory is anchored to the summary file's current location. A published
    checkpoint must exist before the config is returned.
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
            model_revision=report.get("model_revision"),
            dataset_id=report.get("dataset_id"),
            dataset_revision=report.get("dataset_revision"),
            text_field=data["text_field"],
            train_split=data["train_split"],
            validation_split=data["validation_split"],
            args=args,
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
    if tokenizer is None:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                **({"revision": model_revision} if model_revision is not None else {}),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tokenizer for {config.model_name_or_path!r}. "
                "Check the model ID, revision, and Hub access."
            ) from exc
    resolved_tokenizer_revision = (
        _tokenizer_commit(tokenizer) or _resolved_commit(model_revision)
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
        if formatting_func is None and config.text_field not in columns:
            raise ValueError(
                f"Dataset split {split_name!r} has no {config.text_field!r} column."
            )
        try:
            if len(dataset) == 0:
                raise ValueError(f"Dataset split {split_name!r} is empty.")
        except (TypeError, NotImplementedError):
            pass

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

    run_state["phase"] = "tokenizing"
    train_dataset = train_source.map(
        tokenize,
        batched=True,
        remove_columns=[] if data_collator is not None else train_source.column_names,
    )
    validation_dataset = validation_source.map(
        tokenize,
        batched=True,
        remove_columns=(
            [] if data_collator is not None else validation_source.column_names
        ),
    )

    model_load_revision = resolved_tokenizer_revision or model_revision
    run_state["phase"] = "model_loading"
    trainer = TrainLMTrainer.from_pretrained(
        config.model_name_or_path,
        # Pin model loading to the tokenizer's resolved commit when available,
        # so a moving branch cannot select different model and tokenizer files.
        revision=model_load_revision,
        args=config.args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=(
            data_collator
            if data_collator is not None
            else _CausalLMCollator(tokenizer)
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
    if resolved_model_revision and resolved_tokenizer_revision:
        model_tokenizer_revision_match = (
            resolved_model_revision == resolved_tokenizer_revision
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
