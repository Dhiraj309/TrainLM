"""Task-focused fine-tuning of a pretrained causal language model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .api import TrainLMTrainer, TrainLMTrainingArguments
from .training import TrainerCallback


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_FIELD_NAME = re.compile(r"[A-Za-z0-9._+-]{1,96}\Z")
_SNAPSHOT_COMMIT = re.compile(
    r"(?:^|[/\\])snapshots[/\\]([0-9a-f]{40})(?:[/\\]|\Z)"
)


@dataclass(frozen=True, slots=True)
class TrainLMFineTuningConfig:
    """Inputs for fine-tuning a pretrained causal LM on a Hub text dataset.

    The train and validation splits stay separate. TrainLM loads the pretrained
    model and tokenizer, tokenizes the selected text column, then runs the
    familiar trainer lifecycle. Training and evaluation details live in the
    supplied :class:`TrainLMTrainingArguments`.
    """

    model_name_or_path: str
    dataset_id: str
    dataset_revision: str
    args: TrainLMTrainingArguments
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
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty Hugging Face dataset ID.")
        if (
            not isinstance(self.dataset_revision, str)
            or not _COMMIT_SHA.fullmatch(self.dataset_revision)
        ):
            raise ValueError(
                "dataset_revision must be an immutable 40-character commit SHA."
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
    dataset_id: str
    dataset_revision: str
    model_path: Path
    step: int
    train_loss: float | None
    evaluation: Mapping[str, float]
    resolved_model_revision: str | None = None
    resolved_tokenizer_revision: str | None = None
    model_tokenizer_revision_match: bool | None = None
    summary_path: Path | None = None

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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported run-summary value: {type(value).__name__}.")


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
    callbacks: Sequence[TrainerCallback] | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> TrainLMFineTuningResult:
    """Run pretrained text fine-tuning and persist a success or failure summary.

    The JSON summary is written under ``args.output_dir`` before work begins,
    then finalized on success or updated with the failed phase and last known
    checkpoint before the original exception is re-raised.
    """
    if not isinstance(config, TrainLMFineTuningConfig):
        raise TypeError("config must be a TrainLMFineTuningConfig instance.")

    summary_path = Path(config.args.output_dir) / "finetuning-summary.json"
    report = _initial_finetuning_summary(
        config, summary_path, resume_from_checkpoint=resume_from_checkpoint
    )
    _write_summary(summary_path, report)
    run_state: dict[str, Any] = {
        "phase": "validation",
        "trainer": None,
        "state": None,
        "evaluation": {},
    }
    try:
        result = _run_finetuning_impl(
            config,
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
        "data": {
            "text_field": config.text_field,
            "train_split": config.train_split,
            "validation_split": config.validation_split,
        },
        "training_arguments": _json_compatible({
            item.name: getattr(config.args, item.name)
            for item in fields(config.args)
        }),
        "last_known_state": {
            "step": None,
            "train_loss": None,
            "evaluation": {},
            "latest_checkpoint": None,
        },
    }


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
        "train_loss": train_loss,
        "evaluation": _json_compatible(run_state.get("evaluation", {})),
        "latest_checkpoint": latest_checkpoint,
    }


def _run_finetuning_impl(
    config: TrainLMFineTuningConfig,
    *,
    callbacks: Sequence[TrainerCallback] | None = None,
    resume_from_checkpoint: str | Path | None = None,
    run_state: dict[str, Any],
) -> TrainLMFineTuningResult:
    """Load, tokenize, fine-tune, evaluate, and export one pretrained causal LM.

    This convenience path reads immutable Hugging Face dataset revisions and
    requires a separate validation split. It saves a local model and tokenizer
    under ``args.output_dir / "model"``; it never uploads artifacts.
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
    from transformers import AutoTokenizer

    model_revision = config.model_revision
    run_state["phase"] = "tokenizer_loading"
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
    for split_name, dataset in (
        (config.train_split, train_source),
        (config.validation_split, validation_source),
    ):
        if config.text_field not in dataset.column_names:
            raise ValueError(
                f"Dataset split {split_name!r} has no {config.text_field!r} column."
            )
        if len(dataset) == 0:
            raise ValueError(f"Dataset split {split_name!r} is empty.")

    def tokenize(batch: Mapping[str, Any]) -> Mapping[str, Any]:
        texts = batch[config.text_field]
        if not isinstance(texts, list) or any(
            not isinstance(text, str) for text in texts
        ):
            raise ValueError(
                f"Dataset field {config.text_field!r} must contain text strings."
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
        remove_columns=train_source.column_names,
    )
    validation_dataset = validation_source.map(
        tokenize,
        batched=True,
        remove_columns=validation_source.column_names,
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
        data_collator=_CausalLMCollator(tokenizer),
        callbacks=callbacks,
    )
    run_state["trainer"] = trainer
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
    )
    return result
