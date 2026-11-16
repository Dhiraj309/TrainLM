"""Task-focused fine-tuning of a pretrained causal language model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from .api import TrainLMTrainer, TrainLMTrainingArguments
from .training import TrainerCallback


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_FIELD_NAME = re.compile(r"[A-Za-z0-9._+-]{1,96}\Z")


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


def run_finetuning(
    config: TrainLMFineTuningConfig,
    *,
    callbacks: Sequence[TrainerCallback] | None = None,
    resume_from_checkpoint: str | Path | None = None,
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

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "The pretrained tokenizer has no padding or EOS token; configure "
                "a tokenizer with a usable padding token before fine-tuning."
            )
        tokenizer.pad_token = tokenizer.eos_token

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

    trainer = TrainLMTrainer.from_pretrained(
        config.model_name_or_path,
        revision=model_revision,
        args=config.args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=_CausalLMCollator(tokenizer),
        callbacks=callbacks,
    )
    if getattr(trainer.model, "config", None) is not None:
        model_config = trainer.model.config
        if getattr(model_config, "pad_token_id", None) is None:
            model_config.pad_token_id = tokenizer.pad_token_id
    state = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    evaluation = trainer.evaluate()
    model_path = trainer.save_model(Path(config.args.output_dir) / "model")
    raw_loss = getattr(state, "loss", None)
    train_loss = float(raw_loss) if raw_loss is not None else None
    if train_loss is not None and not math.isfinite(train_loss):
        train_loss = None
    return TrainLMFineTuningResult(
        model_name_or_path=config.model_name_or_path,
        model_revision=config.model_revision,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        model_path=Path(model_path),
        step=int(getattr(state, "step", 0)),
        train_loss=train_loss,
        evaluation={str(name): float(value) for name, value in evaluation.items()},
    )
