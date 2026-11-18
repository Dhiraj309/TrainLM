"""Public raw-text preparation for small decoder-only pretraining runs.

The data preparation boundary is intentionally separate from ``TrainLMTrainer``
so the optional Hugging Face Datasets dependency is loaded only when this
workflow is selected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal


DEFAULT_DATASET_ID = "roneneldan/TinyStories"
DEFAULT_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
_DATASET_ID = re.compile(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}\Z")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_FIELD_NAME = re.compile(r"[A-Za-z0-9._+-]{1,96}\Z")
_SPLIT_SELECTOR = re.compile(
    r"(?P<name>[A-Za-z0-9._+-]{1,96})"
    r"(?:\[(?P<start>[0-9]+%?)?:(?P<stop>[0-9]+%?)?\])?\Z"
)
_PREPARATION_CACHE_SCHEMA = 1
_PREPARATION_ALGORITHM = "byte-level-bpe-v1"


def _parse_split_selector(
    value: str,
) -> tuple[str, int | None, int | None, str | None, bool]:
    """Parse the single HF split or numeric/percentage slice we support."""
    if not isinstance(value, str):
        raise ValueError("Dataset split selectors must be strings.")
    match = _SPLIT_SELECTOR.fullmatch(value)
    if match is None:
        raise ValueError(
            "Dataset splits must be names or simple HF slices such as "
            "'train[:90%]' and 'train[90%:]'."
        )
    raw_start, raw_stop = match.group("start", "stop")
    if raw_start is None and raw_stop is None:
        return match.group("name"), None, None, None, False
    units = {
        "%" if raw.endswith("%") else "examples"
        for raw in (raw_start, raw_stop)
        if raw is not None
    }
    if len(units) != 1:
        raise ValueError(
            "A split slice must use either example counts or percentages, not both."
        )
    unit = next(iter(units))
    start = (
        int(raw_start[:-1] if unit == "%" else raw_start)
        if raw_start is not None
        else None
    )
    stop = (
        int(raw_stop[:-1] if unit == "%" else raw_stop)
        if raw_stop is not None
        else None
    )
    if unit == "%" and any(
        value is not None and value > 100 for value in (start, stop)
    ):
        raise ValueError("Percentage split boundaries must be between 0 and 100.")
    if start is not None and stop is not None and start >= stop:
        raise ValueError("A split slice must have a start before its stop.")
    return match.group("name"), start, stop, unit, True


def _validate_disjoint_split_selectors(
    train: tuple[str, int | None, int | None, str | None, bool],
    validation: tuple[str, int | None, int | None, str | None, bool],
) -> None:
    """Reject identical or overlapping selections from one source split."""
    if train[0] != validation[0]:
        return
    if not train[4] or not validation[4] or train[3] != validation[3]:
        raise ValueError(
            "Training and validation splits must be distinct; selecting the same "
            "source split requires both selectors to use disjoint slices in the "
            "same units."
        )
    unit = train[3]
    default_stop = 100 if unit == "%" else math.inf
    train_start = train[1] if train[1] is not None else 0
    train_stop = train[2] if train[2] is not None else default_stop
    validation_start = validation[1] if validation[1] is not None else 0
    validation_stop = validation[2] if validation[2] is not None else default_stop
    if not (
        train_stop <= validation_start or validation_stop <= train_start
    ):
        raise ValueError(
            "Training and validation slices from the same source split "
            "must not overlap."
        )


@dataclass(frozen=True, slots=True)
class TrainLMPretrainingConfig:
    """Settings for causal pretraining from a Hub, Dataset, or local text source.

    The default path preserves the original GPT-2 plus corpus-trained BPE
    workflow. Set ``model_name_or_path`` to use a registered Transformers
    config; TrainLM then reuses that repository's tokenizer by default and
    initializes random weights. Set ``model_initialization="pretrained"`` to
    continue training its weights without resizing embeddings. Existing HF
    tokenizers can be selected independently with ``tokenizer_name_or_path``.
    Set ``data_format="tokens"`` with ``dataset_source="provided"`` to pack
    tokenized Dataset rows. TrainLM creates causal targets from ``input_ids``;
    the dataset must have been tokenized by the selected tokenizer source.
    Set ``tokenizer_mode="train_bpe"`` to explicitly train a corpus tokenizer
    for random initialization. Prepared tokenizers and packed shards are
    integrity-checked and reused when their source identities match.
    The learning-rate schedule defaults to cosine. If neither warmup option is
    set, the run uses the derived floor-of-10%-of-steps warmup (capped at 100).
    ``warmup_ratio`` follows Transformers semantics and rounds the requested
    fraction up to a whole optimizer step. Set at most one of
    ``warmup_steps`` and ``warmup_ratio``. Supported schedules are
    ``constant``, ``linear``, ``cosine``, and ``wsd``. `save_steps` and
    `eval_steps` independently override the default cadence derived from the
    training budget. Best-checkpoint selection uses validation loss; optional
    early stopping counts consecutive evaluations without sufficient progress.
    AdamW `betas` and `eps` set its moment estimates; `max_grad_norm=0` disables
    gradient clipping.
    """

    output_dir: str | Path = Path("runs/pretraining")
    dataset_source: Literal["hub", "provided"] = "hub"
    data_format: Literal["text", "tokens"] = "text"
    dataset_id: str | None = None
    dataset_revision: str | None = None
    train_file: str | Path | None = None
    validation_file: str | Path | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    text_field: str = "text"
    max_tokens: int = 10_000
    sequence_length: int = 128
    max_train_examples: int = 2_000
    max_eval_examples: int = 200
    tokenizer_vocab_size: int = 16_000
    tokenizer_mode: Literal["auto", "train_bpe"] = "auto"
    tokenizer_name_or_path: str | Path | None = None
    tokenizer_revision: str | None = None
    model_embedding_size: int = 768
    model_layers: int = 6
    model_attention_heads: int = 12
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    accelerator: Literal["auto", "cpu", "cuda", "native_tpu"] = "auto"
    bf16: bool = False
    torch_compile: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    max_eval_batches: int = 20
    seed: int = 42
    model_type: str = "gpt2"
    model_name_or_path: str | Path | None = None
    model_revision: str | None = None
    model_initialization: Literal["random", "pretrained"] = "random"
    trust_remote_code: bool = False
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    model_config_overrides: Mapping[str, Any] | None = None
    save_total_limit: int | None = None
    lr_scheduler_type: Literal["constant", "linear", "cosine", "wsd"] = "cosine"
    warmup_steps: int | None = None
    warmup_ratio: float | None = None
    save_steps: int | None = None
    eval_steps: int | None = None
    load_best_model_at_end: bool = False
    early_stopping_patience: int | None = None
    early_stopping_threshold: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    dataset_config_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        for name in ("model_name_or_path", "tokenizer_name_or_path"):
            source = getattr(self, name)
            if source is not None:
                if not isinstance(source, (str, Path)) or not str(source).strip():
                    raise ValueError(f"{name} must be a non-empty Hub ID or local path.")
                object.__setattr__(self, name, str(source))
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)) or not str(self.cache_dir).strip():
                raise ValueError("cache_dir must be a non-empty path when configured.")
            object.__setattr__(self, "cache_dir", str(Path(self.cache_dir).expanduser()))
        for name in ("model_revision", "tokenizer_revision"):
            revision = getattr(self, name)
            if revision is not None and (
                not isinstance(revision, str) or not revision.strip()
            ):
                raise ValueError(f"{name} must be a non-empty revision when configured.")
        if not isinstance(self.model_initialization, str) or self.model_initialization not in {
            "random", "pretrained"
        }:
            raise ValueError("model_initialization must be 'random' or 'pretrained'.")
        if self.model_revision is not None and self.model_name_or_path is None:
            raise ValueError("model_revision requires model_name_or_path.")
        if self.tokenizer_revision is not None and (
            self.tokenizer_name_or_path is None and self.model_name_or_path is None
        ):
            raise ValueError(
                "tokenizer_revision requires tokenizer_name_or_path or model_name_or_path."
            )
        if self.model_initialization == "pretrained" and self.model_name_or_path is None:
            raise ValueError(
                "model_initialization='pretrained' requires model_name_or_path."
            )
        if not isinstance(self.tokenizer_mode, str) or self.tokenizer_mode not in {
            "auto", "train_bpe"
        }:
            raise ValueError("tokenizer_mode must be 'auto' or 'train_bpe'.")
        if not isinstance(self.data_format, str) or self.data_format not in {
            "text", "tokens"
        }:
            raise ValueError("data_format must be 'text' or 'tokens'.")
        if self.tokenizer_mode == "train_bpe" and self.tokenizer_name_or_path is not None:
            raise ValueError(
                "tokenizer_mode='train_bpe' cannot be combined with "
                "tokenizer_name_or_path."
            )
        if self.data_format == "tokens":
            if self.dataset_source != "provided":
                raise ValueError(
                    "data_format='tokens' requires dataset_source='provided'."
                )
            if self.tokenizer_mode == "train_bpe":
                raise ValueError(
                    "Tokenized datasets need the tokenizer that produced their "
                    "input_ids; tokenizer_mode='train_bpe' cannot be used."
                )
            if (
                self.tokenizer_name_or_path is None
                and self.model_name_or_path is None
            ):
                raise ValueError(
                    "data_format='tokens' requires tokenizer_name_or_path or "
                    "model_name_or_path so token IDs have a recorded tokenizer."
                )
        if (
            self.tokenizer_mode == "train_bpe"
            and self.model_initialization == "pretrained"
        ):
            raise ValueError(
                "A pretrained model cannot use a newly trained tokenizer. "
                "Select a tokenizer compatible with its existing embeddings."
            )
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be boolean.")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be boolean.")
        if self.local_files_only and not (
            self.model_name_or_path or self.tokenizer_name_or_path
        ):
            raise ValueError(
                "local_files_only requires a model or tokenizer source."
            )
        if self.model_initialization == "pretrained" and self.model_config_overrides:
            raise ValueError(
                "model_config_overrides are only supported for random initialization; "
                "continued pretraining preserves the selected checkpoint config."
            )
        if isinstance(self.betas, list):
            object.__setattr__(self, "betas", tuple(self.betas))
        if not isinstance(self.dataset_source, str) or self.dataset_source not in {
            "hub", "provided"
        }:
            raise ValueError("dataset_source must be 'hub' or 'provided'.")
        local_files = self.train_file is not None or self.validation_file is not None
        if local_files:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(
                    "train_file and validation_file must be configured together."
                )
            if (
                self.dataset_id is not None
                or self.dataset_config_name is not None
                or self.dataset_revision is not None
                or self.dataset_source != "hub"
            ):
                raise ValueError(
                    "Choose either local text files or a Hugging Face dataset, "
                    "not both; provided Dataset objects are another exclusive source."
                )
            if self.text_field != "text":
                raise ValueError(
                    "Local text files use the 'text' field; set text_field='text'."
                )
            paths = []
            for name in ("train_file", "validation_file"):
                raw_path = getattr(self, name)
                if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
                    raise ValueError(f"{name} must be a non-empty file path.")
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise ValueError(f"{name} must point to a regular text file.")
                object.__setattr__(self, name, path)
                paths.append(path)
            if paths[0] == paths[1]:
                raise ValueError("Training and validation files must be distinct.")
        else:
            if self.dataset_source == "provided":
                if any(value is not None for value in (
                    self.dataset_id, self.dataset_revision, self.dataset_config_name
                )):
                    raise ValueError(
                        "dataset_source='provided' cannot be combined with "
                        "dataset_id, dataset_revision, or dataset_config_name."
                    )
            else:
                if self.dataset_config_name is not None and self.dataset_id is None:
                    raise ValueError(
                        "dataset_config_name requires dataset_id."
                    )
                if self.dataset_id is None and self.dataset_revision is None:
                    object.__setattr__(self, "dataset_id", DEFAULT_DATASET_ID)
                    object.__setattr__(
                        self, "dataset_revision", DEFAULT_DATASET_REVISION
                    )
                if not isinstance(self.dataset_id, str) or not _DATASET_ID.fullmatch(
                    self.dataset_id
                ):
                    raise ValueError("dataset_id must be a Hugging Face owner/name ID.")
                if self.dataset_revision is None:
                    object.__setattr__(self, "dataset_revision", "main")
                if (
                    not isinstance(self.dataset_revision, str)
                    or not self.dataset_revision.strip()
                    or len(self.dataset_revision) > 256
                    or "\x00" in self.dataset_revision
                ):
                    raise ValueError(
                        "dataset_revision must be a non-empty Hub revision."
                    )
        if self.dataset_config_name is not None and (
            not isinstance(self.dataset_config_name, str)
            or not self.dataset_config_name.strip()
            or len(self.dataset_config_name) > 256
            or "\x00" in self.dataset_config_name
        ):
            raise ValueError(
                "dataset_config_name must be a non-empty Hugging Face dataset config."
            )
        if not isinstance(self.text_field, str) or not _FIELD_NAME.fullmatch(
            self.text_field
        ):
            raise ValueError("text_field must be a simple field name.")
        train_selector = _parse_split_selector(self.train_split)
        validation_selector = _parse_split_selector(self.validation_split)
        _validate_disjoint_split_selectors(train_selector, validation_selector)
        if local_files and (train_selector[3] or validation_selector[3]):
            raise ValueError("Split slicing is only supported for Hub datasets.")

        for name in (
            "max_tokens", "sequence_length", "max_train_examples",
            "max_eval_examples", "tokenizer_vocab_size", "model_embedding_size",
            "model_layers", "model_attention_heads",
            "per_device_train_batch_size", "per_device_eval_batch_size",
            "gradient_accumulation_steps", "logging_steps", "max_eval_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two.")
        if self.save_total_limit is not None and (
            isinstance(self.save_total_limit, bool)
            or not isinstance(self.save_total_limit, int)
            or self.save_total_limit < 1
        ):
            raise ValueError("save_total_limit must be positive when configured.")
        external_tokenizer = self.tokenizer_mode == "auto" and (
            self.tokenizer_name_or_path is not None
            or self.model_name_or_path is not None
        )
        if not external_tokenizer and self.tokenizer_vocab_size < 260:
            raise ValueError("Byte-level BPE needs tokenizer_vocab_size >= 260.")
        if not external_tokenizer and self.tokenizer_vocab_size > 1 << 32:
            raise ValueError(
                "tokenizer_vocab_size must fit the packed uint32 token format "
                "(at most 2**32)."
            )
        if (not isinstance(self.model_type, str)
                or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.model_type)):
            raise ValueError("model_type must be a Hugging Face model type name.")
        if self.model_config_overrides is not None:
            if not isinstance(self.model_config_overrides, Mapping):
                raise ValueError("model_config_overrides must be a mapping.")
            overrides = dict(self.model_config_overrides)
            if any(not isinstance(key, str) for key in overrides):
                raise ValueError("model_config_overrides keys must be strings.")
            reserved = {
                "model_type", "vocab_size", "pad_token_id", "bos_token_id",
                "eos_token_id",
            }
            conflicting = reserved.intersection(overrides)
            if conflicting:
                names = ", ".join(sorted(conflicting))
                raise ValueError(
                    "model_config_overrides cannot set tokenizer-owned fields: "
                    f"{names}. TrainLM derives vocabulary and special-token IDs."
                )
            try:
                overrides = json.loads(json.dumps(overrides, allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "model_config_overrides must contain JSON-compatible values."
                ) from exc
            object.__setattr__(self, "model_config_overrides", overrides)
        else:
            object.__setattr__(self, "model_config_overrides", {})
        if self.model_name_or_path is not None and (
            self.model_embedding_size != 768
            or self.model_layers != 6
            or self.model_attention_heads != 12
        ):
            raise ValueError(
                "model_embedding_size, model_layers, and model_attention_heads "
                "are for the built-in GPT-2 config; use model_config_overrides "
                "to adjust a Hub architecture."
            )
        if self.model_name_or_path is None and self.model_type != "gpt2" and (
            self.model_embedding_size != 768
            or self.model_layers != 6
            or self.model_attention_heads != 12
        ):
            raise ValueError(
                "model_embedding_size, model_layers, and model_attention_heads "
                "configure only the default GPT-2 model; use "
                "model_config_overrides for other Hugging Face architectures."
            )
        if self.model_embedding_size % self.model_attention_heads:
            raise ValueError(
                "model_embedding_size must be divisible by model_attention_heads."
            )
        if (isinstance(self.learning_rate, bool)
                or not isinstance(self.learning_rate, Real)
                or not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(self.lr_scheduler_type, str) or self.lr_scheduler_type not in {
            "constant", "linear", "cosine", "wsd"
        }:
            raise ValueError(
                "lr_scheduler_type must be 'constant', 'linear', 'cosine', or 'wsd'."
            )
        if self.warmup_steps is not None and (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
        ):
            raise ValueError(
                "warmup_steps must be a non-negative integer or None for "
                "automatic warmup."
            )
        if self.warmup_ratio is not None and (
            isinstance(self.warmup_ratio, bool)
            or not isinstance(self.warmup_ratio, Real)
            or not math.isfinite(self.warmup_ratio)
            or not 0 <= self.warmup_ratio <= 1
        ):
            raise ValueError("warmup_ratio must be finite and between 0 and 1.")
        if self.warmup_steps is not None and self.warmup_ratio is not None:
            raise ValueError("Set only one of warmup_steps and warmup_ratio.")
        for name in ("save_steps", "eval_steps"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer when configured.")
        if not isinstance(self.load_best_model_at_end, bool):
            raise ValueError("load_best_model_at_end must be boolean.")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be a positive integer.")
        if (
            isinstance(self.early_stopping_threshold, bool)
            or not isinstance(self.early_stopping_threshold, Real)
            or not math.isfinite(self.early_stopping_threshold)
            or self.early_stopping_threshold < 0
        ):
            raise ValueError(
                "early_stopping_threshold must be finite and non-negative."
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
        if self.load_best_model_at_end:
            tokens_per_update = (
                (self.sequence_length - 1)
                * self.per_device_train_batch_size
                * self.gradient_accumulation_steps
            )
            max_steps = math.ceil(self.max_tokens / tokens_per_update)
            cadence = max(1, max_steps // 10)
            save_steps = self.save_steps if self.save_steps is not None else cadence
            eval_steps = self.eval_steps if self.eval_steps is not None else cadence
            if eval_steps > max_steps:
                raise ValueError(
                    "load_best_model_at_end requires at least one evaluation; "
                    "set eval_steps no greater than the planned optimizer steps."
                )
            if save_steps % eval_steps:
                raise ValueError(
                    "save_steps must be a multiple of eval_steps when "
                    "load_best_model_at_end is enabled."
                )
        if (isinstance(self.weight_decay, bool)
                or not isinstance(self.weight_decay, Real)
                or not math.isfinite(self.weight_decay)
                or self.weight_decay < 0):
            raise ValueError("weight_decay must be finite and non-negative.")
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(
                isinstance(beta, bool)
                or not isinstance(beta, Real)
                or not math.isfinite(beta)
                or not 0 <= beta < 1
                for beta in self.betas
            )
        ):
            raise ValueError("betas must contain two finite values in [0, 1).")
        if (
            isinstance(self.eps, bool)
            or not isinstance(self.eps, Real)
            or not math.isfinite(self.eps)
            or self.eps <= 0
        ):
            raise ValueError("eps must be finite and positive.")
        if (
            isinstance(self.max_grad_norm, bool)
            or not isinstance(self.max_grad_norm, Real)
            or not math.isfinite(self.max_grad_norm)
            or self.max_grad_norm < 0
        ):
            raise ValueError("max_grad_norm must be finite and non-negative.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not isinstance(self.accelerator, str) or self.accelerator not in {
            "auto", "cpu", "cuda", "native_tpu"
        }:
            raise ValueError(
                "from_scratch supports auto, cpu, cuda, and native_tpu; "
                "the XLA coordinator cannot reconstruct this random model yet."
            )
        if not all(isinstance(value, bool) for value in (
            self.bf16, self.torch_compile, self.gradient_checkpointing
        )):
            raise ValueError(
                "bf16, torch_compile, and gradient_checkpointing must be booleans."
            )
        if self.torch_compile and self.accelerator != "native_tpu":
            raise ValueError(
                "torch_compile currently requires accelerator='native_tpu'."
            )


def _token_documents(
    dataset: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_examples: int,
    text_field: str = "text",
    batch_size: int = 64,
):
    """Yield EOS-separated token documents without materializing the split."""
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise RuntimeError("The selected tokenizer must define an integer EOS token.")
    pending: list[str] = []
    seen = 0

    def encode(texts: list[str]):
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        for token_ids in encoded["input_ids"]:
            yield [int(token) for token in token_ids] + [eos_token_id]

    for row in dataset:
        if seen >= max_examples:
            break
        text = row.get(text_field) if isinstance(row, Mapping) else None
        if not isinstance(text, str):
            raise ValueError(f"Training rows must contain a string {text_field!r} field.")
        pending.append(text)
        seen += 1
        if len(pending) == batch_size:
            yield from encode(pending)
            pending.clear()
    if pending:
        yield from encode(pending)
    if seen == 0:
        raise ValueError("The selected dataset split contains no text examples.")


def _token_id_documents(
    dataset: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_examples: int,
):
    """Yield supplied token rows as EOS-separated documents.

    TrainLM owns causal labels: the packed reader mirrors ``input_ids`` into
    ``labels`` and the causal task applies exactly one next-token shift.
    Caller-provided labels and loss masks are intentionally unsupported here.
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise RuntimeError("The selected tokenizer must define an integer EOS token.")
    if max_examples < 1:
        raise ValueError("Tokenized dataset example limits must be positive.")

    seen = 0
    for row in dataset:
        if seen >= max_examples:
            break
        if not isinstance(row, Mapping) or "input_ids" not in row:
            raise ValueError("Tokenized rows must contain an 'input_ids' sequence.")
        unsupported = sorted(
            {
                "labels",
                "loss_mask",
                "position_ids",
                "token_type_ids",
                "special_tokens_mask",
            }.intersection(row)
        )
        if unsupported:
            raise ValueError(
                "Tokenized scratch/CPT consumes only input_ids and optional "
                "attention_mask; TrainLM generates next-token targets from "
                f"input_ids. Remove unsupported fields: {', '.join(unsupported)}."
            )

        raw_ids = row["input_ids"]
        if hasattr(raw_ids, "tolist") and callable(raw_ids.tolist):
            raw_ids = raw_ids.tolist()
        if (
            not isinstance(raw_ids, Sequence)
            or isinstance(raw_ids, (str, bytes, bytearray))
            or not raw_ids
        ):
            raise ValueError("Tokenized 'input_ids' values must be non-empty sequences.")
        token_ids: list[int] = []
        for token_id in raw_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise ValueError("Tokenized input_ids values must be integer IDs.")
            token_id = int(token_id)
            if token_id < 0:
                raise ValueError("Tokenized input_ids values must be non-negative.")
            token_ids.append(token_id)

        raw_mask = row.get("attention_mask")
        if raw_mask is not None:
            if hasattr(raw_mask, "tolist") and callable(raw_mask.tolist):
                raw_mask = raw_mask.tolist()
            if (
                not isinstance(raw_mask, Sequence)
                or isinstance(raw_mask, (str, bytes, bytearray))
                or len(raw_mask) != len(token_ids)
            ):
                raise ValueError(
                    "Tokenized attention_mask rows must match input_ids length."
                )
            mask: list[int] = []
            for value in raw_mask:
                if not isinstance(value, Integral) or int(value) not in {0, 1}:
                    raise ValueError("Tokenized attention_mask values must be 0 or 1.")
                mask.append(int(value))
            active_positions = [index for index, value in enumerate(mask) if value]
            if not active_positions:
                raise ValueError("Tokenized attention_mask cannot mask every input ID.")
            start, stop = active_positions[0], active_positions[-1] + 1
            if any(value == 0 for value in mask[start:stop]):
                raise ValueError(
                    "Tokenized attention_mask cannot contain interior padding holes."
                )
            token_ids = token_ids[start:stop]

        if not token_ids:
            raise ValueError("Tokenized rows must contain at least one unmasked token.")
        if token_ids[-1] != eos_token_id:
            token_ids.append(eos_token_id)
        seen += 1
        yield token_ids
    if seen == 0:
        raise ValueError("The selected tokenized dataset split contains no rows.")


def _scratch_dataset_documents(
    dataset: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    config: TrainLMPretrainingConfig,
    *,
    split: Literal["train", "validation"],
):
    """Select text tokenization or validated prepared-token row ingestion."""
    if config.data_format == "tokens":
        max_examples = (
            config.max_train_examples if split == "train" else config.max_eval_examples
        )
        return _token_id_documents(
            dataset, tokenizer, max_examples=max_examples
        )
    return _token_documents(
        dataset,
        tokenizer,
        max_examples=(
            config.max_train_examples if split == "train" else config.max_eval_examples
        ),
        text_field=config.text_field,
    )


def _train_byte_level_tokenizer(
    dataset: Iterable[Mapping[str, Any]],
    *,
    max_examples: int,
    vocab_size: int,
    text_field: str = "text",
):
    """Learn a compact tokenizer from bounded text in the training split."""
    if max_examples < 1:
        raise ValueError("Tokenizer training examples must be positive.")
    if vocab_size < 260:
        raise ValueError("Byte-level BPE vocab_size must be at least 260.")

    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise ImportError(
            "Scratch pretraining needs tokenizers and Transformers installed."
        ) from exc

    seen = 0

    def text_documents():
        nonlocal seen
        for row in dataset:
            if seen >= max_examples:
                break
            text = row.get(text_field) if isinstance(row, Mapping) else None
            if not isinstance(text, str):
                raise ValueError(
                    f"Tokenizer rows must contain a string {text_field!r} field."
                )
            seen += 1
            yield text
        if seen == 0:
            raise ValueError("The tokenizer training split contains no text examples.")

    core = Tokenizer(models.BPE(unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.train_from_iterator(
        text_documents(),
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "<|pad|>", "<|bos|>", "<|eos|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    core.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )
    if not all(
        isinstance(token_id, int)
        for token_id in (
            tokenizer.unk_token_id, tokenizer.pad_token_id,
            tokenizer.bos_token_id, tokenizer.eos_token_id,
        )
    ):
        raise RuntimeError("The trained tokenizer is missing a special token ID.")
    return tokenizer, seen


def _scratch_gpt2_config(
    *,
    vocab_size: int,
    sequence_length: int,
    embedding_size: int,
    layers: int,
    attention_heads: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
):
    """Build the starter GPT-2 config from explicit dimensions."""
    if min(vocab_size, sequence_length, embedding_size, layers, attention_heads) < 1:
        raise ValueError("Scratch GPT-2 dimensions and vocabulary must be positive.")
    if embedding_size % attention_heads:
        raise ValueError("model embedding size must be divisible by attention heads.")
    from transformers import GPT2Config

    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=sequence_length,
        n_ctx=sequence_length,
        n_embd=embedding_size,
        n_layer=layers,
        n_head=attention_heads,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


def _scratch_huggingface_config(
    config: TrainLMPretrainingConfig,
    *,
    vocab_size: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
):
    """Build a registered Transformers config without loading model weights."""
    from transformers import AutoConfig

    architecture_values: dict[str, Any] = {}
    if config.model_type == "gpt2":
        architecture_values.update(
            n_positions=config.sequence_length,
            n_ctx=config.sequence_length,
            n_embd=config.model_embedding_size,
            n_layer=config.model_layers,
            n_head=config.model_attention_heads,
        )
    architecture_values.update(config.model_config_overrides or {})
    # The trained tokenizer is authoritative for vocabulary and token IDs.
    architecture_values.update(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    model_config = AutoConfig.for_model(config.model_type, **architecture_values)

    context_limit = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(context_limit, int):
        context_limit = getattr(model_config, "n_positions", None)
    if isinstance(context_limit, int) and config.sequence_length > context_limit:
        raise ValueError(
            f"sequence_length={config.sequence_length} exceeds the selected "
            f"{config.model_type} context limit ({context_limit})."
        )
    return model_config


def _load_existing_scratch_tokenizer(
    config: TrainLMPretrainingConfig,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Load an explicitly selected tokenizer, or the model repo's tokenizer."""
    if config.tokenizer_mode == "train_bpe":
        return None, None
    source = config.tokenizer_name_or_path or config.model_name_or_path
    if source is None:
        return None, None
    revision = config.tokenizer_revision
    if config.tokenizer_name_or_path is None and revision is None:
        revision = config.model_revision

    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        "use_fast": True,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if config.cache_dir is not None:
        kwargs["cache_dir"] = config.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError(
            "The selected tokenizer must define eos_token_id for document-packed "
            "causal pretraining. Choose a tokenizer with an EOS token."
        )
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError(
                "The selected tokenizer has no pad token or EOS token to use as "
                "padding. Configure a compatible tokenizer before training."
            )
        tokenizer.pad_token = eos_token
    resolved_revision = (
        getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    )
    local_source = Path(source).expanduser().exists()
    same_model_snapshot_contract = (
        config.model_name_or_path is not None
        and source == config.model_name_or_path
        and revision == config.model_revision
    )
    source_relation = (
        "same_model_source_pending_resolution"
        if config.tokenizer_name_or_path is None or same_model_snapshot_contract
        else "independent_source_semantics_unverified"
    )
    metadata = {
        "requested_source": str(source),
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "local_source": local_source,
        "source_relation": source_relation,
        "mapping_sha256": _tokenizer_mapping_sha256(tokenizer),
    }
    tokenizer._trainlm_source_metadata = metadata
    return tokenizer, metadata


def _tokenizer_vocab_capacity(tokenizer: Any) -> int:
    """Return one past the largest token ID, including sparse/added IDs."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else None
    if not isinstance(vocab, Mapping) or not vocab:
        capacity = len(tokenizer)
    else:
        token_ids = list(vocab.values())
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        ):
            raise ValueError("Tokenizer vocabulary IDs must be non-negative integers.")
        capacity = max(len(tokenizer), max(token_ids) + 1)
    special_ids = [
        getattr(tokenizer, f"{name}_token_id", None)
        for name in ("unk", "pad", "bos", "eos")
    ]
    for token_id in special_ids:
        if token_id is None:
            continue
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("Tokenizer special-token IDs must be non-negative integers.")
        capacity = max(capacity, token_id + 1)
    if not 1 <= capacity <= 1 << 32:
        raise ValueError(
            "The tokenizer's highest token ID must fit the packed uint32 format."
        )
    return capacity


def _load_scratch_model(
    config: TrainLMPretrainingConfig,
    *,
    tokenizer: Any,
):
    """Build random HF weights or load pretrained weights without resizing."""
    from trainlm.config import ModelSourceConfig
    from trainlm.model import load_huggingface_causal_lm

    vocab_size = _tokenizer_vocab_capacity(tokenizer)
    overrides = dict(config.model_config_overrides or {})
    if config.model_initialization == "random":
        if config.model_name_or_path is None and config.model_type == "gpt2":
            overrides = {
                "n_positions": config.sequence_length,
                "n_ctx": config.sequence_length,
                "n_embd": config.model_embedding_size,
                "n_layer": config.model_layers,
                "n_head": config.model_attention_heads,
                **overrides,
            }
        overrides.update(
            vocab_size=vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    source = ModelSourceConfig(
        provider="huggingface",
        initialization=(
            "pretrained"
            if config.model_initialization == "pretrained"
            else "config"
        ),
        name_or_path=(
            str(config.model_name_or_path)
            if config.model_name_or_path is not None
            else None
        ),
        model_type=(config.model_type if config.model_name_or_path is None else None),
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        local_files_only=config.local_files_only,
        config_overrides=overrides,
    )
    loaded = load_huggingface_causal_lm(source)
    model = loaded.model
    if config.model_name_or_path is not None:
        model_source_metadata = loaded.metadata.to_dict()
        if config.model_initialization == "random":
            model_source_metadata["effective_vocab_size"] = vocab_size
        model._trainlm_model_source_metadata = model_source_metadata
    tokenizer_source_metadata = getattr(tokenizer, "_trainlm_source_metadata", None)
    if isinstance(tokenizer_source_metadata, Mapping):
        if tokenizer_source_metadata.get("source_relation") == (
            "same_model_source_pending_resolution"
        ):
            tokenizer_revision = tokenizer_source_metadata.get("resolved_revision")
            model_revision = loaded.metadata.resolved_revision
            if (
                isinstance(tokenizer_revision, str)
                and isinstance(model_revision, str)
                and tokenizer_revision != model_revision
            ):
                raise ValueError(
                    "The selected model config and tokenizer resolved to different "
                    "Hub commits. Pin matching model_revision and tokenizer_revision "
                    "values or choose a compatible tokenizer source."
                )
            tokenizer_source_metadata["source_relation"] = (
                "same_model_snapshot"
                if isinstance(tokenizer_revision, str)
                and isinstance(model_revision, str)
                else "same_model_source_revision_unresolved"
            )
        model._trainlm_tokenizer_source_metadata = tokenizer_source_metadata

    model_config = model.config
    context_limit = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(context_limit, int):
        context_limit = getattr(model_config, "n_positions", None)
    if isinstance(context_limit, int) and config.sequence_length > context_limit:
        raise ValueError(
            f"sequence_length={config.sequence_length} exceeds the selected "
            f"{model_config.model_type} context limit ({context_limit})."
        )

    if config.model_initialization == "pretrained":
        embedding_sizes = []
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            rows = getattr(input_embeddings, "num_embeddings", None)
            if isinstance(rows, int):
                embedding_sizes.append(rows)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            rows = getattr(output_embeddings, "out_features", None)
            if isinstance(rows, int):
                embedding_sizes.append(rows)
        max_embedding_rows = min(embedding_sizes) if embedding_sizes else int(
            model.config.vocab_size
        )
        if vocab_size > max_embedding_rows:
            raise ValueError(
                "The selected tokenizer contains token IDs outside the pretrained "
                f"model's {max_embedding_rows} embedding rows. TrainLM will not "
                "resize pretrained embeddings or remap token IDs."
            )
    return model


def _validate_scratch_dataset_source(
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_config_name: str | None = None,
    train_split: str,
    validation_split: str,
    text_field: str,
) -> None:
    """Validate one HF raw-text source and separate train/evaluation slices."""
    if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError("dataset_id must be a Hugging Face owner/name ID.")
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise ValueError("dataset_revision must be a non-empty Hub revision.")
    if dataset_config_name is not None and (
        not isinstance(dataset_config_name, str) or not dataset_config_name.strip()
    ):
        raise ValueError("dataset_config_name must be non-empty when configured.")
    if not isinstance(text_field, str) or not _FIELD_NAME.fullmatch(text_field):
        raise ValueError("text_field must be a simple field name.")
    _validate_disjoint_split_selectors(
        _parse_split_selector(train_split),
        _parse_split_selector(validation_split),
    )


def _resolve_training_steps(
    *,
    max_steps: int | None,
    target_supervised_tokens: int | None,
    from_scratch: bool,
    sequence_length: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    """Resolve a supervised-token budget to an update horizon."""
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
    ):
        raise ValueError("max_steps must be a positive integer when configured.")
    if target_supervised_tokens is not None:
        if (isinstance(target_supervised_tokens, bool)
                or not isinstance(target_supervised_tokens, int)
                or target_supervised_tokens < 1):
            raise ValueError("target_supervised_tokens must be a positive integer.")
        if not from_scratch:
            raise ValueError(
                "target_supervised_tokens is supported for packed scratch runs."
            )
        if max_steps is not None:
            raise ValueError("Set either max_steps or target_supervised_tokens, not both.")
        if sequence_length < 2:
            raise ValueError("Token-budgeted causal training needs sequence_length >= 2.")
        tokens_per_update = (
            (sequence_length - 1) * train_batch_size * gradient_accumulation_steps
        )
        return math.ceil(target_supervised_tokens / tokens_per_update), tokens_per_update

    resolved_steps = 3 if max_steps is None else max_steps
    tokens_per_update = (
        (sequence_length - 1) * train_batch_size * gradient_accumulation_steps
        if sequence_length >= 2 else 0
    )
    return resolved_steps, tokens_per_update


def _preflight_scratch_backend(config: TrainLMPretrainingConfig) -> None:
    """Check package metadata and backend before data is downloaded or packed."""
    from trainlm.runtime.pretraining_profile import require_pretraining_profile

    require_pretraining_profile(accelerator=config.accelerator)
    if config.accelerator not in {"auto", "native_tpu"}:
        return
    import torch

    try:
        import torch_tpu  # noqa: F401 - registers the native backend when installed
    except ModuleNotFoundError as exc:
        if exc.name != "torch_tpu":
            raise
        if config.accelerator == "native_tpu":
            raise RuntimeError(
                "accelerator='native_tpu' requires the torch-tpu package."
            ) from exc

    from trainlm.runtime.torchtpu_discovery import (
        discover_tpu, select_auto_backend,
    )

    report = discover_tpu(
        torch,
        verify_allocation=config.accelerator == "native_tpu",
        include_properties=False,
    )
    if config.accelerator == "native_tpu":
        if report.get("device_type") != "tpu":
            raise RuntimeError(
                "accelerator='native_tpu' was selected, but this process does "
                "not see a native TPU. Select a TPU runtime before pretraining."
            )
        if report.get("status") != "ready":
            status = report.get("status", "unknown")
            error_type = report.get("error_type")
            detail = f", error_type={error_type}" if error_type else ""
            raise RuntimeError(
                f"Native TPU allocation preflight failed (status={status}{detail}). "
                "Resolve the runtime issue before downloading and preparing data."
            )
        if report.get("device_count_visible") != 1:
            raise RuntimeError(
                "This experimental pretraining API requires exactly one visible "
                "native TPU device."
            )
        return
    select_auto_backend(report, cuda_available=torch.cuda.is_available())


def _preparation_cache_inputs(
    config: TrainLMPretrainingConfig,
    *,
    tokenizer_source: Mapping[str, Any] | None = None,
    resolved_dataset_revision: str | None = None,
    provided_datasets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the settings that define tokenizer and packed-shard contents."""
    inputs = {
        "algorithm": (
            "prepared_hf_token_ids_v1"
            if config.data_format == "tokens"
            else (
                "huggingface_tokenizer"
                if tokenizer_source is not None
                else _PREPARATION_ALGORITHM
            )
        ),
        "dataset_id": config.dataset_id,
        "dataset_revision": resolved_dataset_revision or config.dataset_revision,
        "train_split": config.train_split,
        "validation_split": config.validation_split,
        "text_field": config.text_field,
        "max_train_examples": config.max_train_examples,
        "max_eval_examples": config.max_eval_examples,
        "tokenizer_vocab_size": (
            None if tokenizer_source is not None else config.tokenizer_vocab_size
        ),
    }
    if config.dataset_config_name is not None:
        inputs["dataset_config_name"] = config.dataset_config_name
    if config.dataset_source == "provided":
        if not isinstance(provided_datasets, Mapping) or set(provided_datasets) != {
            "train", "validation"
        }:
            raise ValueError(
                "Provided Dataset inputs need stable train and validation identities."
            )
        inputs["provided_datasets"] = dict(provided_datasets)
        if config.data_format == "tokens":
            inputs["data_format"] = "tokens"
    elif provided_datasets is not None:
        raise ValueError(
            "Pass Dataset objects only when dataset_source='provided'."
        )
    if tokenizer_source is not None:
        inputs["tokenizer_source"] = {
            key: value
            for key, value in tokenizer_source.items()
            if key != "source_relation"
        }
    if config.train_file is not None and config.validation_file is not None:
        inputs["local_text_files"] = {
            "train": {
                "path": str(config.train_file),
                "sha256": _sha256_file(config.train_file),
            },
            "validation": {
                "path": str(config.validation_file),
                "sha256": _sha256_file(config.validation_file),
            },
        }
    return inputs


def _preparation_cache_fingerprint(inputs: Mapping[str, Any]) -> str:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_scratch_text_source(
    load_dataset: Any,
    config: TrainLMPretrainingConfig,
    *,
    split: Literal["train", "validation"],
    resolved_dataset_revision: str | None = None,
    provided_datasets: Mapping[str, Any] | None = None,
):
    """Open one pinned Hub split or one local UTF-8 text file as a stream."""
    split_name = config.train_split if split == "train" else config.validation_split
    local_file = config.train_file if split == "train" else config.validation_file
    if config.dataset_source == "provided":
        if not isinstance(provided_datasets, Mapping):
            raise ValueError(
                "dataset_source='provided' requires train_dataset and eval_dataset."
            )
        return provided_datasets[split]
    if local_file is not None:
        return load_dataset(
            "text",
            data_files={split_name: str(local_file)},
            split=split_name,
            streaming=True,
        )
    kwargs: dict[str, Any] = {
        "revision": resolved_dataset_revision or config.dataset_revision,
        "split": split_name,
        "streaming": True,
    }
    if config.dataset_config_name is not None:
        kwargs["name"] = config.dataset_config_name
    if config.cache_dir is not None:
        kwargs["cache_dir"] = config.cache_dir
    return load_dataset(config.dataset_id, **kwargs)


def _resolve_scratch_dataset_revision(
    config: TrainLMPretrainingConfig,
) -> str | None:
    """Resolve a friendly Hub ref once so both splits and caches are immutable."""
    if config.dataset_id is None:
        return None
    revision = config.dataset_revision or "main"
    if _COMMIT_SHA.fullmatch(revision):
        return revision
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Resolving a named Hugging Face dataset revision needs "
            "'huggingface_hub'. Install the TrainLM Hub dependencies."
        ) from exc
    info = HfApi().dataset_info(repo_id=config.dataset_id, revision=revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError(
            "Hugging Face did not resolve the dataset source to a lowercase "
            "40-character commit SHA."
        )
    return resolved


def _scratch_dataset_object_identity(
    config: TrainLMPretrainingConfig,
    train_dataset: Any | None,
    eval_dataset: Any | None,
) -> dict[str, dict[str, Any]] | None:
    """Validate direct HF Dataset inputs and return cache-safe identities."""
    if config.dataset_source != "provided":
        if train_dataset is not None or eval_dataset is not None:
            raise ValueError(
                "Pass train_dataset and eval_dataset only with "
                "dataset_source='provided'."
            )
        return None
    if train_dataset is None or eval_dataset is None:
        raise ValueError(
            "dataset_source='provided' requires both train_dataset and eval_dataset."
        )
    if train_dataset is eval_dataset:
        raise ValueError("Training and evaluation Dataset objects must be distinct.")

    identities: dict[str, dict[str, Any]] = {}
    for split, dataset in (("train", train_dataset), ("validation", eval_dataset)):
        columns = getattr(dataset, "column_names", None)
        fingerprint = getattr(dataset, "_fingerprint", None)
        if not isinstance(columns, (list, tuple)):
            raise ValueError(
                f"The {split} Dataset must expose Hugging Face column_names."
            )
        if config.data_format == "tokens":
            if "input_ids" not in columns:
                raise ValueError(
                    f"The {split} token Dataset must contain an 'input_ids' column."
                )
            unsupported = sorted(
                {
                    "labels",
                    "loss_mask",
                    "position_ids",
                    "token_type_ids",
                    "special_tokens_mask",
                }
                .intersection(columns)
            )
            if unsupported:
                raise ValueError(
                    "Tokenized scratch/CPT currently consumes only input_ids and "
                    "optional attention_mask; TrainLM generates its own causal "
                    f"targets. Remove unsupported columns: {', '.join(unsupported)}."
                )
        elif config.text_field not in columns:
            raise ValueError(
                f"The {split} Dataset must expose the configured text column "
                f"{config.text_field!r}."
            )
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError(
                f"The {split} Dataset must expose a stable Hugging Face "
                "fingerprint for cache and resume safety."
            )
        try:
            rows = len(dataset)
            iter(dataset)
        except (TypeError, AttributeError) as exc:
            raise TypeError(
                f"The {split} source must be a map-style Hugging Face Dataset."
            ) from exc
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError(f"The {split} Dataset must contain at least one row.")
        identities[split] = {"fingerprint": fingerprint, "rows": rows}
    return identities


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenizer_cache_hashes(tokenizer_dir: Path) -> dict[str, str]:
    if tokenizer_dir.is_symlink() or not tokenizer_dir.is_dir():
        raise ValueError("Prepared tokenizer directory is missing or unsafe.")
    files: dict[str, str] = {}
    for path in sorted(tokenizer_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("Prepared tokenizer cache cannot contain symlinks.")
        if path.is_file():
            files[path.relative_to(tokenizer_dir).as_posix()] = _sha256_file(path)
    if "tokenizer.json" not in files:
        raise ValueError("Prepared tokenizer cache is missing tokenizer.json.")
    return files


def _scratch_tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    identity = {
        "vocab_size": len(tokenizer),
        "special_token_ids": {
            name: getattr(tokenizer, f"{name}_token_id")
            for name in ("unk", "pad", "bos", "eos")
        },
    }
    if isinstance(getattr(tokenizer, "_trainlm_source_metadata", None), Mapping):
        identity["mapping_sha256"] = _tokenizer_mapping_sha256(tokenizer)
    return identity


def _tokenizer_mapping_sha256(tokenizer: Any) -> str:
    """Fingerprint tokenizer behavior without recording authentication data."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialize = getattr(backend, "to_str", None)
    if callable(serialize):
        payload = serialize()
    else:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        vocab = get_vocab() if callable(get_vocab) else {}
        payload = json.dumps(
            {
                "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "vocab": sorted(
                    (str(token), int(token_id))
                    for token, token_id in vocab.items()
                ),
                "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_scratch_dataset_sizes(
    config: TrainLMPretrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
) -> None:
    required_train_sequences = math.ceil(
        config.max_tokens / (config.sequence_length - 1)
    )
    if len(train_dataset) < required_train_sequences:
        available = len(train_dataset)
        train_dataset.close()
        eval_dataset.close()
        raise ValueError(
            f"Packed training text provides {available} sequences, but the "
            f"token budget needs at least {required_train_sequences}. Increase "
            "max_train_examples or lower max_tokens."
        )
    if len(eval_dataset) == 0:
        train_dataset.close()
        eval_dataset.close()
        raise ValueError("The validation split does not contain a full sequence.")


def _load_scratch_preparation_cache(
    config: TrainLMPretrainingConfig,
    *,
    root: Path,
    packed_dataset_type: Any,
    inputs: Mapping[str, Any] | None = None,
    candidate_tokenizer: Any | None = None,
) -> tuple[Any, Any, Any] | None:
    """Load only a matching cache whose tokenizer and packed shards validate."""
    cache_path = root / "preparation_cache.json"
    if cache_path.is_symlink():
        raise ValueError(f"Prepared-data cache cannot be a symlink: {cache_path}")
    if not cache_path.exists():
        return None
    if not cache_path.is_file():
        raise ValueError(f"Prepared-data cache is not a regular file: {cache_path}")
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read prepared-data cache {cache_path}; use a fresh output_dir."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _PREPARATION_CACHE_SCHEMA
    ):
        raise RuntimeError(
            f"Prepared-data cache {cache_path} has an unsupported format; "
            "use a fresh output_dir."
        )
    inputs = _preparation_cache_inputs(config) if inputs is None else dict(inputs)
    if (
        payload.get("inputs") != inputs
        or payload.get("fingerprint") != _preparation_cache_fingerprint(inputs)
    ):
        raise ValueError(
            "This output_dir already contains prepared data for different "
            "dataset/tokenizer settings. Choose a fresh output_dir to preserve "
            "the existing data and checkpoints."
        )

    tokenizer_dir = root / "tokenizer"
    expected_tokenizer_files = payload.get("tokenizer_files")
    if not isinstance(expected_tokenizer_files, dict):
        raise RuntimeError(
            f"Prepared tokenizer cache metadata is malformed: {cache_path}"
        )
    actual_tokenizer_files = _tokenizer_cache_hashes(tokenizer_dir)
    if actual_tokenizer_files != expected_tokenizer_files:
        raise RuntimeError(
            f"Prepared tokenizer cache failed integrity validation at {tokenizer_dir}; "
            "use a fresh output_dir to preserve existing checkpoints."
        )

    expected_shards = payload.get("shard_manifests")
    if not isinstance(expected_shards, dict):
        raise RuntimeError(f"Prepared shard cache metadata is malformed: {cache_path}")
    for split in ("train", "validation"):
        manifest_path = root / "data" / split / f"{split}.manifest.json"
        expected_digest = expected_shards.get(split)
        if (
            not isinstance(expected_digest, str)
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or _sha256_file(manifest_path) != expected_digest
        ):
            raise RuntimeError(
                f"Prepared {split} shard manifest failed integrity validation at "
                f"{manifest_path}; use a fresh output_dir to preserve checkpoints."
            )

    train_dataset = None
    try:
        if candidate_tokenizer is None:
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                tokenizer_dir, local_files_only=True
            )
        else:
            tokenizer = candidate_tokenizer
        if _scratch_tokenizer_identity(tokenizer) != payload.get("tokenizer"):
            raise ValueError("Cached tokenizer identity does not match its metadata.")
        train_dataset = packed_dataset_type.from_directory(
            root / "data" / "train",
            sequence_length=config.sequence_length,
            split="train",
            seed=config.seed,
        )
        eval_dataset = packed_dataset_type.from_directory(
            root / "data" / "validation",
            sequence_length=config.sequence_length,
            split="validation",
        )
    except Exception as exc:
        if train_dataset is not None:
            train_dataset.close()
        raise RuntimeError(
            f"Prepared-data cache failed validation under {root}; use a fresh "
            "output_dir rather than overwriting data associated with checkpoints."
        ) from exc
    _check_scratch_dataset_sizes(config, train_dataset, eval_dataset)
    return tokenizer, train_dataset, eval_dataset


def _write_scratch_preparation_cache(
    config: TrainLMPretrainingConfig,
    *,
    root: Path,
    tokenizer: Any,
    inputs: Mapping[str, Any] | None = None,
) -> None:
    inputs = _preparation_cache_inputs(config) if inputs is None else dict(inputs)
    payload = {
        "schema_version": _PREPARATION_CACHE_SCHEMA,
        "inputs": inputs,
        "fingerprint": _preparation_cache_fingerprint(inputs),
        "tokenizer": _scratch_tokenizer_identity(tokenizer),
        "tokenizer_files": _tokenizer_cache_hashes(root / "tokenizer"),
        "shard_manifests": {
            split: _sha256_file(
                root / "data" / split / f"{split}.manifest.json"
            )
            for split in ("train", "validation")
        },
    }
    cache_path = root / "preparation_cache.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".preparation-cache-", suffix=".tmp", dir=root
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _enable_scratch_gradient_checkpointing(model: Any) -> None:
    """Enable the Hugging Face checkpointing hook and disable KV caching."""
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise ValueError(
            "The selected Hugging Face model does not support gradient checkpointing."
        )
    model_config = getattr(model, "config", None)
    has_use_cache = model_config is not None and hasattr(model_config, "use_cache")
    original_use_cache = getattr(model_config, "use_cache", None)
    if has_use_cache:
        model_config.use_cache = False
    try:
        enable()
    except Exception:
        if has_use_cache:
            model_config.use_cache = original_use_cache
        raise
    if has_use_cache:
        model._trainlm_original_use_cache = original_use_cache


def _prepare_scratch_training(
    config: TrainLMPretrainingConfig,
    *,
    train_dataset: Any | None = None,
    eval_dataset: Any | None = None,
):
    """Build the model and reuse integrity-checked prepared data when possible."""
    import torch

    from trainlm.data import PackedBinDataset
    from trainlm.data.writer import write_packed_binary_shard

    root = Path(config.output_dir)
    data_root = root / "data"
    root.mkdir(parents=True, exist_ok=True)
    provided_datasets = _scratch_dataset_object_identity(
        config, train_dataset, eval_dataset
    )
    dataset_objects = (
        {"train": train_dataset, "validation": eval_dataset}
        if provided_datasets is not None
        else None
    )
    resolved_dataset_revision = _resolve_scratch_dataset_revision(config)
    tokenizer, tokenizer_source = _load_existing_scratch_tokenizer(config)
    preparation_inputs = _preparation_cache_inputs(
        config,
        tokenizer_source=tokenizer_source,
        resolved_dataset_revision=resolved_dataset_revision,
        provided_datasets=provided_datasets,
    )
    cached = _load_scratch_preparation_cache(
        config,
        root=root,
        packed_dataset_type=PackedBinDataset,
        inputs=preparation_inputs,
        candidate_tokenizer=tokenizer,
    )
    if cached is not None:
        tokenizer, train_dataset, eval_dataset = cached
    else:
        load_dataset = None
        if dataset_objects is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "Raw-text pretraining needs the optional 'datasets' package. "
                    "Install datasets, then restart the Python runtime."
                ) from exc

        if tokenizer is None:
            if config.data_format == "tokens":
                raise RuntimeError(
                    "Tokenized Dataset inputs require their matching tokenizer."
                )
            train_source = _load_scratch_text_source(
                load_dataset,
                config,
                split="train",
                resolved_dataset_revision=resolved_dataset_revision,
                provided_datasets=dataset_objects,
            )
            tokenizer, _ = _train_byte_level_tokenizer(
                train_source,
                max_examples=config.max_train_examples,
                vocab_size=config.tokenizer_vocab_size,
                text_field=config.text_field,
            )
        tokenizer.save_pretrained(root / "tokenizer")

        # Re-read each source before packing its documents into fixed sequences.
        train_source = _load_scratch_text_source(
            load_dataset,
            config,
            split="train",
            resolved_dataset_revision=resolved_dataset_revision,
            provided_datasets=dataset_objects,
        )
        validation_source = _load_scratch_text_source(
            load_dataset,
            config,
            split="validation",
            resolved_dataset_revision=resolved_dataset_revision,
            provided_datasets=dataset_objects,
        )
        write_packed_binary_shard(
            _scratch_dataset_documents(
                train_source,
                tokenizer,
                config,
                split="train",
            ),
            data_root / "train",
            shard_id="train",
            vocab_size=_tokenizer_vocab_capacity(tokenizer),
        )
        write_packed_binary_shard(
            _scratch_dataset_documents(
                validation_source,
                tokenizer,
                config,
                split="validation",
            ),
            data_root / "validation",
            shard_id="validation",
            vocab_size=_tokenizer_vocab_capacity(tokenizer),
        )

        train_dataset = PackedBinDataset.from_directory(
            data_root / "train",
            sequence_length=config.sequence_length,
            split="train",
            seed=config.seed,
        )
        try:
            eval_dataset = PackedBinDataset.from_directory(
                data_root / "validation",
                sequence_length=config.sequence_length,
                split="validation",
            )
        except Exception:
            train_dataset.close()
            raise
        if _preparation_cache_inputs(
            config,
            tokenizer_source=tokenizer_source,
            resolved_dataset_revision=resolved_dataset_revision,
            provided_datasets=(
                _scratch_dataset_object_identity(
                    config, dataset_objects["train"], dataset_objects["validation"]
                )
                if dataset_objects is not None
                else None
            ),
        ) != preparation_inputs:
            train_dataset.close()
            eval_dataset.close()
            raise RuntimeError(
                "The source text files changed while TrainLM was preparing data. "
                "Restore the original files or choose a fresh output_dir."
            )
        _check_scratch_dataset_sizes(config, train_dataset, eval_dataset)
        _write_scratch_preparation_cache(
            config,
            root=root,
            tokenizer=tokenizer,
            inputs=preparation_inputs,
        )

    dataset_source_metadata = {
        "source_type": (
            "huggingface_dataset_objects"
            if provided_datasets is not None
            else (
                "huggingface_hub"
                if config.dataset_id is not None
                else "local_text_files"
            )
        ),
        "dataset_id": config.dataset_id,
        "dataset_config_name": config.dataset_config_name,
        "requested_revision": config.dataset_revision,
        "resolved_revision": resolved_dataset_revision,
        "train_split": config.train_split,
        "validation_split": config.validation_split,
        "text_field": config.text_field,
        "data_format": config.data_format,
    }
    if provided_datasets is not None:
        dataset_source_metadata["provided_datasets"] = provided_datasets
    train_dataset._trainlm_dataset_source_metadata = dataset_source_metadata
    eval_dataset._trainlm_dataset_source_metadata = dataset_source_metadata

    torch.manual_seed(config.seed)
    model = _load_scratch_model(config, tokenizer=tokenizer)
    if config.gradient_checkpointing:
        _enable_scratch_gradient_checkpointing(model)
    return model, tokenizer, train_dataset, eval_dataset


__all__ = ["TrainLMPretrainingConfig"]
