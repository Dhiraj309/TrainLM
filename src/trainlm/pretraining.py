"""Public raw-text preparation for small decoder-only pretraining runs.

The data preparation boundary is intentionally separate from ``TrainLMTrainer``
so the optional Hugging Face Datasets dependency is loaded only when this
workflow is selected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Literal


DEFAULT_DATASET_ID = "roneneldan/TinyStories"
DEFAULT_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
_DATASET_ID = re.compile(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}\Z")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_FIELD_NAME = re.compile(r"[A-Za-z0-9._+-]{1,96}\Z")


@dataclass(frozen=True, slots=True)
class TrainLMPretrainingConfig:
    """Settings for starting a GPT-2 causal LM from random weights on HF text.

    The starter path uses a revision-pinned streaming dataset, learns a
    byte-level BPE tokenizer from the training split, and writes validated
    packed shards under ``output_dir`` before constructing a trainer.
    """

    output_dir: str | Path = Path("runs/pretraining")
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_revision: str = DEFAULT_DATASET_REVISION
    train_split: str = "train"
    validation_split: str = "validation"
    text_field: str = "text"
    max_tokens: int = 10_000
    sequence_length: int = 128
    max_train_examples: int = 2_000
    max_eval_examples: int = 200
    tokenizer_vocab_size: int = 16_000
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
    logging_steps: int = 10
    max_eval_batches: int = 20
    seed: int = 42
    model_type: str = "gpt2"
    model_config_overrides: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        if not isinstance(self.dataset_id, str) or not _DATASET_ID.fullmatch(
            self.dataset_id
        ):
            raise ValueError("dataset_id must be a Hugging Face owner/name ID.")
        if (not isinstance(self.dataset_revision, str)
                or not _COMMIT_SHA.fullmatch(self.dataset_revision)):
            raise ValueError(
                "dataset_revision must be an immutable 40-character commit SHA."
            )
        if not all(
            isinstance(value, str) and _FIELD_NAME.fullmatch(value)
            for value in (self.train_split, self.validation_split, self.text_field)
        ):
            raise ValueError("Dataset splits and text_field must be simple names.")
        if self.train_split == self.validation_split:
            raise ValueError(
                "Training and validation split names must be distinct."
            )

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
        if self.tokenizer_vocab_size < 260:
            raise ValueError("Byte-level BPE needs tokenizer_vocab_size >= 260.")
        if self.tokenizer_vocab_size > 65_536:
            raise ValueError(
                "tokenizer_vocab_size must fit the packed uint16 token format "
                "(at most 65536)."
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
        if self.model_type != "gpt2" and (
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
        if (isinstance(self.weight_decay, bool)
                or not isinstance(self.weight_decay, Real)
                or not math.isfinite(self.weight_decay)
                or self.weight_decay < 0):
            raise ValueError("weight_decay must be finite and non-negative.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not isinstance(self.accelerator, str) or self.accelerator not in {
            "auto", "cpu", "cuda", "native_tpu"
        }:
            raise ValueError(
                "from_scratch supports auto, cpu, cuda, and native_tpu; "
                "the XLA coordinator cannot reconstruct this random model yet."
            )
        if not isinstance(self.bf16, bool) or not isinstance(self.torch_compile, bool):
            raise ValueError("bf16 and torch_compile must be booleans.")
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


def _validate_scratch_dataset_source(
    *,
    dataset_id: str,
    dataset_revision: str,
    train_split: str,
    validation_split: str,
    text_field: str,
) -> None:
    """Validate the revision-pinned source contract used by the recipe."""
    if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError("dataset_id must be a Hugging Face owner/name ID.")
    if (not isinstance(dataset_revision, str)
            or not _COMMIT_SHA.fullmatch(dataset_revision)):
        raise ValueError("dataset_revision must be an immutable 40-character commit SHA.")
    if not all(
        isinstance(value, str) and _FIELD_NAME.fullmatch(value)
        for value in (train_split, validation_split, text_field)
    ):
        raise ValueError("Dataset splits and text_field must be simple names.")
    if train_split == validation_split:
        raise ValueError("Training and validation split names must be distinct.")


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
    """Register/detect TPU before a raw-text stream is downloaded or packed."""
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


def _prepare_scratch_training(config: TrainLMPretrainingConfig):
    """Build the model and validated packed streams for the public trainer."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Raw-text Hub pretraining needs the optional 'datasets' package. "
            "Install datasets, then restart the Python runtime."
        ) from exc

    import torch
    from transformers import AutoModelForCausalLM

    from trainlm.data import PackedBinDataset
    from trainlm.data.writer import write_packed_binary_shard

    root = Path(config.output_dir)
    data_root = root / "data"
    train_source = load_dataset(
        config.dataset_id,
        revision=config.dataset_revision,
        split=config.train_split,
        streaming=True,
    )
    tokenizer, _ = _train_byte_level_tokenizer(
        train_source,
        max_examples=config.max_train_examples,
        vocab_size=config.tokenizer_vocab_size,
        text_field=config.text_field,
    )
    tokenizer.save_pretrained(root / "tokenizer")

    # Reopen the immutable stream after fitting the tokenizer on training text.
    train_source = load_dataset(
        config.dataset_id,
        revision=config.dataset_revision,
        split=config.train_split,
        streaming=True,
    )
    validation_source = load_dataset(
        config.dataset_id,
        revision=config.dataset_revision,
        split=config.validation_split,
        streaming=True,
    )
    train_manifest = write_packed_binary_shard(
        _token_documents(
            train_source,
            tokenizer,
            max_examples=config.max_train_examples,
            text_field=config.text_field,
        ),
        data_root / "train",
        shard_id="train",
        vocab_size=len(tokenizer),
    )
    validation_manifest = write_packed_binary_shard(
        _token_documents(
            validation_source,
            tokenizer,
            max_examples=config.max_eval_examples,
            text_field=config.text_field,
        ),
        data_root / "validation",
        shard_id="validation",
        vocab_size=len(tokenizer),
    )
    del train_manifest, validation_manifest

    train_dataset = PackedBinDataset.from_directory(
        data_root / "train",
        sequence_length=config.sequence_length,
        split="train",
        seed=config.seed,
    )
    eval_dataset = PackedBinDataset.from_directory(
        data_root / "validation",
        sequence_length=config.sequence_length,
        split="validation",
    )
    required_train_sequences = math.ceil(config.max_tokens / (config.sequence_length - 1))
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

    model_config = _scratch_huggingface_config(
        config,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(config.seed)
    model = AutoModelForCausalLM.from_config(model_config)
    return model, tokenizer, train_dataset, eval_dataset


__all__ = ["TrainLMPretrainingConfig"]
