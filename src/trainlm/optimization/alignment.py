"""Tensor-core / TPU MXU 128-alignment inspection and optional vocab padding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Summary of hardware tile-alignment checks for a dense causal LM."""

    alignment: int
    vocab_size: int | None
    padded_vocab_size: int | None
    vocab_aligned: bool
    hidden_size: int | None
    intermediate_size: int | None
    sequence_length: int | None
    warnings: tuple[str, ...] = ()


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def inspect_model_alignment(
    model: nn.Module,
    *,
    sequence_length: int | None = None,
    alignment: int = 128,
) -> AlignmentReport:
    """Inspect model dimensions and sequence length for TPU MXU / GPU tile alignment."""
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 1:
        raise ValueError("alignment must be a positive integer.")

    config: Any = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        emb = getattr(model, "get_input_embeddings", lambda: None)()
        weight = getattr(emb, "weight", None)
        vocab_size = int(weight.shape[0]) if weight is not None and weight.ndim == 2 else None

    hidden_size = getattr(config, "hidden_size", None)
    if not isinstance(hidden_size, int):
        hidden_size = getattr(config, "n_embd", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    if not isinstance(intermediate_size, int):
        intermediate_size = getattr(config, "n_inner", None)

    warnings: list[str] = []
    vocab_aligned = True
    padded_vocab_size = vocab_size
    if isinstance(vocab_size, int) and vocab_size > 0 and vocab_size % alignment != 0:
        vocab_aligned = False
        padded_vocab_size = _round_up(vocab_size, alignment)
        warnings.append(
            f"vocab_size={vocab_size} is not a multiple of {alignment}; "
            f"set pad_vocab_to_multiple={alignment} (padded={padded_vocab_size}) "
            "to avoid unaligned lm_head GEMM padding."
        )
    if isinstance(hidden_size, int) and hidden_size > 0 and hidden_size % alignment != 0:
        warnings.append(
            f"hidden_size={hidden_size} is not a multiple of {alignment}; "
            "projection GEMMs may incur tile padding."
        )
    if isinstance(intermediate_size, int) and intermediate_size > 0 and intermediate_size % alignment != 0:
        warnings.append(
            f"intermediate_size={intermediate_size} is not a multiple of {alignment}; "
            "MLP GEMMs may incur tile padding."
        )
    if isinstance(sequence_length, int) and sequence_length > 0 and sequence_length % alignment != 0:
        warnings.append(
            f"sequence_length={sequence_length} is not a multiple of {alignment}; "
            f"prefer multiples of {alignment} for optimal TPU MXU / FlashAttention tiling."
        )

    return AlignmentReport(
        alignment=alignment,
        vocab_size=vocab_size,
        padded_vocab_size=padded_vocab_size,
        vocab_aligned=vocab_aligned,
        hidden_size=hidden_size if isinstance(hidden_size, int) else None,
        intermediate_size=intermediate_size if isinstance(intermediate_size, int) else None,
        sequence_length=sequence_length,
        warnings=tuple(warnings),
    )


def pad_vocab_to_multiple(model: nn.Module, multiple: int = 128) -> int | None:
    """Pad model embeddings/lm_head to a multiple of ``multiple`` when supported.

    Returns the new vocabulary size if resized, or ``None`` if already aligned
    or if the model does not expose ``resize_token_embeddings``.
    """
    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple < 1:
        raise ValueError("multiple must be a positive integer.")

    report = inspect_model_alignment(model, alignment=multiple)
    if report.vocab_size is None or report.vocab_aligned:
        return None
    resize_fn = getattr(model, "resize_token_embeddings", None)
    if not callable(resize_fn):
        return None
    target_vocab = _round_up(report.vocab_size, multiple)
    resize_fn(target_vocab)
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        tie_fn = getattr(model, "tie_weights", None)
        if callable(tie_fn):
            tie_fn()
    return target_vocab
