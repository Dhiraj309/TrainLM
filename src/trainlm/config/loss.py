"""Language model loss policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Configure loss ownership and numerically relevant behavior."""

    implementation: Literal["causal_lm", "model"] = "causal_lm"
    ignore_index: int = -100
    normalization: Literal["supervised_tokens", "batch"] = "supervised_tokens"
    z_loss: float = 0.0
    logits_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.implementation not in {"causal_lm", "model"}:
            raise ValueError(
                f"Unsupported loss implementation: {self.implementation}"
            )
        if self.normalization not in {"supervised_tokens", "batch"}:
            raise ValueError(
                f"Unsupported loss normalization: {self.normalization}"
            )
        if self.z_loss < 0:
            raise ValueError("'loss.z_loss' must be non-negative.")
        if self.logits_chunk_size is not None and self.logits_chunk_size <= 0:
            raise ValueError("'loss.logits_chunk_size' must be positive.")
