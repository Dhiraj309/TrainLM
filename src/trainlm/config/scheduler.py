"""
Learning-rate scheduler configuration.

This module defines the configuration required to construct a PyTorch
learning-rate scheduler. Scheduler implementations are provided by
``torch.optim.lr_scheduler`` and instantiated by the SchedulerFactory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """
    Learning-rate scheduler configuration.

    Notes
    -----
    The scheduler horizon is intentionally separated from the trainer's
    stopping criteria. This enables staged or continuous pretraining
    without reshaping the learning-rate schedule after resuming.
    """

    name: Literal[
        "constant",
        "linear",
        "cosine",
        "wsd",
    ] = "cosine"

    warmup_steps: int = 0

    horizon_steps: int | None = None

    min_lr_ratio: float = 0.0

    horizon_tokens: int | None = None

    warmup_fraction: float = 0.0

    stable_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in {"constant", "linear", "cosine", "wsd"}:
            raise ValueError(f"Unsupported scheduler: {self.name}")
        for name in ("warmup_steps", "horizon_steps"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative when configured.")
        if (
            not isinstance(self.min_lr_ratio, (int, float))
            or not 0 <= self.min_lr_ratio <= 1
        ):
            raise ValueError("min_lr_ratio must be between 0 and 1.")
        if self.horizon_tokens is not None and (
            isinstance(self.horizon_tokens, bool)
            or not isinstance(self.horizon_tokens, int)
            or self.horizon_tokens < 1
        ):
            raise ValueError("horizon_tokens must be positive when configured.")
        for name in ("warmup_fraction", "stable_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.warmup_fraction + self.stable_fraction > 1:
            raise ValueError("warmup_fraction + stable_fraction must be <= 1.")
        if self.name == "wsd" and self.horizon_tokens is None:
            raise ValueError("WSD requires horizon_tokens.")
