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
    ] = "cosine"

    warmup_steps: int = 0

    horizon_steps: int | None = None

    min_lr_ratio: float = 0.0
