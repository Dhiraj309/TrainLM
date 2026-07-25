"""
Optimizer configuration.

This module defines the configuration required to construct a PyTorch
optimizer. The actual optimizer implementation is provided by
``torch.optim`` and is created by the OptimizerFactory.

The configuration intentionally describes *what* optimizer to construct,
not *how* parameter groups are created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """
    Optimizer configuration.

    Notes
    -----
    TrainLM delegates optimizer implementations to ``torch.optim``.
    """

    name: Literal[
        "adamw",
    ] = "adamw"

    learning_rate: float = 3e-4

    betas: tuple[float, float] = (0.9, 0.95)

    eps: float = 1e-8

    weight_decay: float = 0.1

    fused: bool = True
