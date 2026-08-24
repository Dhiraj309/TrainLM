"""
Trainer configuration.

This module defines the behaviour of the training loop. It specifies
when training starts and stops, how gradients are accumulated, and
other trainer-specific settings.

The trainer configuration intentionally excludes runtime, optimizer,
scheduler, logging, and checkpoint settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """
    Configuration for the training loop.
    """

    max_steps: int | None = None

    max_tokens: int | None = None

    gradient_accumulation_steps: int = 1

    max_grad_norm: float = 1.0

    seed: int = 42

    def __post_init__(self) -> None:
        if (
            isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be positive.")
        if (
            isinstance(self.max_grad_norm, bool)
            or not isinstance(self.max_grad_norm, (int, float))
            or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be non-negative.")
