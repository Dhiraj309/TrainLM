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

    # Maximum number of training batches to consume. Public epoch budgets are
    # resolved to this finite cursor before the engine starts.
    max_train_batches: int | None = None

    gradient_accumulation_steps: int = 1

    # Host materialization is intentionally configurable because ``.item()``
    # on an XLA loss forces a device synchronization. Keep the compatibility
    # default at every step; TPU jobs can align this with logging cadence.
    materialize_loss_every_steps: int = 1

    max_grad_norm: float = 1.0

    seed: int = 42

    # Flush accumulated gradients at a finite dataset boundary so epoch-based
    # evaluation and checkpoint actions run at an exact epoch end.
    stop_at_epoch_end: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stop_at_epoch_end, bool):
            raise ValueError("stop_at_epoch_end must be a boolean.")
        if (
            isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.max_train_batches is not None and (
            isinstance(self.max_train_batches, bool)
            or not isinstance(self.max_train_batches, int)
            or self.max_train_batches < 1
        ):
            raise ValueError("max_train_batches must be positive when configured.")
        if (
            isinstance(self.materialize_loss_every_steps, bool)
            or not isinstance(self.materialize_loss_every_steps, int)
            or self.materialize_loss_every_steps < 1
        ):
            raise ValueError("materialize_loss_every_steps must be positive.")
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
