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
import math
from numbers import Real


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

    # Optional deterministic cyclic schedule of loop depths evaluated host-side
    # across microbatches so DDP ranks stay lockstep with bounded compiled graphs.
    loop_depth_schedule: tuple[int, ...] | None = None

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
            or not isinstance(self.max_grad_norm, Real)
            or not math.isfinite(self.max_grad_norm)
            or self.max_grad_norm < 0
        ):
            raise ValueError("max_grad_norm must be finite and non-negative.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be non-negative.")
        if self.loop_depth_schedule is not None:
            if not isinstance(self.loop_depth_schedule, tuple) or not self.loop_depth_schedule:
                raise ValueError(
                    "loop_depth_schedule must be a non-empty tuple of positive ints."
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.loop_depth_schedule
            ):
                raise ValueError(
                    "loop_depth_schedule entries must be positive integers."
                )
            if len(set(self.loop_depth_schedule)) > 6:
                raise ValueError(
                    "loop_depth_schedule supports at most 6 unique depths to bound XLA compilation cache."
                )
