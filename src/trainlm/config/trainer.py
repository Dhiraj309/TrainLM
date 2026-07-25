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
