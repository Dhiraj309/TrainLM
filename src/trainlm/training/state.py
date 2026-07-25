"""
Trainer state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainerState:
    """
    Mutable state describing the progress of a training run.
    """

    step: int = 0

    epoch: int = 0

    tokens_seen: int = 0

    samples_seen: int = 0

    global_batch_size: int = 0

    learning_rate: float = 0.0

    loss: float | None = None

    is_training: bool = False

    should_stop: bool = False
