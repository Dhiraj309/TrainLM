"""
Evaluation configuration.

This module defines the evaluation policy used during training.

The Trainer decides when evaluation is triggered, while the evaluation
subsystem is responsible for executing the configured evaluators and
producing structured evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """
    Evaluation configuration.
    """

    enabled: bool = True

    eval_every_steps: int | None = None

    eval_every_tokens: int | None = None

    max_batches: int | None = None

    eval_at_epoch_end: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.eval_at_epoch_end, bool):
            raise ValueError("'eval_at_epoch_end' must be a boolean.")
