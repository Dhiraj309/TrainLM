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
