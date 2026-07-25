"""
Cross-configuration validation.

This module validates relationships between configuration domains that
cannot be checked by individual configuration classes.
"""

from __future__ import annotations

from .train import TrainConfig


def validate_config(config: TrainConfig) -> None:
    """
    Validate cross-domain configuration invariants.
    """

    _validate_training_stop(config)
    _validate_scheduler(config)


def _validate_training_stop(config: TrainConfig) -> None:
    """
    Training must define at least one stopping criterion.
    """

    trainer = config.trainer

    if (
        trainer.max_steps is None
        and trainer.max_tokens is None
    ):
        raise ValueError(
            "Either 'trainer.max_steps' or "
            "'trainer.max_tokens' must be specified."
        )


def _validate_scheduler(config: TrainConfig) -> None:
    """
    Validate scheduler configuration.
    """

    trainer = config.trainer
    scheduler = config.scheduler

    if (
        trainer.max_steps is not None
        and scheduler.horizon_steps is not None
        and scheduler.horizon_steps < trainer.max_steps
    ):
        raise ValueError(
            "'scheduler.horizon_steps' must be greater than or equal "
            "to 'trainer.max_steps'."
        )

    if (
        trainer.max_tokens is not None
        and scheduler.horizon_steps is None
    ):
        # Nothing to validate until token-based scheduler horizons
        # are introduced.
        return
