"""
Trainer callback interface.
"""

from __future__ import annotations

from abc import ABC

from .control import TrainerControl
from .state import TrainerState


class TrainerCallback(ABC):
    """
    Base class for trainer callbacks.

    All hook methods are optional.
    """

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Called before training begins."""

    def on_resume(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Called after a checkpoint has been restored."""

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Called after training completes."""

    def on_step_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Called before an optimisation step."""

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Called after an optimisation step."""

    def on_evaluate(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Called after evaluation."""

    def on_save_checkpoint(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Called after a training checkpoint is written."""
