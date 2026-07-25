"""
TrainLM trainer.
"""

from __future__ import annotations

from collections.abc import Sequence

from trainlm.config import TrainConfig

from .callback import TrainerCallback
from .callback_handler import CallbackHandler
from .control import TrainerControl
from .state import TrainerState


class Trainer:
    """
    High-level training orchestrator.
    """

    def __init__(
        self,
        config: TrainConfig,
        *,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        self.config = config

        self.state = TrainerState()

        self.control = TrainerControl()

        self.callback_handler = CallbackHandler(callbacks)

    def train(self) -> TrainerState:
        """
        Run training.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Run evaluation.
        """
        raise NotImplementedError

    def save_model(self) -> None:
        """
        Export an inference checkpoint.
        """
        raise NotImplementedError

    def save_checkpoint(self) -> None:
        """
        Save a training checkpoint.
        """
        raise NotImplementedError

    def load_checkpoint(self) -> None:
        """
        Restore a training checkpoint.
        """
        raise NotImplementedError
