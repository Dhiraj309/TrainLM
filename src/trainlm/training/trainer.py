from __future__ import annotations

from collections.abc import Sequence

from trainlm.config import TrainConfig

from .callback import TrainerCallback
from .callback_handler import CallbackHandler
from .control import TrainerControl
from .state import TrainerState


class Trainer:
    """Coordinates the end-to-end training lifecycle.

    The Trainer owns orchestration only. Execution details such as automatic
    mixed precision, distributed execution, compilation, and checkpoint I/O
    are delegated to dedicated subsystems.
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
        """Run the training loop."""

        self.state.is_training = True

        self.callback_handler.on_train_begin(self.state, self.control)

        try:
            while not self.control.should_stop:
                self.control.reset()

                self.callback_handler.on_step_begin(
                    self.state,
                    self.control,
                )

                self._train_step()

                self.callback_handler.on_step_end(
                    self.state,
                    self.control,
                )

                if self.control.should_evaluate:
                    self.evaluate()

                if self.control.should_save_checkpoint:
                    self.save_checkpoint()

        finally:
            self.state.is_training = False

            self.callback_handler.on_train_end(
                self.state,
                self.control,
            )

        return self.state

    def _train_step(self) -> None:
        """Execute a single optimization step."""
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError
