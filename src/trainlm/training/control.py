"""
Trainer control flow.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainerControl:
    """
    Control flags used by callbacks to influence training.

    The trainer resets these flags after handling them.
    """

    should_log: bool = False

    should_evaluate: bool = False

    should_save_checkpoint: bool = False

    should_stop: bool = False

    def reset(self) -> None:
        """
        Reset one-shot control flags.

        The stop request is preserved until the trainer exits.
        """

        self.should_log = False
        self.should_evaluate = False
        self.should_save_checkpoint = False

    def request_stop(self) -> None:
        """Request a safe stop at the next trainer loop boundary."""

        self.should_stop = True
