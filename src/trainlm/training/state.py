"""
Trainer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainerPhase(str, Enum):
    """Explicit lifecycle phases owned by the training engine."""

    CREATED = "created"
    PREPARED = "prepared"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SAVING = "saving"
    RESUMING = "resuming"
    STOPPING = "stopping"
    FINALIZED = "finalized"
    FAILED = "failed"


class InvalidTrainerTransition(RuntimeError):
    """Raised when a lifecycle operation is used in the wrong phase."""


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

    phase: TrainerPhase = TrainerPhase.CREATED

    failure: str | None = None

    def transition(self, phase: TrainerPhase) -> None:
        """Move to a valid lifecycle phase."""

        if not isinstance(phase, TrainerPhase):
            raise TypeError("phase must be a TrainerPhase.")
        if phase == self.phase:
            return
        allowed = {
            TrainerPhase.CREATED: {
                TrainerPhase.PREPARED,
                TrainerPhase.EVALUATING,
                TrainerPhase.SAVING,
                TrainerPhase.RESUMING,
                TrainerPhase.FAILED,
            },
            TrainerPhase.PREPARED: {
                TrainerPhase.TRAINING,
                TrainerPhase.EVALUATING,
                TrainerPhase.SAVING,
                TrainerPhase.RESUMING,
                TrainerPhase.STOPPING,
                TrainerPhase.FAILED,
            },
            TrainerPhase.TRAINING: {
                TrainerPhase.EVALUATING,
                TrainerPhase.SAVING,
                TrainerPhase.STOPPING,
                TrainerPhase.FAILED,
            },
            TrainerPhase.EVALUATING: {
                TrainerPhase.CREATED,
                TrainerPhase.PREPARED,
                TrainerPhase.TRAINING,
                TrainerPhase.FAILED,
            },
            TrainerPhase.SAVING: {
                TrainerPhase.CREATED,
                TrainerPhase.PREPARED,
                TrainerPhase.TRAINING,
                TrainerPhase.FAILED,
            },
            TrainerPhase.RESUMING: {
                TrainerPhase.PREPARED,
                TrainerPhase.TRAINING,
                TrainerPhase.FAILED,
            },
            TrainerPhase.STOPPING: {
                TrainerPhase.FINALIZED,
                TrainerPhase.FAILED,
            },
            TrainerPhase.FAILED: {TrainerPhase.STOPPING, TrainerPhase.FINALIZED},
            TrainerPhase.FINALIZED: set(),
        }
        if phase not in allowed[self.phase]:
            raise InvalidTrainerTransition(
                f"Cannot transition trainer from '{self.phase.value}' "
                f"to '{phase.value}'."
            )
        self.phase = phase

    def mark_failed(self, error: BaseException | str) -> None:
        """Record a failure and make it visible to lifecycle observers."""

        self.failure = str(error)
        self.should_stop = True
        if self.phase not in {TrainerPhase.FAILED, TrainerPhase.FINALIZED}:
            self.phase = TrainerPhase.FAILED
