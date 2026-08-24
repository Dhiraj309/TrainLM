"""Trainer lifecycle transitions, cleanup, stop, and resume boundaries."""

from __future__ import annotations

import pytest

from trainlm.training import (
    InvalidTrainerTransition,
    TrainerCallback,
    TrainerPhase,
    TrainerState,
)

from .test_trainer import create_trainer


def test_state_machine_rejects_terminal_reuse():
    state = TrainerState()
    state.transition(TrainerPhase.PREPARED)
    state.transition(TrainerPhase.TRAINING)
    state.transition(TrainerPhase.STOPPING)
    state.transition(TrainerPhase.FINALIZED)

    with pytest.raises(InvalidTrainerTransition):
        state.transition(TrainerPhase.TRAINING)


def test_normal_training_finalizes_resources():
    trainer = create_trainer()

    state = trainer.train()

    assert state.phase == TrainerPhase.FINALIZED
    assert state.is_training is False
    assert state.failure is None


class StopAfterFirstStep(TrainerCallback):
    def on_step_end(self, state, control):
        del state
        control.request_stop()


def test_callback_stop_is_safe_and_persistent():
    trainer = create_trainer()
    trainer.callback_handler.add_callback(StopAfterFirstStep())

    state = trainer.train()

    assert state.phase == TrainerPhase.FINALIZED
    assert state.step == 1
    assert state.should_stop is True


class FailOnStepBegin(TrainerCallback):
    def on_step_begin(self, state, control):
        del state, control
        raise RuntimeError("callback failure")


def test_failure_records_error_and_still_finalizes():
    trainer = create_trainer()
    trainer.callback_handler.add_callback(FailOnStepBegin())

    with pytest.raises(RuntimeError, match="callback failure"):
        trainer.train()

    assert trainer.state.phase == TrainerPhase.FINALIZED
    assert trainer.state.failure == "callback failure"
    assert trainer.state.should_stop is True


def test_save_and_resume_hooks_have_explicit_boundaries():
    events = []

    def save(trainer, destination):
        events.append(("save", trainer.state.phase, destination))
        return {"destination": destination}

    def load(trainer, source):
        events.append(("load", trainer.state.phase, source))
        return {"source": source}

    trainer = create_trainer(
        checkpoint_saver=save,
        checkpoint_loader=load,
    )
    trainer.prepare()

    assert trainer.save_checkpoint("step-1") == {"destination": "step-1"}
    assert trainer.state.phase == TrainerPhase.PREPARED
    assert trainer.load_checkpoint("step-1") == {"source": "step-1"}
    assert trainer.state.phase == TrainerPhase.PREPARED
    assert events == [
        ("save", TrainerPhase.SAVING, "step-1"),
        ("load", TrainerPhase.RESUMING, "step-1"),
    ]
    trainer.finalize()
    assert trainer.state.phase == TrainerPhase.FINALIZED
