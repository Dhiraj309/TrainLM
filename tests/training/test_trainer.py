import pytest

from trainlm.training import Trainer


class DummyConfig:
    pass


def test_trainer_initializes():
    trainer = Trainer(DummyConfig())

    assert trainer.config is not None
    assert trainer.state.step == 0
    assert trainer.state.is_training is False


def test_train_step_not_implemented():
    trainer = Trainer(DummyConfig())

    with pytest.raises(NotImplementedError):
        trainer._train_step()


def test_train_invokes_train_step():
    trainer = Trainer(DummyConfig())

    with pytest.raises(NotImplementedError):
        trainer.train()
