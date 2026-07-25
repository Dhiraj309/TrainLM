import pytest

from trainlm.config import TrainConfig
from trainlm.training import Trainer


def test_trainer_initialization():
    trainer = Trainer(TrainConfig())

    assert trainer.config is not None
    assert trainer.state.step == 0
    assert trainer.control.should_stop is False


def test_train_not_implemented():
    trainer = Trainer(TrainConfig())

    with pytest.raises(NotImplementedError):
        trainer.train()
