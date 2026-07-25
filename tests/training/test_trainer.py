import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from trainlm.runtime import Runtime
from trainlm.training import Trainer


class DummyConfig:
    pass


def create_trainer() -> Trainer:
    model = nn.Linear(4, 2)

    optimizer = SGD(model.parameters(), lr=0.1)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    dataset = TensorDataset(torch.randn(8, 4))

    dataloader = DataLoader(
        dataset,
        batch_size=2,
    )

    runtime = Runtime()

    return Trainer(
        config=DummyConfig(),
        model=model,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloader,
    )


def test_trainer_initializes():
    trainer = create_trainer()

    assert trainer.model is not None
    assert trainer.runtime is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.train_dataloader is not None
    assert trainer.eval_dataloader is None


def test_state_initialized():
    trainer = create_trainer()

    assert trainer.state.step == 0
    assert trainer.state.is_training is False


def test_control_initialized():
    trainer = create_trainer()

    assert trainer.control.should_stop is False


def test_callback_handler_initialized():
    trainer = create_trainer()

    assert trainer.callback_handler.callbacks == ()
