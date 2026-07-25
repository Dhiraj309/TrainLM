from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from trainlm.runtime import Runtime
from trainlm.training import Trainer
from trainlm.training.loss import LanguageModelLoss


class DummyOutput:

    def __init__(self, loss):
        self.loss = loss


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, input_ids):
        output = self.linear(input_ids)
        return DummyOutput(output.mean())


class DummyDataset(torch.utils.data.Dataset):

    def __len__(self):
        return 8

    def __getitem__(self, index):
        del index
        return {
            "input_ids": torch.randn(4),
        }


class DummyTrainerConfig:

    max_steps = 1
    max_grad_norm = 1.0


class DummyConfig:

    trainer = DummyTrainerConfig()


class ConstantLoss:

    def __call__(self, model, batch, runtime):
        del batch
        del runtime

        return model.linear.weight.sum()


def create_trainer(loss_fn=None):
    model = DummyModel()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    dataloader = DataLoader(
        DummyDataset(),
        batch_size=2,
    )

    return Trainer(
        config=DummyConfig(),
        model=model,
        runtime=Runtime(),
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn or LanguageModelLoss(),
        train_dataloader=dataloader,
    )


def test_train_runs_one_step():
    trainer = create_trainer()

    state = trainer.train()

    assert state.step == 1
    assert state.loss is not None
    assert state.learning_rate > 0.0


def test_parameters_are_updated():
    trainer = create_trainer()

    before = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]

    trainer.train()

    after = list(trainer.model.parameters())

    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, after)
    )


def test_custom_loss_function():
    trainer = create_trainer(
        loss_fn=ConstantLoss(),
    )

    trainer.train()

    assert trainer.state.step == 1
    assert trainer.state.loss is not None

def test_current_learning_rate():
    trainer = create_trainer()

    assert trainer._current_learning_rate() == 0.1


def test_update_state():
    trainer = create_trainer()

    batch = {
        "input_ids": torch.randn(2, 4),
    }

    loss = torch.tensor(2.5)

    trainer._update_state(
        batch=batch,
        loss=loss,
    )

    assert trainer.state.step == 1
    assert trainer.state.loss == 2.5
    assert trainer.state.learning_rate == 0.1
