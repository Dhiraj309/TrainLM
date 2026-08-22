from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from trainlm.runtime import Runtime
from trainlm.tasks import TaskResult, TokenCounts
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
    max_tokens = None
    max_grad_norm = 1.0


class DummyConfig:

    trainer = DummyTrainerConfig()


class ConstantLoss:

    def __call__(self, model, batch, runtime):
        del batch
        del runtime

        return model.linear.weight.sum()


class CountingTask:

    name = "counting"

    def training_step(self, model, batch, backend):
        del batch
        del backend
        return TaskResult(
            loss=model.linear.weight.sum(),
            tokens=TokenCounts(
                sequences=2,
                input_tokens=8,
                target_tokens=6,
                supervised_tokens=5,
                ignored_tokens=1,
            ),
        )

    def evaluation_step(self, model, batch, backend):
        return self.training_step(model, batch, backend)

    def aggregate_evaluation(self, results):
        return {
            "eval_loss": sum(result.loss.item() for result in results)
            / len(results)
        }


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


def test_trainer_consumes_task_result_and_exact_token_counts():
    trainer = create_trainer(task=CountingTask())

    state = trainer.train()

    assert state.step == 1
    assert state.tokens_seen == 5
    assert state.samples_seen == 2


def test_current_learning_rate():
    trainer = create_trainer()

    assert trainer._current_learning_rate() == 0.1


def test_update_state():
    trainer = create_trainer()

    trainer._update_state(
        result=TaskResult(
            loss=torch.tensor(2.5),
            tokens=TokenCounts(
                sequences=2,
                input_tokens=8,
                target_tokens=6,
                supervised_tokens=5,
                ignored_tokens=1,
            ),
        ),
    )

    assert trainer.state.step == 1
    assert trainer.state.loss == 2.5
    assert trainer.state.learning_rate == 0.1
    assert trainer.state.tokens_seen == 5
    assert trainer.state.samples_seen == 2


def create_trainer(loss_fn=None, runtime=None, task=None):
    model = DummyModel()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    train_dataloader = DataLoader(
        DummyDataset(),
        batch_size=2,
    )

    eval_dataloader = DataLoader(
        DummyDataset(),
        batch_size=2,
    )

    return Trainer(
        config=DummyConfig(),
        model=model,
        runtime=runtime or Runtime(),
        optimizer=optimizer,
        scheduler=scheduler,
        task=task,
        loss_fn=None if task is not None else (loss_fn or LanguageModelLoss()),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )


def test_evaluate_returns_metrics():
    trainer = create_trainer()

    metrics = trainer.evaluate()

    assert "eval_loss" in metrics
    assert isinstance(metrics["eval_loss"], float)


def test_evaluate_restores_train_mode():
    trainer = create_trainer()

    trainer.model.train()

    trainer.evaluate()

    assert trainer.model.training


def test_evaluate_without_dataloader():
    trainer = create_trainer()

    trainer.eval_dataloader = None

    with pytest.raises(RuntimeError):
        trainer.evaluate()


def test_evaluate_average_loss():
    trainer = create_trainer()

    metrics = trainer.evaluate()

    assert metrics["eval_loss"] >= 0.0


class RecordingRuntime(Runtime):

    def __init__(self):
        super().__init__()
        self.events = []

    def initialize(self):
        self.events.append("initialize")

    def on_train_begin(self):
        self.events.append("train_begin")

    def on_step_begin(self, step):
        self.events.append(("step_begin", step))

    def on_step_end(self, step):
        self.events.append(("step_end", step))

    def on_train_end(self):
        self.events.append("train_end")

    def finalize(self):
        self.events.append("finalize")


def test_trainer_uses_backend_lifecycle_hooks():
    runtime = RecordingRuntime()
    trainer = create_trainer(runtime=runtime)

    trainer.train()

    assert runtime.events == [
        "initialize",
        "train_begin",
        ("step_begin", 0),
        ("step_end", 1),
        "train_end",
        "finalize",
    ]
