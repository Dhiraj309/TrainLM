from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from trainlm.runtime import Runtime
from trainlm.tasks import CausalLMTask, TaskResult, TokenCounts
from trainlm.training import Trainer, TrainerCallback
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
        # Keep the fixture's model-owned loss mathematically valid while
        # preserving a differentiable signal for optimizer/lifecycle tests.
        return DummyOutput(output.square().mean())


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


class DummyLoggingConfig:

    log_every_steps = 1


class DummyConfig:

    trainer = DummyTrainerConfig()
    logging = DummyLoggingConfig()


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


class StreamingCountingTask(CountingTask):

    def __init__(self):
        self.received_iterator = False
        self.result_count = 0

    def aggregate_evaluation_stream(self, results):
        self.received_iterator = iter(results) is results
        self.result_count = sum(1 for _ in results)
        return {"eval_loss": float(self.result_count)}


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


def create_trainer(
    loss_fn=None,
    runtime=None,
    task=None,
    checkpoint_saver=None,
    checkpoint_loader=None,
):
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
        checkpoint_saver=checkpoint_saver,
        checkpoint_loader=checkpoint_loader,
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


def test_evaluate_uses_streaming_task_aggregator():
    task = StreamingCountingTask()
    trainer = create_trainer(task=task)

    metrics = trainer.evaluate()

    assert task.received_iterator
    assert task.result_count == len(trainer.eval_dataloader)
    assert metrics["eval_loss"] == float(task.result_count)


def test_streaming_causal_evaluation_matches_reference_without_mutation():
    class CausalFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 4)
            self.output = nn.Linear(4, 16, bias=False)

        def forward(self, input_ids, attention_mask=None):
            del attention_mask
            return SimpleNamespace(logits=self.output(self.embedding(input_ids)))

    batches = (
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
        {
            "input_ids": torch.tensor([[5, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        },
    )
    model = CausalFixture()
    task = CausalLMTask(loss_implementation="causal_lm")
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        config=DummyConfig(),
        model=model,
        runtime=Runtime(),
        optimizer=optimizer,
        scheduler=LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
        task=task,
        train_dataloader=DataLoader(batches, batch_size=None),
        eval_dataloader=DataLoader(batches, batch_size=None),
    )
    reference = task.aggregate_evaluation(
        tuple(task.evaluation_step(model, batch, trainer.runtime) for batch in batches)
    )
    state_before = deepcopy(trainer.state)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    metrics = trainer.evaluate()

    assert metrics == pytest.approx(reference)
    assert trainer.state == state_before
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(parameters_before[name], parameter)
        for name, parameter in model.state_dict().items()
    )


class MetricsRecordingCallback(TrainerCallback):

    def __init__(self):
        self.metrics = []

    def on_metrics(self, state, control, metrics):
        del state, control
        self.metrics.append(metrics)


def test_training_emits_sparse_materialized_metrics():
    callback = MetricsRecordingCallback()
    trainer = create_trainer()
    trainer.callback_handler.add_callback(callback)

    trainer.train()

    assert len(callback.metrics) == 1
    assert callback.metrics[0]["step"] == 1.0
    assert isinstance(callback.metrics[0]["loss"], float)


def test_callbacks_do_not_extract_live_scalars_between_logging_steps(monkeypatch):
    class ThreeStepTrainerConfig(DummyTrainerConfig):
        max_steps = 3
        materialize_loss_every_steps = 3

    class SparseLoggingConfig:
        log_every_steps = 3

    class SparseConfig:
        trainer = ThreeStepTrainerConfig()
        logging = SparseLoggingConfig()

    class StateRecordingCallback(TrainerCallback):
        def __init__(self):
            self.step_losses = []
            self.metrics = []

        def on_step_end(self, state, control):
            del control
            self.step_losses.append(state.loss)

        def on_metrics(self, state, control, metrics):
            del state, control
            self.metrics.append(dict(metrics))

    callback = StateRecordingCallback()
    model = DummyModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        config=SparseConfig(),
        model=model,
        runtime=Runtime(),
        optimizer=optimizer,
        scheduler=LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
        loss_fn=ConstantLoss(),
        # Use an already materialized iterable so this assertion measures
        # trainer loss materialization rather than DataLoader's internal
        # iterator seed extraction, which also calls Tensor.item().
        train_dataloader=[
            {"input_ids": torch.randn(2, 4)}
            for _ in range(ThreeStepTrainerConfig.max_steps)
        ],
        callbacks=(callback,),
    )
    original_item = torch.Tensor.item
    item_calls = 0

    def counted_item(tensor, *args, **kwargs):
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    trainer.train()

    assert item_calls == 1
    assert callback.step_losses[:2] == [None, None]
    assert isinstance(callback.step_losses[2], float)
    assert len(callback.metrics) == 1
    assert all(isinstance(value, float) for value in callback.metrics[0].values())
    assert callback.metrics[0]["step"] == 3.0


def test_training_honors_evaluation_and_checkpoint_cadence():
    events = []
    trainer = create_trainer(checkpoint_saver=lambda trainer, path: events.append(path))
    trainer.config.evaluation = type("Evaluation", (), {"eval_every_steps": 1})()
    trainer.config.checkpoint = type(
        "Checkpoint", (), {"save_training_every_steps": 1}
    )()

    trainer.train()

    assert events == ["checkpoint-1"]


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
