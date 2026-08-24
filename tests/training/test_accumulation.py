"""Token-normalized accumulation tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

import pytest

from trainlm.runtime import Runtime
from trainlm.tasks import TaskResult, TokenCounts
from trainlm.training import Trainer


class AccumulationConfig:
    max_steps = 1
    max_tokens = None
    gradient_accumulation_steps = 2
    max_grad_norm = 1.0


class AccumulationTrainConfig:
    trainer = AccumulationConfig()


class VariableTokenDataset(Dataset):
    def __init__(self, values):
        self.values = tuple(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        scale, tokens = self.values[index]
        return {"scale": torch.tensor(scale), "tokens": tokens}


class VariableTokenTask:
    name = "variable_token_test"

    def __init__(self, *, exact=True):
        self.exact = exact

    def training_step(self, model, batch, backend):
        del backend
        if not self.exact:
            return TaskResult(
                loss=model.weight.sum(),
                tokens=TokenCounts(
                    sequences=0,
                    input_tokens=0,
                    target_tokens=0,
                    supervised_tokens=0,
                    ignored_tokens=0,
                    exact=False,
                ),
            )
        tokens = int(batch["tokens"].item())
        return TaskResult(
            loss=model.weight.sum() * batch["scale"].reshape(()),
            tokens=TokenCounts(
                sequences=1,
                input_tokens=tokens,
                target_tokens=tokens,
                supervised_tokens=tokens,
                ignored_tokens=0,
            ),
        )

    def evaluation_step(self, model, batch, backend):
        return self.training_step(model, batch, backend)

    def aggregate_evaluation(self, results):
        return {"eval_loss": sum(result.loss.item() for result in results)}


def _trainer(task):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    dataloader = DataLoader(
        VariableTokenDataset(((1.0, 1), (3.0, 3))),
        batch_size=1,
    )
    return Trainer(
        config=AccumulationTrainConfig(),
        model=model,
        runtime=Runtime(),
        optimizer=optimizer,
        scheduler=scheduler,
        task=task,
        train_dataloader=dataloader,
    )


def test_accumulation_uses_supervised_token_weighting():
    trainer = _trainer(VariableTokenTask())

    state = trainer.train()

    assert state.step == 1
    assert state.micro_step == 2
    assert state.tokens_seen == 4
    assert state.samples_seen == 2
    assert state.global_batch_size == 2
    assert state.loss == pytest.approx(5.0)


def test_inexact_loss_rejects_multi_microbatch_accumulation():
    trainer = _trainer(VariableTokenTask(exact=False))

    with pytest.raises(ValueError, match="exact positive supervised-token"):
        trainer.train()

    assert trainer.state.step == 0
    assert trainer.state.micro_step == 1
