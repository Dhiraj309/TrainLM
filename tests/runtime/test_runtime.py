from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD

from trainlm.runtime import Runtime


def test_prepare_model_returns_same_model():
    runtime = Runtime()
    model = nn.Linear(4, 2)

    assert runtime.prepare_model(model) is model


def test_prepare_batch_returns_same_batch():
    runtime = Runtime()

    batch = {
        "input_ids": torch.ones(2, 4),
    }

    assert runtime.prepare_batch(batch) is batch


def test_autocast_context():
    runtime = Runtime()

    with runtime.autocast():
        x = torch.tensor([1.0])

    assert x.item() == 1.0


def test_backward():
    runtime = Runtime()

    model = nn.Linear(4, 2)

    x = torch.randn(2, 4)

    loss = model(x).sum()

    runtime.backward(loss)

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_clip_gradients():
    runtime = Runtime()

    model = nn.Linear(4, 2)

    x = torch.randn(2, 4)

    loss = model(x).sum()

    runtime.backward(loss)

    runtime.clip_gradients(
        model.parameters(),
        max_norm=1.0,
    )


def test_optimizer_step():
    runtime = Runtime()

    model = nn.Linear(4, 2)

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    x = torch.randn(2, 4)

    loss = model(x).sum()

    runtime.backward(loss)

    runtime.optimizer_step(optimizer)


def test_zero_grad():
    runtime = Runtime()

    model = nn.Linear(4, 2)

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    x = torch.randn(2, 4)

    loss = model(x).sum()

    runtime.backward(loss)

    runtime.zero_grad(optimizer)

    for parameter in model.parameters():
        assert parameter.grad is None


def test_synchronize():
    runtime = Runtime()

    runtime.synchronize()


def test_state_dict():
    runtime = Runtime()

    assert runtime.state_dict() == {}


def test_load_state_dict():
    runtime = Runtime()

    runtime.load_state_dict({})
