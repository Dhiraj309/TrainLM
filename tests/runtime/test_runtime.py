from __future__ import annotations

import torch
from torch import nn

from trainlm.runtime import Runtime


def test_prepare_model_returns_same_model():
    runtime = Runtime()
    model = nn.Linear(4, 2)

    prepared = runtime.prepare_model(model)

    assert prepared is model


def test_prepare_batch_returns_same_batch():
    runtime = Runtime()

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }

    prepared = runtime.prepare_batch(batch)

    assert prepared is batch


def test_autocast_is_context_manager():
    runtime = Runtime()

    with runtime.autocast():
        x = torch.tensor([1.0])

    assert torch.equal(x, torch.tensor([1.0]))


def test_backward_computes_gradients():
    runtime = Runtime()

    model = nn.Linear(4, 2)

    inputs = torch.randn(2, 4)
    outputs = model(inputs)
    loss = outputs.sum()

    runtime.backward(loss)

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_clip_gradients():
    runtime = Runtime()

    model = nn.Linear(4, 2)

    inputs = torch.randn(2, 4)
    outputs = model(inputs)
    loss = outputs.sum()

    runtime.backward(loss)

    runtime.clip_gradients(
        model.parameters(),
        max_norm=1.0,
    )


def test_synchronize_does_not_raise():
    runtime = Runtime()

    runtime.synchronize()


def test_state_dict_returns_empty_dict():
    runtime = Runtime()

    assert runtime.state_dict() == {}


def test_load_state_dict_accepts_empty_dict():
    runtime = Runtime()

    runtime.load_state_dict({})
