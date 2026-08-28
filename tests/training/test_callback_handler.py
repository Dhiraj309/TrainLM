import pytest
import torch

from trainlm.training import (
    CallbackHandler,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


class DummyCallback(TrainerCallback):
    def __init__(self):
        self.calls = 0

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        self.calls += 1


def test_dispatch():
    callback = DummyCallback()

    handler = CallbackHandler([callback])

    handler.on_step_end(
        TrainerState(),
        TrainerControl(),
    )

    assert callback.calls == 1


def test_add_callback():
    handler = CallbackHandler()

    callback = DummyCallback()

    handler.add_callback(callback)

    handler.on_step_end(
        TrainerState(),
        TrainerControl(),
    )

    assert callback.calls == 1


def test_callbacks_property():
    callback = DummyCallback()

    handler = CallbackHandler([callback])

    assert handler.callbacks == (callback,)


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = []

    def on_metrics(self, state, control, metrics):
        del state, control
        self.metrics.append(metrics)


def test_metrics_are_read_only_host_scalars():
    callback = MetricsCallback()
    handler = CallbackHandler([callback])

    handler.on_metrics(
        TrainerState(),
        TrainerControl(),
        {"loss": 1.5, "step": 2},
    )

    snapshot = callback.metrics[0]
    assert snapshot["loss"] == 1.5
    with pytest.raises(TypeError):
        snapshot["loss"] = 3.0


def test_metrics_reject_live_tensors():
    handler = CallbackHandler()

    with pytest.raises(TypeError, match="materialized"):
        handler.on_metrics(
            TrainerState(),
            TrainerControl(),
            {"loss": torch.tensor(1.5)},
        )
