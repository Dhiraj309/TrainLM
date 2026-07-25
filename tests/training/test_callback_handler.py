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
