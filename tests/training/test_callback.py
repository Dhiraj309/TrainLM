from trainlm.training import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


class DummyCallback(TrainerCallback):
    def __init__(self):
        self.called = False

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        self.called = True


def test_callback_hooks():
    callback = DummyCallback()

    callback.on_step_end(
        TrainerState(),
        TrainerControl(),
    )

    assert callback.called
