from trainlm.training import TrainerControl


def test_default_control():
    control = TrainerControl()

    assert control.should_log is False
    assert control.should_evaluate is False
    assert control.should_save_checkpoint is False
    assert control.should_stop is False


def test_reset():
    control = TrainerControl(
        should_log=True,
        should_evaluate=True,
        should_save_checkpoint=True,
        should_stop=True,
    )

    control.reset()

    assert control.should_log is False
    assert control.should_evaluate is False
    assert control.should_save_checkpoint is False

    # Stop requests persist until training exits.
    assert control.should_stop is True
