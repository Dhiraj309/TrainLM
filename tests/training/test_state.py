from trainlm.training import TrainerState


def test_default_state():
    state = TrainerState()

    assert state.step == 0
    assert state.tokens_seen == 0
    assert state.should_stop is False
