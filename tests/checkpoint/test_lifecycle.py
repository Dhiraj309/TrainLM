import pytest

from trainlm.checkpoint import AsyncCheckpointLifecycle


def test_async_lifecycle_does_not_report_durability_before_completion():
    lifecycle = AsyncCheckpointLifecycle(max_retained=2)

    started = lifecycle.begin("step-1")
    assert started.phase == "in_flight"
    assert started.durable is False
    assert started.retained_checkpoints == ()

    completed = lifecycle.complete()
    assert completed.phase == "committed"
    assert completed.durable is True
    assert completed.retained_checkpoints == ("step-1",)


def test_async_lifecycle_applies_retention_only_to_committed_checkpoints():
    lifecycle = AsyncCheckpointLifecycle(max_retained=2)
    lifecycle.begin("step-1")
    lifecycle.complete()
    lifecycle.begin("step-2")
    lifecycle.fail("write failed")
    lifecycle.begin("step-3")
    snapshot = lifecycle.complete()

    assert snapshot.retained_checkpoints == ("step-1", "step-3")
    assert snapshot.error is None

    lifecycle.begin("step-4")
    assert lifecycle.complete().retained_checkpoints == ("step-3", "step-4")


def test_async_lifecycle_failure_is_never_durable():
    lifecycle = AsyncCheckpointLifecycle()
    lifecycle.begin("step-1")
    failed = lifecycle.fail("backend error")

    assert failed.phase == "failed"
    assert failed.durable is False
    assert failed.retained_checkpoints == ()
    assert failed.error == "backend error"


def test_async_lifecycle_shutdown_invalidates_in_flight_work():
    lifecycle = AsyncCheckpointLifecycle()
    lifecycle.begin("step-1")
    stopped = lifecycle.shutdown()

    assert stopped.phase == "shutdown"
    assert stopped.durable is False
    assert "shutdown" in stopped.error
    with pytest.raises(RuntimeError, match="shut down"):
        lifecycle.begin("step-2")


def test_async_lifecycle_rejects_invalid_transitions_and_configuration():
    with pytest.raises(ValueError, match="max_retained"):
        AsyncCheckpointLifecycle(max_retained=0)

    lifecycle = AsyncCheckpointLifecycle()
    with pytest.raises(ValueError, match="transaction_id"):
        lifecycle.begin("")
    with pytest.raises(RuntimeError, match="expected in_flight"):
        lifecycle.complete()
    with pytest.raises(RuntimeError, match="expected in_flight"):
        lifecycle.fail("no transaction")
