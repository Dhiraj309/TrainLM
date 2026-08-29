import pytest

from trainlm.monitoring import StepTelemetry, TelemetryRecorder


def test_telemetry_snapshot_aggregates_synchronized_samples():
    recorder = TelemetryRecorder()
    recorder.record_step(
        StepTelemetry(
            supervised_tokens=100,
            wall_seconds=2.0,
            device_seconds=1.0,
            compile_seconds=0.5,
            peak_hbm_gib=4.0,
            input_idle_fraction=0.1,
            collective_seconds=0.2,
            compile_count=1,
        )
    )
    recorder.record_step(
        StepTelemetry(
            supervised_tokens=200,
            wall_seconds=2.0,
            device_seconds=1.5,
            peak_hbm_gib=5.0,
            input_idle_fraction=0.3,
            collective_seconds=0.4,
            cpu_fallback_count=2,
        )
    )

    snapshot = recorder.snapshot()
    assert snapshot.measured_steps == 2
    assert snapshot.supervised_tokens == 300
    assert snapshot.global_tokens_per_second == 75.0
    assert snapshot.device_tokens_per_second == 120.0
    assert snapshot.compile_seconds == 0.5
    assert snapshot.compile_count == 1
    assert snapshot.peak_hbm_gib == 5.0
    assert snapshot.input_idle_fraction == 0.2
    assert snapshot.collective_seconds_median == 0.3
    assert snapshot.cpu_fallback_count == 2
    assert snapshot.synchronized is True


def test_telemetry_rejects_unsynchronized_or_invalid_samples():
    with pytest.raises(ValueError, match="synchronized"):
        StepTelemetry(1, 1.0, 1.0, synchronized=False)
    with pytest.raises(ValueError, match="device_seconds"):
        StepTelemetry(1, 1.0, 2.0)
    with pytest.raises(TypeError, match="StepTelemetry"):
        TelemetryRecorder().record_step(object())


def test_telemetry_requires_samples_and_can_clear():
    recorder = TelemetryRecorder()
    with pytest.raises(RuntimeError, match="No synchronized"):
        recorder.snapshot()
    recorder.record_step(StepTelemetry(1, 1.0, 1.0))
    assert recorder.snapshot().measured_steps == 1
    recorder.clear()
    with pytest.raises(RuntimeError, match="No synchronized"):
        recorder.snapshot()
