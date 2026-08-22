import json
from pathlib import Path

import pytest

from trainlm.benchmark import BenchmarkResult


REPOSITORY_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "schemas"
    / "benchmark_result_v1.schema.json"
)


def _result(**overrides):
    values = {
        "schema_version": 1,
        "run_id": "run-001",
        "workload_id": "laughlm-135m-v5e8",
        "workload_version": 1,
        "framework": "TrainLM",
        "framework_revision": "test-revision",
        "backend": "pytorch-xla",
        "accelerator_type": "v5e-8",
        "measurement_kind": "steady_state",
        "cache_state": "warm",
        "device_synchronized": True,
        "device_count": 2,
        "host_count": 1,
        "data_parallel_replicas": 2,
        "warmup_steps": 10,
        "measured_steps": 2,
        "scheduled_tokens_per_update": 100,
        "supervised_tokens_per_replica": (90, 80),
        "ignored_tokens_per_replica": (10, 20),
        "measurement_wall_seconds": 2.0,
        "measurement_device_seconds": 1.8,
        "total_step_seconds_median": 1.0,
        "device_step_seconds_median": 0.9,
        "compile_seconds": 5.0,
        "peak_hbm_gib": 4.0,
        "input_idle_fraction": 0.05,
        "collective_seconds_median": 0.01,
        "compile_count": 2,
        "unexpected_compile_count": 0,
        "cpu_fallback_count": 0,
        "non_embedding_flops_per_token": 100.0,
        "logits_inclusive_flops_per_token": 125.0,
        "peak_flops_per_device": 1_000.0,
        "metadata": {"cache": "warm"},
    }
    values.update(overrides)
    derived = {
        "supervised_tokens_per_replica",
        "ignored_tokens_per_replica",
        "measurement_wall_seconds",
        "measurement_device_seconds",
        "non_embedding_flops_per_token",
        "logits_inclusive_flops_per_token",
        "peak_flops_per_device",
    }
    measurement = {key: values.pop(key) for key in derived}
    return BenchmarkResult.from_measurement(**measurement, **values)


def test_measurement_counts_actual_supervised_tokens_across_replicas():
    result = _result()

    assert result.total_supervised_tokens == 170
    assert result.total_ignored_tokens == 30
    assert result.global_tokens_per_second == 85.0
    assert result.device_tokens_per_second == pytest.approx(94.444444)
    assert result.non_embedding_mfu == pytest.approx(4.722222)
    assert result.logits_inclusive_mfu == pytest.approx(5.902778)


def test_result_round_trips_through_versioned_json():
    result = _result()

    restored = BenchmarkResult.from_json(result.to_json())

    assert restored == result
    assert isinstance(restored.supervised_tokens_per_replica, tuple)
    assert restored.metadata == {"cache": "warm"}


def test_python_result_fields_match_json_schema_required_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    result = _result().to_dict()

    assert schema["properties"]["schema_version"]["const"] == 1
    assert set(schema["required"]) == set(result)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"data_parallel_replicas": 3}, "data-parallel replica"),
        ({"measurement_wall_seconds": 0.0}, "greater than zero"),
        ({"measurement_device_seconds": 3.0}, "synchronized wall-clock"),
        ({"input_idle_fraction": 1.1}, "between zero and one"),
        ({"unexpected_compile_count": 3}, "cannot exceed compile_count"),
        ({"device_synchronized": False}, "must be true"),
    ],
)
def test_result_rejects_invalid_measurements(overrides, message):
    with pytest.raises(ValueError, match=message):
        _result(**overrides)
