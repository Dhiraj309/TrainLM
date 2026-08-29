from pathlib import Path

import pytest

from trainlm.benchmark import (
    BenchmarkResult,
    evaluate_baseline,
    load_baseline_workload,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
MANIFEST = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "manifests"
    / "laughlm_135m_v5e8_v1.json"
)


def _result(*, tokens_per_replica=131_072, **overrides):
    values = {
        "schema_version": 1,
        "run_id": "trainlm-baseline-test",
        "workload_id": "laughlm-135m-v5e8",
        "workload_version": 1,
        "framework": "trainlm",
        "framework_revision": "test",
        "backend": "pytorch-xla",
        "accelerator_type": "v5e-8",
        "measurement_kind": "steady_state",
        "cache_state": "warm",
        "device_synchronized": True,
        "device_count": 8,
        "host_count": 1,
        "data_parallel_replicas": 8,
        "warmup_steps": 3,
        "measured_steps": 10,
        "scheduled_tokens_per_update": 1_048_576,
        "total_step_seconds_median": 1.0,
        "device_step_seconds_median": 1.0,
        "compile_seconds": 1.0,
        "peak_hbm_gib": 5.0,
        "input_idle_fraction": 0.01,
        "collective_seconds_median": 0.1,
        "compile_count": 1,
        "unexpected_compile_count": 0,
        "cpu_fallback_count": 0,
        "metadata": {},
    }
    values.update(overrides)
    return BenchmarkResult.from_measurement(
        supervised_tokens_per_replica=(tokens_per_replica,) * 8,
        ignored_tokens_per_replica=(0,) * 8,
        measurement_wall_seconds=1.0,
        measurement_device_seconds=1.0,
        non_embedding_flops_per_token=1.0e9,
        logits_inclusive_flops_per_token=1.1e9,
        peak_flops_per_device=1.0e12,
        **values,
    )


def test_locked_baseline_workload_loads_reference_geometry():
    workload = load_baseline_workload(MANIFEST)

    assert workload.accelerator_type == "v5e-8"
    assert workload.sequence_length == 2048
    assert workload.micro_batch_per_device == 2
    assert workload.gradient_accumulation_steps == 32
    assert workload.expected_tokens_per_update == 1_048_576


def test_baseline_evaluation_accepts_synchronized_steady_state_result():
    workload = load_baseline_workload(MANIFEST)

    evaluation = evaluate_baseline(_result(), workload)

    assert evaluation.passed
    assert evaluation.reasons == ()


def test_baseline_evaluation_reports_gate_and_fallback_failures():
    workload = load_baseline_workload(MANIFEST)
    result = _result(
        tokens_per_replica=60_000,
        measurement_kind="compile",
        unexpected_compile_count=1,
        cpu_fallback_count=2,
    )

    evaluation = evaluate_baseline(result, workload)

    assert not evaluation.passed
    assert len(evaluation.reasons) == 3
    assert evaluation.warnings == ("CPU fallback counters were observed",)
