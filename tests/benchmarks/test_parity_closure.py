from trainlm.benchmark import (
    BenchmarkResult,
    ParityClosureEvidence,
    evaluate_parity_closure,
)


def result(*, tokens=1_000_000, **overrides):
    values = dict(
        schema_version=1,
        run_id="m11-closure",
        workload_id="laughlm-135m-v5e8",
        workload_version=1,
        framework="trainlm",
        framework_revision="test",
        backend="pytorch-xla",
        accelerator_type="v5e-8",
        measurement_kind="steady_state",
        cache_state="warm",
        device_synchronized=True,
        device_count=8,
        host_count=1,
        data_parallel_replicas=8,
        warmup_steps=3,
        measured_steps=10,
        scheduled_tokens_per_update=1_048_576,
        total_step_seconds_median=1.0,
        device_step_seconds_median=1.0,
        compile_seconds=1.0,
        peak_hbm_gib=5.0,
        input_idle_fraction=0.01,
        collective_seconds_median=0.05,
        compile_count=1,
        unexpected_compile_count=0,
        cpu_fallback_count=0,
        metadata={},
    )
    values.update(overrides)
    return BenchmarkResult.from_measurement(
        supervised_tokens_per_replica=(tokens // 8,) * 8,
        ignored_tokens_per_replica=(0,) * 8,
        measurement_wall_seconds=1.0,
        measurement_device_seconds=1.0,
        non_embedding_flops_per_token=4_000_000.0,
        logits_inclusive_flops_per_token=4_500_000.0,
        peak_flops_per_device=1_000_000_000_000.0,
        **values,
    )


def evidence(**values):
    defaults = dict(
        result=result(),
        hlo_fingerprint="sha256:closure",
        transpose_count=0,
        layout_copy_count=0,
        host_sync_count=0,
        full_logits_materialized=False,
    )
    return ParityClosureEvidence(**{**defaults, **values})


def test_complete_closure_evidence_passes_hard_gate():
    evaluation = evaluate_parity_closure(evidence())
    assert evaluation.passed
    assert evaluation.collective_fraction == 0.05


def test_known_graph_and_host_bottlenecks_are_all_reported():
    bad_result = result(
        tokens=800_000,
        unexpected_compile_count=1,
        cpu_fallback_count=1,
        input_idle_fraction=0.1,
        collective_seconds_median=0.2,
    )
    evaluation = evaluate_parity_closure(
        evidence(
            result=bad_result,
            transpose_count=1,
            layout_copy_count=1,
            host_sync_count=1,
            full_logits_materialized=True,
        )
    )
    assert not evaluation.passed
    assert len(evaluation.reasons) == 10


def test_wrong_accelerator_is_rejected():
    evaluation = evaluate_parity_closure(
        evidence(result=result(accelerator_type="v4-8"))
    )
    assert "parity closure requires v5e-8" in evaluation.reasons
