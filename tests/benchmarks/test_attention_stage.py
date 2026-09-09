from trainlm.benchmark import BenchmarkResult, evaluate_attention_stage


def result(*, tokens=900_000, peak_hbm_gib=4.0, metadata=None, **overrides):
    values = dict(
        schema_version=1,
        run_id="m10-stage",
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
        peak_hbm_gib=peak_hbm_gib,
        input_idle_fraction=0.01,
        collective_seconds_median=0.1,
        compile_count=1,
        unexpected_compile_count=0,
        cpu_fallback_count=0,
        metadata=metadata or {},
    )
    values.update(overrides)
    return BenchmarkResult.from_measurement(
        supervised_tokens_per_replica=(tokens // 8,) * 8,
        ignored_tokens_per_replica=(0,) * 8,
        measurement_wall_seconds=1.0,
        measurement_device_seconds=1.0,
        non_embedding_flops_per_token=1.0e9,
        logits_inclusive_flops_per_token=1.1e9,
        peak_flops_per_device=1.0e12,
        **values,
    )


def evidence():
    return {
        "full_logits_materialized": False,
        "attention_provider": "trainlm.pallas_grouped_attention",
        "loss_provider": "trainlm.pallas_linear_ce",
        "hlo_fingerprint": "sha256:test",
    }


def test_matching_optimized_stage_passes_all_gates():
    evaluation = evaluate_attention_stage(
        result(metadata=evidence()), result(tokens=300_000, peak_hbm_gib=6.0)
    )
    assert evaluation.passed


def test_gate_reports_throughput_hbm_graph_fallback_and_missing_evidence():
    evaluation = evaluate_attention_stage(
        result(
            tokens=800_000,
            peak_hbm_gib=6.0,
            unexpected_compile_count=1,
            cpu_fallback_count=1,
        ),
        result(tokens=300_000, peak_hbm_gib=6.0),
    )
    assert not evaluation.passed
    assert evaluation.reasons == (
        "global throughput is below the 850K stage gate",
        "peak HBM is not lower than the matched reference",
        "unexpected compilation occurred after warmup",
        "CPU fallback counters were observed",
        "full-logits elimination is not proven",
        "attention provider evidence is missing",
        "loss provider evidence is missing",
        "HLO fingerprint evidence is missing",
    )


def test_mismatched_workload_is_rejected():
    evaluation = evaluate_attention_stage(
        result(metadata=evidence(), workload_version=2),
        result(tokens=300_000, peak_hbm_gib=6.0),
    )
    assert "workload version does not match the reference" in evaluation.reasons
    assert evaluation.reasons == ()


def test_gate_reports_throughput_hbm_graph_fallback_and_missing_evidence():
    optimized = result(
        tokens=800_000,
        peak_hbm_gib=6.0,
        unexpected_compile_count=1,
        cpu_fallback_count=1,
    )
    evaluation = evaluate_attention_stage(
        optimized, result(tokens=300_000, peak_hbm_gib=5.0)
    )
    assert not evaluation.passed
    assert len(evaluation.reasons) == 8


def test_mismatched_workload_geometry_is_rejected():
    evaluation = evaluate_attention_stage(
        result(metadata=evidence()),
        result(tokens=300_000, peak_hbm_gib=6.0, scheduled_tokens_per_update=42),
    )
    assert not evaluation.passed
    assert "scheduled tokens per update does not match the reference" in evaluation.reasons
