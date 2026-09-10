from dataclasses import replace

from trainlm.benchmark import (
    FSDPScalingEvidence,
    FSDPScalingTarget,
    evaluate_fsdp_scaling,
)


def target():
    return FSDPScalingTarget(
        workload_id="dense-ar-1.3b-v1",
        workload_version=1,
        parameter_count=1_300_000_000,
        accelerator_type="v5e-8",
        device_count=8,
        data_replicas=4,
        fsdp_shards=2,
        global_tokens_per_second=41_900.0,
        mfu=0.426,
        source="reviewed LaughLM comparison",
        locked=True,
    )


def evidence():
    expected = target()
    return FSDPScalingEvidence(
        workload_id=expected.workload_id,
        workload_version=expected.workload_version,
        parameter_count=expected.parameter_count,
        accelerator_type=expected.accelerator_type,
        device_count=expected.device_count,
        data_replicas=expected.data_replicas,
        fsdp_shards=expected.fsdp_shards,
        global_tokens_per_second=42_000.0,
        mfu=0.43,
        peak_hbm_gib=14.0,
        unexpected_compile_count=0,
        cpu_fallback_count=0,
        correctness_passed=True,
        resume_passed=True,
        export_passed=True,
        collective_artifact="profiles/collectives.json",
        hbm_artifact="profiles/hbm.json",
    )


def test_matched_1_3b_result_passes_locked_target():
    result = evaluate_fsdp_scaling(evidence(), target())

    assert result.passed
    assert result.throughput_ratio > 1
    assert result.mfu_ratio > 1


def test_unlocked_or_mismatched_target_cannot_certify():
    result = evaluate_fsdp_scaling(
        replace(evidence(), fsdp_shards=4), replace(target(), locked=False)
    )

    assert not result.passed
    assert any("review-locked" in reason for reason in result.reasons)
    assert any("fsdp_shards" in reason for reason in result.reasons)


def test_performance_graph_and_lifecycle_gates_are_required():
    measured = replace(
        evidence(),
        global_tokens_per_second=40_000.0,
        mfu=0.40,
        unexpected_compile_count=1,
        cpu_fallback_count=2,
        resume_passed=False,
    )
    result = evaluate_fsdp_scaling(measured, target())

    assert not result.passed
    assert any("throughput" in reason for reason in result.reasons)
    assert any("MFU" in reason for reason in result.reasons)
    assert any("compilation" in reason for reason in result.reasons)
    assert any("CPU fallback" in reason for reason in result.reasons)
    assert any("resume evidence failed" in reason for reason in result.reasons)
