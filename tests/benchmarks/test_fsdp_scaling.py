from dataclasses import asdict, replace
import json

import pytest

from trainlm.benchmark import (
    FSDPScalingEvidence,
    FSDPScalingTarget,
    evaluate_fsdp_scaling,
    load_fsdp_scaling_evaluation,
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
        replace(evidence(), device_count=16, fsdp_shards=4),
        replace(target(), locked=False),
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


def _write_bundle(tmp_path, *, measured=None, **extra):
    measured = measured or evidence()
    for reference in (measured.collective_artifact, measured.hbm_artifact):
        artifact = tmp_path / reference
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("evidence", encoding="utf-8")
    path = tmp_path / "fsdp-scaling.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": asdict(target()),
                "evidence": asdict(measured),
                **extra,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_fsdp_scaling_loader_evaluates_complete_bundle(tmp_path):
    result = load_fsdp_scaling_evaluation(_write_bundle(tmp_path))

    assert result.passed
    assert result.throughput_ratio > 1


def test_fsdp_scaling_loader_rejects_schema_drift_and_missing_profiles(tmp_path):
    path = _write_bundle(tmp_path, unexpected=True)
    with pytest.raises(ValueError, match="manifest keys"):
        load_fsdp_scaling_evaluation(path)

    path = _write_bundle(tmp_path)
    (tmp_path / evidence().hbm_artifact).unlink()
    with pytest.raises(ValueError, match="escapes the manifest or is missing"):
        load_fsdp_scaling_evaluation(path)


def test_fsdp_scaling_loader_rejects_escaping_profile_path(tmp_path):
    outside = tmp_path.parent / "outside-hbm.json"
    outside.write_text("evidence", encoding="utf-8")
    measured = replace(evidence(), hbm_artifact="../outside-hbm.json")
    path = _write_bundle(tmp_path, measured=measured)

    with pytest.raises(ValueError, match="escapes the manifest or is missing"):
        load_fsdp_scaling_evaluation(path)
