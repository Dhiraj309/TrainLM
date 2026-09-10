from dataclasses import replace

import pytest

from trainlm.benchmark import CrossFamilyRecord, evaluate_cross_family_matrix


def _record(family_id, **changes):
    values = dict(
        family_id=family_id,
        support_level="certified",
        workload_id="dense-ar-135m-v1",
        accelerator_type="v5e-8",
        device_count=8,
        scheduled_tokens_per_update=524_288,
        global_tokens_per_second=900_000.0,
        model_flops_per_token=2_000_000_000.0,
        peak_device_tflops=400.0,
        full_attention=True,
        correctness_passed=True,
        graph_passed=True,
        export_passed=True,
        evidence_artifact=f"evidence/{family_id}.json",
    )
    values.update(changes)
    return CrossFamilyRecord(**values)


def test_complete_matched_matrix_reports_architecture_adjusted_mfu():
    records = (_record("gpt2"), _record("llama"))
    result = evaluate_cross_family_matrix(
        records, advertised_families=("gpt2", "llama")
    )

    assert result.complete
    assert result.certified
    assert records[0].architecture_adjusted_mfu == pytest.approx(0.5625)


def test_matrix_reports_missing_family_geometry_and_evidence_failures():
    record = _record("gpt2", support_level="experimental", graph_passed=False)
    result = evaluate_cross_family_matrix(
        (record,), advertised_families=("gpt2", "llama")
    )

    assert not result.complete
    assert not result.certified
    assert any("missing record" in reason for reason in result.reasons)
    assert any("experimental" in reason for reason in result.reasons)
    assert any("graph evidence failed" in reason for reason in result.reasons)


def test_full_attention_record_must_meet_mfu_gate():
    record = replace(_record("gpt2"), global_tokens_per_second=100_000.0)
    result = evaluate_cross_family_matrix(
        (record,), advertised_families=("gpt2",)
    )

    assert result.complete
    assert not result.certified
    assert any("MFU" in reason for reason in result.reasons)


def test_matrix_rejects_duplicate_ids_and_invalid_threshold():
    record = _record("gpt2")
    with pytest.raises(ValueError, match="record IDs must be unique"):
        evaluate_cross_family_matrix(
            (record, record), advertised_families=("gpt2",)
        )
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        evaluate_cross_family_matrix(
            (record,), advertised_families=("gpt2",), minimum_full_attention_mfu=0
        )
