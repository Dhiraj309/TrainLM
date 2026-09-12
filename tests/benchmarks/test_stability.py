from dataclasses import asdict
import json

import pytest

from trainlm.benchmark import (
    RealShardStabilityEvidence,
    evaluate_real_shard_stability,
    load_real_shard_stability_evaluation,
)


def evidence(**values):
    defaults = dict(
        completed_updates=200,
        real_data_revision="dataset@abc123",
        shard_count=4,
        evaluation_completed=True,
        checkpoint_resumed=True,
        integrity_checks_passed=True,
        data_cursor_continuous=True,
        export_completed=True,
        unexpected_compile_count=0,
        cpu_fallback_count=0,
        minimum_loss=1.0,
        maximum_loss=10.0,
        minimum_gradient_norm=0.1,
        maximum_gradient_norm=5.0,
    )
    return RealShardStabilityEvidence(**{**defaults, **values})


def test_complete_200_update_run_passes():
    assert evaluate_real_shard_stability(evidence()).passed


def test_all_lifecycle_and_integrity_failures_are_reported():
    result = evaluate_real_shard_stability(
        evidence(
            completed_updates=199,
            shard_count=1,
            evaluation_completed=False,
            checkpoint_resumed=False,
            integrity_checks_passed=False,
            data_cursor_continuous=False,
            export_completed=False,
            unexpected_compile_count=1,
            cpu_fallback_count=1,
        )
    )
    assert not result.passed
    assert len(result.reasons) == 9


def test_non_finite_loss_is_rejected():
    with pytest.raises(ValueError, match="minimum_loss"):
        evidence(minimum_loss=float("nan"))


def test_loss_and_gradient_ranges_must_be_ordered():
    with pytest.raises(ValueError, match="minimum_loss"):
        evidence(minimum_loss=2.0, maximum_loss=1.0)
    with pytest.raises(ValueError, match="minimum_gradient_norm"):
        evidence(minimum_gradient_norm=2.0, maximum_gradient_norm=1.0)


def _write_artifact(path, *, evidence_values=None, requirements=None, **extra):
    payload = {
        "schema_version": 1,
        "evidence": asdict(evidence()) if evidence_values is None else evidence_values,
        "requirements": (
            {"required_updates": 200, "minimum_shards": 2}
            if requirements is None
            else requirements
        ),
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stability_artifact_loader_evaluates_versioned_evidence(tmp_path):
    artifact = tmp_path / "stability.json"
    _write_artifact(artifact)

    result = load_real_shard_stability_evaluation(artifact)

    assert result.passed
    assert result.reasons == ()


def test_stability_artifact_loader_rejects_unknown_or_missing_fields(tmp_path):
    artifact = tmp_path / "stability.json"
    _write_artifact(artifact, unexpected=True)
    with pytest.raises(ValueError, match="artifact keys"):
        load_real_shard_stability_evaluation(artifact)

    values = asdict(evidence())
    values.pop("export_completed")
    _write_artifact(artifact, evidence_values=values)
    with pytest.raises(ValueError, match="evidence keys"):
        load_real_shard_stability_evaluation(artifact)


def test_stability_artifact_loader_rejects_invalid_schema_and_types(tmp_path):
    artifact = tmp_path / "stability.json"
    _write_artifact(
        artifact,
        requirements={"required_updates": True, "minimum_shards": 2},
    )
    with pytest.raises(ValueError, match="required_updates"):
        load_real_shard_stability_evaluation(artifact)

    artifact.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid real-shard"):
        load_real_shard_stability_evaluation(artifact)
