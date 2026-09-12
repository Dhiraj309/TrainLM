from dataclasses import asdict
import json

import pytest

from trainlm.benchmark import (
    PlainHFExportEvidence,
    evaluate_plain_hf_export,
    load_plain_hf_export_evaluation,
)


def evidence(**values):
    defaults = dict(
        transformers_version="4.56.0",
        trainlm_installed=False,
        canonical_state_dict_only=True,
        tied_aliases_preserved=True,
        logits_max_abs_error=1e-6,
        loss_abs_error=1e-7,
    )
    return PlainHFExportEvidence(**{**defaults, **values})


def test_clean_transformers_only_reload_passes():
    assert evaluate_plain_hf_export(evidence()).passed


def test_all_interoperability_failures_are_reported():
    result = evaluate_plain_hf_export(
        evidence(
            trainlm_installed=True,
            canonical_state_dict_only=False,
            tied_aliases_preserved=False,
            missing_keys=("model.embed_tokens.weight",),
            unexpected_keys=("model.packed_qkv.weight",),
            logits_max_abs_error=1e-3,
            loss_abs_error=1e-3,
        )
    )
    assert not result.passed
    assert len(result.reasons) == 7


def test_boundary_errors_are_accepted():
    result = evaluate_plain_hf_export(
        evidence(logits_max_abs_error=1e-5, loss_abs_error=1e-5),
        tolerance=1e-5,
    )
    assert result.passed


def test_non_finite_error_is_rejected():
    with pytest.raises(ValueError, match="logits_max_abs_error"):
        evidence(logits_max_abs_error=float("nan"))


def _write_artifact(path, *, evidence_values=None, tolerance=1e-5, **extra):
    values = asdict(evidence()) if evidence_values is None else evidence_values
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence": values,
                "tolerance": tolerance,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def test_plain_hf_artifact_loader_evaluates_clean_process_evidence(tmp_path):
    artifact = tmp_path / "plain-hf.json"
    _write_artifact(artifact)

    result = load_plain_hf_export_evaluation(artifact)

    assert result.passed
    assert result.tolerance == 1e-5


def test_plain_hf_artifact_loader_preserves_reload_key_failures(tmp_path):
    artifact = tmp_path / "plain-hf.json"
    values = asdict(evidence())
    values["missing_keys"] = ["model.embed_tokens.weight"]
    _write_artifact(artifact, evidence_values=values)

    result = load_plain_hf_export_evaluation(artifact)

    assert not result.passed
    assert "plain Transformers reload reported missing keys" in result.reasons


def test_plain_hf_artifact_loader_rejects_bad_schema_and_key_arrays(tmp_path):
    artifact = tmp_path / "plain-hf.json"
    _write_artifact(artifact, unexpected=True)
    with pytest.raises(ValueError, match="artifact keys"):
        load_plain_hf_export_evaluation(artifact)

    values = asdict(evidence())
    values["unexpected_keys"] = "model.packed_qkv.weight"
    _write_artifact(artifact, evidence_values=values)
    with pytest.raises(ValueError, match="unexpected_keys"):
        load_plain_hf_export_evaluation(artifact)


def test_plain_hf_artifact_loader_rejects_malformed_json(tmp_path):
    artifact = tmp_path / "plain-hf.json"
    artifact.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid plain-HF"):
        load_plain_hf_export_evaluation(artifact)
