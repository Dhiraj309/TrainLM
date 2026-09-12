import copy
import json
from pathlib import Path

import pytest

from trainlm.benchmark import (
    compare_numerical_alignment,
    load_numerical_alignment_report,
)


MANIFEST = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "manifests"
    / "laughlm_135m_v5e8_v1.json"
)


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_matching_semantics_and_early_updates_are_aligned():
    reference = manifest()
    report = compare_numerical_alignment(
        reference,
        copy.deepcopy(reference),
        deterministic_update_max_abs_errors=(0.0, 2e-7, 5e-7),
    )
    assert report.aligned
    assert report.differences == ()


def test_unjustified_semantic_difference_blocks_alignment():
    reference, candidate = manifest(), manifest()
    candidate["loss"]["label_shift"] = "same_token"
    report = compare_numerical_alignment(
        reference, candidate, deterministic_update_max_abs_errors=(0.0,)
    )
    assert not report.aligned
    assert report.differences[0].path == "loss.label_shift"
    assert report.differences[0].status == "mismatch"


def test_documented_difference_is_preserved_as_justified():
    reference, candidate = manifest(), manifest()
    candidate["initialization"]["parameter_std"] = 0.02
    report = compare_numerical_alignment(
        reference,
        candidate,
        justifications={"initialization.parameter_std": "Imported checkpoint values."},
        deterministic_update_max_abs_errors=(0.0,),
    )
    assert report.aligned
    assert report.differences[0].status == "justified"


def test_missing_or_divergent_update_evidence_blocks_alignment():
    reference = manifest()
    assert not compare_numerical_alignment(reference, reference).aligned
    assert not compare_numerical_alignment(
        reference,
        reference,
        deterministic_update_max_abs_errors=(2e-6,),
        update_tolerance=1e-6,
    ).aligned


def test_unknown_justification_path_is_rejected():
    with pytest.raises(ValueError, match="known path"):
        compare_numerical_alignment(
            manifest(),
            manifest(),
            justifications={"model.family": "not a semantic contract"},
        )


def test_artifact_loader_combines_semantics_and_update_evidence(tmp_path):
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    evidence_path = tmp_path / "updates.json"
    reference = manifest()
    candidate = copy.deepcopy(reference)
    candidate["initialization"]["parameter_std"] = 0.02
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deterministic_update_max_abs_errors": [0.0, 5e-7],
                "update_tolerance": 1e-6,
                "justifications": {
                    "initialization.parameter_std": "Imported checkpoint values."
                },
            }
        ),
        encoding="utf-8",
    )

    report = load_numerical_alignment_report(
        reference_path=reference_path,
        candidate_path=candidate_path,
        update_evidence_path=evidence_path,
    )

    assert report.aligned
    assert report.deterministic_update_max_abs_errors == (0.0, 5e-7)
    assert report.differences[0].status == "justified"


def test_artifact_loader_rejects_unversioned_update_fields(tmp_path):
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    evidence_path = tmp_path / "updates.json"
    reference_path.write_text(json.dumps(manifest()), encoding="utf-8")
    candidate_path.write_text(json.dumps(manifest()), encoding="utf-8")
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deterministic_update_max_abs_errors": [0.0],
                "update_tolerance": 1e-6,
                "justifications": {},
                "unversioned": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="versioned schema"):
        load_numerical_alignment_report(
            reference_path=reference_path,
            candidate_path=candidate_path,
            update_evidence_path=evidence_path,
        )


@pytest.mark.parametrize(
    "value",
    ["not-a-list", {"step": 0.0}, None],
)
def test_artifact_loader_rejects_invalid_update_error_sequences(tmp_path, value):
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    evidence_path = tmp_path / "updates.json"
    reference_path.write_text(json.dumps(manifest()), encoding="utf-8")
    candidate_path.write_text(json.dumps(manifest()), encoding="utf-8")
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deterministic_update_max_abs_errors": value,
                "update_tolerance": 1e-6,
                "justifications": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="list or tuple"):
        load_numerical_alignment_report(
            reference_path=reference_path,
            candidate_path=candidate_path,
            update_evidence_path=evidence_path,
        )
