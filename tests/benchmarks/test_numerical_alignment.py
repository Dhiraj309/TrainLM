import copy
import json
from pathlib import Path

import pytest

from trainlm.benchmark import compare_numerical_alignment


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
