import json

import pytest

from trainlm.benchmark import ParityReport


def report(**values):
    defaults = dict(
        schema_version=1,
        report_id="laughlm-135m-v5e8-run-1",
        environment=(("python", "3.12.3"), ("torch_xla", "2.8.0")),
        commands=("python scripts/trainlm_tpu_worker.py --config run.json",),
        configuration_artifacts=(
            "benchmarks/manifests/laughlm_135m_v5e8_v1.json",
        ),
        metric_artifacts=("artifacts/run-1/summary.json",),
        profile_artifact="artifacts/run-1/profile.json",
        hlo_artifact="artifacts/run-1/hlo.txt",
        limitations=("Certified only for the exact 135M workload.",),
        numerical_alignment_passed=True,
        repeated_benchmark_passed=True,
        real_shard_stability_passed=True,
        plain_hf_export_passed=True,
    )
    return ParityReport(**{**defaults, **values})


def test_complete_report_is_certified_and_serializable():
    value = report()
    assert value.certified
    assert json.loads(value.to_json())["report_id"] == value.report_id


def test_any_failed_gate_prevents_certification():
    assert not report(repeated_benchmark_passed=False).certified


def test_markdown_contains_reproduction_and_evidence():
    markdown = report().to_markdown()
    assert "**Status:** CERTIFIED" in markdown
    assert "python scripts/trainlm_tpu_worker.py" in markdown
    assert "artifacts/run-1/hlo.txt" in markdown
    assert "Certified only for the exact 135M workload." in markdown


def test_report_requires_reproducibility_artifacts():
    with pytest.raises(ValueError, match="commands"):
        report(commands=())
    with pytest.raises(ValueError, match="profile_artifact"):
        report(profile_artifact="")
