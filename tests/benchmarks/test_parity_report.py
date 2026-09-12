import json

import pytest

from trainlm.benchmark import ParityReport, load_parity_report


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


def _write_report_bundle(tmp_path, value=None):
    value = value or report(
        configuration_artifacts=("config.json",),
        metric_artifacts=("metrics.json",),
        profile_artifact="profile.json",
        hlo_artifact="hlo.txt",
    )
    for artifact in (
        *value.configuration_artifacts,
        *value.metric_artifacts,
        value.profile_artifact,
        value.hlo_artifact,
    ):
        target = tmp_path / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("evidence", encoding="utf-8")
    path = tmp_path / "report.json"
    path.write_text(value.to_json(), encoding="utf-8")
    return path


def test_report_loader_round_trips_complete_confined_bundle(tmp_path):
    path = _write_report_bundle(tmp_path)

    loaded = load_parity_report(path)

    assert loaded == report(
        configuration_artifacts=("config.json",),
        metric_artifacts=("metrics.json",),
        profile_artifact="profile.json",
        hlo_artifact="hlo.txt",
    )
    assert loaded.certified


def test_report_loader_rejects_unknown_fields_and_missing_artifacts(tmp_path):
    path = _write_report_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="report keys"):
        load_parity_report(path)

    path = _write_report_bundle(tmp_path)
    (tmp_path / "hlo.txt").unlink()
    with pytest.raises(ValueError, match="escapes the report or is missing"):
        load_parity_report(path)


def test_report_loader_rejects_escaping_artifact_path(tmp_path):
    outside = tmp_path.parent / "outside-profile.json"
    outside.write_text("evidence", encoding="utf-8")
    value = report(
        configuration_artifacts=("config.json",),
        metric_artifacts=("metrics.json",),
        profile_artifact="../outside-profile.json",
        hlo_artifact="hlo.txt",
    )
    path = _write_report_bundle(tmp_path, value)

    with pytest.raises(ValueError, match="escapes the report or is missing"):
        load_parity_report(path)
