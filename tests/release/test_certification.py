from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from trainlm.release import (
    CertificationTierEvidence,
    evaluate_release_certification,
    load_release_certification,
)


SHA = "a" * 40
NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


def tier(number):
    return CertificationTierEvidence(
        tier=number,
        commit_sha=SHA,
        completed_at=NOW - timedelta(days=1),
        passed=True,
        artifact=f"certification/tier-{number}.json",
    )


def test_release_requires_current_tier_zero_through_three_for_same_commit():
    result = evaluate_release_certification(
        tuple(tier(number) for number in range(4)),
        release_commit_sha=SHA,
        now=NOW,
    )

    assert result.certified
    assert result.reasons == ()
    assert [item.tier for item in result.evidence] == [0, 1, 2, 3]


def test_missing_tier_three_prevents_release():
    result = evaluate_release_certification(
        tuple(tier(number) for number in range(3)),
        release_commit_sha=SHA,
        now=NOW,
    )

    assert not result.certified
    assert result.reasons == (
        "Tier 3 (release_performance_stability) evidence is missing",
    )


def test_stale_failed_or_wrong_commit_evidence_is_rejected():
    evidence = (
        replace(tier(0), passed=False),
        replace(tier(1), completed_at=NOW - timedelta(days=15)),
        replace(tier(2), commit_sha="b" * 40),
        tier(3),
    )
    result = evaluate_release_certification(
        evidence, release_commit_sha=SHA, now=NOW
    )

    assert not result.certified
    assert any("did not pass" in reason for reason in result.reasons)
    assert any("stale" in reason for reason in result.reasons)
    assert any("different commit" in reason for reason in result.reasons)


def test_duplicate_tiers_and_naive_timestamps_are_invalid():
    with pytest.raises(ValueError, match="tiers must be unique"):
        evaluate_release_certification((tier(0), tier(0)), release_commit_sha=SHA)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(tier(0), completed_at=datetime(2026, 9, 10))


def _write_manifest(tmp_path, *, evidence=None, **changes):
    evidence = evidence or tuple(tier(number) for number in range(4))
    payload_evidence = []
    for item in evidence:
        artifact = tmp_path / item.artifact
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("evidence", encoding="utf-8")
        values = asdict(item)
        values["completed_at"] = item.completed_at.isoformat()
        payload_evidence.append(values)
    payload = {
        "schema_version": 1,
        "release_commit_sha": SHA,
        "evaluated_at": NOW.isoformat(),
        "maximum_age_seconds": 14 * 24 * 60 * 60,
        "evidence": payload_evidence,
        **changes,
    }
    path = tmp_path / "certification.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_certification_loader_evaluates_complete_tier_bundle(tmp_path):
    result = load_release_certification(_write_manifest(tmp_path))

    assert result.certified
    assert tuple(item.tier for item in result.evidence) == (0, 1, 2, 3)


def test_certification_loader_rejects_schema_drift_and_missing_artifact(tmp_path):
    path = _write_manifest(tmp_path, unexpected=True)
    with pytest.raises(ValueError, match="manifest keys"):
        load_release_certification(path)

    path = _write_manifest(tmp_path)
    (tmp_path / tier(3).artifact).unlink()
    with pytest.raises(ValueError, match="escapes the manifest or is missing"):
        load_release_certification(path)


def test_certification_loader_rejects_naive_timestamp_and_path_escape(tmp_path):
    path = _write_manifest(tmp_path, evaluated_at="2026-09-10T00:00:00")
    with pytest.raises(ValueError, match="evaluated_at must be timezone-aware"):
        load_release_certification(path)

    outside = tmp_path.parent / "outside-tier.json"
    outside.write_text("evidence", encoding="utf-8")
    escaped = replace(tier(3), artifact="../outside-tier.json")
    path = _write_manifest(
        tmp_path,
        evidence=tuple(tier(number) for number in range(3)) + (escaped,),
    )
    with pytest.raises(ValueError, match="escapes the manifest or is missing"):
        load_release_certification(path)
