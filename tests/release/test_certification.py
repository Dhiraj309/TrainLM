from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from trainlm.release import CertificationTierEvidence, evaluate_release_certification


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
