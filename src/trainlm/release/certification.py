"""Current-evidence gate for dense-AR release certification tiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

CertificationTier = Literal[0, 1, 2, 3]

TIER_NAMES = {
    0: "cpu",
    1: "cuda",
    2: "v5e_correctness",
    3: "release_performance_stability",
}


@dataclass(frozen=True, slots=True)
class CertificationTierEvidence:
    tier: CertificationTier
    commit_sha: str
    completed_at: datetime
    passed: bool
    artifact: str

    def __post_init__(self) -> None:
        if self.tier not in TIER_NAMES:
            raise ValueError(f"Unsupported certification tier: {self.tier!r}.")
        if not isinstance(self.commit_sha, str) or len(self.commit_sha) != 40:
            raise ValueError("commit_sha must be a full 40-character Git SHA.")
        try:
            int(self.commit_sha, 16)
        except ValueError as exc:
            raise ValueError("commit_sha must be hexadecimal.") from exc
        if not isinstance(self.completed_at, datetime):
            raise TypeError("completed_at must be a datetime.")
        if self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware.")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be boolean.")
        if not isinstance(self.artifact, str) or not self.artifact.strip():
            raise ValueError("artifact cannot be empty.")


@dataclass(frozen=True, slots=True)
class ReleaseCertification:
    certified: bool
    reasons: tuple[str, ...]
    evidence: tuple[CertificationTierEvidence, ...]


def evaluate_release_certification(
    evidence: tuple[CertificationTierEvidence, ...],
    *,
    release_commit_sha: str,
    now: datetime | None = None,
    maximum_age: timedelta = timedelta(days=14),
) -> ReleaseCertification:
    """Require current passing Tier 0-3 evidence for the release commit."""

    if not isinstance(release_commit_sha, str) or len(release_commit_sha) != 40:
        raise ValueError("release_commit_sha must be a full 40-character Git SHA.")
    try:
        int(release_commit_sha, 16)
    except ValueError as exc:
        raise ValueError("release_commit_sha must be hexadecimal.") from exc
    if any(not isinstance(item, CertificationTierEvidence) for item in evidence):
        raise TypeError("evidence must contain CertificationTierEvidence values.")
    tiers = [item.tier for item in evidence]
    if len(tiers) != len(set(tiers)):
        raise ValueError("Certification evidence tiers must be unique.")
    if not isinstance(maximum_age, timedelta) or maximum_age <= timedelta(0):
        raise ValueError("maximum_age must be a positive timedelta.")
    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo is None:
        raise ValueError("now must be timezone-aware.")

    by_tier = {item.tier: item for item in evidence}
    reasons: list[str] = []
    for tier, name in TIER_NAMES.items():
        item = by_tier.get(tier)
        if item is None:
            reasons.append(f"Tier {tier} ({name}) evidence is missing")
            continue
        if not item.passed:
            reasons.append(f"Tier {tier} ({name}) did not pass")
        if item.commit_sha != release_commit_sha:
            reasons.append(f"Tier {tier} ({name}) was run for a different commit")
        age = observed_now.astimezone(timezone.utc) - item.completed_at.astimezone(
            timezone.utc
        )
        if age < timedelta(0):
            reasons.append(f"Tier {tier} ({name}) completion time is in the future")
        elif age > maximum_age:
            reasons.append(f"Tier {tier} ({name}) evidence is stale")
    return ReleaseCertification(
        certified=not reasons,
        reasons=tuple(reasons),
        evidence=tuple(sorted(evidence, key=lambda item: item.tier)),
    )


__all__ = [
    "CertificationTier",
    "CertificationTierEvidence",
    "ReleaseCertification",
    "TIER_NAMES",
    "evaluate_release_certification",
]
