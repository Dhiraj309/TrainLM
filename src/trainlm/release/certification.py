"""Current-evidence gate for dense-AR release certification tiers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Literal, Mapping

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
        if isinstance(self.tier, bool) or self.tier not in TIER_NAMES:
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


def load_release_certification(manifest_path: str | Path) -> ReleaseCertification:
    """Load current tier evidence from a strict, self-contained manifest."""

    if not isinstance(manifest_path, (str, Path)):
        raise TypeError("manifest_path must be a path.")
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("manifest_path must reference an existing file.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid release certification manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Release certification manifest must contain a JSON object.")
    if set(manifest) != {
        "schema_version",
        "release_commit_sha",
        "evaluated_at",
        "maximum_age_seconds",
        "evidence",
    }:
        raise ValueError(
            "Release certification manifest keys must match schema version 1."
        )
    version = manifest["schema_version"]
    if isinstance(version, bool) or version != 1:
        raise ValueError(
            "Release certification manifest supports schema_version=1 only."
        )
    evidence_payload = manifest["evidence"]
    if isinstance(evidence_payload, (str, bytes)) or not isinstance(
        evidence_payload, list
    ):
        raise ValueError("Release certification evidence must be a JSON array.")
    maximum_age_seconds = manifest["maximum_age_seconds"]
    if (
        isinstance(maximum_age_seconds, bool)
        or not isinstance(maximum_age_seconds, int)
        or maximum_age_seconds <= 0
    ):
        raise ValueError("maximum_age_seconds must be a positive integer.")
    evaluated_at = _parse_timestamp("evaluated_at", manifest["evaluated_at"])

    expected = {field.name for field in fields(CertificationTierEvidence)}
    root = path.resolve().parent
    evidence = []
    for index, payload in enumerate(evidence_payload):
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError(f"Tier evidence {index} keys must match the schema.")
        values = dict(payload)
        values["completed_at"] = _parse_timestamp(
            f"evidence[{index}].completed_at", values["completed_at"]
        )
        try:
            item = CertificationTierEvidence(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid tier evidence {index}: {exc}") from exc
        artifact = Path(item.artifact)
        if artifact.is_absolute():
            raise ValueError("Certification artifact paths must be relative.")
        resolved = (root / artifact).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(
                "Certification artifact path escapes the manifest or is missing."
            )
        evidence.append(item)
    return evaluate_release_certification(
        tuple(evidence),
        release_commit_sha=manifest["release_commit_sha"],
        now=evaluated_at,
        maximum_age=timedelta(seconds=maximum_age_seconds),
    )


def _parse_timestamp(name: str, value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return parsed


__all__ = [
    "CertificationTier",
    "CertificationTierEvidence",
    "ReleaseCertification",
    "TIER_NAMES",
    "evaluate_release_certification",
    "load_release_certification",
]
