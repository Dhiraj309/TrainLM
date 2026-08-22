"""Shared immutable records for checkpoint and export manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Any, Literal

CommitStatus = Literal["staging", "complete", "failed"]
AtomicStrategy = Literal["directory_rename", "commit_marker"]


def tuple_field(name: str, value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    return tuple(value)


def require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty.")
    return value


def validate_utc_timestamp(name: str, value: str) -> None:
    require_text(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must include a UTC offset.")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One content-addressed file in a checkpoint transaction."""

    artifact_id: str
    role: str
    path: str
    format: str
    sha256: str
    size_bytes: int
    required: bool = True
    shard_group: str | None = None
    shard_index: int | None = None
    shard_count: int | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        for name in ("artifact_id", "role", "path", "format"):
            require_text(name, getattr(self, name))
        path = PurePosixPath(self.path)
        if (
            "\\" in self.path
            or path.is_absolute()
            or self.path in {".", ".."}
            or ".." in path.parts
        ):
            raise ValueError("Artifact paths must be safe relative POSIX paths.")
        if not isinstance(self.sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.sha256
        ):
            raise ValueError("Artifact sha256 must be lowercase hexadecimal.")
        if not isinstance(self.required, bool):
            raise ValueError("Artifact required must be boolean.")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("Artifact size_bytes must be non-negative.")
        shard_values = (self.shard_group, self.shard_index, self.shard_count)
        if any(value is not None for value in shard_values):
            if any(value is None for value in shard_values):
                raise ValueError("Shard group, index, and count must be set together.")
            require_text("shard_group", self.shard_group)
            if (
                isinstance(self.shard_count, bool)
                or not isinstance(self.shard_count, int)
                or isinstance(self.shard_index, bool)
                or not isinstance(self.shard_index, int)
                or self.shard_count < 1
                or not 0 <= self.shard_index < self.shard_count
            ):
                raise ValueError("Artifact shard index must be within shard count.")
        if self.rank is not None and (
            isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0
        ):
            raise ValueError("Artifact rank must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class AtomicCommit:
    """Atomic publication state and incomplete-save behavior."""

    status: CommitStatus
    transaction_id: str
    strategy: AtomicStrategy
    staging_suffix: str = ".incomplete"
    completion_marker: str = "COMPLETED"
    incomplete_policy: Literal["ignore"] = "ignore"

    def __post_init__(self) -> None:
        if self.status not in {"staging", "complete", "failed"}:
            raise ValueError(f"Unsupported commit status: {self.status}")
        if self.strategy not in {"directory_rename", "commit_marker"}:
            raise ValueError(f"Unsupported atomic strategy: {self.strategy}")
        require_text("transaction_id", self.transaction_id)
        require_text("staging_suffix", self.staging_suffix)
        require_text("completion_marker", self.completion_marker)
        if "/" in self.completion_marker or "\\" in self.completion_marker:
            raise ValueError("Completion marker must be a filename.")
        if self.incomplete_policy != "ignore":
            raise ValueError("Incomplete checkpoint transactions must be ignored.")

    @property
    def is_committed(self) -> bool:
        return self.status == "complete"


def validate_artifacts(
    artifacts: tuple[ArtifactRecord, ...],
    *,
    require_complete_shards: bool,
) -> None:
    if any(not isinstance(item, ArtifactRecord) for item in artifacts):
        raise ValueError("Artifacts must be ArtifactRecord objects.")
    ids = [item.artifact_id for item in artifacts]
    paths = [item.path for item in artifacts]
    if len(ids) != len(set(ids)):
        raise ValueError("Artifact IDs must be unique.")
    if len(paths) != len(set(paths)):
        raise ValueError("Artifact paths must be unique.")

    groups: dict[str, list[ArtifactRecord]] = {}
    for artifact in artifacts:
        if artifact.shard_group is not None:
            groups.setdefault(artifact.shard_group, []).append(artifact)
    for name, shards in groups.items():
        counts = {item.shard_count for item in shards}
        indices = {item.shard_index for item in shards}
        expected = next(iter(counts)) if len(counts) == 1 else None
        if expected is None or (
            require_complete_shards and indices != set(range(expected))
        ):
            raise ValueError(f"Shard group '{name}' must be complete and contiguous.")
