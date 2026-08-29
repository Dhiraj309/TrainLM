"""Validation and planning for distributed resume checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

from .resume import ResumeManifest


_REQUIRED_STATES = (
    "model",
    "optimizer",
    "scheduler",
    "trainer",
    "runtime",
    "rng",
    "data",
)


@dataclass(frozen=True, slots=True)
class DistributedResumePlan:
    """Validated topology and state ownership for a distributed save."""

    backend: str
    world_size: int
    direct_shard_io: bool
    state_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend cannot be empty.")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise ValueError("world_size must be a positive integer.")
        if self.world_size < 1:
            raise ValueError("world_size must be a positive integer.")
        if not isinstance(self.direct_shard_io, bool):
            raise ValueError("direct_shard_io must be boolean.")
        object.__setattr__(self, "state_names", tuple(self.state_names))
        if self.state_names != _REQUIRED_STATES:
            raise ValueError(
                "state_names must contain the canonical distributed state order."
            )


def plan_distributed_resume(
    manifest: ResumeManifest,
    *,
    backend: str | None = None,
    world_size: int | None = None,
    direct_shard_io: bool = True,
) -> DistributedResumePlan:
    """Validate a committed manifest before distributed shard I/O begins."""

    if not isinstance(manifest, ResumeManifest):
        raise TypeError("manifest must be a ResumeManifest.")
    manifest.assert_resumable()
    expected_backend = manifest.topology.backend if backend is None else backend
    expected_world_size = (
        manifest.topology.world_size if world_size is None else world_size
    )
    if expected_backend != manifest.topology.backend:
        raise ValueError("Resume backend does not match checkpoint topology.")
    if expected_world_size != manifest.topology.world_size:
        raise ValueError("Resume world_size does not match checkpoint topology.")
    if not isinstance(direct_shard_io, bool):
        raise ValueError("direct_shard_io must be boolean.")

    states = {state.name: state for state in manifest.states}
    if set(states) != set(_REQUIRED_STATES):
        raise ValueError("Resume states must cover every canonical state.")
    if manifest.topology.world_size > 1:
        if states["model"].layout != "sharded":
            raise ValueError("Distributed model state must be sharded.")
        if states["optimizer"].layout != "sharded":
            raise ValueError("Distributed optimizer state must be sharded.")
        if states["rng"].layout != "per_rank":
            raise ValueError("Distributed RNG state must be per-rank.")
        if states["data"].layout != "per_worker":
            raise ValueError("Distributed data state must be per-worker.")
    ranks = {cursor.replica_rank for cursor in manifest.data_cursors}
    if ranks != set(range(manifest.topology.world_size)):
        raise ValueError("Distributed resume requires one exact cursor per rank.")
    return DistributedResumePlan(
        backend=manifest.topology.backend,
        world_size=manifest.topology.world_size,
        direct_shard_io=direct_shard_io,
        state_names=_REQUIRED_STATES,
    )
