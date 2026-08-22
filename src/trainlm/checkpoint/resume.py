"""Versioned exact-training-resume manifest contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, ClassVar, Literal, Mapping

from .contracts import (
    ArtifactRecord,
    AtomicCommit,
    require_text,
    tuple_field,
    validate_artifacts,
    validate_utc_timestamp,
)


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    optimizer_step: int
    micro_step: int
    epoch: int
    tokens_seen: int
    samples_seen: int

    def __post_init__(self) -> None:
        for name in (
            "optimizer_step", "micro_step", "epoch", "tokens_seen", "samples_seen"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Training progress {name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class MeshAxis:
    name: str
    size: int

    def __post_init__(self) -> None:
        require_text("mesh axis name", self.name)
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise ValueError("Mesh axis size must be positive.")


@dataclass(frozen=True, slots=True)
class ResumeTopology:
    backend: str
    world_size: int
    precision: str
    state_layout: Literal["canonical", "sharded"]
    mesh: tuple[MeshAxis, ...]

    def __post_init__(self) -> None:
        for name in ("backend", "precision", "state_layout"):
            require_text(name, getattr(self, name))
        if self.state_layout not in {"canonical", "sharded"}:
            raise ValueError(
                f"Unsupported resume topology state_layout: {self.state_layout}"
            )
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 1
        ):
            raise ValueError("Resume topology world_size must be positive.")
        object.__setattr__(self, "mesh", tuple_field("mesh", self.mesh))
        if not self.mesh or any(not isinstance(axis, MeshAxis) for axis in self.mesh):
            raise ValueError("Resume topology mesh must contain MeshAxis objects.")
        if len({axis.name for axis in self.mesh}) != len(self.mesh):
            raise ValueError("Resume topology mesh axis names must be unique.")
        mesh_size = 1
        for axis in self.mesh:
            mesh_size *= axis.size
        if mesh_size != self.world_size:
            raise ValueError("Resume topology mesh must equal world_size.")


@dataclass(frozen=True, slots=True)
class StateDescriptor:
    """Version and artifact membership for one resumable state component."""

    name: str
    implementation: str
    format_version: int
    layout: Literal["canonical", "sharded", "replicated", "per_rank", "per_worker"]
    keying: str
    artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("name", "implementation", "keying"):
            require_text(f"state descriptor {name}", getattr(self, name))
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version < 1
        ):
            raise ValueError("State descriptor format_version must be positive.")
        if self.layout not in {
            "canonical", "sharded", "replicated", "per_rank", "per_worker"
        }:
            raise ValueError(f"Unsupported state descriptor layout: {self.layout}")
        object.__setattr__(
            self,
            "artifact_ids",
            tuple_field("state artifact_ids", self.artifact_ids),
        )
        if not self.artifact_ids or any(
            not isinstance(item, str) or not item.strip() for item in self.artifact_ids
        ):
            raise ValueError("State descriptors require artifact IDs.")
        if len(self.artifact_ids) != len(set(self.artifact_ids)):
            raise ValueError("State descriptor artifact IDs must be unique.")


@dataclass(frozen=True, slots=True)
class DataCursor:
    """Exact cursor for one data-parallel replica and loader worker."""

    replica_rank: int
    worker_id: int
    dataset_fingerprint: str
    state_artifact_id: str
    source_index: int
    shard_index: int
    document_index: int
    token_offset: int
    epoch: int
    samples_consumed: int
    exact: bool = True

    def __post_init__(self) -> None:
        require_text("dataset_fingerprint", self.dataset_fingerprint)
        require_text("state_artifact_id", self.state_artifact_id)
        if not isinstance(self.exact, bool):
            raise ValueError("Data cursor exact must be boolean.")
        for name in (
            "replica_rank", "worker_id", "source_index", "shard_index",
            "document_index", "token_offset", "epoch", "samples_consumed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Data cursor {name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class LayoutState:
    capability_fingerprint: str
    execution_plan_id: str
    parameter_layout: str
    applied_transform_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.capability_fingerprint, str) or len(
            self.capability_fingerprint
        ) != 64 or any(
            value not in "0123456789abcdef" for value in self.capability_fingerprint
        ):
            raise ValueError("Layout capability fingerprint must be SHA-256.")
        require_text("execution_plan_id", self.execution_plan_id)
        require_text("parameter_layout", self.parameter_layout)
        object.__setattr__(
            self,
            "applied_transform_ids",
            tuple_field("applied_transform_ids", self.applied_transform_ids),
        )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.applied_transform_ids
        ):
            raise ValueError("Applied transformation IDs cannot be empty.")
        if len(self.applied_transform_ids) != len(set(self.applied_transform_ids)):
            raise ValueError("Applied transformation IDs must be unique.")


@dataclass(frozen=True, slots=True)
class ResumeManifest:
    """Manifest required for exact continuation of a TrainLM run."""

    schema_version: int
    checkpoint_id: str
    created_at: str
    framework_version: str
    framework_revision: str
    progress: TrainingProgress
    topology: ResumeTopology
    layout: LayoutState
    states: tuple[StateDescriptor, ...]
    data_cursors: tuple[DataCursor, ...]
    artifacts: tuple[ArtifactRecord, ...]
    commit: AtomicCommit
    manifest_type: str = "trainlm_resume"

    _REQUIRED_ROLES: ClassVar[set[str]] = {
        "model", "optimizer", "scheduler", "trainer", "runtime", "rng", "data"
    }

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ResumeManifest supports schema_version=1 only.")
        if self.manifest_type != "trainlm_resume":
            raise ValueError("Resume manifest_type must be 'trainlm_resume'.")
        for name in ("checkpoint_id", "framework_version", "framework_revision"):
            require_text(name, getattr(self, name))
        validate_utc_timestamp("created_at", self.created_at)
        if not isinstance(self.progress, TrainingProgress):
            raise ValueError("Resume progress must be TrainingProgress.")
        if not isinstance(self.topology, ResumeTopology):
            raise ValueError("Resume topology must be ResumeTopology.")
        if not isinstance(self.layout, LayoutState):
            raise ValueError("Resume layout must be LayoutState.")
        if not isinstance(self.commit, AtomicCommit):
            raise ValueError("Resume commit must be AtomicCommit.")
        object.__setattr__(
            self, "data_cursors", tuple_field("data_cursors", self.data_cursors)
        )
        object.__setattr__(self, "states", tuple_field("states", self.states))
        object.__setattr__(self, "artifacts", tuple_field("artifacts", self.artifacts))
        validate_artifacts(
            self.artifacts,
            require_complete_shards=self.commit.is_committed,
        )
        if any(not isinstance(cursor, DataCursor) for cursor in self.data_cursors):
            raise ValueError("Resume data cursors must be DataCursor objects.")
        if any(not isinstance(state, StateDescriptor) for state in self.states):
            raise ValueError("Resume states must be StateDescriptor objects.")
        state_names = [state.name for state in self.states]
        if len(state_names) != len(set(state_names)):
            raise ValueError("Resume state descriptor names must be unique.")
        cursor_ids = [(cursor.replica_rank, cursor.worker_id) for cursor in self.data_cursors]
        if len(cursor_ids) != len(set(cursor_ids)):
            raise ValueError("Resume data cursors must be unique per replica/worker.")
        if self.commit.is_committed:
            roles = {item.role for item in self.artifacts if item.required}
            missing = self._REQUIRED_ROLES - roles
            if missing:
                raise ValueError(
                    "Committed resume checkpoint is missing required roles: "
                    + ", ".join(sorted(missing))
                )
            missing_states = self._REQUIRED_ROLES - set(state_names)
            if missing_states:
                raise ValueError(
                    "Committed resume checkpoint is missing state descriptors: "
                    + ", ".join(sorted(missing_states))
                )
            artifacts_by_id = {item.artifact_id: item for item in self.artifacts}
            for state in self.states:
                state_artifacts = []
                for artifact_id in state.artifact_ids:
                    artifact = artifacts_by_id.get(artifact_id)
                    if (
                        artifact is None
                        or not artifact.required
                        or artifact.role != state.name
                    ):
                        raise ValueError(
                            f"State '{state.name}' references an invalid artifact."
                        )
                    state_artifacts.append(artifact)
                if state.layout == "sharded" and self.topology.world_size > 1:
                    if any(
                        artifact.shard_count != self.topology.world_size
                        for artifact in state_artifacts
                    ):
                        raise ValueError(
                            f"Sharded state '{state.name}' must cover world size."
                        )
                if state.layout == "per_rank":
                    ranks = {artifact.rank for artifact in state_artifacts}
                    if ranks != set(range(self.topology.world_size)):
                        raise ValueError(
                            f"Per-rank state '{state.name}' must cover every rank."
                        )
            optimizer_state = next(
                state for state in self.states if state.name == "optimizer"
            )
            if optimizer_state.keying != "canonical_parameter_name":
                raise ValueError(
                    "Optimizer state must be keyed by canonical parameter name."
                )
            if not self.data_cursors or any(not cursor.exact for cursor in self.data_cursors):
                raise ValueError("Committed resume checkpoints require exact data cursors.")
            for cursor in self.data_cursors:
                state = artifacts_by_id.get(cursor.state_artifact_id)
                if state is None or state.role != "data" or not state.required:
                    raise ValueError(
                        "Every exact data cursor must reference a data-state artifact."
                    )
            ranks = {cursor.replica_rank for cursor in self.data_cursors}
            if ranks != set(range(self.topology.world_size)):
                raise ValueError("Committed resume checkpoints require every replica cursor.")

    @property
    def is_resumable(self) -> bool:
        return self.commit.is_committed

    def assert_resumable(self) -> None:
        if not self.is_resumable:
            raise RuntimeError(
                "Checkpoint transaction is incomplete and must be ignored."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResumeManifest":
        values = dict(data)
        values["progress"] = TrainingProgress(**values["progress"])
        topology = dict(values["topology"])
        topology["mesh"] = tuple(MeshAxis(**axis) for axis in topology["mesh"])
        values["topology"] = ResumeTopology(**topology)
        values["layout"] = LayoutState(**values["layout"])
        values["states"] = tuple(
            StateDescriptor(**state) for state in values.get("states", ())
        )
        values["data_cursors"] = tuple(
            DataCursor(**cursor) for cursor in values.get("data_cursors", ())
        )
        values["artifacts"] = tuple(
            ArtifactRecord(**artifact) for artifact in values.get("artifacts", ())
        )
        values["commit"] = AtomicCommit(**values["commit"])
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "ResumeManifest":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Resume manifest JSON root must be an object.")
        return cls.from_dict(data)
