"""Versioned plain-Hugging-Face export manifest contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, ClassVar, Mapping

from .contracts import (
    ArtifactRecord,
    AtomicCommit,
    require_text,
    tuple_field,
    validate_artifacts,
    validate_utc_timestamp,
)


@dataclass(frozen=True, slots=True)
class TiedWeight:
    canonical_name: str
    alias_name: str

    def __post_init__(self) -> None:
        require_text("canonical weight name", self.canonical_name)
        require_text("alias weight name", self.alias_name)
        if self.canonical_name == self.alias_name:
            raise ValueError("Tied-weight canonical and alias names must differ.")


@dataclass(frozen=True, slots=True)
class ExportLayout:
    """Proof that runtime transforms were reversed for HF serialization."""

    source_parameter_layout: str
    target_parameter_layout: str
    reversed_transform_ids: tuple[str, ...] = field(default_factory=tuple)
    remaining_transform_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_text("source_parameter_layout", self.source_parameter_layout)
        require_text("target_parameter_layout", self.target_parameter_layout)
        object.__setattr__(
            self,
            "reversed_transform_ids",
            tuple_field("reversed_transform_ids", self.reversed_transform_ids),
        )
        object.__setattr__(
            self,
            "remaining_transform_ids",
            tuple_field("remaining_transform_ids", self.remaining_transform_ids),
        )
        for item in (*self.reversed_transform_ids, *self.remaining_transform_ids):
            require_text("transformation ID", item)
        if len(self.reversed_transform_ids) != len(set(self.reversed_transform_ids)):
            raise ValueError("Reversed transformation IDs must be unique.")
        if len(self.remaining_transform_ids) != len(set(self.remaining_transform_ids)):
            raise ValueError("Remaining transformation IDs must be unique.")
        if set(self.reversed_transform_ids) & set(self.remaining_transform_ids):
            raise ValueError("A transformation cannot be both reversed and remaining.")


@dataclass(frozen=True, slots=True)
class HFExportManifest:
    """Manifest for a directory loadable by plain Transformers APIs."""

    schema_version: int
    export_id: str
    created_at: str
    model_type: str
    architecture: str
    config_class: str
    transformers_version: str
    dtype: str
    safe_serialization: bool
    layout: ExportLayout
    tied_weights: tuple[TiedWeight, ...]
    artifacts: tuple[ArtifactRecord, ...]
    commit: AtomicCommit
    source_checkpoint_id: str | None = None
    manifest_type: str = "huggingface_export"

    _REQUIRED_ROLES: ClassVar[set[str]] = {"config", "model_weights"}
    _FORBIDDEN_ROLES: ClassVar[set[str]] = {
        "optimizer", "scheduler", "trainer", "runtime", "rng", "data"
    }

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("HFExportManifest supports schema_version=1 only.")
        if self.manifest_type != "huggingface_export":
            raise ValueError("HF export manifest_type must be 'huggingface_export'.")
        for name in (
            "export_id", "model_type", "architecture", "config_class",
            "transformers_version", "dtype",
        ):
            require_text(name, getattr(self, name))
        if self.source_checkpoint_id is not None:
            require_text("source_checkpoint_id", self.source_checkpoint_id)
        validate_utc_timestamp("created_at", self.created_at)
        if self.safe_serialization is not True:
            raise ValueError("HF exports require safe_serialization with safetensors.")
        if not isinstance(self.layout, ExportLayout):
            raise ValueError("HF export layout must be ExportLayout.")
        if not isinstance(self.commit, AtomicCommit):
            raise ValueError("HF export commit must be AtomicCommit.")
        object.__setattr__(
            self, "tied_weights", tuple_field("tied_weights", self.tied_weights)
        )
        object.__setattr__(self, "artifacts", tuple_field("artifacts", self.artifacts))
        if any(not isinstance(item, TiedWeight) for item in self.tied_weights):
            raise ValueError("HF export ties must be TiedWeight objects.")
        aliases = [item.alias_name for item in self.tied_weights]
        if len(aliases) != len(set(aliases)):
            raise ValueError("HF export tied-weight aliases must be unique.")
        validate_artifacts(
            self.artifacts,
            require_complete_shards=self.commit.is_committed,
        )
        roles = {item.role for item in self.artifacts}
        required_roles = {item.role for item in self.artifacts if item.required}
        forbidden = roles & self._FORBIDDEN_ROLES
        if forbidden:
            raise ValueError(
                "HF exports cannot contain training-only roles: "
                + ", ".join(sorted(forbidden))
            )
        if self.commit.is_committed:
            missing = self._REQUIRED_ROLES - required_roles
            if missing:
                raise ValueError(
                    "Committed HF export is missing required roles: "
                    + ", ".join(sorted(missing))
                )
            if self.layout.target_parameter_layout != "huggingface":
                raise ValueError("Committed HF exports require Hugging Face layout.")
            if self.layout.remaining_transform_ids:
                raise ValueError(
                    "Committed HF exports cannot retain runtime transformations."
                )
            weight_formats = {
                item.format for item in self.artifacts if item.role == "model_weights"
            }
            if weight_formats != {"safetensors"}:
                raise ValueError("HF model weights must use safetensors format.")
            weight_files = [
                item for item in self.artifacts if item.role == "model_weights"
            ]
            if len(weight_files) > 1 and "weight_index" not in required_roles:
                raise ValueError("Sharded HF weights require a weight index artifact.")

    @property
    def is_loadable(self) -> bool:
        return self.commit.is_committed

    def assert_loadable(self) -> None:
        if not self.is_loadable:
            raise RuntimeError("HF export transaction is incomplete and must be ignored.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HFExportManifest":
        values = dict(data)
        values["layout"] = ExportLayout(**values["layout"])
        values["tied_weights"] = tuple(
            TiedWeight(**item) for item in values.get("tied_weights", ())
        )
        values["artifacts"] = tuple(
            ArtifactRecord(**item) for item in values.get("artifacts", ())
        )
        values["commit"] = AtomicCommit(**values["commit"])
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "HFExportManifest":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("HF export manifest JSON root must be an object.")
        return cls.from_dict(data)
