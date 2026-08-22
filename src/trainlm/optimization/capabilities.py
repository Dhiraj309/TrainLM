"""Serializable model capability schema with no model mutation behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, ClassVar, Literal, Mapping

CapabilityStatus = Literal["known", "inferred", "unknown", "unsupported"]
Scalar = str | int | float | bool | None


def _tuple_field(name: str, value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class CapabilityFact:
    """One scalar capability fact and its optional evidence source."""

    name: str
    value: Scalar
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Capability fact name cannot be empty.")
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source.strip()
        ):
            raise ValueError("Capability fact source cannot be empty.")
        if self.value is not None and not isinstance(
            self.value, (str, int, float, bool)
        ):
            raise ValueError("Capability fact values must be JSON scalars.")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Capability fact floats must be finite.")


@dataclass(frozen=True, slots=True)
class ComponentCapability:
    """Semantic description of one model component."""

    status: CapabilityStatus
    kind: str | None = None
    facts: tuple[CapabilityFact, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in {"known", "inferred", "unknown", "unsupported"}:
            raise ValueError(f"Unsupported capability status: {self.status}")
        object.__setattr__(self, "facts", _tuple_field("facts", self.facts))
        object.__setattr__(
            self,
            "evidence",
            _tuple_field("evidence", self.evidence),
        )
        object.__setattr__(self, "notes", _tuple_field("notes", self.notes))
        if self.kind is not None and (
            not isinstance(self.kind, str) or not self.kind.strip()
        ):
            raise ValueError("Capability kind must be non-empty text.")
        if self.status in {"known", "inferred"} and self.kind is None:
            raise ValueError(f"Capability status '{self.status}' requires a kind.")
        if self.status == "unknown" and self.kind is not None:
            raise ValueError("Unknown capabilities cannot claim a component kind.")
        if any(not isinstance(fact, CapabilityFact) for fact in self.facts):
            raise ValueError("Capability facts must be CapabilityFact objects.")
        names = [fact.name for fact in self.facts]
        if len(names) != len(set(names)):
            raise ValueError("Capability fact names must be unique per component.")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*self.evidence, *self.notes)
        ):
            raise ValueError("Capability evidence and notes cannot be empty.")

    @classmethod
    def unknown(cls, note: str | None = None) -> "ComponentCapability":
        return cls(status="unknown", notes=(() if note is None else (note,)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComponentCapability":
        return cls(
            status=data["status"],
            kind=data.get("kind"),
            facts=tuple(CapabilityFact(**fact) for fact in data.get("facts", ())),
            evidence=tuple(data.get("evidence", ())),
            notes=tuple(data.get("notes", ())),
        )


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Versioned semantic capability report for a dense causal LM."""

    schema_version: int
    model_type: str
    model_class: str
    config_class: str
    source_provider: str
    architectures: tuple[str, ...]
    attention: ComponentCapability
    position: ComponentCapability
    normalization: ComponentCapability
    mlp: ComponentCapability
    residual: ComponentCapability
    projections: ComponentCapability
    embedding: ComponentCapability
    lm_head: ComponentCapability
    checkpointing: ComponentCapability
    warnings: tuple[str, ...] = field(default_factory=tuple)

    _COMPONENTS: ClassVar[tuple[str, ...]] = (
        "attention",
        "position",
        "normalization",
        "mlp",
        "residual",
        "projections",
        "embedding",
        "lm_head",
        "checkpointing",
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ModelCapabilities supports schema_version=1 only.")
        for name in ("model_type", "model_class", "config_class", "source_provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty.")
        object.__setattr__(
            self,
            "architectures",
            _tuple_field("architectures", self.architectures),
        )
        object.__setattr__(
            self,
            "warnings",
            _tuple_field("warnings", self.warnings),
        )
        if len(self.architectures) != len(set(self.architectures)):
            raise ValueError("Model architectures must be unique.")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*self.architectures, *self.warnings)
        ):
            raise ValueError("Architectures and warnings cannot contain empty text.")
        for name in self._COMPONENTS:
            if not isinstance(getattr(self, name), ComponentCapability):
                raise ValueError(f"{name} must be a ComponentCapability.")

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def component(self, name: str) -> ComponentCapability:
        if name not in self._COMPONENTS:
            raise KeyError(f"Unknown capability component: {name}")
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelCapabilities":
        values = dict(data)
        for name in cls._COMPONENTS:
            values[name] = ComponentCapability.from_dict(values[name])
        values["architectures"] = tuple(values.get("architectures", ()))
        values["warnings"] = tuple(values.get("warnings", ()))
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "ModelCapabilities":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Model capability JSON root must be an object.")
        return cls.from_dict(data)
