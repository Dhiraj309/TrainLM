"""Declarative eligibility constraints shared by adapters and provider planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .capabilities import CapabilityFact, ModelCapabilities


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text.")


def _strings(name: str, values) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of unique strings.")
    result = tuple(values)
    for value in result:
        _text(name, value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return tuple(sorted(result))


def _versions(values) -> tuple[tuple[str, str], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    result = []
    for name, value in items:
        _text("distribution", name)
        _text("version", value)
        result.append((canonicalize_name(name), value))
    if len(dict(result)) != len(result):
        raise ValueError("Distribution versions must be unique after normalization.")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class VersionRequirement:
    distribution: str
    specifier: str

    def __post_init__(self) -> None:
        _text("distribution", self.distribution)
        _text("specifier", self.specifier)
        object.__setattr__(self, "distribution", canonicalize_name(self.distribution))
        object.__setattr__(self, "specifier", str(SpecifierSet(self.specifier)))

    def rejection(self, versions: Mapping[str, str]) -> str | None:
        value = versions.get(self.distribution)
        if value is None:
            return f"missing version for {self.distribution} ({self.specifier})"
        try:
            version = Version(value)
        except InvalidVersion:
            return f"invalid version for {self.distribution}: {value}"
        if version not in SpecifierSet(self.specifier):
            return f"{self.distribution}=={value} does not satisfy {self.specifier}"
        return None


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    component: str
    kinds: tuple[str, ...]
    facts: tuple[CapabilityFact, ...] = field(default_factory=tuple)
    allow_inferred: bool = False

    def __post_init__(self) -> None:
        if self.component not in ModelCapabilities.COMPONENT_NAMES:
            raise ValueError(f"Unknown capability component: {self.component}")
        object.__setattr__(self, "kinds", _strings("kinds", self.kinds))
        if not self.kinds:
            raise ValueError("Capability requirements must name supported kinds.")
        facts = tuple(self.facts)
        if any(not isinstance(fact, CapabilityFact) for fact in facts):
            raise ValueError("Required facts must be CapabilityFact objects.")
        if len({fact.name for fact in facts}) != len(facts):
            raise ValueError("Required fact names must be unique.")
        object.__setattr__(self, "facts", tuple(sorted(facts, key=lambda fact: fact.name)))
        if not isinstance(self.allow_inferred, bool):
            raise ValueError("allow_inferred must be boolean.")

    def rejections(self, capabilities: ModelCapabilities) -> tuple[str, ...]:
        component = capabilities.component(self.component)
        allowed = {"known", "inferred"} if self.allow_inferred else {"known"}
        if component.status not in allowed:
            return (f"{self.component} status {component.status} is not accepted",)
        errors = []
        if not component.evidence:
            errors.append(f"{self.component} has no semantic evidence")
        if component.kind not in self.kinds:
            errors.append(f"{self.component} kind {component.kind} is not in {self.kinds}")
        facts = {fact.name: fact for fact in component.facts}
        for expected in self.facts:
            actual = facts.get(expected.name)
            if actual is None:
                errors.append(f"missing {self.component}.{expected.name}")
            elif actual.source is None:
                errors.append(f"{self.component}.{expected.name} has no evidence source")
            elif type(actual.value) is not type(expected.value) or actual.value != expected.value:
                errors.append(f"{self.component}.{expected.name} must equal {expected.value!r}")
        return tuple(errors)


@dataclass(frozen=True, slots=True)
class ShapeConstraint:
    name: str
    minimum: int = 1
    maximum: int | None = None
    multiple_of: int = 1

    def __post_init__(self) -> None:
        _text("shape dimension", self.name)
        for name in ("minimum", "maximum", "multiple_of"):
            value = getattr(self, name)
            if value is None and name == "maximum":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Shape {name} must be a positive integer.")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("Shape maximum must be at least minimum.")

    def rejection(self, shapes: Mapping[str, int]) -> str | None:
        value = shapes.get(self.name)
        if value is None:
            return f"missing shape dimension {self.name}"
        if (value < self.minimum or (self.maximum is not None and value > self.maximum)
                or value % self.multiple_of):
            return (f"shape {self.name}={value} requires minimum={self.minimum}, "
                    f"maximum={self.maximum}, multiple_of={self.multiple_of}")
        return None


@dataclass(frozen=True, slots=True)
class PlanningContext:
    backend: str
    precision: str
    mask_kind: str
    training: bool = True
    shapes: tuple[tuple[str, int], ...] | Mapping[str, int] = field(default_factory=tuple)
    versions: tuple[tuple[str, str], ...] | Mapping[str, str] = field(default_factory=tuple)
    adapter_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("backend", "precision", "mask_kind"):
            _text(name, getattr(self, name))
        if not isinstance(self.training, bool):
            raise ValueError("training must be boolean.")
        if self.adapter_id is not None:
            _text("adapter_id", self.adapter_id)
        shapes = self.shapes.items() if isinstance(self.shapes, Mapping) else self.shapes
        shapes = tuple(shapes)
        for name, value in shapes:
            _text("shape dimension", name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Shape dimensions must be positive integers.")
        if len(dict(shapes)) != len(shapes):
            raise ValueError("Shape dimension names must be unique.")
        object.__setattr__(self, "shapes", tuple(sorted(shapes)))
        object.__setattr__(self, "versions", _versions(self.versions))
