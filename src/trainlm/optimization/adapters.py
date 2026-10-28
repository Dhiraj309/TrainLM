"""Explicit, version-guarded model adapter registration and resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .capabilities import ModelCapabilities


def _text_tuple(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a tuple of non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique.")
    return values


@dataclass(frozen=True, slots=True)
class PackageVersionGuard:
    """An explicit allow-list of package versions validated by an adapter."""

    package: str
    supported_versions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.package, str) or not self.package.strip():
            raise ValueError("Version guard package cannot be empty.")
        object.__setattr__(
            self,
            "supported_versions",
            _text_tuple("supported_versions", self.supported_versions),
        )
        if not self.supported_versions:
            raise ValueError("A version guard requires at least one tested version.")


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Declarative eligibility contract for one optional model adapter.

    Class names select an explicit integration boundary. Capability guards are
    still mandatory so a familiar class name cannot stand in for semantics.
    """

    adapter_id: str
    model_classes: tuple[str, ...]
    config_classes: tuple[str, ...] = ()
    source_providers: tuple[str, ...] = ("huggingface",)
    capability_kinds: tuple[tuple[str, str], ...] = ()
    version_guards: tuple[PackageVersionGuard, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or not self.adapter_id.strip():
            raise ValueError("Adapter ID cannot be empty.")
        object.__setattr__(self, "model_classes", _text_tuple("model_classes", self.model_classes))
        object.__setattr__(self, "config_classes", _text_tuple("config_classes", self.config_classes))
        object.__setattr__(self, "source_providers", _text_tuple("source_providers", self.source_providers))
        if not self.model_classes:
            raise ValueError("Adapters must name at least one explicit model class.")
        if not self.source_providers:
            raise ValueError("Adapters must name at least one source provider.")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("Adapter priority must be an integer.")
        seen_components: set[str] = set()
        for guard in self.capability_kinds:
            if not isinstance(guard, tuple) or len(guard) != 2:
                raise ValueError("Capability guards must be (component, kind) tuples.")
            component, kind = guard
            if component not in ModelCapabilities.COMPONENT_NAMES or not kind:
                raise ValueError(f"Invalid adapter capability guard: {guard!r}.")
            if component in seen_components:
                raise ValueError("Adapter capability guards must be unique by component.")
            seen_components.add(component)
        if any(not isinstance(guard, PackageVersionGuard) for guard in self.version_guards):
            raise ValueError("version_guards must contain PackageVersionGuard values.")
        packages = [guard.package for guard in self.version_guards]
        if len(packages) != len(set(packages)):
            raise ValueError("Adapter package version guards must be unique.")


@dataclass(frozen=True, slots=True)
class AdapterCandidate:
    """One deterministic adapter eligibility result for explanation output."""

    adapter_id: str
    eligible: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AdapterResolution:
    """Selected adapter and the complete ordered candidate audit."""

    selected: AdapterSpec | None
    candidates: tuple[AdapterCandidate, ...]


class ModelAdapterRegistry:
    """Registry that resolves adapters without importing or mutating models."""

    def __init__(self) -> None:
        self._adapters: dict[str, AdapterSpec] = {}

    def register(self, adapter: AdapterSpec) -> None:
        if not isinstance(adapter, AdapterSpec):
            raise TypeError("adapter must be an AdapterSpec.")
        if adapter.adapter_id in self._adapters:
            raise ValueError(f"Adapter already registered: {adapter.adapter_id}")
        self._adapters[adapter.adapter_id] = adapter

    def resolve(
        self,
        capabilities: ModelCapabilities,
        *,
        package_versions: Mapping[str, str],
    ) -> AdapterResolution:
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities.")
        ordered = sorted(
            self._adapters.values(), key=lambda item: (-item.priority, item.adapter_id)
        )
        candidates: list[AdapterCandidate] = []
        selected: AdapterSpec | None = None
        for adapter in ordered:
            reasons = self._rejection_reasons(adapter, capabilities, package_versions)
            eligible = not reasons
            candidates.append(AdapterCandidate(adapter.adapter_id, eligible, tuple(reasons)))
            if eligible and selected is None:
                selected = adapter
        return AdapterResolution(selected, tuple(candidates))

    @staticmethod
    def _rejection_reasons(
        adapter: AdapterSpec,
        capabilities: ModelCapabilities,
        package_versions: Mapping[str, str],
    ) -> list[str]:
        reasons: list[str] = []
        if capabilities.model_class not in adapter.model_classes:
            reasons.append(f"model class {capabilities.model_class!r} is not explicitly supported")
        if adapter.config_classes and capabilities.config_class not in adapter.config_classes:
            reasons.append(f"config class {capabilities.config_class!r} is not explicitly supported")
        if capabilities.source_provider not in adapter.source_providers:
            reasons.append(f"source provider {capabilities.source_provider!r} is not supported")
        for component, expected_kind in adapter.capability_kinds:
            actual = capabilities.component(component)
            if actual.status not in {"known", "inferred"} or actual.kind != expected_kind:
                reasons.append(
                    f"{component} requires {expected_kind!r}; observed "
                    f"status={actual.status!r}, kind={actual.kind!r}"
                )
        for guard in adapter.version_guards:
            installed = package_versions.get(guard.package)
            if installed is None:
                reasons.append(f"package {guard.package!r} version is unavailable")
            elif installed not in guard.supported_versions:
                reasons.append(
                    f"package {guard.package!r} version {installed!r} is untested; "
                    f"supported={guard.supported_versions!r}"
                )
        return reasons


__all__ = [
    "AdapterCandidate",
    "AdapterResolution",
    "AdapterSpec",
    "ModelAdapterRegistry",
    "PackageVersionGuard",
]
