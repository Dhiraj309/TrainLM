"""Optional exact-class adapters that enrich evidence without editing models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from torch import nn

from .capabilities import ComponentCapability, ModelCapabilities
from .inspection import inspect_dense_causal_lm
from .requirements import CapabilityRequirement, VersionRequirement, _text, _versions


def _class_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


@dataclass(frozen=True, slots=True)
class AdapterDefinition:
    adapter_id: str
    model_class: str
    config_class: str
    versions: tuple[VersionRequirement, ...]
    overrides: tuple[tuple[str, ComponentCapability], ...] = field(default_factory=tuple)
    requirements: tuple[CapabilityRequirement, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("adapter_id", "model_class", "config_class"):
            _text(name, getattr(self, name))
        if "." not in self.model_class or "." not in self.config_class:
            raise ValueError("Adapter class selectors must be fully qualified.")
        versions = tuple(self.versions)
        if not versions or any(not isinstance(item, VersionRequirement) for item in versions):
            raise ValueError("Adapters require explicit version guards.")
        if len({item.distribution for item in versions}) != len(versions):
            raise ValueError("Adapter version guards must have unique distributions.")
        object.__setattr__(self, "versions", tuple(sorted(versions, key=lambda item: item.distribution)))
        overrides = tuple(self.overrides)
        if len({name for name, _ in overrides}) != len(overrides):
            raise ValueError("Adapter capability overrides must be unique.")
        for name, capability in overrides:
            if name not in ModelCapabilities.COMPONENT_NAMES:
                raise ValueError(f"Unknown capability component: {name}")
            if (not isinstance(capability, ComponentCapability)
                    or capability.status not in {"known", "inferred"} or not capability.evidence):
                raise ValueError("Adapter overrides require known/inferred semantics and evidence.")
            if any(fact.source is None for fact in capability.facts):
                raise ValueError("Adapter facts require evidence sources.")
        object.__setattr__(self, "overrides", tuple(sorted(overrides)))
        requirements = tuple(self.requirements)
        if any(not isinstance(item, CapabilityRequirement) for item in requirements):
            raise ValueError("Adapter requirements must be CapabilityRequirement objects.")
        object.__setattr__(self, "requirements", tuple(sorted(requirements, key=lambda item: item.component)))


@dataclass(frozen=True, slots=True)
class AdapterResolution:
    capabilities: ModelCapabilities
    adapter_id: str | None
    rejections: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AdapterRegistry:
    adapters: tuple[AdapterDefinition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        adapters = tuple(self.adapters)
        if any(not isinstance(item, AdapterDefinition) for item in adapters):
            raise ValueError("Registry entries must be AdapterDefinition objects.")
        if len({item.adapter_id for item in adapters}) != len(adapters):
            raise ValueError("Adapter IDs must be unique.")
        object.__setattr__(self, "adapters", tuple(sorted(adapters, key=lambda item: item.adapter_id)))

    def resolve(self, model: nn.Module, *, versions: Mapping[str, str],
                source_provider: str = "external") -> AdapterResolution:
        report = inspect_dense_causal_lm(model, source_provider=source_provider)
        versions = dict(_versions(versions))
        model_class = _class_name(model)
        config_class = _class_name(getattr(model, "config", None))
        eligible = []
        rejections = []
        for adapter in self.adapters:
            if adapter.model_class != model_class:
                continue
            errors = []
            if adapter.config_class != config_class:
                errors.append(f"config class is {config_class}, not {adapter.config_class}")
            errors.extend(error for guard in adapter.versions
                          if (error := guard.rejection(versions)) is not None)
            for requirement in adapter.requirements:
                errors.extend(requirement.rejections(report))
            if errors:
                rejections.append(f"adapter {adapter.adapter_id}: " + "; ".join(errors))
            else:
                eligible.append(adapter)
        if len(eligible) > 1:
            raise ValueError("Ambiguous adapters: " + ", ".join(item.adapter_id for item in eligible))
        if not eligible:
            return AdapterResolution(report, None, tuple(rejections))
        adapter = eligible[0]
        overrides = {}
        for name, replacement in adapter.overrides:
            original = report.component(name)
            if original.status == "unsupported":
                raise ValueError(f"Adapter cannot override unsupported {name} semantics.")
            facts = {fact.name: fact for fact in original.facts}
            if original.status in {"known", "inferred"} and original.kind != replacement.kind:
                raise ValueError(f"Adapter contradicts inspected {name} kind.")
            for fact in replacement.facts:
                previous = facts.get(fact.name)
                if previous is not None and (type(previous.value) is not type(fact.value)
                                             or previous.value != fact.value):
                    raise ValueError(f"Adapter contradicts inspected {name}.{fact.name}.")
                facts[fact.name] = fact
            overrides[name] = replace(
                replacement, facts=tuple(sorted(facts.values(), key=lambda fact: fact.name)),
                evidence=tuple(sorted(set((*original.evidence, *replacement.evidence,
                                          f"adapter={adapter.adapter_id}; class={model_class}")))),
                notes=tuple(sorted(set((*original.notes, *replacement.notes)))),
            )
        return AdapterResolution(replace(report, **overrides), adapter.adapter_id, tuple(rejections))
