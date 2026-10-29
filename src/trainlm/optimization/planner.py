"""Pure, deterministic provider selection; no kernels are imported or applied."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Sequence

from .capabilities import ModelCapabilities
from .plan import ExecutionPlan, ModelTransformation, OptimizationPolicy, ProviderDecision
from .requirements import (
    CapabilityRequirement, PlanningContext, ShapeConstraint, VersionRequirement,
    _strings, _text,
)


@dataclass(frozen=True, slots=True)
class OperationRequest:
    component: str
    operation: str
    provider_id: str | None = None

    def __post_init__(self) -> None:
        if self.component not in ModelCapabilities.COMPONENT_NAMES:
            raise ValueError(f"Unknown capability component: {self.component}")
        _text("operation", self.operation)
        if self.provider_id is not None:
            _text("provider_id", self.provider_id)


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    provider_id: str
    operation: str
    capability: CapabilityRequirement
    backends: tuple[str, ...]
    precisions: tuple[str, ...]
    masks: tuple[str, ...]
    supports_backward: bool = False
    shapes: tuple[ShapeConstraint, ...] = field(default_factory=tuple)
    versions: tuple[VersionRequirement, ...] = field(default_factory=tuple)
    adapter_ids: tuple[str, ...] = field(default_factory=tuple)
    transformations: tuple[ModelTransformation, ...] = field(default_factory=tuple)
    priority: int = 0

    def __post_init__(self) -> None:
        _text("provider_id", self.provider_id)
        if self.provider_id in {"auto", "model_reference"}:
            raise ValueError("Provider ID is reserved for planning/fallback.")
        _text("operation", self.operation)
        if not isinstance(self.capability, CapabilityRequirement):
            raise ValueError("Provider capability must be a CapabilityRequirement.")
        for name in ("backends", "precisions", "masks", "adapter_ids"):
            object.__setattr__(self, name, _strings(name, getattr(self, name)))
            if name != "adapter_ids" and not getattr(self, name):
                raise ValueError(f"Provider {name} cannot be empty.")
        if not isinstance(self.supports_backward, bool):
            raise ValueError("supports_backward must be boolean.")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("Provider priority must be an integer.")
        for name, kind, key in (("shapes", ShapeConstraint, "name"),
                                ("versions", VersionRequirement, "distribution"),
                                ("transformations", ModelTransformation, "transform_id")):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, kind) for value in values):
                raise ValueError(f"Provider {name} contains an invalid declaration.")
            if len({getattr(value, key) for value in values}) != len(values):
                raise ValueError(f"Provider {name} must have unique identifiers.")
            # Transform application order is intentional; never reorder it.
            object.__setattr__(self, name, values if name == "transformations" else
                               tuple(sorted(values, key=lambda value: getattr(value, key))))
        for transform in self.transformations:
            if transform.provider != self.provider_id or transform.component != self.capability.component:
                raise ValueError("Transformation ownership must match its provider/component.")

    def rejections(self, capabilities: ModelCapabilities, context: PlanningContext) -> tuple[str, ...]:
        errors = list(self.capability.rejections(capabilities))
        if context.backend not in self.backends:
            errors.append(f"backend {context.backend} is unsupported")
        if context.precision not in self.precisions:
            errors.append(f"precision {context.precision} is unsupported")
        if context.mask_kind not in self.masks:
            errors.append(f"mask {context.mask_kind} is unsupported")
        if context.training and not self.supports_backward:
            errors.append("backward is unsupported")
        if self.adapter_ids and context.adapter_id not in self.adapter_ids:
            errors.append(f"requires adapter in {self.adapter_ids}; selected={context.adapter_id}")
        facts = {fact.name: fact for fact in capabilities.component(self.capability.component).facts}
        for shape in self.shapes:
            if (error := shape.rejection(dict(context.shapes))) is not None:
                errors.append(error)
            observed = facts.get(shape.name)
            requested = dict(context.shapes).get(shape.name)
            if observed is not None and requested is not None and (
                type(observed.value) is not int or observed.value != requested
            ):
                errors.append(f"shape {shape.name}={requested} contradicts inspected fact {observed.value!r}")
        for guard in self.versions:
            if (error := guard.rejection(dict(context.versions))) is not None:
                errors.append(error)
        return tuple(errors)

    def requirements(self) -> tuple[str, ...]:
        return (
            f"capability={self.capability.component}; kinds={self.capability.kinds}; "
            f"allow_inferred={self.capability.allow_inferred}",
            *(f"fact {self.capability.component}.{fact.name}={fact.value!r}" for fact in self.capability.facts),
            f"backends={self.backends}", f"precisions={self.precisions}", f"masks={self.masks}",
            f"supports_backward={self.supports_backward}", f"adapters={self.adapter_ids}",
            *(f"shape {shape.name}: min={shape.minimum}; max={shape.maximum}; multiple={shape.multiple_of}"
              for shape in self.shapes),
            *(f"version {guard.distribution}{guard.specifier}" for guard in self.versions),
        )


def plan_optimizations(
    capabilities: ModelCapabilities, *, context: PlanningContext,
    requests: Sequence[OperationRequest], providers: Sequence[ProviderDefinition] = (),
    policy: OptimizationPolicy = "auto", allow_fallbacks: bool | None = None,
    expected_capability_fingerprint: str | None = None,
) -> ExecutionPlan:
    """Select declared eligible providers without inspecting a live environment.

    Explicit provider requests use the same policy: auto may fall back when
    allowed; required and allow_fallbacks=False block rather than substitute.
    Missing/unknown facts never become evidence of optimization eligibility.
    """
    if not isinstance(capabilities, ModelCapabilities) or not isinstance(context, PlanningContext):
        raise TypeError("Planning requires ModelCapabilities and PlanningContext.")
    if policy not in {"auto", "required", "disabled"}:
        raise ValueError(f"Unsupported optimization policy: {policy}")
    if allow_fallbacks is None:
        allow_fallbacks = policy != "required"
    if not isinstance(allow_fallbacks, bool):
        raise ValueError("allow_fallbacks must be boolean.")
    if policy == "required" and allow_fallbacks:
        raise ValueError("Required optimization policy cannot allow fallbacks.")
    requests, providers = tuple(requests), tuple(providers)
    if any(not isinstance(item, OperationRequest) for item in requests):
        raise ValueError("Requests must be OperationRequest objects.")
    if any(not isinstance(item, ProviderDefinition) for item in providers):
        raise ValueError("Providers must be ProviderDefinition objects.")
    requests = tuple(sorted(requests, key=lambda item: (item.component, item.operation)))
    providers = tuple(sorted(providers, key=lambda item: (item.provider_id, item.capability.component, item.operation)))
    if len({(item.component, item.operation) for item in requests}) != len(requests):
        raise ValueError("Operation requests must be unique per component/operation.")
    if len({(item.provider_id, item.capability.component, item.operation) for item in providers}) != len(providers):
        raise ValueError("Provider declarations must be unique per ID/component/operation.")
    decisions, transformations, errors, warnings = [], {}, [], list(capabilities.warnings)
    stale = (expected_capability_fingerprint is not None
             and expected_capability_fingerprint != capabilities.fingerprint)
    if stale:
        errors.append("Capability fingerprint does not match the requested planning snapshot.")
    for request in requests:
        decision_id = f"{request.component}.{request.operation}"
        requested = request.provider_id or "auto"
        candidates = [item for item in providers if item.capability.component == request.component
                      and item.operation == request.operation]
        selected = None
        evidence = [f"capability={capabilities.fingerprint}",
                    f"backend={context.backend}; precision={context.precision}; mask={context.mask_kind}; training={context.training}"]
        if context.adapter_id is not None:
            evidence.append(f"adapter={context.adapter_id}")
        if stale:
            status, reason = "blocked", errors[0]
        elif policy == "disabled":
            status, reason = "skipped", "Optimization is disabled; keep the model reference implementation."
        else:
            checks = [(item, item.rejections(capabilities, context)) for item in candidates]
            eligible = sorted((item for item, reasons in checks if not reasons),
                              key=lambda item: (-item.priority, item.provider_id))
            requested_match = next((item for item in eligible if item.provider_id == requested), None)
            if request.provider_id is None:
                selected = eligible[0] if eligible else None
            elif requested_match is not None:
                selected = requested_match
            elif allow_fallbacks:
                selected = eligible[0] if eligible else None
            reports = [f"{item.provider_id}: " + ("; ".join(reasons) if reasons else "eligible")
                       for item, reasons in checks]
            if request.provider_id is not None and not any(item.provider_id == requested for item in candidates):
                reports.insert(0, f"{requested}: not registered for this operation")
            diagnosis = " | ".join(reports) or "No providers are registered for this operation."
            if selected is not None:
                status = "fallback" if request.provider_id is not None and selected.provider_id != requested else "selected"
                reason = f"Selected {selected.provider_id} by eligibility, descending priority, then provider ID. {diagnosis}"
                component = capabilities.component(request.component)
                evidence.extend(component.evidence)
                evidence.extend(f"fact {fact.name}: {fact.source}" for fact in component.facts if fact.source)
                evidence.extend(f"shape {name}={value}" for name, value in context.shapes)
                evidence.extend(f"version {name}={value}" for name, value in context.versions)
                for transform in selected.transformations:
                    previous = transformations.get(transform.transform_id)
                    if previous is not None and previous != transform:
                        errors.append(f"Conflicting transformation declarations: {transform.transform_id}")
                    transformations[transform.transform_id] = transform
            elif policy == "required" or not allow_fallbacks:
                status, reason = "blocked", f"No eligible requested optimization. {diagnosis}"
                errors.append(f"{decision_id}: {reason}")
            else:
                status, reason = "fallback", f"Keep the unchanged model reference implementation. {diagnosis}"
                warnings.append(f"{decision_id}: {reason}")
        selected_provider = (selected.provider_id if selected is not None else
                             "model_reference" if status == "fallback" else None)
        decisions.append(ProviderDecision(
            decision_id=decision_id, component=request.component, operation=request.operation,
            status=status, reason=reason, selected_provider=selected_provider,
            requested_provider=requested,
            requirements=tuple(requirement for item in candidates for requirement in
                               (f"provider={item.provider_id}", *item.requirements())),
            evidence=tuple(evidence),
        ))
    blocked = bool(errors)
    plan = ExecutionPlan(
        schema_version=1, plan_id="pending", status="blocked" if blocked else
        "noop" if not decisions or all(item.status == "skipped" for item in decisions) else "ready",
        policy=policy, capability_fingerprint=capabilities.fingerprint,
        backend=context.backend, precision=context.precision, decisions=tuple(decisions),
        transformations=() if blocked else tuple(transformations.values()),
        warnings=tuple(warnings), errors=tuple(dict.fromkeys(errors)),
    )
    contents = plan.to_dict()
    contents.pop("plan_id")
    digest = hashlib.sha256(json.dumps({
        "plan": contents, "context": asdict(context), "allow_fallbacks": allow_fallbacks,
        "requests": [asdict(item) for item in requests],
        "providers": [asdict(item) for item in providers],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return replace(plan, plan_id=f"plan-{digest}")
