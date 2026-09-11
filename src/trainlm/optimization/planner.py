"""Pure deterministic provider selection for optimization execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable

from .adapters import AdapterResolution
from .capabilities import ModelCapabilities
from .plan import ExecutionPlan, ModelTransformation, OptimizationPolicy, ProviderDecision


def _strings(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a tuple of non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique.")
    return values


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Declarative eligibility and transform output for one provider."""

    provider_id: str
    component: str
    operation: str
    backends: tuple[str, ...]
    precisions: tuple[str, ...]
    capability_kinds: tuple[str, ...] = ()
    supported_requirements: tuple[str, ...] = ()
    transformations: tuple[ModelTransformation, ...] = ()
    fallback: bool = False
    priority: int = 0
    runtime_requirements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("provider_id", "component", "operation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Provider {name} cannot be empty.")
        if self.component not in ModelCapabilities.COMPONENT_NAMES:
            raise ValueError(f"Unknown provider component: {self.component}")
        for name in (
            "backends",
            "precisions",
            "capability_kinds",
            "supported_requirements",
            "runtime_requirements",
        ):
            object.__setattr__(self, name, _strings(name, getattr(self, name)))
        if not self.backends or not self.precisions:
            raise ValueError("Providers require at least one backend and precision.")
        if any(
            not isinstance(transform, ModelTransformation)
            for transform in self.transformations
        ):
            raise ValueError("Provider transformations must be ModelTransformation values.")
        if any(transform.provider != self.provider_id for transform in self.transformations):
            raise ValueError("Provider transformations must name their owning provider.")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("Provider priority must be an integer.")


@dataclass(frozen=True, slots=True)
class OperationRequest:
    """One operation the planner must resolve."""

    component: str
    operation: str
    requirements: tuple[str, ...] = field(default_factory=tuple)
    requested_provider: str | None = None

    def __post_init__(self) -> None:
        if self.component not in ModelCapabilities.COMPONENT_NAMES:
            raise ValueError(f"Unknown requested component: {self.component}")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("Requested operation cannot be empty.")
        object.__setattr__(self, "requirements", _strings("requirements", self.requirements))
        if self.requested_provider is not None and (
            not isinstance(self.requested_provider, str) or not self.requested_provider.strip()
        ):
            raise ValueError("requested_provider cannot be empty.")


class OptimizationPlanner:
    """Build execution plans without importing providers or mutating models."""

    def __init__(self, providers: Iterable[ProviderSpec] = ()) -> None:
        self._providers: dict[str, ProviderSpec] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ProviderSpec) -> None:
        if not isinstance(provider, ProviderSpec):
            raise TypeError("provider must be a ProviderSpec.")
        if provider.provider_id in self._providers:
            raise ValueError(f"Provider already registered: {provider.provider_id}")
        self._providers[provider.provider_id] = provider

    def plan(
        self,
        capabilities: ModelCapabilities,
        *,
        backend: str,
        precision: str,
        policy: OptimizationPolicy,
        requests: Iterable[OperationRequest],
        adapter_resolution: AdapterResolution | None = None,
        runtime_features: tuple[str, ...] = (),
    ) -> ExecutionPlan:
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities.")
        for name, value in (("backend", backend), ("precision", precision)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(policy, str) or policy not in {
            "disabled",
            "auto",
            "required",
        }:
            raise ValueError(f"Unsupported optimization policy: {policy}")
        if adapter_resolution is not None and not isinstance(
            adapter_resolution,
            AdapterResolution,
        ):
            raise TypeError("adapter_resolution must be AdapterResolution or None.")
        requests = tuple(requests)
        if any(not isinstance(item, OperationRequest) for item in requests):
            raise TypeError("requests must contain OperationRequest values.")
        runtime_features = _strings("runtime_features", runtime_features)
        if len({(item.component, item.operation) for item in requests}) != len(requests):
            raise ValueError("Operation requests must be unique by component and operation.")
        adapter_id = (
            adapter_resolution.selected.adapter_id
            if adapter_resolution is not None and adapter_resolution.selected is not None
            else None
        )
        identity = {
            "capabilities": capabilities.fingerprint,
            "backend": backend,
            "precision": precision,
            "policy": policy,
            "adapter": adapter_id,
            "runtime_features": runtime_features,
            "requests": [
                [item.component, item.operation, item.requirements, item.requested_provider]
                for item in requests
            ],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        if policy == "disabled":
            decisions = tuple(
                ProviderDecision(
                    decision_id=f"{item.component}.{item.operation}",
                    component=item.component,
                    operation=item.operation,
                    status="skipped",
                    reason="Optimization policy is disabled.",
                    requested_provider=item.requested_provider,
                    requirements=item.requirements,
                )
                for item in requests
            )
            return ExecutionPlan(
                1, f"plan-{digest}", "noop", policy, capabilities.fingerprint,
                backend, precision, decisions=decisions,
            )

        decisions: list[ProviderDecision] = []
        transformations: list[ModelTransformation] = []
        errors: list[str] = []
        warnings: list[str] = []
        for request in requests:
            eligible, rejected = self._eligible(
                request,
                capabilities,
                backend=backend,
                precision=precision,
                runtime_features=runtime_features,
            )
            preferred = [item for item in eligible if not item.fallback]
            fallbacks = [item for item in eligible if item.fallback]
            requested = request.requested_provider
            chosen = next((item for item in eligible if item.provider_id == requested), None)
            if requested is None:
                candidates = preferred if policy == "required" else preferred or fallbacks
                chosen = (candidates or [None])[0]
                if chosen is None and policy == "required" and fallbacks:
                    rejected.extend(
                        f"{item.provider_id}: fallback-only provider does not "
                        "satisfy required policy"
                        for item in fallbacks
                    )
            if chosen is None and policy == "auto" and fallbacks:
                chosen = fallbacks[0]
            if chosen is None:
                reason = self._failure_reason(request, requested, rejected)
                status = "blocked" if policy == "required" or requested is not None else "skipped"
                decisions.append(ProviderDecision(
                    decision_id=f"{request.component}.{request.operation}",
                    component=request.component,
                    operation=request.operation,
                    status=status,
                    reason=reason,
                    requested_provider=requested,
                    requirements=request.requirements,
                    evidence=tuple(rejected),
                ))
                if status == "blocked":
                    errors.append(reason)
                else:
                    warnings.append(reason)
                continue
            is_fallback = (
                requested is None and chosen.fallback
            ) or (
                requested is not None and chosen.provider_id != requested
            )
            decisions.append(ProviderDecision(
                decision_id=f"{request.component}.{request.operation}",
                component=request.component,
                operation=request.operation,
                status="fallback" if is_fallback else "selected",
                reason=("Selected compatible fallback provider." if is_fallback else "All provider eligibility guards passed."),
                selected_provider=chosen.provider_id,
                requested_provider=(requested or "auto") if is_fallback else requested,
                requirements=request.requirements,
                evidence=(f"capability.{request.component}",)
                + ((f"adapter.{adapter_id}",) if adapter_id else ())
                + (tuple(rejected) if is_fallback else ()),
            ))
            transformations.extend(chosen.transformations)
            if is_fallback:
                warnings.append(f"{request.component}.{request.operation} uses {chosen.provider_id} fallback.")

        status = "blocked" if errors else "ready" if any(
            decision.status in {"selected", "fallback"} for decision in decisions
        ) else "noop"
        if status != "ready":
            transformations = []
        return ExecutionPlan(
            1, f"plan-{digest}", status, policy, capabilities.fingerprint,
            backend, precision, decisions=tuple(decisions),
            transformations=tuple(transformations), warnings=tuple(warnings),
            errors=tuple(errors),
        )

    def _eligible(
        self, request, capabilities, *, backend, precision, runtime_features
    ):
        eligible: list[ProviderSpec] = []
        rejected: list[str] = []
        component = capabilities.component(request.component)
        providers = sorted(
            (item for item in self._providers.values() if item.component == request.component and item.operation == request.operation),
            key=lambda item: (-item.priority, item.provider_id),
        )
        for provider in providers:
            reasons = []
            if backend not in provider.backends:
                reasons.append(f"backend {backend!r}")
            if precision not in provider.precisions:
                reasons.append(f"precision {precision!r}")
            if provider.capability_kinds and component.kind not in provider.capability_kinds:
                reasons.append(f"capability kind {component.kind!r}")
            missing = sorted(set(request.requirements) - set(provider.supported_requirements))
            if missing:
                reasons.append(f"requirements {missing!r}")
            missing_runtime = sorted(
                set(provider.runtime_requirements) - set(runtime_features)
            )
            if missing_runtime:
                reasons.append(f"runtime features {missing_runtime!r}")
            if reasons:
                rejected.append(f"{provider.provider_id}: rejected by {', '.join(reasons)}")
            else:
                eligible.append(provider)
        return eligible, rejected

    @staticmethod
    def _failure_reason(request, requested, rejected):
        target = f"requested provider {requested!r}" if requested else "any provider"
        detail = "; ".join(rejected) or "no provider is registered"
        return f"No eligible {target} for {request.component}.{request.operation}: {detail}."


__all__ = ["OperationRequest", "OptimizationPlanner", "ProviderSpec"]
