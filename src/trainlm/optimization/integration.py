"""Public trainer orchestration for explicit, opt-in optimization profiles."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
from importlib import import_module, metadata
import json
from typing import Mapping

from .adapters import AdapterRegistry
from .application import TransformRegistry, apply_execution_plan, _preserve_rng
from .capabilities import ComponentCapability, ModelCapabilities
from .planner import OperationRequest, ProviderDefinition, plan_optimizations
from .requirements import PlanningContext


class OptimizationError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class OptimizationOptions:
    requests: tuple[OperationRequest, ...] = (OperationRequest("attention", "forward_backward"),)
    providers: tuple[ProviderDefinition, ...] = field(default_factory=tuple)
    handlers: TransformRegistry = TransformRegistry()
    adapters: AdapterRegistry = AdapterRegistry()
    shapes: tuple[tuple[str, int], ...] | Mapping[str, int] = field(default_factory=tuple)
    mask_kind: str = "unknown"

    def __post_init__(self):
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(self, "providers", tuple(self.providers))
        if any(not isinstance(item, OperationRequest) for item in self.requests):
            raise ValueError("Optimization requests must be OperationRequest entries.")
        if any(not isinstance(item, ProviderDefinition) for item in self.providers):
            raise ValueError("Optimization providers must be ProviderDefinition entries.")
        if not isinstance(self.handlers, TransformRegistry) or not isinstance(self.adapters, AdapterRegistry):
            raise ValueError("Optimization options require handler and adapter registries.")
        context = PlanningContext("profile", "profile", self.mask_kind, shapes=self.shapes)
        object.__setattr__(self, "shapes", context.shapes)
        for provider in self.providers:
            if not provider.transformations:
                raise ValueError("Public trainer providers require executable model transformations.")
            for transform in provider.transformations:
                self.handlers.resolve(transform)


def load_optimization_options(value=None, *, policy="auto"):
    if value is None or (policy == "disabled" and isinstance(value, str)):
        return OptimizationOptions()
    if isinstance(value, OptimizationOptions):
        return value
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError("optimization must be OptimizationOptions or an importable module:factory name.")
    module, name = value.split(":")
    if not module or not all(part.isidentifier() for part in module.split(".")) or not name.isidentifier():
        raise ValueError("Invalid optimization module:factory name.")
    factory = getattr(import_module(module), name)
    if not callable(factory):
        raise ValueError("Optimization factory is not callable.")
    options = factory()
    if not isinstance(options, OptimizationOptions):
        raise ValueError("Optimization factory must return OptimizationOptions.")
    return options


def _versions(options):
    names = {"torch", "transformers", "huggingface-hub"}
    names.update(guard.distribution for provider in options.providers for guard in provider.versions)
    names.update(guard.distribution for adapter in options.adapters.adapters for guard in adapter.versions)
    result = {}
    for name in sorted(names):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass  # Missing metadata remains a failed guard, never a guessed version.
    return result


def deferred_optimization_report(policy, precision, factory=None):
    return {"status": "deferred", "policy": policy, "backend": "xla", "precision": precision,
            "profile": factory, "capabilities": None, "execution_plan": None,
            "applied_transform_ids": [], "explanation": "Model inspection and planning run in host preflight/worker setup.",
            "graph": {"compile_requested": False, "compile_verified": False},
            "certification": {"hardware_validated": False, "support_level": "compatible"}}


def reject_unavailable_strict_options(options, *, policy, allow_fallbacks, precision):
    # This route needs neither model weights nor an accelerator runtime.
    if options.providers or policy == "disabled" or (policy == "auto" and allow_fallbacks is not False):
        return
    unknown = ComponentCapability.unknown("Model not loaded: no optimization providers are registered.")
    capabilities = ModelCapabilities(1, "unresolved", "unresolved", "unresolved", "external", (),
                                     **{name: unknown for name in ModelCapabilities.COMPONENT_NAMES})
    plan = plan_optimizations(capabilities, context=PlanningContext("xla", precision, options.mask_kind),
                             requests=options.requests, policy=policy, allow_fallbacks=allow_fallbacks)
    if not plan.is_executable:
        raise OptimizationError(plan.explain(), {"execution_plan": plan.to_dict(), "status": "blocked"})


class OptimizationSession:
    def __init__(self, model, options, *, backend, precision, batch_size, sequence_length,
                 accumulation_steps, policy="auto", allow_fallbacks=None, source_provider="external",
                 eval_batch_size=None, world_size=1):
        self.options, self.source_provider = options, source_provider
        self.allow_fallbacks = allow_fallbacks
        versions = _versions(options)
        self.resolution = options.adapters.resolve(model, versions=versions, source_provider=source_provider)
        shapes = {"batch_size": batch_size, "sequence_length": sequence_length,
                  "accumulation_steps": accumulation_steps, "world_size": world_size,
                  "global_batch_size": batch_size * accumulation_steps * world_size}
        if eval_batch_size is not None:
            shapes["eval_batch_size"] = eval_batch_size
        for name, value in options.shapes:
            if name in shapes and shapes[name] != value:
                raise ValueError(f"Profile shape {name} contradicts training arguments.")
            shapes[name] = value
        self.context = PlanningContext(backend, precision, options.mask_kind, shapes=shapes,
                                       versions=versions, adapter_id=self.resolution.adapter_id)
        self.providers = options.providers
        self.excluded = []
        if eval_batch_size is not None:
            evaluation = replace(self.context, training=False, shapes={**shapes, "batch_size": eval_batch_size,
                                                                      "global_batch_size": eval_batch_size * world_size})
            available = []
            for provider in self.providers:
                errors = provider.rejections(self.resolution.capabilities, evaluation)
                if errors:
                    self.excluded.append({"provider": provider.provider_id, "component": provider.capability.component,
                                          "operation": provider.operation, "reason": "Evaluation: " + "; ".join(errors)})
                else:
                    available.append(provider)
            self.providers = tuple(available)
        self.plan = plan_optimizations(self.resolution.capabilities, context=self.context,
            requests=options.requests, providers=self.providers, policy=policy, allow_fallbacks=allow_fallbacks)
        self.applied = None
        self.compile_requested = False
        self.runtime_adapters = []
        if not self.plan.is_executable:
            raise OptimizationError(self.report()["explanation"], self.report())

    def apply(self, model, optimizer=None):
        if self.plan.transformations:
            self.applied = apply_execution_plan(model, self.plan, context=self.context,
                requests=self.options.requests, providers=self.providers, registry=self.options.handlers,
                adapters=self.options.adapters, source_provider=self.source_provider,
                allow_fallbacks=self.allow_fallbacks, optimizer=optimizer)
        return self.applied

    def validate_before_placement(self, model):
        if self.plan.transformations:
            if any(item.parameter_layout_change for item in self.plan.transformations) and any(
                parameter.grad is not None for parameter in model.parameters()
            ):
                raise OptimizationError("Layout-changing transforms require cleared gradients before runtime setup.", self.report())
            # Validate handlers/targets/layouts on an isolated host graph, so
            # strict failures cannot allocate a runtime or edit caller modules.
            with _preserve_rng():
                isolated = deepcopy(model)
                apply_execution_plan(isolated, self.plan, context=self.context,
                    requests=self.options.requests, providers=self.providers, registry=self.options.handlers,
                    adapters=self.options.adapters, source_provider=self.source_provider,
                    allow_fallbacks=self.allow_fallbacks)

    def report(self):
        active = self.applied is not None and not self.applied.is_reverted
        return {"status": "applied" if active else self.plan.status, "policy": self.plan.policy,
                "backend": self.context.backend, "precision": self.context.precision,
                "capabilities": self.resolution.capabilities.to_dict(),
                "adapter": {"selected": self.resolution.adapter_id, "rejections": list(self.resolution.rejections)},
                "execution_plan": self.plan.to_dict(), "excluded_providers": self.excluded,
                "runtime_adapters": list(self.runtime_adapters),
                "explanation": self.plan.explain() + "".join(f"\nUnavailable {item['provider']}: {item['reason']}" for item in self.excluded),
                "applied_transform_ids": list(self.applied.applied_transform_ids) if active else [],
                "graph": {"compile_requested": self.compile_requested, "compile_verified": False},
                "certification": {"hardware_validated": False, "support_level": "compatible"}}


def write_optimization_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
