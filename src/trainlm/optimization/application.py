"""Staged module replacement with canonical tensor ownership and explicit undo."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import random

import torch
from torch import nn

from .adapters import AdapterRegistry
from .plan import ExecutionPlan, ModelTransformation
from .planner import OperationRequest, ProviderDefinition, plan_optimizations
from .requirements import PlanningContext, _text


class TransformationError(RuntimeError):
    """A plan could not be applied or reversed safely."""


@dataclass(frozen=True, slots=True)
class TransformHandler:
    transform_id: str
    inverse_transform_id: str
    provider: str
    component: str
    source_types: tuple[type[nn.Module], ...]
    forward: Callable[[nn.Module], nn.Module]
    inverse: Callable[[nn.Module], nn.Module]
    validate: Callable[[nn.Module, nn.Module], None] | None = None

    def __post_init__(self) -> None:
        for name in ("transform_id", "inverse_transform_id", "provider", "component"):
            _text(name, getattr(self, name))
        if self.transform_id == self.inverse_transform_id:
            raise ValueError("Forward and inverse transform IDs must differ.")
        types = tuple(self.source_types)
        if not types or any(not isinstance(kind, type) or not issubclass(kind, nn.Module) for kind in types):
            raise ValueError("Handlers require explicit source module classes.")
        object.__setattr__(self, "source_types", types)
        if not callable(self.forward) or not callable(self.inverse):
            raise ValueError("Forward and inverse handlers must be callable.")
        if self.validate is not None and not callable(self.validate):
            raise ValueError("Handler validation must be callable.")


@dataclass(frozen=True, slots=True)
class TransformRegistry:
    handlers: tuple[TransformHandler, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        handlers = tuple(self.handlers)
        if any(not isinstance(handler, TransformHandler) for handler in handlers):
            raise ValueError("Registry entries must be TransformHandler objects.")
        if len({handler.transform_id for handler in handlers}) != len(handlers):
            raise ValueError("Transform handler IDs must be unique.")
        object.__setattr__(self, "handlers", tuple(sorted(handlers, key=lambda handler: handler.transform_id)))

    def resolve(self, transform: ModelTransformation) -> TransformHandler:
        handler = next((item for item in self.handlers if item.transform_id == transform.transform_id), None)
        if handler is None:
            raise TransformationError(f"Missing transform handler: {transform.transform_id}")
        if (handler.inverse_transform_id != transform.inverse_transform_id
                or handler.provider != transform.provider or handler.component != transform.component):
            raise TransformationError(f"Handler ownership/inverse mismatch: {transform.transform_id}")
        return handler


@contextmanager
def _preserve_rng():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python = random.getstate()
    try:
        yield
    finally:
        random.setstate(python)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _bindings(module: nn.Module):
    """Include None registrations and nonpersistent buffers, including aliases."""
    result = {}
    for path, child in module.named_modules(remove_duplicate=False):
        for kind, entries in (("parameter", child._parameters), ("buffer", child._buffers)):
            for name, value in entries.items():
                key = f"{path}.{name}" if path else name
                persistent = kind == "parameter" or name not in child._non_persistent_buffers_set
                result[(kind, key)] = (value, persistent)
    return result


def _aliases(bindings):
    groups = {}
    for name, (tensor, _) in bindings.items():
        if tensor is not None:
            groups.setdefault(id(tensor), []).append(name)
    return sorted(tuple(sorted(names)) for names in groups.values())


def _check_state(source: nn.Module, candidate: nn.Module) -> None:
    if not isinstance(candidate, nn.Module):
        raise TransformationError("A transform must return a torch.nn.Module.")
    before, after = _bindings(source), _bindings(candidate)
    if before.keys() != after.keys() or _aliases(before) != _aliases(after):
        raise TransformationError("Canonical tensor names or aliases changed; layout conversion is required.")
    for name, (original, persistent) in before.items():
        tensor, new_persistent = after[name]
        if persistent != new_persistent or (original is None) != (tensor is None):
            raise TransformationError(f"Tensor registration changed: {name}")
        if original is None:
            continue
        if (type(original) is not type(tensor) or original.shape != tensor.shape
                or original.dtype != tensor.dtype or original.device != tensor.device
                or original.layout != tensor.layout or original.requires_grad != tensor.requires_grad):
            raise TransformationError(f"Tensor metadata changed: {name}")
        if original.layout != torch.strided or original.device.type == "meta":
            raise TransformationError("This application path requires materialized dense tensors.")
        if not torch.equal(original.detach(), tensor.detach()):
            raise TransformationError(f"Canonical tensor values changed: {name}")


_HOOK_FIELDS = (
    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
    "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
)


def _check_hooks(source: nn.Module, candidate: nn.Module) -> None:
    originals = dict(source.named_modules(remove_duplicate=False))
    candidates = dict(candidate.named_modules(remove_duplicate=False))
    for path, child in candidates.items():
        original = originals.get(path)
        for field_name in _HOOK_FIELDS:
            before = getattr(original, field_name, {}) if original is not None else {}
            after = getattr(child, field_name, {})
            if before.keys() != after.keys() or any(after[key] is not hook for key, hook in before.items()):
                raise TransformationError(f"Transform added or changed hooks at {path or '<target>'}.")
    for path, original in originals.items():
        if path not in candidates and any(
            getattr(original, name, {}) for name in _HOOK_FIELDS
        ):
            raise TransformationError(f"Transform removed hooked module at {path}.")


def _check_modes(source: nn.Module, candidate: nn.Module) -> None:
    modes = {path: child.training for path, child in source.named_modules(remove_duplicate=False)}
    for path, child in candidate.named_modules(remove_duplicate=False):
        if child.training != modes.get(path, source.training):
            raise TransformationError(f"Transform changed training mode at {path or '<target>'}.")


def _check_canonical_classes(source: nn.Module, candidate: nn.Module) -> None:
    before = {path: type(child) for path, child in source.named_modules(remove_duplicate=False)}
    after = {path: type(child) for path, child in candidate.named_modules(remove_duplicate=False)}
    if before != after:
        raise TransformationError("Inverse did not restore canonical module classes/topology.")


def _module_alias_groups(module: nn.Module):
    groups = {}
    for path, child in module.named_modules(remove_duplicate=False):
        groups.setdefault(id(child), []).append(path)
    return tuple(tuple(paths) for paths in groups.values() if len(paths) > 1)


def _check_module_aliases(groups, candidate: nn.Module) -> None:
    modules = dict(candidate.named_modules(remove_duplicate=False))
    for paths in groups:
        if any(path not in modules for path in paths) or any(
            modules[path] is not modules[paths[0]] for path in paths
        ):
            raise TransformationError(f"Shared module aliases changed: {paths}")


def _reuse_tensors(source: nn.Module, candidate: nn.Module) -> None:
    for (kind, path), (tensor, _) in _bindings(source).items():
        parent_path, _, name = path.rpartition(".")
        parent = candidate.get_submodule(parent_path) if parent_path else candidate
        entries = parent._parameters if kind == "parameter" else parent._buffers
        entries[name] = tensor
    # Keep RemovableHandle references valid: registrations alone are not enough
    # when the user's handle still points to the original hook dictionary.
    originals = dict(source.named_modules(remove_duplicate=False))
    for path, child in candidate.named_modules(remove_duplicate=False):
        original = originals.get(path)
        if original is not None:
            for name in (*_HOOK_FIELDS, "_is_full_backward_hook"):
                if name in original.__dict__:
                    child.__dict__[name] = original.__dict__[name]


@dataclass(frozen=True, slots=True)
class _Replacement:
    path: str
    original: nn.Module
    replacement: nn.Module
    handler: TransformHandler


class AppliedExecutionPlan:
    """Undo handle; canonical Parameter objects survive application and training."""

    def __init__(self, model, plan, replacements):
        self.model = model
        self.plan = plan
        self._replacements = tuple(replacements)
        self._bindings = _bindings(model)
        self._reverted = False

    @property
    def applied_transform_ids(self) -> tuple[str, ...]:
        return tuple(transform.transform_id for transform in self.plan.transformations)

    @property
    def is_reverted(self) -> bool:
        return self._reverted

    def revert(self) -> nn.Module:
        if self._reverted:
            return self.model
        try:
            for record in self._replacements:
                if self.model.get_submodule(record.path) is not record.replacement:
                    raise TransformationError(f"Replacement is no longer installed: {record.path}")
            current = _bindings(self.model)
            if current.keys() != self._bindings.keys() or any(
                current[key][0] is not value[0] or current[key][1] != value[1]
                for key, value in self._bindings.items()
            ):
                raise TransformationError("Tensor ownership changed since application; refusing stale undo.")
            with _preserve_rng():
                for record in self._replacements:
                    restored = record.handler.inverse(deepcopy(record.replacement))
                    _check_state(record.replacement, restored)
                    _check_hooks(record.replacement, restored)
                    _check_canonical_classes(record.original, restored)
            # Original modules already share the current trained tensors. Carry
            # current train/eval flags back as well, without replaying __setattr__.
            for record in reversed(self._replacements):
                modes = dict(record.replacement.named_modules(remove_duplicate=False))
                for path, child in record.original.named_modules(remove_duplicate=False):
                    child.__dict__["training"] = modes.get(path, record.replacement).training
                _install(self.model, record.path, record.original)
        except Exception as exc:
            if isinstance(exc, TransformationError):
                raise
            raise TransformationError(f"Inverse validation failed; live replacements remain installed: {exc}") from exc
        self._reverted = True
        return self.model


def _install(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    # Preflight requires ordinary PyTorch registration dictionaries. Updating
    # them avoids user-defined attribute setters during commit and rollback.
    parent._modules[name] = replacement


def _preflight(model, plan, registry):
    modules = dict(model.named_modules(remove_duplicate=False))
    targets = []
    for transform in plan.transformations:
        if transform.parameter_layout_change is not False:
            raise TransformationError("Parameter-layout changes require M8-F5 conversion support.")
        if not any(decision.component == transform.component and decision.selected_provider == transform.provider
                   and decision.status in {"selected", "fallback"} for decision in plan.decisions):
            raise TransformationError(f"Transformation has no selected provider: {transform.transform_id}")
        handler = registry.resolve(transform)
        matched = set()
        for pattern in transform.target_paths:
            parts = pattern.split(".")
            if any(not part or (part != "*" and any(char in part for char in "*?[]")) for part in parts):
                raise TransformationError(f"Invalid target selector: {pattern}")
            found = {path for path in modules if path and len(path.split(".")) == len(parts)
                     and all(left == "*" or left == right for left, right in zip(parts, path.split(".")))}
            if not found:
                raise TransformationError(f"Target selector matched no modules: {pattern}")
            matched.update(found)
        for path in sorted(matched):
            target = modules[path]
            if type(target) not in handler.source_types:
                raise TransformationError(f"Unsupported target class at {path}: {type(target).__qualname__}")
            aliases = {name for name, module in modules.items() if module is target}
            if not aliases <= matched:
                raise TransformationError(f"All aliases of target {path} must be selected together: {sorted(aliases)}")
            parent_path = path.rpartition(".")[0]
            parent = modules[parent_path]
            if type(parent._modules) not in {dict, OrderedDict}:
                raise TransformationError("Custom module registration dictionaries are unsupported.")
            targets.append((path, target, handler))
    paths = [path for path, _, _ in targets]
    if len(set(paths)) != len(paths) or any(
        first != second and second.startswith(first + ".") for first in paths for second in paths
    ):
        raise TransformationError("Transform targets overlap; compose them in one validated handler.")
    return targets


def apply_execution_plan(
    model: nn.Module, plan: ExecutionPlan, *, context: PlanningContext,
    requests: Sequence[OperationRequest], providers: Sequence[ProviderDefinition] = (),
    registry: TransformRegistry = TransformRegistry(), adapters: AdapterRegistry = AdapterRegistry(),
    source_provider: str = "external", allow_fallbacks: bool | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> AppliedExecutionPlan:
    """Validate/replay planning inputs, stage every target, then commit together.

    Call before constructing an optimizer or compiling/placing a model. Handlers
    may only mutate their isolated argument; global patches and external side
    effects are outside the provider contract. Layout-changing transforms are
    intentionally refused until canonical conversion is available.
    """
    if not isinstance(model, nn.Module) or not isinstance(plan, ExecutionPlan):
        raise TypeError("Application requires a torch module and ExecutionPlan.")
    if optimizer is not None:
        raise TransformationError("Apply transforms before optimizer construction.")
    if not plan.is_executable:
        raise TransformationError("Cannot apply a blocked execution plan.")
    resolution = adapters.resolve(model, versions=dict(context.versions), source_provider=source_provider)
    if resolution.adapter_id != context.adapter_id:
        raise TransformationError("Resolved adapter does not match the planning context.")
    current = plan_optimizations(resolution.capabilities, context=context, requests=requests,
                                 providers=providers, policy=plan.policy, allow_fallbacks=allow_fallbacks)
    if current != plan:
        raise TransformationError("Plan is stale or changed: capabilities, context, requests, or providers differ.")
    targets = _preflight(model, plan, registry)
    staged, cache, committed = [], {}, []
    original_bindings = _bindings(model)
    module_aliases = _module_alias_groups(model)
    try:
        with _preserve_rng():
            for path, original, handler in targets:
                key = (id(original), handler.transform_id)
                if key not in cache:
                    replacement = handler.forward(deepcopy(original))
                    _check_state(original, replacement)
                    _check_hooks(original, replacement)
                    _check_modes(original, replacement)
                    _check_module_aliases(_module_alias_groups(original), replacement)
                    if handler.validate is not None:
                        handler.validate(deepcopy(original), replacement)
                        _check_state(original, replacement)
                        _check_hooks(original, replacement)
                        _check_modes(original, replacement)
                        _check_module_aliases(_module_alias_groups(original), replacement)
                    restored = handler.inverse(deepcopy(replacement))
                    _check_state(original, restored)
                    _check_hooks(original, restored)
                    _check_modes(original, restored)
                    _check_canonical_classes(original, restored)
                    _reuse_tensors(original, replacement)
                    cache[key] = replacement
                staged.append(_Replacement(path, original, cache[key], handler))
        if any(model.get_submodule(record.path) is not record.original for record in staged):
            raise TransformationError("Module topology changed while staging transforms.")
        for record in staged:
            committed.append(record)
            _install(model, record.path, record.replacement)
        after = _bindings(model)
        _check_module_aliases(module_aliases, model)
        if after.keys() != original_bindings.keys() or any(
            after[key][0] is not value[0] or after[key][1] != value[1] for key, value in original_bindings.items()
        ):
            raise TransformationError("Application changed canonical tensor ownership.")
    except BaseException as exc:
        for record in reversed(committed):
            _install(model, record.path, record.original)
        if isinstance(exc, (TransformationError, KeyboardInterrupt, SystemExit)):
            raise
        raise TransformationError(f"Transform staging failed; original model restored: {exc}") from exc
    return AppliedExecutionPlan(model, plan, staged)
