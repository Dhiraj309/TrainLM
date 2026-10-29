"""Explicit, lossless canonical/packed tensor and named-optimizer conversion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json

import torch
from torch import nn

from .requirements import _text


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return deepcopy(value)


def _equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and left.device == right.device and torch.equal(left, right)
    return type(left) is type(right) and left == right


@dataclass(frozen=True, slots=True)
class StateTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "parameter"
    requires_grad: bool = True
    persistent: bool = True

    def __post_init__(self):
        _text("tensor name", self.name)
        shape = tuple(self.shape)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape):
            raise ValueError("Tensor shape dimensions must be positive integers.")
        object.__setattr__(self, "shape", shape)
        if self.dtype not in {str(getattr(torch, name)) for name in (
            "float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool",
        )}:
            raise ValueError("Unsupported tensor dtype in state layout.")
        if self.kind not in {"parameter", "buffer"}:
            raise ValueError("State tensor kind must be parameter or buffer.")
        if not isinstance(self.requires_grad, bool) or not isinstance(self.persistent, bool):
            raise ValueError("State tensor flags must be boolean.")
        if self.kind == "parameter" and not self.persistent:
            raise ValueError("Parameters must be persistent.")


@dataclass(frozen=True, slots=True)
class TensorPack:
    sources: tuple[str, ...]
    target: str
    dimension: int = 0

    def __post_init__(self):
        sources = tuple(self.sources)
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("Pack sources must be nonempty and unique.")
        for name in (*sources, self.target):
            _text("pack tensor name", name)
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 0:
            raise ValueError("Pack dimension must be a nonnegative integer.")
        object.__setattr__(self, "sources", sources)


def module_tensors(module: nn.Module, *, persistent_only=False):
    result = {}
    for path, child in module.named_modules(remove_duplicate=False):
        for kind, entries in (("parameter", child._parameters), ("buffer", child._buffers)):
            for name, value in entries.items():
                if value is None or (persistent_only and kind == "buffer" and name in child._non_persistent_buffers_set):
                    continue
                result[f"{path}.{name}" if path else name] = value
    return result


@dataclass(frozen=True, slots=True)
class StateLayout:
    tensors: tuple[StateTensor, ...]
    packs: tuple[TensorPack, ...] = field(default_factory=tuple)
    aliases: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("StateLayout supports schema_version=1 only.")
        tensors, packs, aliases = tuple(self.tensors), tuple(self.packs), tuple(tuple(pair) for pair in self.aliases)
        if any(not isinstance(item, StateTensor) for item in tensors) or any(not isinstance(item, TensorPack) for item in packs):
            raise ValueError("Invalid state tensor or pack declaration.")
        specs = {item.name: item for item in tensors}
        if len(specs) != len(tensors):
            raise ValueError("State tensor names must be unique.")
        alias_map = dict(aliases)
        if len(alias_map) != len(aliases):
            raise ValueError("Alias names must be unique.")
        for alias, owner in aliases:
            if alias not in specs or owner not in specs or alias == owner or owner in alias_map:
                raise ValueError("Aliases must reference a declared canonical owner without chains/cycles.")
            if replace_name(specs[alias], owner) != specs[owner]:
                raise ValueError("Alias tensor metadata does not match its owner.")
        used, targets = set(), set()
        for pack in packs:
            if pack.target in specs or pack.target in targets:
                raise ValueError("Packed target collides with an existing tensor.")
            if any(name not in specs or name in used for name in pack.sources):
                raise ValueError("Pack sources must be declared and disjoint.")
            if any(name in alias_map or name in alias_map.values() for name in pack.sources):
                raise ValueError("Packing tied parameters requires an explicit alias/view implementation.")
            sources = [specs[name] for name in pack.sources]
            first = sources[0]
            for spec in sources:
                if (spec.kind != "parameter" or not spec.dtype.startswith("torch.float") and spec.dtype != "torch.bfloat16"
                        or spec.dtype != first.dtype or spec.requires_grad != first.requires_grad
                        or pack.dimension >= len(spec.shape) or len(spec.shape) != len(first.shape)
                        or any(size != first.shape[index] for index, size in enumerate(spec.shape) if index != pack.dimension)):
                    raise ValueError("Packed parameters require compatible shapes, floating dtypes, and grad flags.")
            targets.add(pack.target)
            used.update(pack.sources)
        object.__setattr__(self, "tensors", tuple(sorted(tensors, key=lambda item: item.name)))
        object.__setattr__(self, "packs", tuple(sorted(packs, key=lambda item: item.target)))
        object.__setattr__(self, "aliases", tuple(sorted(aliases)))

    @classmethod
    def from_module(cls, module: nn.Module, packs=()):
        tensors, groups = [], {}
        for path, child in module.named_modules(remove_duplicate=False):
            for kind, entries in (("parameter", child._parameters), ("buffer", child._buffers)):
                for name, value in entries.items():
                    if value is None:
                        continue
                    if value.layout != torch.strided or value.device.type == "meta":
                        raise ValueError("Layouts require materialized dense tensors.")
                    name = f"{path}.{name}" if path else name
                    tensors.append(StateTensor(name, tuple(value.shape), str(value.dtype), kind,
                                               value.requires_grad, kind == "parameter" or name.rsplit(".", 1)[-1] not in child._non_persistent_buffers_set))
                    groups.setdefault(id(value), []).append(name)
        aliases = tuple((alias, sorted(names)[0]) for names in groups.values() for alias in sorted(names)[1:])
        return cls(tuple(tensors), tuple(packs), aliases)

    @property
    def transformed_tensors(self):
        sources = {name for pack in self.packs for name in pack.sources}
        specs = {item.name: item for item in self.tensors}
        result = [item for item in self.tensors if item.name not in sources]
        for pack in self.packs:
            first = specs[pack.sources[0]]
            shape = list(first.shape)
            shape[pack.dimension] = sum(specs[name].shape[pack.dimension] for name in pack.sources)
            result.append(StateTensor(pack.target, tuple(shape), first.dtype, "parameter", first.requires_grad))
        return tuple(sorted(result, key=lambda item: item.name))

    @property
    def fingerprint(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(StateTensor(**item) for item in value["tensors"]),
                   tuple(TensorPack(**item) for item in value.get("packs", ())),
                   tuple(tuple(pair) for pair in value.get("aliases", ())), value.get("schema_version", 1))

    def validate(self, state, *, transformed=False, include_nonpersistent=False):
        specs = self.transformed_tensors if transformed else self.tensors
        expected = {item.name: item for item in specs if item.persistent or include_nonpersistent}
        if set(state) != set(expected):
            raise ValueError("State keys do not match the declared layout.")
        for name, spec in expected.items():
            tensor = state[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.device.type == "meta" or tuple(tensor.shape) != spec.shape or str(tensor.dtype) != spec.dtype):
                raise ValueError(f"State tensor shape/dtype/layout mismatch: {name}")
        for alias, owner in self.aliases:
            if alias in state and (state[alias].device != state[owner].device or not _equal(state[alias], state[owner])):
                raise ValueError(f"Aliased state values disagree: {alias} and {owner}")

    def convert(self, state, *, to_canonical=False, include_nonpersistent=False):
        self.validate(state, transformed=to_canonical, include_nonpersistent=include_nonpersistent)
        result = {name: _clone(value) for name, value in state.items()}
        specs = {item.name: item for item in self.tensors}
        for pack in self.packs:
            if to_canonical:
                packed = result.pop(pack.target)
                sizes = [specs[name].shape[pack.dimension] for name in pack.sources]
                result.update(zip(pack.sources, (part.clone() for part in packed.split(sizes, dim=pack.dimension))))
            else:
                values = [result.pop(name) for name in pack.sources]
                if len({value.device for value in values}) != 1:
                    raise ValueError("Packed source devices must match.")
                result[pack.target] = torch.cat(values, dim=pack.dimension)
        for alias, owner in self.aliases:
            if alias in result:
                result[alias] = result[owner]
        self.validate(result, transformed=not to_canonical, include_nonpersistent=include_nonpersistent)
        return result

    def prefixed(self, prefix):
        name = lambda value: f"{prefix}.{value}" if prefix else value
        return StateLayout(tuple(replace_name(item, name(item.name)) for item in self.tensors),
            tuple(TensorPack(tuple(name(value) for value in pack.sources), name(pack.target), pack.dimension) for pack in self.packs),
            tuple((name(alias), name(owner)) for alias, owner in self.aliases))

    def check_module(self, module, *, transformed=False):
        expected = StateLayout(self.transformed_tensors if transformed else self.tensors, aliases=self.aliases)
        if StateLayout.from_module(module) != expected:
            raise ValueError("Module tensor metadata/aliases do not match the declared layout.")


def replace_name(spec, name):
    return StateTensor(name, spec.shape, spec.dtype, spec.kind, spec.requires_grad, spec.persistent)


_TENSOR_SLOTS = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}
_SCALAR_SLOTS = {"step"}


def _validate_optimizer(named, specs, aliases):
    if any(group.get("capturable") or group.get("differentiable") or group.get("fused")
           for group in named["param_groups"]):
        raise ValueError("Canonical optimizer conversion requires noncapturable, nonfused standard state.")
    parameters = {item.name: item for item in specs if item.kind == "parameter" and item.name not in dict(aliases)}
    owners = [name for group in named["param_groups"] for name in group["params"]]
    if len(set(owners)) != len(owners) or any(name not in parameters for name in owners):
        raise ValueError("Optimizer parameter ownership is invalid for this layout.")
    if any(name not in owners for name in named["state"]):
        raise ValueError("Optimizer state has unowned parameters.")
    for name, slots in named["state"].items():
        for key, value in slots.items():
            if key in _TENSOR_SLOTS:
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != parameters[name].shape
                        or not value.is_floating_point() or value.layout != torch.strided or value.device.type == "meta"):
                    raise ValueError(f"Optimizer slot shape/type mismatch: {name}.{key}")
            elif key in _SCALAR_SLOTS:
                if isinstance(value, torch.Tensor):
                    if (value.ndim != 0 or value.device.type == "meta" or value.dtype == torch.bool
                            or value.is_complex() or value.layout != torch.strided
                            or not torch.isfinite(value).item() or value.item() < 0):
                        raise ValueError("Optimizer step must be a finite nonnegative scalar.")
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                    raise ValueError("Optimizer step must be a finite nonnegative scalar.")
            else:
                raise ValueError(f"Unsupported optimizer slot: {key}")


def convert_optimizer_state(layout: StateLayout, named, *, to_canonical=False):
    """Convert standard AdamW/SGD named slots without casting their precision."""
    _validate_optimizer(named, layout.transformed_tensors if to_canonical else layout.tensors, layout.aliases)
    result = _clone(named)
    specs = {item.name: item for item in layout.tensors}
    for pack in layout.packs:
        sources = (pack.target,) if to_canonical else pack.sources
        groups = [index for index, group in enumerate(result["param_groups"]) if any(name in group["params"] for name in sources)]
        if not groups:
            continue
        group = result["param_groups"][groups[0]]
        if len(groups) != 1 or not set(sources) <= set(group["params"]):
            raise ValueError("Packed optimizer parameters must share one complete parameter group.")
        states = [result["state"].get(name) for name in sources]
        if any(item is not None for item in states) and not all(item is not None for item in states):
            raise ValueError("Packed optimizer parameters have inconsistent initialized state.")
        replacements = pack.sources if to_canonical else (pack.target,)
        if all(item is not None for item in states):
            if any(set(item) != set(states[0]) for item in states):
                raise ValueError("Packed optimizer slot sets differ.")
            converted = {name: {} for name in replacements}
            for slot in states[0]:
                values = [item[slot] for item in states]
                if slot in _TENSOR_SLOTS:
                    if to_canonical:
                        sizes = [specs[name].shape[pack.dimension] for name in pack.sources]
                        parts = values[0].split(sizes, dim=pack.dimension)
                    else:
                        if len({(value.dtype, value.device) for value in values}) != 1:
                            raise ValueError("Packed optimizer moment precision/devices must match.")
                        parts = (torch.cat(values, dim=pack.dimension),)
                    for name, value in zip(replacements, parts):
                        converted[name][slot] = _clone(value)
                else:
                    if any(not _equal(values[0], value) for value in values[1:]):
                        raise ValueError("Packed optimizer scalar steps must match.")
                    for name in replacements:
                        converted[name][slot] = _clone(values[0])
            for name in sources:
                result["state"].pop(name)
            result["state"].update(converted)
        updated, inserted = [], False
        for name in group["params"]:
            if name in sources:
                if not inserted:
                    updated.extend(replacements)
                    inserted = True
            else:
                updated.append(name)
        group["params"] = updated
    _validate_optimizer(result, layout.tensors if to_canonical else layout.transformed_tensors, layout.aliases)
    return result


def named_optimizer_state(model, optimizer):
    layout = StateLayout.from_module(model)
    tensors = module_tensors(model)
    aliases = dict(layout.aliases)
    names = {id(value): name for name, value in sorted(tensors.items(), reverse=True) if name not in aliases}
    if any(id(parameter) not in names for group in optimizer.param_groups for parameter in group["params"]):
        raise ValueError("Optimizer references parameters outside this model.")
    named = {"state": {names[id(parameter)]: _clone(value) for parameter, value in optimizer.state.items()},
             "param_groups": [{**_clone({key: value for key, value in group.items() if key != "params"}),
                               "params": [names[id(parameter)] for parameter in group["params"]]} for group in optimizer.param_groups]}
    _validate_optimizer(named, layout.tensors, layout.aliases)
    return named


def prepare_optimizer_state(model, named):
    layout = StateLayout.from_module(model)
    _validate_optimizer(named, layout.tensors, layout.aliases)
    tensors = module_tensors(model)
    state = defaultdict(dict, {tensors[name]: _clone(value) for name, value in named["state"].items()})
    groups = [{**_clone({key: value for key, value in group.items() if key != "params"}),
               "params": [tensors[name] for name in group["params"]]} for group in named["param_groups"]]
    for parameter, slots in state.items():
        for key in _TENSOR_SLOTS & slots.keys():
            slots[key] = slots[key].to(device=parameter.device)
    return state, groups
