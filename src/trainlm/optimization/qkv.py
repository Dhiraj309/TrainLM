"""Validated reversible layouts for compatible separate Q/K/V projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .plan import ModelTransformation
from .state_dict import ParameterLayoutMapping, StateDictLayoutConverter
from .transforms import TransformHandler


@dataclass(frozen=True, slots=True)
class QKVProjectionSpec:
    """Canonical projection geometry used to construct reversible mappings."""

    prefix: str
    query_heads: int
    key_value_heads: int
    head_dim: int
    input_size: int
    q_weight_key: str
    k_weight_key: str
    v_weight_key: str
    packed_weight_key: str
    q_bias_key: str | None = None
    k_bias_key: str | None = None
    v_bias_key: str | None = None
    packed_bias_key: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError("prefix cannot be empty.")
        for name in ("query_heads", "key_value_heads", "head_dim", "input_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.query_heads % self.key_value_heads:
            raise ValueError("query_heads must be divisible by key_value_heads.")
        weight_keys = (
            self.q_weight_key,
            self.k_weight_key,
            self.v_weight_key,
            self.packed_weight_key,
        )
        if any(not isinstance(key, str) or not key for key in weight_keys):
            raise ValueError("QKV weight keys cannot be empty.")
        if len(set(weight_keys)) != len(weight_keys):
            raise ValueError("QKV weight keys must be unique.")
        bias_keys = (
            self.q_bias_key,
            self.k_bias_key,
            self.v_bias_key,
            self.packed_bias_key,
        )
        configured_biases = tuple(key is not None for key in bias_keys)
        if any(configured_biases) and not all(configured_biases):
            raise ValueError("QKV bias keys must be configured together.")
        if all(configured_biases) and (
            any(not isinstance(key, str) or not key for key in bias_keys)
            or len(set(bias_keys)) != len(bias_keys)
        ):
            raise ValueError("QKV bias keys must be non-empty and unique.")
        if self.dtype is not None and (
            not isinstance(self.dtype, str) or not self.dtype
        ):
            raise ValueError("dtype cannot be empty.")

    @property
    def q_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.key_value_heads * self.head_dim

    def converter(self) -> StateDictLayoutConverter:
        """Create mappings for optimized resume and canonical HF export."""

        mappings = [
            ParameterLayoutMapping(
                mapping_id=f"{self.prefix}.qkv.weight",
                canonical_keys=(
                    self.q_weight_key,
                    self.k_weight_key,
                    self.v_weight_key,
                ),
                transformed_key=self.packed_weight_key,
                canonical_shapes=(
                    (self.q_size, self.input_size),
                    (self.kv_size, self.input_size),
                    (self.kv_size, self.input_size),
                ),
                axis=0,
                dtype=self.dtype,
            )
        ]
        if self.packed_bias_key is not None:
            mappings.append(
                ParameterLayoutMapping(
                    mapping_id=f"{self.prefix}.qkv.bias",
                    canonical_keys=(
                        self.q_bias_key,
                        self.k_bias_key,
                        self.v_bias_key,
                    ),
                    transformed_key=self.packed_bias_key,
                    canonical_shapes=((self.q_size,), (self.kv_size,), (self.kv_size,)),
                    axis=0,
                    dtype=self.dtype,
                )
            )
        return StateDictLayoutConverter(tuple(mappings))


class PackedQKVProjection(nn.Module):
    """One linear projection that returns compact query, key, and value views."""

    def __init__(self, spec: QKVProjectionSpec, weight: Tensor, bias: Tensor | None) -> None:
        super().__init__()
        expected_weight = (spec.q_size + 2 * spec.kv_size, spec.input_size)
        if tuple(weight.shape) != expected_weight:
            raise ValueError(
                f"Packed QKV weight shape must be {expected_weight}, got {tuple(weight.shape)}."
            )
        expected_bias = (expected_weight[0],)
        if bias is not None and tuple(bias.shape) != expected_bias:
            raise ValueError(
                f"Packed QKV bias shape must be {expected_bias}, got {tuple(bias.shape)}."
            )
        self.spec = spec
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_separate(
        cls,
        spec: QKVProjectionSpec,
        q_projection: nn.Linear,
        k_projection: nn.Linear,
        v_projection: nn.Linear,
    ) -> "PackedQKVProjection":
        """Pack three validated linear projections before optimizer construction."""

        projections = (q_projection, k_projection, v_projection)
        if any(not isinstance(projection, nn.Linear) for projection in projections):
            raise TypeError("QKV packing requires three torch.nn.Linear projections.")
        expected_outputs = (spec.q_size, spec.kv_size, spec.kv_size)
        for name, projection, output_size in zip(
            ("query", "key", "value"), projections, expected_outputs
        ):
            if projection.in_features != spec.input_size or projection.out_features != output_size:
                raise ValueError(
                    f"{name} projection geometry does not match the QKV specification."
                )
        bias_presence = tuple(projection.bias is not None for projection in projections)
        if len(set(bias_presence)) != 1:
            raise ValueError("QKV projections must either all have bias or all be bias-free.")
        if (spec.packed_bias_key is not None) != bias_presence[0]:
            raise ValueError("QKV projection bias layout does not match the specification.")
        weights = tuple(projection.weight.detach().clone() for projection in projections)
        if len({(weight.dtype, weight.device) for weight in weights}) != 1:
            raise ValueError("QKV projection weights must share dtype and device.")
        bias = None
        if bias_presence[0]:
            biases = tuple(projection.bias.detach().clone() for projection in projections)
            if len({(value.dtype, value.device) for value in biases}) != 1:
                raise ValueError("QKV projection biases must share dtype and device.")
            bias = torch.cat(biases, dim=0)
        return cls(spec, torch.cat(weights, dim=0), bias)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        packed = F.linear(hidden_states, self.weight, self.bias)
        return packed.split((self.spec.q_size, self.spec.kv_size, self.spec.kv_size), dim=-1)


def qkv_pack_transform_handler(
    spec: QKVProjectionSpec,
    *,
    transform_id: str = "pack-qkv",
    inverse_transform_id: str = "unpack-qkv",
) -> TransformHandler:
    """Create an explicit reversible handler for a QKV projection wrapper path.

    The selected target must expose ``q_proj``, ``k_proj``, and ``v_proj`` linear
    modules and represent a call site whose output is the corresponding tuple.
    Family adapters remain responsible for proving that interface.
    """

    def capture(model: nn.Module, transformation: ModelTransformation) -> Any:
        target = _resolve_module(model, transformation.target_paths[0])
        return target

    def apply(model: nn.Module, transformation: ModelTransformation) -> None:
        if len(transformation.target_paths) != 1:
            raise ValueError("QKV packing requires exactly one explicit wrapper path.")
        target = _resolve_module(model, transformation.target_paths[0])
        packed = PackedQKVProjection.from_separate(
            spec, target.q_proj, target.k_proj, target.v_proj
        )
        _replace_module(model, transformation.target_paths[0], packed)

    def rollback(
        model: nn.Module, transformation: ModelTransformation, snapshot: nn.Module
    ) -> None:
        _replace_module(model, transformation.target_paths[0], snapshot)

    return TransformHandler(transform_id, inverse_transform_id, capture, apply, rollback)


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    target: Any = model
    for component in path.split("."):
        if not component or not hasattr(target, component):
            raise ValueError(f"Model has no module at path {path!r}.")
        target = getattr(target, component)
    if not isinstance(target, nn.Module):
        raise TypeError(f"Target path {path!r} does not resolve to a module.")
    return target


def _replace_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    components = path.split(".")
    parent: Any = model
    for component in components[:-1]:
        if not component or not hasattr(parent, component):
            raise ValueError(f"Model has no module at path {path!r}.")
        parent = getattr(parent, component)
    name = components[-1]
    if not name or not hasattr(parent, name):
        raise ValueError(f"Model has no module at path {path!r}.")
    setattr(parent, name, replacement)


__all__ = ["PackedQKVProjection", "QKVProjectionSpec", "qkv_pack_transform_handler"]
