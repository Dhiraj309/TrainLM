"""Reversible state-dict layouts for compatible gated MLP projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .plan import ModelTransformation
from .state_dict import ParameterLayoutMapping, StateDictLayoutConverter
from .transforms import TransformHandler

MLPActivation = Literal["swiglu", "geglu", "gelu"]


@dataclass(frozen=True, slots=True)
class GatedMLPProjectionSpec:
    """Explicit gate/up projection geometry without architecture-name guesses."""

    prefix: str
    activation: MLPActivation
    input_size: int
    intermediate_size: int
    gate_weight_key: str | None = None
    up_weight_key: str | None = None
    packed_weight_key: str | None = None
    gate_bias_key: str | None = None
    up_bias_key: str | None = None
    packed_bias_key: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError("prefix cannot be empty.")
        if self.activation not in {"swiglu", "geglu", "gelu"}:
            raise ValueError(f"Unsupported MLP activation: {self.activation}")
        for name in ("input_size", "intermediate_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        weight_keys = (
            self.gate_weight_key,
            self.up_weight_key,
            self.packed_weight_key,
        )
        bias_keys = (self.gate_bias_key, self.up_bias_key, self.packed_bias_key)
        if self.activation == "gelu":
            if any(key is not None for key in weight_keys + bias_keys):
                raise ValueError("GELU paths must remain unpacked.")
        else:
            self._validate_key_group("weight", weight_keys, required=True)
            self._validate_key_group("bias", bias_keys, required=False)
        if self.dtype is not None and (
            not isinstance(self.dtype, str) or not self.dtype
        ):
            raise ValueError("dtype cannot be empty.")

    @staticmethod
    def _validate_key_group(
        label: str,
        keys: tuple[str | None, ...],
        *,
        required: bool,
    ) -> None:
        configured = tuple(key is not None for key in keys)
        if required and not all(configured):
            raise ValueError(f"Gated MLP {label} keys must be configured together.")
        if any(configured) and not all(configured):
            raise ValueError(f"Gated MLP {label} keys must be configured together.")
        if all(configured) and (
            any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError(f"Gated MLP {label} keys must be non-empty and unique.")

    def converter(self) -> StateDictLayoutConverter | None:
        """Return no transform for GELU, preserving its single projection path."""

        if self.activation == "gelu":
            return None
        mappings = [
            ParameterLayoutMapping(
                mapping_id=f"{self.prefix}.gate_up.weight",
                canonical_keys=(self.gate_weight_key, self.up_weight_key),
                transformed_key=self.packed_weight_key,
                canonical_shapes=(
                    (self.intermediate_size, self.input_size),
                    (self.intermediate_size, self.input_size),
                ),
                axis=0,
                dtype=self.dtype,
            )
        ]
        if self.packed_bias_key is not None:
            mappings.append(
                ParameterLayoutMapping(
                    mapping_id=f"{self.prefix}.gate_up.bias",
                    canonical_keys=(self.gate_bias_key, self.up_bias_key),
                    transformed_key=self.packed_bias_key,
                    canonical_shapes=(
                        (self.intermediate_size,),
                        (self.intermediate_size,),
                    ),
                    axis=0,
                    dtype=self.dtype,
                )
            )
        return StateDictLayoutConverter(tuple(mappings))


class PackedGatedMLPProjection(nn.Module):
    """One packed gate/up projection with an explicit activation callable."""

    def __init__(
        self,
        spec: GatedMLPProjectionSpec,
        weight: Tensor,
        bias: Tensor | None,
        activation_fn: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        if spec.activation == "gelu":
            raise ValueError("GELU paths do not use a gated packed projection.")
        if not callable(activation_fn):
            raise TypeError("activation_fn must be callable.")
        expected_weight = (2 * spec.intermediate_size, spec.input_size)
        if tuple(weight.shape) != expected_weight:
            raise ValueError(
                f"Packed gated MLP weight shape must be {expected_weight}, "
                f"got {tuple(weight.shape)}."
            )
        if bias is not None and tuple(bias.shape) != (expected_weight[0],):
            raise ValueError("Packed gated MLP bias shape does not match its weight.")
        self.spec = spec
        self.activation_fn = activation_fn
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_separate(
        cls,
        spec: GatedMLPProjectionSpec,
        gate_projection: nn.Linear,
        up_projection: nn.Linear,
        activation_fn: Callable[[Tensor], Tensor],
    ) -> "PackedGatedMLPProjection":
        projections = (gate_projection, up_projection)
        if any(not isinstance(projection, nn.Linear) for projection in projections):
            raise TypeError("Gated MLP packing requires two torch.nn.Linear projections.")
        for name, projection in zip(("gate", "up"), projections):
            if (
                projection.in_features != spec.input_size
                or projection.out_features != spec.intermediate_size
            ):
                raise ValueError(
                    f"{name} projection geometry does not match the gated MLP specification."
                )
        bias_presence = tuple(projection.bias is not None for projection in projections)
        if len(set(bias_presence)) != 1:
            raise ValueError("Gate/up projections must both have bias or both be bias-free.")
        if (spec.packed_bias_key is not None) != bias_presence[0]:
            raise ValueError("Gate/up projection bias layout does not match the specification.")
        weights = tuple(projection.weight.detach().clone() for projection in projections)
        if len({(weight.dtype, weight.device) for weight in weights}) != 1:
            raise ValueError("Gate/up weights must share dtype and device.")
        bias = None
        if bias_presence[0]:
            biases = tuple(projection.bias.detach().clone() for projection in projections)
            if len({(value.dtype, value.device) for value in biases}) != 1:
                raise ValueError("Gate/up biases must share dtype and device.")
            bias = torch.cat(biases, dim=0)
        return cls(spec, torch.cat(weights, dim=0), bias, activation_fn)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = F.linear(hidden_states, self.weight, self.bias).chunk(2, dim=-1)
        return self.activation_fn(gate) * up


def gated_mlp_pack_transform_handler(
    spec: GatedMLPProjectionSpec,
    activation_fn: Callable[[Tensor], Tensor],
    *,
    transform_id: str = "pack-gated-mlp",
    inverse_transform_id: str = "unpack-gated-mlp",
) -> TransformHandler:
    """Create a reversible handler for an adapter-proven gated wrapper path."""

    if spec.activation == "gelu":
        raise ValueError("GELU paths must remain on their original projection.")

    def capture(model: nn.Module, transformation: ModelTransformation) -> nn.Module:
        return _resolve_module(model, transformation.target_paths[0])

    def apply(model: nn.Module, transformation: ModelTransformation) -> None:
        if len(transformation.target_paths) != 1:
            raise ValueError("Gated MLP packing requires exactly one wrapper path.")
        target = _resolve_module(model, transformation.target_paths[0])
        packed = PackedGatedMLPProjection.from_separate(
            spec, target.gate_proj, target.up_proj, activation_fn
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


__all__ = [
    "GatedMLPProjectionSpec",
    "MLPActivation",
    "PackedGatedMLPProjection",
    "gated_mlp_pack_transform_handler",
]
