"""Reversible state-dict layouts for compatible gated MLP projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .state_dict import ParameterLayoutMapping, StateDictLayoutConverter

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


__all__ = ["GatedMLPProjectionSpec", "MLPActivation"]
