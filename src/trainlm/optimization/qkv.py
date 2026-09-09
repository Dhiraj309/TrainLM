"""Validated reversible layouts for compatible separate Q/K/V projections."""

from __future__ import annotations

from dataclasses import dataclass

from .state_dict import ParameterLayoutMapping, StateDictLayoutConverter


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


__all__ = ["QKVProjectionSpec"]
