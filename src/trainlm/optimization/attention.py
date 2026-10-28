"""Canonical, model-family-neutral causal attention specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Literal, Mapping

from .capabilities import ModelCapabilities

AttentionLayout = Literal["mha", "gqa", "mqa"]
PositionEncoding = Literal["learned", "rope", "alibi", "none"]
MaskLayout = Literal["causal", "causal_sliding_window"]
OutputLayout = Literal["bsh"]


@dataclass(frozen=True, slots=True)
class AttentionMaskSpec:
    """Mask semantics consumed by an attention provider."""

    layout: MaskLayout = "causal"
    sliding_window: int | None = None
    segment_ids: bool = False

    def __post_init__(self) -> None:
        if self.layout not in {"causal", "causal_sliding_window"}:
            raise ValueError(f"Unsupported attention mask layout: {self.layout}")
        if self.layout == "causal" and self.sliding_window is not None:
            raise ValueError("Full causal attention cannot declare a sliding window.")
        if self.layout == "causal_sliding_window" and (
            isinstance(self.sliding_window, bool)
            or not isinstance(self.sliding_window, int)
            or self.sliding_window < 1
        ):
            raise ValueError("Sliding-window attention requires a positive window.")
        if not isinstance(self.segment_ids, bool):
            raise ValueError("segment_ids must be boolean.")


@dataclass(frozen=True, slots=True)
class CanonicalAttentionSpec:
    """Complete semantic input to portable or optimized attention providers."""

    schema_version: int
    layout: AttentionLayout
    query_heads: int
    key_value_heads: int
    head_dim: int
    position_encoding: PositionEncoding
    mask: AttentionMaskSpec
    scale: float | None = None
    dropout: float = 0.0
    soft_cap: float | None = None
    qk_normalization: bool = False
    output_layout: OutputLayout = "bsh"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("CanonicalAttentionSpec supports schema_version=1 only.")
        if self.layout not in {"mha", "gqa", "mqa"}:
            raise ValueError(f"Unsupported attention layout: {self.layout}")
        for name in ("query_heads", "key_value_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.key_value_heads > self.query_heads or (
            self.query_heads % self.key_value_heads
        ):
            raise ValueError("Query heads must be divisible by key/value heads.")
        expected_layout = (
            "mha" if self.key_value_heads == self.query_heads
            else "mqa" if self.key_value_heads == 1
            else "gqa"
        )
        if self.layout != expected_layout:
            raise ValueError("Attention layout conflicts with head geometry.")
        if self.position_encoding not in {"learned", "rope", "alibi", "none"}:
            raise ValueError(f"Unsupported position encoding: {self.position_encoding}")
        if not isinstance(self.mask, AttentionMaskSpec):
            raise TypeError("mask must be an AttentionMaskSpec.")
        if self.scale is not None and (
            not isinstance(self.scale, (int, float))
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("scale must be finite and positive when configured.")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.soft_cap is not None and (
            not isinstance(self.soft_cap, (int, float))
            or not math.isfinite(self.soft_cap)
            or self.soft_cap <= 0
        ):
            raise ValueError("soft_cap must be finite and positive when configured.")
        if not isinstance(self.qk_normalization, bool):
            raise ValueError("qk_normalization must be boolean.")
        if self.output_layout != "bsh":
            raise ValueError("Only batch-sequence-hidden output is currently supported.")

    @property
    def effective_scale(self) -> float:
        return float(self.scale) if self.scale is not None else self.head_dim ** -0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalAttentionSpec":
        values = dict(data)
        values["mask"] = AttentionMaskSpec(**values["mask"])
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "CanonicalAttentionSpec":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Canonical attention JSON root must be an object.")
        return cls.from_dict(data)

    @classmethod
    def from_capabilities(
        cls,
        capabilities: ModelCapabilities,
        *,
        head_dim: int,
        mask: AttentionMaskSpec | None = None,
        scale: float | None = None,
        dropout: float = 0.0,
        soft_cap: float | None = None,
        qk_normalization: bool = False,
    ) -> "CanonicalAttentionSpec":
        """Map inspected semantics without consulting architecture names."""

        attention = capabilities.attention
        position = capabilities.position
        if attention.status not in {"known", "inferred"}:
            raise ValueError("Attention semantics are not proven by inspection.")
        if position.status not in {"known", "inferred"}:
            raise ValueError("Position semantics are not proven by inspection.")
        facts = {fact.name: fact.value for fact in attention.facts}
        heads, kv_heads = facts.get("heads"), facts.get("kv_heads")
        if not isinstance(heads, int) or not isinstance(kv_heads, int):
            raise ValueError("Inspected attention is missing integer head geometry.")
        return cls(
            schema_version=1,
            layout=attention.kind,
            query_heads=heads,
            key_value_heads=kv_heads,
            head_dim=head_dim,
            position_encoding=position.kind,
            mask=mask or AttentionMaskSpec(),
            scale=scale,
            dropout=dropout,
            soft_cap=soft_cap,
            qk_normalization=qk_normalization,
        )


__all__ = [
    "AttentionLayout",
    "AttentionMaskSpec",
    "CanonicalAttentionSpec",
    "MaskLayout",
    "OutputLayout",
    "PositionEncoding",
]
