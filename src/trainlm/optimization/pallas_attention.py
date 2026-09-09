"""Version-guarded bridge for an explicitly supplied XLA Pallas MHA kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from .attention import CanonicalAttentionSpec
from .hf_attention import HFAttentionProvider


@dataclass(frozen=True, slots=True)
class PallasAttentionRuntime:
    """Evidence required before a Pallas attention kernel becomes selectable."""

    torch_xla_version: str
    tested_torch_xla_versions: tuple[str, ...]
    kernel: Callable[..., Any]
    backward_verified: bool
    hlo_custom_call_verified: bool = False
    grouped_query_verified: bool = False
    alibi_verified: bool = False
    sliding_window_verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.torch_xla_version, str) or not self.torch_xla_version:
            raise ValueError("torch_xla_version cannot be empty.")
        if (
            not isinstance(self.tested_torch_xla_versions, tuple)
            or not self.tested_torch_xla_versions
            or any(
                not isinstance(version, str) or not version
                for version in self.tested_torch_xla_versions
            )
        ):
            raise ValueError("tested_torch_xla_versions must contain versions.")
        if not callable(self.kernel):
            raise TypeError("kernel must be callable.")
        if not isinstance(self.backward_verified, bool):
            raise TypeError("backward_verified must be boolean.")
        if not isinstance(self.hlo_custom_call_verified, bool):
            raise TypeError("hlo_custom_call_verified must be boolean.")
        if not isinstance(self.grouped_query_verified, bool):
            raise TypeError("grouped_query_verified must be boolean.")
        if not isinstance(self.alibi_verified, bool):
            raise TypeError("alibi_verified must be boolean.")
        if not isinstance(self.sliding_window_verified, bool):
            raise TypeError("sliding_window_verified must be boolean.")

    def require_supported(self) -> None:
        if self.torch_xla_version not in self.tested_torch_xla_versions:
            raise RuntimeError(
                "Pallas attention is disabled for untested torch_xla version "
                f"{self.torch_xla_version!r}."
            )
        if not self.backward_verified:
            raise RuntimeError(
                "Pallas attention requires explicit backward correctness evidence."
            )


def pallas_mha_provider(
    runtime: PallasAttentionRuntime,
    *,
    provider_id: str = "trainlm.pallas_mha",
) -> HFAttentionProvider:
    """Build a guarded HF provider without importing optional XLA packages.

    ``runtime.kernel`` is an explicit adapter with the stable TrainLM call
    boundary ``kernel(query, key, value, *, causal, scale)``. This avoids
    guessing private torch_xla module paths or kernel signatures.
    """

    runtime.require_supported()

    def attention_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any | None = None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        del module, kwargs
        if attention_mask is not None:
            raise ValueError(
                "Pallas MHA currently accepts only its registered causal mask."
            )
        if dropout != 0.0:
            raise ValueError("Pallas MHA dropout is not implemented.")
        output = runtime.kernel(query, key, value, causal=True, scale=scaling)
        return output, None

    def causal_mask_factory(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        # The guarded kernel applies causality internally. Returning no tensor
        # prevents materialization of a dense quadratic mask.
        return None

    return HFAttentionProvider(
        provider_id=provider_id,
        attention_forward=attention_forward,
        mask_factory=causal_mask_factory,
        layouts=("mha",),
        mask_layouts=("causal",),
        position_encodings=("learned", "rope", "none"),
    )


@dataclass(frozen=True, slots=True)
class KVHeadMapping:
    """Logical query-to-KV ownership without materializing repeated K/V heads."""

    query_heads: int
    key_value_heads: int

    def __post_init__(self) -> None:
        for name in ("query_heads", "key_value_heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.query_heads % self.key_value_heads:
            raise ValueError("query_heads must be divisible by key_value_heads.")

    @property
    def queries_per_kv_head(self) -> int:
        return self.query_heads // self.key_value_heads

    def kv_head_for_query(self, query_head: int) -> int:
        if (
            isinstance(query_head, bool)
            or not isinstance(query_head, int)
            or not 0 <= query_head < self.query_heads
        ):
            raise ValueError("query_head is outside the configured head range.")
        return query_head // self.queries_per_kv_head

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.kv_head_for_query(head) for head in range(self.query_heads))


def pallas_grouped_attention_provider(
    runtime: PallasAttentionRuntime,
    spec: CanonicalAttentionSpec,
    *,
    provider_id: str = "trainlm.pallas_grouped_attention",
    alibi_slopes: tuple[float, ...] | None = None,
) -> HFAttentionProvider:
    """Build an MHA/GQA/MQA provider that preserves compact K/V head storage."""

    runtime.require_supported()
    if spec.layout != "mha" and not runtime.grouped_query_verified:
        raise RuntimeError(
            "Grouped-query Pallas attention requires explicit GQA/MQA evidence."
        )
    if spec.mask.segment_ids:
        raise ValueError("Grouped Pallas attention does not support segment masks.")
    if (
        spec.mask.layout == "causal_sliding_window"
        and not runtime.sliding_window_verified
    ):
        raise RuntimeError(
            "Sliding-window Pallas attention requires explicit runtime evidence."
        )
    if spec.position_encoding == "alibi":
        if not runtime.alibi_verified:
            raise RuntimeError("Pallas ALiBi attention requires explicit runtime evidence.")
        if (
            not isinstance(alibi_slopes, tuple)
            or len(alibi_slopes) != spec.query_heads
            or any(
                isinstance(slope, bool) or not isinstance(slope, (int, float))
                or not math.isfinite(slope)
                for slope in alibi_slopes
            )
        ):
            raise ValueError("ALiBi requires one explicit numeric slope per query head.")
    elif alibi_slopes is not None:
        raise ValueError("ALiBi slopes cannot be supplied for another position encoding.")
    mapping = KVHeadMapping(spec.query_heads, spec.key_value_heads)

    def attention_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any | None = None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        del module, kwargs
        if attention_mask is not None:
            raise ValueError("Grouped Pallas attention uses its registered causal mask.")
        if dropout != 0.0:
            raise ValueError("Grouped Pallas attention dropout is not implemented.")
        output = runtime.kernel(
            query,
            key,
            value,
            causal=True,
            scale=spec.effective_scale if scaling is None else scaling,
            query_heads=mapping.query_heads,
            key_value_heads=mapping.key_value_heads,
            sliding_window=spec.mask.sliding_window,
            alibi_slopes=alibi_slopes,
        )
        return output, None

    def causal_mask_factory(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    return HFAttentionProvider(
        provider_id=provider_id,
        attention_forward=attention_forward,
        mask_factory=causal_mask_factory,
        layouts=(spec.layout,),
        mask_layouts=(spec.mask.layout,),
        position_encodings=(spec.position_encoding,),
    )


__all__ = [
    "KVHeadMapping",
    "PallasAttentionRuntime",
    "pallas_grouped_attention_provider",
    "pallas_mha_provider",
]
