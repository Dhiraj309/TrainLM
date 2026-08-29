"""Conservative attention-layout detection for dense causal models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AttentionKind = Literal["mha", "gqa", "mqa", "unknown"]
ProjectionKind = Literal["packed", "separate", "unknown"]


@dataclass(frozen=True, slots=True)
class AttentionLayout:
    """Attention head and QKV projection layout with explicit evidence."""

    kind: AttentionKind
    projection: ProjectionKind
    query_heads: int | None
    kv_heads: int | None
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"mha", "gqa", "mqa", "unknown"}:
            raise ValueError(f"Unsupported attention kind: {self.kind}")
        if self.projection not in {"packed", "separate", "unknown"}:
            raise ValueError(f"Unsupported projection kind: {self.projection}")
        for name, value in (
            ("query_heads", self.query_heads),
            ("kv_heads", self.kv_heads),
        ):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None.")
        if not self.evidence or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence
        ):
            raise ValueError("Attention layout requires non-empty evidence.")


def _first_int(config: Any, names: tuple[str, ...]) -> tuple[str, int] | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return name, value
    return None


def detect_attention_layout(
    config: Any,
    model: Any | None = None,
) -> AttentionLayout:
    """Classify MHA/GQA/MQA and QKV projection packing without family guesses."""

    if config is None:
        raise TypeError("config is required.")

    query = _first_int(
        config,
        ("num_attention_heads", "num_heads", "n_head", "num_q_heads"),
    )
    kv = _first_int(
        config,
        (
            "num_key_value_heads",
            "num_kv_heads",
            "n_head_kv",
            "num_kv_attention_heads",
        ),
    )
    multi_query = any(
        getattr(config, name, None) is True
        for name in ("multi_query", "multi_query_attention", "use_mqa")
    )
    evidence = []
    if query is not None:
        evidence.append(f"config.{query[0]}={query[1]}")
    if kv is not None:
        evidence.append(f"config.{kv[0]}={kv[1]}")
    if multi_query:
        evidence.append("explicit multi-query flag")

    query_heads = query[1] if query is not None else None
    kv_heads = kv[1] if kv is not None else None
    kind: AttentionKind = "unknown"
    if query_heads is not None and kv_heads is None:
        kv_heads = 1 if multi_query else query_heads
    if query_heads is not None and kv_heads is not None:
        if multi_query or kv_heads == 1:
            kind = "mqa"
        elif kv_heads == query_heads:
            kind = "mha"
        elif query_heads % kv_heads == 0 and kv_heads < query_heads:
            kind = "gqa"

    projection: ProjectionKind = "unknown"
    fused = getattr(config, "fused_qkv", None)
    if fused is None:
        fused = getattr(config, "use_fused_qkv", None)
    if isinstance(fused, bool):
        projection = "packed" if fused else "separate"
        evidence.append(f"config.fused_qkv={fused}")

    # Module inspection is a fallback for models whose configs do not describe
    # projection packing. It is intentionally based on structure, not family.
    if model is not None:
        modules = getattr(model, "modules", None)
        if callable(modules):
            names = tuple(type(module).__name__.lower() for module in modules())
            packed_tokens = ("qkv", "querykeyvalue", "c_attn")
            has_packed = any(
                token in name for name in names for token in packed_tokens
            )
            has_separate = all(
                any(token in name for name in names)
                for token in ("q_proj", "k_proj", "v_proj")
            )
            if projection == "unknown" and has_packed != has_separate:
                projection = "packed" if has_packed else "separate"
                evidence.append("model module structure exposes QKV projection layout")

    if not evidence:
        evidence.append("no explicit attention head or projection declaration")
    if query_heads is None or (kv is None and not multi_query):
        evidence.append("query/KV head counts are incomplete")
    if kind == "unknown" and query_heads is not None and kv_heads is not None:
        evidence.append("head counts are incompatible or ambiguous")
    return AttentionLayout(kind, projection, query_heads, kv_heads, tuple(evidence))
