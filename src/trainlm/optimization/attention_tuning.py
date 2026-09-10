"""Deterministic attention autotuning keys, candidates, and result cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class AttentionTuningKey:
    hardware: str
    provider_id: str
    provider_version: str
    dtype: str
    batch_size: int
    query_heads: int
    key_value_heads: int
    sequence_length: int
    head_dim: int
    mask_layout: str
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "hardware",
            "provider_id",
            "provider_version",
            "dtype",
            "mask_layout",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} cannot be empty.")
        for name in (
            "batch_size",
            "query_heads",
            "key_value_heads",
            "sequence_length",
            "head_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.query_heads % self.key_value_heads:
            raise ValueError("query_heads must be divisible by key_value_heads.")
        if self.mask_layout not in {"causal", "causal_sliding_window"}:
            raise ValueError(f"Unsupported attention mask layout: {self.mask_layout}")
        if self.mask_layout == "causal_sliding_window":
            if (
                isinstance(self.sliding_window, bool)
                or not isinstance(self.sliding_window, int)
                or self.sliding_window < 1
            ):
                raise ValueError("Sliding masks require a positive window.")
        elif self.sliding_window is not None:
            raise ValueError("Only sliding masks may declare a window.")

    @property
    def cache_id(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class AttentionTuningCandidate:
    candidate_id: str
    block_q: int
    block_k: int
    num_warps: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id cannot be empty.")
        for name in ("block_q", "block_k", "num_warps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class AttentionTuningResult:
    key: AttentionTuningKey
    candidate: AttentionTuningCandidate
    score: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise ValueError("score must be finite.")


class AttentionTuningCache:
    """Exact-key cache whose selection is stable across candidate orderings."""

    def __init__(self) -> None:
        self._results: dict[str, AttentionTuningResult] = {}

    def get(self, key: AttentionTuningKey) -> AttentionTuningResult | None:
        return self._results.get(key.cache_id)

    def tune(
        self,
        key: AttentionTuningKey,
        candidates: Iterable[AttentionTuningCandidate],
        benchmark: Callable[[AttentionTuningCandidate], float],
    ) -> AttentionTuningResult:
        cached = self.get(key)
        if cached is not None:
            return cached
        unique: dict[str, AttentionTuningCandidate] = {}
        for candidate in candidates:
            if not isinstance(candidate, AttentionTuningCandidate):
                raise TypeError("candidates must be AttentionTuningCandidate instances.")
            previous = unique.setdefault(candidate.candidate_id, candidate)
            if previous != candidate:
                raise ValueError(f"Conflicting candidate ID: {candidate.candidate_id}")
        if not unique:
            raise ValueError("At least one tuning candidate is required.")
        measured = []
        for candidate_id in sorted(unique):
            candidate = unique[candidate_id]
            score = benchmark(candidate)
            measured.append(AttentionTuningResult(key, candidate, score))
        # Higher scores win; candidate ID is the stable tie breaker.
        result = min(measured, key=lambda item: (-item.score, item.candidate.candidate_id))
        self._results[key.cache_id] = result
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        payload = [
            {
                "key": asdict(result.key),
                "candidate": asdict(result.candidate),
                "score": result.score,
            }
            for _, result in sorted(self._results.items())
        ]
        return json.dumps(payload, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "AttentionTuningCache":
        payload = json.loads(value)
        if not isinstance(payload, list):
            raise ValueError("Attention tuning cache root must be a list.")
        cache = cls()
        for entry in payload:
            if not isinstance(entry, Mapping):
                raise ValueError("Attention tuning cache entries must be objects.")
            result = AttentionTuningResult(
                key=AttentionTuningKey(**entry["key"]),
                candidate=AttentionTuningCandidate(**entry["candidate"]),
                score=entry["score"],
            )
            if cache.get(result.key) is not None:
                raise ValueError("Attention tuning cache contains a duplicate key.")
            cache._results[result.key.cache_id] = result
        return cache


__all__ = [
    "AttentionTuningCache",
    "AttentionTuningCandidate",
    "AttentionTuningKey",
    "AttentionTuningResult",
]
