"""Immutable, host-materialized metrics delivered to callbacks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class MetricSnapshot(Mapping[str, float]):
    """Read-only scalar metrics safe to hand to host callbacks."""

    _values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for name, value in self._values.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Metric names must be non-empty strings.")
            if isinstance(value, torch.Tensor):
                raise TypeError(
                    "Callback metrics must be materialized Python scalars; "
                    f"metric '{name}' is a tensor."
                )
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"Metric '{name}' must be a materialized real scalar."
                )
            normalized[name] = float(value)
        object.__setattr__(self, "_values", MappingProxyType(normalized))

    def __getitem__(self, name: str) -> float:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @classmethod
    def from_mapping(cls, metrics: Mapping[str, Any]) -> "MetricSnapshot":
        """Validate and freeze a host-side metric mapping."""

        return cls(metrics)
