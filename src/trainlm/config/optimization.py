"""Optimization request policy, separate from backend execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Declare optimization intent without selecting implementations."""

    policy: Literal["disabled", "auto", "required"] = "auto"
    compile: bool = False
    allow_fallbacks: bool = True
    requested: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.policy not in {"disabled", "auto", "required"}:
            raise ValueError(f"Unsupported optimization policy: {self.policy}")
        object.__setattr__(self, "requested", tuple(self.requested))
        if self.policy == "required" and self.allow_fallbacks:
            raise ValueError(
                "Required optimization policy cannot allow fallbacks."
            )
