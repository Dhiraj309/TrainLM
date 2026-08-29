"""Optimization request policy, separate from backend execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Declare optimization intent without selecting implementations."""

    policy: Literal["disabled", "auto", "required"] = "auto"
    compile: bool = False
    allow_fallbacks: bool = True
    requested: tuple[str, ...] = field(default_factory=tuple)
    compilation_cache_dir: str | Path | None = None
    compilation_cache_readonly: bool = False
    accumulation_strategy: Literal[
        "auto", "microstep", "unrolled", "xla_loop", "native"
    ] = "auto"

    def __post_init__(self) -> None:
        if self.policy not in {"disabled", "auto", "required"}:
            raise ValueError(f"Unsupported optimization policy: {self.policy}")
        object.__setattr__(self, "requested", tuple(self.requested))
        if self.compilation_cache_dir is not None:
            cache_dir = str(self.compilation_cache_dir).strip()
            if not cache_dir:
                raise ValueError("compilation_cache_dir cannot be empty.")
            object.__setattr__(self, "compilation_cache_dir", cache_dir)
        if self.accumulation_strategy not in {
            "auto",
            "microstep",
            "unrolled",
            "xla_loop",
            "native",
        }:
            raise ValueError(
                f"Unsupported accumulation strategy: {self.accumulation_strategy}"
            )
        if self.policy == "required" and self.allow_fallbacks:
            raise ValueError(
                "Required optimization policy cannot allow fallbacks."
            )
