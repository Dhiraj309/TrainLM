"""Accumulation strategy selection for compiled accelerator updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AccumulationStrategy = Literal["microstep", "unrolled", "xla_loop", "native"]
AccumulationRequest = Literal[
    "auto", "microstep", "unrolled", "xla_loop", "native"
]


@dataclass(frozen=True, slots=True)
class AccumulationEvidence:
    """Measured/provider facts used to select an accumulation implementation."""

    backend: str
    compile_supported: bool = False
    dispatch_supported: bool = False
    hbm_headroom: bool = False
    xla_loop_supported: bool = False
    native_supported: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("Accumulation evidence backend cannot be empty.")


@dataclass(frozen=True, slots=True)
class AccumulationPlan:
    """Explainable selection for one static gradient-accumulation shape."""

    requested: AccumulationRequest
    selected: AccumulationStrategy
    fallback_from: AccumulationStrategy | None
    reason: str
    micro_batch: int
    sequence_length: int
    accumulation_steps: int
    evidence: AccumulationEvidence

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("Accumulation plan reason cannot be empty.")
        for name in ("micro_batch", "sequence_length", "accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.fallback_from == self.selected:
            raise ValueError("Accumulation fallback must change strategy.")


def select_accumulation_plan(
    *,
    requested: AccumulationRequest = "auto",
    allow_fallbacks: bool = True,
    micro_batch: int,
    sequence_length: int,
    accumulation_steps: int,
    evidence: AccumulationEvidence,
) -> AccumulationPlan:
    """Select a strategy only when compile/dispatch/HBM evidence is ready."""

    valid_requests = {"auto", "microstep", "unrolled", "xla_loop", "native"}
    if requested not in valid_requests:
        raise ValueError(f"Unsupported accumulation strategy: {requested}")

    supports = {
        "microstep": True,
        "unrolled": (
            evidence.compile_supported
            and evidence.dispatch_supported
            and evidence.hbm_headroom
        ),
        "xla_loop": evidence.xla_loop_supported and evidence.hbm_headroom,
        "native": evidence.native_supported and evidence.hbm_headroom,
    }

    if requested == "auto":
        candidates = ("native", "xla_loop", "unrolled", "microstep")
        selected = next(strategy for strategy in candidates if supports[strategy])
        reason = (
            "Selected from measured provider and HBM evidence."
            if selected != "microstep"
            else (
                "No optimized strategy has complete compile, dispatch, and "
                "HBM evidence."
            )
        )
        return AccumulationPlan(
            requested=requested,
            selected=selected,
            fallback_from=None,
            reason=reason,
            micro_batch=micro_batch,
            sequence_length=sequence_length,
            accumulation_steps=accumulation_steps,
            evidence=evidence,
        )

    if supports[requested]:
        return AccumulationPlan(
            requested=requested,
            selected=requested,
            fallback_from=None,
            reason="Requested strategy satisfies provider and HBM evidence.",
            micro_batch=micro_batch,
            sequence_length=sequence_length,
            accumulation_steps=accumulation_steps,
            evidence=evidence,
        )

    if not allow_fallbacks:
        raise RuntimeError(
            f"Accumulation strategy '{requested}' lacks compile, dispatch, "
            "or HBM evidence and fallbacks are disabled."
        )
    return AccumulationPlan(
        requested=requested,
        selected="microstep",
        fallback_from=requested,
        reason=(
            f"Requested '{requested}' lacks complete compile, dispatch, or HBM "
            "evidence; using the generic microstep path."
        ),
        micro_batch=micro_batch,
        sequence_length=sequence_length,
        accumulation_steps=accumulation_steps,
        evidence=evidence,
    )
