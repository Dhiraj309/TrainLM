"""Evidence-backed native-versus-custom fusion audit decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Literal

FusionComponent = Literal["normalization", "rope", "residual", "mlp"]
FusionDecision = Literal["native", "custom", "blocked"]


@dataclass(frozen=True, slots=True)
class HLOFusionObservation:
    component: FusionComponent
    hlo_fingerprint: str
    native_fused: bool | None
    copy_count: int
    transpose_count: int
    materialization_count: int
    custom_call_count: int
    fallback_count: int = 0

    def __post_init__(self) -> None:
        if self.component not in {"normalization", "rope", "residual", "mlp"}:
            raise ValueError(f"Unsupported fusion component: {self.component}")
        if not isinstance(self.hlo_fingerprint, str) or not self.hlo_fingerprint:
            raise ValueError("hlo_fingerprint cannot be empty.")
        if self.native_fused is not None and not isinstance(self.native_fused, bool):
            raise TypeError("native_fused must be true, false, or unknown.")
        for name in (
            "copy_count",
            "transpose_count",
            "materialization_count",
            "custom_call_count",
            "fallback_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    @classmethod
    def from_hlo_text(
        cls,
        *,
        component: FusionComponent,
        hlo_text: str,
        native_fused: bool | None,
        materialization_patterns: tuple[str, ...] = (),
        fallback_count: int = 0,
    ) -> "HLOFusionObservation":
        """Build reproducible evidence from captured textual HLO.

        Fusion itself remains explicit evidence: textual opcode counting cannot
        prove semantic fusion. Materializations are similarly caller-declared
        patterns because their HLO form depends on the inspected operation.
        """

        if not isinstance(hlo_text, str) or not hlo_text.strip():
            raise ValueError("hlo_text cannot be empty.")
        if isinstance(materialization_patterns, (str, bytes)) or not isinstance(
            materialization_patterns, (tuple, list)
        ):
            raise TypeError("materialization_patterns must be a tuple or list.")
        patterns = tuple(materialization_patterns)
        if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise ValueError("materialization patterns cannot be empty.")
        try:
            materialization_count = sum(
                len(re.findall(pattern, hlo_text, flags=re.MULTILINE))
                for pattern in patterns
            )
        except re.error as exc:
            raise ValueError(f"Invalid materialization pattern: {exc}") from exc
        normalized = hlo_text.replace("\r\n", "\n").strip() + "\n"
        return cls(
            component=component,
            hlo_fingerprint="sha256:" + hashlib.sha256(normalized.encode()).hexdigest(),
            native_fused=native_fused,
            copy_count=_count_hlo_opcode(normalized, "copy"),
            transpose_count=_count_hlo_opcode(normalized, "transpose"),
            materialization_count=materialization_count,
            custom_call_count=_count_hlo_opcode(normalized, "custom-call"),
            fallback_count=fallback_count,
        )


@dataclass(frozen=True, slots=True)
class HLOFusionDecision:
    component: FusionComponent
    decision: FusionDecision
    reason: str
    observation: HLOFusionObservation


@dataclass(frozen=True, slots=True)
class HLOFusionAudit:
    schema_version: int
    decisions: tuple[HLOFusionDecision, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("HLOFusionAudit supports schema_version=1 only.")
        components = [item.component for item in self.decisions]
        if len(components) != len(set(components)):
            raise ValueError("Fusion audit components must be unique.")

    @property
    def ready(self) -> bool:
        return bool(self.decisions) and all(
            item.decision != "blocked" for item in self.decisions
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def audit_hlo_fusions(
    observations: tuple[HLOFusionObservation, ...],
) -> HLOFusionAudit:
    """Choose native/custom only from supplied HLO and fallback evidence."""

    if not observations:
        raise ValueError("At least one HLO fusion observation is required.")
    decisions = []
    for observation in sorted(observations, key=lambda item: item.component):
        if observation.fallback_count:
            decision = "blocked"
            reason = "Fallbacks must be removed before selecting a fusion path."
        elif observation.native_fused is None:
            decision = "blocked"
            reason = "Native fusion is unknown; inspect the captured HLO."
        elif (
            observation.native_fused
            and observation.copy_count == 0
            and observation.transpose_count == 0
            and observation.materialization_count == 0
            and observation.custom_call_count == 0
        ):
            decision = "native"
            reason = "Captured HLO proves clean native fusion without layout overhead."
        else:
            decision = "custom"
            reason = (
                "Captured HLO proves the native path is unfused or retains layout "
                "overhead/custom calls; benchmark an eligible custom provider."
            )
        decisions.append(
            HLOFusionDecision(
                component=observation.component,
                decision=decision,
                reason=reason,
                observation=observation,
            )
        )
    return HLOFusionAudit(schema_version=1, decisions=tuple(decisions))


def _count_hlo_opcode(hlo_text: str, opcode: str) -> int:
    return len(re.findall(rf"(?:^|[=, (]){re.escape(opcode)}\(", hlo_text))


__all__ = [
    "FusionComponent",
    "FusionDecision",
    "HLOFusionAudit",
    "HLOFusionDecision",
    "HLOFusionObservation",
    "audit_hlo_fusions",
]
