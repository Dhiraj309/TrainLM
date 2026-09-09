"""Evidence-driven decoder rematerialization policy selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

RematerializationScope = Literal["none", "block", "attention", "mlp", "loss_chunk"]


@dataclass(frozen=True, slots=True)
class RematerializationPolicy:
    policy_id: str
    scopes: tuple[RematerializationScope, ...]
    apply_before_fsdp: bool

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id cannot be empty.")
        if not self.scopes:
            raise ValueError("At least one rematerialization scope is required.")
        allowed = {"none", "block", "attention", "mlp", "loss_chunk"}
        if any(scope not in allowed for scope in self.scopes):
            raise ValueError("Unsupported rematerialization scope.")
        if len(self.scopes) != len(set(self.scopes)):
            raise ValueError("Rematerialization scopes must be unique.")
        if "none" in self.scopes and len(self.scopes) != 1:
            raise ValueError("'none' cannot be combined with rematerialized scopes.")
        if not isinstance(self.apply_before_fsdp, bool):
            raise TypeError("apply_before_fsdp must be boolean.")
        if self.scopes != ("none",) and not self.apply_before_fsdp:
            raise ValueError("Rematerialization must be applied before FSDP wrapping.")


@dataclass(frozen=True, slots=True)
class RematerializationMeasurement:
    policy: RematerializationPolicy
    step_seconds: float
    peak_hbm_gib: float
    gradients_match: bool
    graph_stable: bool

    def __post_init__(self) -> None:
        for name in ("step_seconds", "peak_hbm_gib"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.gradients_match, bool):
            raise TypeError("gradients_match must be boolean.")
        if not isinstance(self.graph_stable, bool):
            raise TypeError("graph_stable must be boolean.")


@dataclass(frozen=True, slots=True)
class RematerializationSelection:
    selected: RematerializationMeasurement
    rejected: tuple[tuple[str, str], ...]
    maximum_slowdown: float


def select_rematerialization_policy(
    measurements: tuple[RematerializationMeasurement, ...],
    *,
    maximum_slowdown: float = 0.10,
) -> RematerializationSelection:
    """Select the lowest-HBM correct policy within a measured slowdown budget."""

    if (
        isinstance(maximum_slowdown, bool)
        or not isinstance(maximum_slowdown, (int, float))
        or not math.isfinite(maximum_slowdown)
        or maximum_slowdown < 0
    ):
        raise ValueError("maximum_slowdown must be finite and non-negative.")
    if not measurements:
        raise ValueError("At least one rematerialization measurement is required.")
    ids = [item.policy.policy_id for item in measurements]
    if len(ids) != len(set(ids)):
        raise ValueError("Rematerialization policy IDs must be unique.")
    baselines = [item for item in measurements if item.policy.scopes == ("none",)]
    if len(baselines) != 1:
        raise ValueError("Exactly one non-rematerialized baseline is required.")
    baseline = baselines[0]
    if not baseline.gradients_match or not baseline.graph_stable:
        raise ValueError("The non-rematerialized baseline must be correct and stable.")
    step_limit = baseline.step_seconds * (1.0 + maximum_slowdown)
    eligible = []
    rejected = []
    for item in sorted(measurements, key=lambda value: value.policy.policy_id):
        if not item.gradients_match:
            rejected.append((item.policy.policy_id, "gradient parity failed"))
        elif not item.graph_stable:
            rejected.append((item.policy.policy_id, "compiled graph is unstable"))
        elif item.step_seconds > step_limit:
            rejected.append((item.policy.policy_id, "step-time slowdown exceeds budget"))
        else:
            eligible.append(item)
    selected = min(
        eligible,
        key=lambda item: (
            item.peak_hbm_gib,
            item.step_seconds,
            item.policy.policy_id,
        ),
    )
    return RematerializationSelection(
        selected=selected,
        rejected=tuple(rejected),
        maximum_slowdown=float(maximum_slowdown),
    )


__all__ = [
    "RematerializationMeasurement",
    "RematerializationPolicy",
    "RematerializationScope",
    "RematerializationSelection",
    "select_rematerialization_policy",
]
