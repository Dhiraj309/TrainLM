"""Offline checks for generic save/resume and Hugging Face round trips."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class RoundTripReport:
    """Numerical and key-set evidence for one round-trip comparison."""

    state_max_abs_error: float
    exported_state_max_abs_error: float
    next_update_max_abs_error: float
    exported_logits_max_abs_error: float
    missing_state_keys: tuple[str, ...]
    unexpected_state_keys: tuple[str, ...]
    tolerance: float

    def __post_init__(self) -> None:
        for name in (
            "state_max_abs_error",
            "exported_state_max_abs_error",
            "next_update_max_abs_error",
            "exported_logits_max_abs_error",
            "tolerance",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or math.isnan(value):
                raise ValueError(f"{name} must be numeric and not NaN.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")

    @property
    def passed(self) -> bool:
        return (
            not self.missing_state_keys
            and not self.unexpected_state_keys
            and self.state_max_abs_error <= self.tolerance
            and self.exported_state_max_abs_error <= self.tolerance
            and self.next_update_max_abs_error <= self.tolerance
            and self.exported_logits_max_abs_error <= self.tolerance
        )


def _max_abs_error(left: Any, right: Any) -> float:
    left_tensor = torch.as_tensor(left).detach().to(device="cpu")
    right_tensor = torch.as_tensor(right).detach().to(device="cpu")
    if left_tensor.shape != right_tensor.shape:
        return math.inf
    if not (left_tensor.is_floating_point() or left_tensor.is_complex()):
        return 0.0 if torch.equal(left_tensor, right_tensor) else math.inf
    delta = (left_tensor.to(torch.float64) - right_tensor.to(torch.float64)).abs()
    return float(delta.max().item()) if delta.numel() else 0.0


def _compare_states(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    missing = tuple(sorted(reference_keys - candidate_keys))
    unexpected = tuple(sorted(candidate_keys - reference_keys))
    errors = [
        _max_abs_error(reference[name], candidate[name])
        for name in sorted(reference_keys & candidate_keys)
    ]
    return (max(errors, default=0.0), missing, unexpected)


def evaluate_round_trip(
    *,
    reference_state: Mapping[str, Any],
    resumed_state: Mapping[str, Any],
    exported_state: Mapping[str, Any],
    reference_next_update: Any,
    resumed_next_update: Any,
    reference_logits: Any,
    exported_logits: Any,
    tolerance: float = 1e-5,
) -> RoundTripReport:
    """Compare continuation and canonical-HF outputs at an evidence boundary.

    The caller supplies states and outputs after loading each artifact.  This
    helper performs no checkpoint I/O and is intentionally outside the training
    hot path; TPU callers should invoke it only after synchronizing a run.
    """

    if not isinstance(reference_state, Mapping):
        raise TypeError("reference_state must be a mapping.")
    if not isinstance(resumed_state, Mapping):
        raise TypeError("resumed_state must be a mapping.")
    if not isinstance(exported_state, Mapping):
        raise TypeError("exported_state must be a mapping.")
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    state_error, missing, unexpected = _compare_states(
        reference_state, resumed_state
    )
    exported_error, export_missing, export_unexpected = _compare_states(
        reference_state, exported_state
    )
    return RoundTripReport(
        state_max_abs_error=state_error,
        exported_state_max_abs_error=exported_error,
        next_update_max_abs_error=_max_abs_error(
            reference_next_update, resumed_next_update
        ),
        exported_logits_max_abs_error=_max_abs_error(
            reference_logits, exported_logits
        ),
        missing_state_keys=tuple(sorted(set(missing) | set(export_missing))),
        unexpected_state_keys=tuple(
            sorted(set(unexpected) | set(export_unexpected))
        ),
        tolerance=float(tolerance),
    )
