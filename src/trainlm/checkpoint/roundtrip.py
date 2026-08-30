"""Offline checks for generic save/resume and Hugging Face round trips."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any


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
    left_values = _materialize(left)
    right_values = _materialize(right)
    left_shape = _shape(left_values)
    right_shape = _shape(right_values)
    if left_shape != right_shape:
        return math.inf
    errors = []
    for left_item, right_item in zip(
        _flatten(left_values), _flatten(right_values), strict=True
    ):
        try:
            error = abs(left_item - right_item)
            error = float(error)
        except (TypeError, ValueError, OverflowError):
            error = 0.0 if left_item == right_item else math.inf
        if math.isnan(error):
            return math.inf
        errors.append(error)
    return max(errors, default=0.0)


def _materialize(value: Any) -> Any:
    """Convert tensor-like values through optional duck-typed host methods."""

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(current, (list, tuple)):
        return type(current)(_materialize(item) for item in current)
    return current


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    children = tuple(_shape(item) for item in value)
    if children and any(child != children[0] for child in children[1:]):
        return (-1,)
    return (len(value),) + (children[0] if children else ())


def _flatten(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        return (value,)
    flattened: list[Any] = []
    for item in value:
        flattened.extend(_flatten(item))
    return tuple(flattened)


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
