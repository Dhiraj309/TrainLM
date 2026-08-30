"""Sparse, host-boundary training-integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch


@dataclass(frozen=True, slots=True)
class IntegrityPolicy:
    """Checks enabled at a configured integrity interval."""

    check_loss: bool = True
    check_gradients: bool = True
    check_parameters: bool = True
    max_update_norm: float | None = None
    require_token_delta: bool = True
    require_cursor_continuity: bool = False

    def __post_init__(self) -> None:
        for name in (
            "check_loss", "check_gradients", "check_parameters",
            "require_token_delta", "require_cursor_continuity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        if self.max_update_norm is not None and (
            not isinstance(self.max_update_norm, (int, float))
            or not math.isfinite(self.max_update_norm)
            or self.max_update_norm <= 0
        ):
            raise ValueError("max_update_norm must be finite and positive.")


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Result of sparse checks; violations are safe to log on the host."""

    checked: tuple[str, ...]
    violations: tuple[str, ...]
    loss_finite: bool
    gradients_finite: bool
    parameters_finite: bool
    token_delta_valid: bool
    cursor_continuous: bool

    @property
    def passed(self) -> bool:
        return not self.violations


def _is_finite(value: Any) -> bool:
    try:
        return bool(torch.isfinite(torch.as_tensor(value)).all())
    except (TypeError, ValueError, RuntimeError):
        return False


def _all_finite(values: Iterable[Any]) -> bool:
    return all(_is_finite(value) for value in values)


def check_training_integrity(
    *,
    loss: Any,
    gradients: Iterable[Any] = (),
    parameters: Iterable[Any] = (),
    update_norm: float | None = None,
    expected_tokens: int | None = None,
    actual_tokens: int | None = None,
    expected_cursor: Any = None,
    actual_cursor: Any = None,
    policy: IntegrityPolicy | None = None,
) -> IntegrityReport:
    """Run configured checks after a step outside the compiled hot path."""

    policy = IntegrityPolicy() if policy is None else policy
    if not isinstance(policy, IntegrityPolicy):
        raise TypeError("policy must be an IntegrityPolicy.")
    checked: list[str] = []
    violations: list[str] = []
    loss_finite = gradients_finite = parameters_finite = True
    token_delta_valid = cursor_continuous = True

    if policy.check_loss:
        checked.append("loss")
        loss_finite = _is_finite(loss)
        if not loss_finite:
            violations.append("loss is non-finite")
    if policy.check_gradients:
        checked.append("gradients")
        gradients_finite = _all_finite(gradients)
        if not gradients_finite:
            violations.append("gradients contain non-finite values")
    if policy.check_parameters:
        checked.append("parameters")
        parameters_finite = _all_finite(parameters)
        if not parameters_finite:
            violations.append("parameters contain non-finite values")
    if update_norm is not None:
        checked.append("update_norm")
        if not isinstance(update_norm, (int, float)) or not math.isfinite(update_norm):
            violations.append("update norm is non-finite")
        elif (
            policy.max_update_norm is not None
            and update_norm > policy.max_update_norm
        ):
            violations.append("update norm exceeds configured limit")
    if expected_tokens is not None or actual_tokens is not None:
        checked.append("tokens")
        token_delta_valid = (
            isinstance(expected_tokens, int) and not isinstance(expected_tokens, bool)
            and expected_tokens >= 0 and isinstance(actual_tokens, int)
            and not isinstance(actual_tokens, bool) and actual_tokens == expected_tokens
        )
        if policy.require_token_delta and not token_delta_valid:
            violations.append("token delta is discontinuous")
    if expected_cursor is not None or actual_cursor is not None:
        checked.append("cursor")
        cursor_continuous = actual_cursor == expected_cursor
        if policy.require_cursor_continuity and not cursor_continuous:
            violations.append("data cursor is discontinuous")
    return IntegrityReport(
        checked=tuple(checked), violations=tuple(violations),
        loss_finite=loss_finite, gradients_finite=gradients_finite,
        parameters_finite=parameters_finite, token_delta_valid=token_delta_valid,
        cursor_continuous=cursor_continuous,
    )
