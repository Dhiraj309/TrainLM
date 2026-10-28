"""Transactional application of declarative model transformation plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torch import nn

from .plan import ExecutionPlan, ModelTransformation


class TransformApplicationError(RuntimeError):
    """Raised after a failed transform has been rolled back."""


Capture = Callable[[nn.Module, ModelTransformation], Any]
Apply = Callable[[nn.Module, ModelTransformation], None]
Rollback = Callable[[nn.Module, ModelTransformation, Any], None]


@dataclass(frozen=True, slots=True)
class TransformHandler:
    """Implementation hooks for one transform and its declared inverse."""

    transform_id: str
    inverse_transform_id: str
    capture: Capture
    apply: Apply
    rollback: Rollback

    def __post_init__(self) -> None:
        for name in ("transform_id", "inverse_transform_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Transform handler {name} cannot be empty.")
        if self.transform_id == self.inverse_transform_id:
            raise ValueError("Transform and inverse handler IDs must differ.")
        for name in ("capture", "apply", "rollback"):
            if not callable(getattr(self, name)):
                raise TypeError(f"Transform handler {name} must be callable.")


@dataclass(frozen=True, slots=True)
class _AppliedTransform:
    specification: ModelTransformation
    handler: TransformHandler
    snapshot: Any


class TransformTransaction:
    """Applied transformations that can be committed or rolled back once."""

    def __init__(self, model: nn.Module, applied: list[_AppliedTransform]) -> None:
        self.model = model
        self._applied = applied
        self._closed = False
        self.committed = False

    @property
    def applied_transform_ids(self) -> tuple[str, ...]:
        return tuple(item.specification.transform_id for item in self._applied)

    def commit(self) -> nn.Module:
        if self._closed:
            raise RuntimeError("Transform transaction is already closed.")
        self._closed = True
        self.committed = True
        return self.model

    def rollback(self) -> nn.Module:
        if self._closed:
            raise RuntimeError("Transform transaction is already closed.")
        self._rollback_all()
        self._closed = True
        return self.model

    def _rollback_all(self) -> None:
        failures: list[str] = []
        for item in reversed(self._applied):
            try:
                item.handler.rollback(self.model, item.specification, item.snapshot)
            except Exception as exc:  # rollback must attempt every earlier transform
                failures.append(f"{item.specification.transform_id}: {exc}")
        if failures:
            raise TransformApplicationError(
                "Transformation rollback was incomplete: " + "; ".join(failures)
            )

    def __enter__(self) -> "TransformTransaction":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        if not self._closed:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        return False


class ModelTransformRegistry:
    """Apply a ready plan using explicitly registered reversible handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, TransformHandler] = {}

    def register(self, handler: TransformHandler) -> None:
        if not isinstance(handler, TransformHandler):
            raise TypeError("handler must be a TransformHandler.")
        if handler.transform_id in self._handlers:
            raise ValueError(f"Transform handler already registered: {handler.transform_id}")
        self._handlers[handler.transform_id] = handler

    def apply(self, model: nn.Module, plan: ExecutionPlan) -> TransformTransaction:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan.")
        if not plan.is_executable:
            raise TransformApplicationError(
                f"Cannot apply blocked execution plan {plan.plan_id}."
            )
        if plan.status == "noop":
            return TransformTransaction(model, [])

        missing = [
            item.transform_id
            for item in plan.transformations
            if item.transform_id not in self._handlers
        ]
        if missing:
            raise TransformApplicationError(
                "No registered handler for planned transforms: " + ", ".join(missing)
            )

        applied: list[_AppliedTransform] = []
        try:
            for specification in plan.transformations:
                handler = self._handlers[specification.transform_id]
                if handler.inverse_transform_id != specification.inverse_transform_id:
                    raise TransformApplicationError(
                        f"Transform {specification.transform_id} declares inverse "
                        f"{specification.inverse_transform_id}, but its handler provides "
                        f"{handler.inverse_transform_id}."
                    )
                aliases_before = _parameter_aliases(model)
                snapshot = handler.capture(model, specification)
                current = _AppliedTransform(specification, handler, snapshot)
                applied.append(current)
                handler.apply(model, specification)
                if (
                    not specification.parameter_layout_change
                    and _parameter_aliases(model) != aliases_before
                ):
                    raise TransformApplicationError(
                        f"Transform {specification.transform_id} changed parameter aliases "
                        "without declaring a layout change."
                    )
        except Exception as exc:
            transaction = TransformTransaction(model, applied)
            try:
                transaction.rollback()
            except Exception as rollback_exc:
                raise TransformApplicationError(
                    f"Transform application failed ({exc}); rollback also failed: "
                    f"{rollback_exc}"
                ) from exc
            if isinstance(exc, TransformApplicationError):
                raise
            raise TransformApplicationError(
                f"Transform application failed and was rolled back: {exc}"
            ) from exc
        return TransformTransaction(model, applied)


def _parameter_aliases(model: nn.Module) -> tuple[tuple[str, ...], ...]:
    groups: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    return tuple(sorted(tuple(sorted(names)) for names in groups.values()))


__all__ = [
    "ModelTransformRegistry",
    "TransformApplicationError",
    "TransformHandler",
    "TransformTransaction",
]
