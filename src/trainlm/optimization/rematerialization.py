"""Evidence-driven decoder rematerialization policy selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal

from torch import nn
from torch.utils.checkpoint import checkpoint

from .plan import ModelTransformation
from .transforms import TransformHandler

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


def module_rematerialization_transform_handler(
    policy: RematerializationPolicy,
    *,
    transform_id: str = "rematerialize-module",
    inverse_transform_id: str = "restore-module-forward",
) -> TransformHandler:
    """Create a reversible non-reentrant checkpoint transform for module paths.

    Adapters supply the exact block, attention, or MLP paths. Patching ``forward``
    leaves module structure, parameter aliases, and canonical state keys intact.
    """

    active_scopes = tuple(scope for scope in policy.scopes if scope != "none")
    if not active_scopes:
        raise ValueError("The no-rematerialization policy does not need a handler.")
    if "loss_chunk" in active_scopes:
        raise ValueError("Loss-chunk rematerialization is owned by the loss provider.")

    def capture(
        model: nn.Module, transformation: ModelTransformation
    ) -> tuple[tuple[nn.Module, Callable[..., Any]], ...]:
        _validate_transformation_scope(transformation, active_scopes)
        return tuple(
            (module, module.forward)
            for module in _resolve_unique_modules(model, transformation.target_paths)
        )

    def apply(model: nn.Module, transformation: ModelTransformation) -> None:
        _validate_transformation_scope(transformation, active_scopes)
        for module in _resolve_unique_modules(model, transformation.target_paths):
            original_forward = module.forward

            def checkpointed_forward(
                *args: Any,
                _forward: Callable[..., Any] = original_forward,
                **kwargs: Any,
            ) -> Any:
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)

            module.forward = checkpointed_forward

    def rollback(
        model: nn.Module,
        transformation: ModelTransformation,
        snapshot: tuple[tuple[nn.Module, Callable[..., Any]], ...],
    ) -> None:
        del model, transformation
        for module, original_forward in snapshot:
            module.forward = original_forward

    return TransformHandler(transform_id, inverse_transform_id, capture, apply, rollback)


def _validate_transformation_scope(
    transformation: ModelTransformation,
    active_scopes: tuple[RematerializationScope, ...],
) -> None:
    if transformation.component not in active_scopes:
        raise ValueError(
            f"Transformation component {transformation.component!r} is not enabled "
            "by the rematerialization policy."
        )
    if transformation.parameter_layout_change:
        raise ValueError("Rematerialization cannot declare a parameter layout change.")


def _resolve_unique_modules(
    model: nn.Module, paths: tuple[str, ...]
) -> tuple[nn.Module, ...]:
    modules: list[nn.Module] = []
    identities: set[int] = set()
    for path in paths:
        target: Any = model
        for component in path.split("."):
            if not component or not hasattr(target, component):
                raise ValueError(f"Model has no module at path {path!r}.")
            target = getattr(target, component)
        if not isinstance(target, nn.Module):
            raise TypeError(f"Target path {path!r} does not resolve to a module.")
        if id(target) in identities:
            raise ValueError("Rematerialization target paths must resolve uniquely.")
        identities.add(id(target))
        modules.append(target)
    return tuple(modules)


__all__ = [
    "RematerializationMeasurement",
    "RematerializationPolicy",
    "RematerializationScope",
    "RematerializationSelection",
    "module_rematerialization_transform_handler",
    "select_rematerialization_policy",
]
