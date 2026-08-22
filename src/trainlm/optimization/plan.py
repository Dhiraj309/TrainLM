"""Serializable, explainable optimization execution-plan schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal, Mapping

DecisionStatus = Literal["selected", "fallback", "skipped", "blocked"]
PlanStatus = Literal["ready", "noop", "blocked"]
OptimizationPolicy = Literal["disabled", "auto", "required"]


def _tuple_field(name: str, value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ProviderDecision:
    """Provider selection or explicit non-selection for one operation."""

    decision_id: str
    component: str
    operation: str
    status: DecisionStatus
    reason: str
    selected_provider: str | None = None
    requested_provider: str | None = None
    requirements: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("decision_id", "component", "operation", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Provider decision {name} cannot be empty.")
        if self.status not in {"selected", "fallback", "skipped", "blocked"}:
            raise ValueError(f"Unsupported provider decision status: {self.status}")
        object.__setattr__(
            self,
            "requirements",
            _tuple_field("requirements", self.requirements),
        )
        object.__setattr__(
            self,
            "evidence",
            _tuple_field("evidence", self.evidence),
        )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*self.requirements, *self.evidence)
        ):
            raise ValueError("Decision requirements and evidence cannot be empty.")
        for name in ("selected_provider", "requested_provider"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"Provider decision {name} cannot be empty.")
        if self.status in {"selected", "fallback"} and not self.selected_provider:
            raise ValueError(f"Decision status '{self.status}' requires a provider.")
        if self.status in {"skipped", "blocked"} and self.selected_provider:
            raise ValueError(f"Decision status '{self.status}' cannot select a provider.")
        if self.status == "fallback" and not self.requested_provider:
            raise ValueError("Fallback decisions must record the requested provider.")
        if (
            self.status == "fallback"
            and self.requested_provider == self.selected_provider
        ):
            raise ValueError("Fallback decisions must select a different provider.")


@dataclass(frozen=True, slots=True)
class ModelTransformation:
    """Declarative reversible model transformation; contains no callable."""

    transform_id: str
    component: str
    provider: str
    target_paths: tuple[str, ...]
    inverse_transform_id: str
    reason: str
    parameter_layout_change: bool = False

    def __post_init__(self) -> None:
        for name in (
            "transform_id",
            "component",
            "provider",
            "inverse_transform_id",
            "reason",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Model transformation {name} cannot be empty.")
        object.__setattr__(
            self,
            "target_paths",
            _tuple_field("target_paths", self.target_paths),
        )
        if not self.target_paths or any(
            not isinstance(path, str) or not path.strip()
            for path in self.target_paths
        ):
            raise ValueError("Model transformations require non-empty target paths.")
        if len(self.target_paths) != len(set(self.target_paths)):
            raise ValueError("Model transformation target paths must be unique.")
        if self.transform_id == self.inverse_transform_id:
            raise ValueError("A transformation and its inverse must have different IDs.")


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Versioned provider and transformation plan produced before application."""

    schema_version: int
    plan_id: str
    status: PlanStatus
    policy: OptimizationPolicy
    capability_fingerprint: str
    backend: str
    precision: str
    decisions: tuple[ProviderDecision, ...] = field(default_factory=tuple)
    transformations: tuple[ModelTransformation, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ExecutionPlan supports schema_version=1 only.")
        for name in ("plan_id", "capability_fingerprint", "backend", "precision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Execution plan {name} cannot be empty.")
        if len(self.capability_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.capability_fingerprint
        ):
            raise ValueError("Capability fingerprint must be lowercase SHA-256.")
        if self.status not in {"ready", "noop", "blocked"}:
            raise ValueError(f"Unsupported execution plan status: {self.status}")
        if self.policy not in {"disabled", "auto", "required"}:
            raise ValueError(f"Unsupported optimization policy: {self.policy}")
        object.__setattr__(
            self,
            "decisions",
            _tuple_field("decisions", self.decisions),
        )
        object.__setattr__(
            self,
            "transformations",
            _tuple_field("transformations", self.transformations),
        )
        object.__setattr__(
            self,
            "warnings",
            _tuple_field("warnings", self.warnings),
        )
        object.__setattr__(self, "errors", _tuple_field("errors", self.errors))
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*self.warnings, *self.errors)
        ):
            raise ValueError("Execution plan warnings and errors cannot be empty.")
        if any(not isinstance(item, ProviderDecision) for item in self.decisions):
            raise ValueError("Plan decisions must be ProviderDecision objects.")
        if any(
            not isinstance(item, ModelTransformation)
            for item in self.transformations
        ):
            raise ValueError(
                "Plan transformations must be ModelTransformation objects."
            )
        decision_ids = [decision.decision_id for decision in self.decisions]
        transform_ids = [item.transform_id for item in self.transformations]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("Execution plan decision IDs must be unique.")
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("Execution plan transformation IDs must be unique.")
        if self.status == "blocked" and not self.errors:
            raise ValueError("Blocked execution plans must explain their errors.")
        if self.status != "blocked" and self.errors:
            raise ValueError("Only blocked execution plans may contain errors.")
        if self.status == "noop" and self.transformations:
            raise ValueError("No-op execution plans cannot contain transformations.")
        if self.status == "noop" and any(
            decision.status in {"selected", "fallback"}
            for decision in self.decisions
        ):
            raise ValueError("No-op execution plans cannot select providers.")
        if any(decision.status == "blocked" for decision in self.decisions):
            if self.status != "blocked":
                raise ValueError("A blocked decision requires a blocked plan.")

    @property
    def is_executable(self) -> bool:
        return self.status in {"ready", "noop"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def explain(self) -> str:
        lines = [
            f"Execution plan {self.plan_id}",
            f"Status: {self.status}",
            f"Backend: {self.backend} ({self.precision})",
            f"Policy: {self.policy}",
            f"Capabilities: {self.capability_fingerprint}",
            "Provider decisions:",
        ]
        if self.decisions:
            for decision in self.decisions:
                provider = decision.selected_provider or "none"
                requested = decision.requested_provider or "auto"
                requirements = ", ".join(decision.requirements) or "none"
                evidence = ", ".join(decision.evidence) or "none"
                lines.append(
                    f"- {decision.component}.{decision.operation}: "
                    f"{decision.status}; requested={requested}; "
                    f"selected={provider}; {decision.reason}; "
                    f"requirements={requirements}; evidence={evidence}"
                )
        else:
            lines.append("- none")
        lines.append("Model transformations:")
        if self.transformations:
            for transform in self.transformations:
                targets = ", ".join(transform.target_paths)
                lines.append(
                    f"- {transform.transform_id}: {transform.provider} at "
                    f"{targets}; inverse={transform.inverse_transform_id}; "
                    f"layout_change={transform.parameter_layout_change}; "
                    f"{transform.reason}"
                )
        else:
            lines.append("- none")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in self.errors)
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionPlan":
        values = dict(data)
        values["decisions"] = tuple(
            ProviderDecision(**decision) for decision in values.get("decisions", ())
        )
        values["transformations"] = tuple(
            ModelTransformation(**transform)
            for transform in values.get("transformations", ())
        )
        values["warnings"] = tuple(values.get("warnings", ()))
        values["errors"] = tuple(values.get("errors", ()))
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "ExecutionPlan":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Execution plan JSON root must be an object.")
        return cls.from_dict(data)
