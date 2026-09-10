"""Stable human and JSON reporting for optimization decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal

from .adapters import AdapterResolution
from .capabilities import ModelCapabilities
from .plan import ExecutionPlan

CertificationStatus = Literal["unverified", "compatible", "optimized", "certified"]


@dataclass(frozen=True, slots=True)
class OptimizationExplanation:
    """Complete, serializable audit of a selected execution path.

    Missing planning evidence is represented explicitly instead of being
    interpreted as an optimized or certified path.
    """

    backend: str
    precision: str
    selected_path: str
    certification: CertificationStatus = "unverified"
    capabilities: ModelCapabilities | None = None
    adapter_resolution: AdapterResolution | None = None
    execution_plan: ExecutionPlan | None = None
    graph_evidence: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("backend", "precision", "selected_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Explanation {name} cannot be empty.")
        if self.certification not in {
            "unverified", "compatible", "optimized", "certified"
        }:
            raise ValueError(f"Unsupported certification state: {self.certification}")
        for name in ("graph_evidence", "limitations"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValueError(f"{name} must be a tuple of non-empty strings.")

    @property
    def strict_errors(self) -> tuple[str, ...]:
        errors = list(self.execution_plan.errors if self.execution_plan else ())
        if self.execution_plan is not None and not self.execution_plan.is_executable:
            errors.append("The optimization execution plan is not executable.")
        if self.capabilities is not None:
            unknown = [
                name for name in self.capabilities.component_names
                if self.capabilities.component(name).status in {"unknown", "unsupported"}
            ]
            if unknown:
                errors.append("Unproven model capabilities: " + ", ".join(unknown) + ".")
        return tuple(dict.fromkeys(errors))

    def require_supported(self) -> None:
        """Fail before launch when the report contains unsupported evidence."""

        errors = self.strict_errors
        if errors:
            raise RuntimeError("Strict optimization requirements failed: " + " ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = 1
        data["support_level"] = self.certification
        data["strict_errors"] = list(self.strict_errors)
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_text(self) -> str:
        lines = [
            "TrainLM optimization explanation",
            f"Path: {self.selected_path}",
            f"Backend: {self.backend} ({self.precision})",
            f"Certification: {self.certification}",
            "Capabilities: " + (
                self.capabilities.fingerprint if self.capabilities else "not inspected"
            ),
            "Adapter: " + (
                self.adapter_resolution.selected.adapter_id
                if self.adapter_resolution and self.adapter_resolution.selected
                else "none"
            ),
            "Execution plan: " + (
                f"{self.execution_plan.plan_id} ({self.execution_plan.status})"
                if self.execution_plan else "none"
            ),
            "Graph evidence: " + (", ".join(self.graph_evidence) or "none"),
            "Limitations: " + ("; ".join(self.limitations) or "none"),
        ]
        if self.execution_plan:
            lines.append(self.execution_plan.explain())
        if self.strict_errors:
            lines.append("Strict errors: " + " ".join(self.strict_errors))
        return "\n".join(lines)


__all__ = ["CertificationStatus", "OptimizationExplanation"]
