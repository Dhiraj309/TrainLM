"""Stable compatibility explanation for the generic Hugging Face path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal, Mapping

from trainlm.optimization import (
    ComponentCapability,
    ExecutionPlan,
    ModelCapabilities,
    ProviderDecision,
)

from .huggingface import LoadedCausalLM

SupportLevel = Literal["compatible", "optimized", "certified"]


@dataclass(frozen=True, slots=True)
class ModelCompatibilityExplanation:
    """Serializable support statement for one acquired causal LM."""

    schema_version: int
    support_level: SupportLevel
    selected_path: str
    adapter: str | None
    capabilities: ModelCapabilities
    execution_plan: ExecutionPlan
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "ModelCompatibilityExplanation supports schema_version=1 only."
            )
        if self.support_level not in {"compatible", "optimized", "certified"}:
            raise ValueError(f"Unsupported support level: {self.support_level}")
        if not isinstance(self.selected_path, str) or not self.selected_path.strip():
            raise ValueError("Selected compatibility path cannot be empty.")
        if self.adapter is not None and (
            not isinstance(self.adapter, str) or not self.adapter.strip()
        ):
            raise ValueError("Compatibility adapter cannot be empty.")
        if not isinstance(self.capabilities, ModelCapabilities):
            raise ValueError("capabilities must be ModelCapabilities.")
        if not isinstance(self.execution_plan, ExecutionPlan):
            raise ValueError("execution_plan must be an ExecutionPlan.")
        if (
            self.execution_plan.capability_fingerprint
            != self.capabilities.fingerprint
        ):
            raise ValueError("Execution plan does not match model capabilities.")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if any(not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise ValueError("Compatibility evidence cannot contain empty text.")

    @property
    def fallbacks(self) -> tuple[ProviderDecision, ...]:
        return tuple(
            decision
            for decision in self.execution_plan.decisions
            if decision.status == "fallback"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def explain(self) -> str:
        lines = [
            "TrainLM model compatibility",
            f"Support level: {self.support_level.capitalize()}",
            f"Model: {self.capabilities.model_type} "
            f"({self.capabilities.model_class})",
            f"Selected path: {self.selected_path}",
            f"Adapter: {self.adapter or 'none'}",
            "Capabilities:",
        ]
        for name in self.capabilities.component_names:
            capability = self.capabilities.component(name)
            kind = capability.kind or "unknown"
            lines.append(f"- {name}: {capability.status}; kind={kind}")
        lines.append("Fallbacks:")
        if self.fallbacks:
            for decision in self.fallbacks:
                lines.append(
                    f"- {decision.component}.{decision.operation}: "
                    f"requested={decision.requested_provider}; "
                    f"selected={decision.selected_provider}; {decision.reason}"
                )
        else:
            lines.append("- none")
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in self.evidence)
        if self.capabilities.warnings or self.execution_plan.warnings:
            lines.append("Warnings:")
            lines.extend(
                f"- {item}"
                for item in (
                    *self.capabilities.warnings,
                    *self.execution_plan.warnings,
                )
            )
        return "\n".join(lines)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ModelCompatibilityExplanation":
        values = dict(data)
        values["capabilities"] = ModelCapabilities.from_dict(
            values["capabilities"]
        )
        values["execution_plan"] = ExecutionPlan.from_dict(
            values["execution_plan"]
        )
        values["evidence"] = tuple(values.get("evidence", ()))
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "ModelCompatibilityExplanation":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Compatibility explanation JSON root must be an object.")
        return cls.from_dict(data)


def explain_huggingface_compatibility(
    loaded: LoadedCausalLM,
) -> ModelCompatibilityExplanation:
    """Describe the non-mutating generic path selected for an HF causal LM."""

    if not isinstance(loaded, LoadedCausalLM):
        raise TypeError("loaded must be a LoadedCausalLM.")

    unknown = {
        name: ComponentCapability.unknown(
            "Generic compatibility does not infer architecture structure; "
            "the M8 capability inspector must provide evidence."
        )
        for name in ModelCapabilities.COMPONENT_NAMES
    }
    capabilities = ModelCapabilities(
        schema_version=1,
        model_type=loaded.metadata.model_type,
        model_class=loaded.metadata.model_class,
        config_class=loaded.metadata.config_class,
        source_provider="huggingface",
        architectures=loaded.metadata.architectures,
        **unknown,
        warnings=(
            "Structural capabilities are intentionally unknown on the generic path.",
        ),
    )
    decisions = (
        ProviderDecision(
            decision_id="model-forward",
            component="model",
            operation="forward_backward",
            status="selected",
            reason="Preserve the official AutoModelForCausalLM implementation.",
            selected_provider="huggingface",
            evidence=(loaded.metadata.model_class,),
        ),
        ProviderDecision(
            decision_id="batch-dispatch",
            component="batch",
            operation="dispatch",
            status="selected",
            reason="Filter batches using the model's declared forward signature.",
            selected_provider="trainlm.forward_signature",
        ),
        ProviderDecision(
            decision_id="causal-task",
            component="loss",
            operation="forward_backward",
            status="selected",
            reason="Use the backend-neutral causal-language-model task protocol.",
            selected_provider="trainlm.causal_lm",
        ),
        ProviderDecision(
            decision_id="architecture-optimization",
            component="model",
            operation="optimization",
            status="fallback",
            reason=(
                "No evidence-backed architecture optimization plan has been "
                "applied; preserve the generic Hugging Face model."
            ),
            requested_provider="trainlm.architecture_optimized",
            selected_provider="huggingface.generic",
            requirements=("M8 capability inspection and reversible planning",),
        ),
    )
    precision = loaded.metadata.resolved_dtype or "unknown"
    plan = ExecutionPlan(
        schema_version=1,
        plan_id="huggingface-generic-dense-ar-v1",
        status="ready",
        policy="auto",
        capability_fingerprint=capabilities.fingerprint,
        backend="unprepared",
        precision=precision,
        decisions=decisions,
        warnings=(
            "Compatible does not mean TPU Optimized or hardware Certified.",
        ),
    )
    return ModelCompatibilityExplanation(
        schema_version=1,
        support_level="compatible",
        selected_path="huggingface.generic_causal_lm",
        adapter=None,
        capabilities=capabilities,
        execution_plan=plan,
        evidence=(
            "Loaded through the generic Hugging Face causal-model provider.",
            "M2 CPU conformance covers the representative dense-AR matrix.",
        ),
    )
