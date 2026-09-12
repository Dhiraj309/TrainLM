import pytest
import torch
from torch import nn

from trainlm.optimization import (
    ExecutionPlan,
    ModelTransformation,
    ProviderDecision,
)

from .test_capabilities import capabilities


def plan():
    report = capabilities()
    return ExecutionPlan(
        schema_version=1,
        plan_id="plan-001",
        status="ready",
        policy="auto",
        capability_fingerprint=report.fingerprint,
        backend="pytorch-xla",
        precision="bf16",
        decisions=(
            ProviderDecision(
                decision_id="attention-provider",
                component="attention",
                operation="forward_backward",
                status="fallback",
                requested_provider="pallas_splash",
                selected_provider="torch_sdpa",
                reason="Pallas provider is not certified for this mask.",
                requirements=("causal_mask", "backward"),
                evidence=("capability.attention",),
            ),
            ProviderDecision(
                decision_id="projection-provider",
                component="projections",
                operation="transform",
                status="selected",
                selected_provider="trainlm.qkv_pack",
                reason="Packed QKV provider is compatible.",
                evidence=("capability.projections",),
            ),
        ),
        transformations=(
            ModelTransformation(
                transform_id="pack-qkv",
                component="projections",
                provider="trainlm.qkv_pack",
                target_paths=("model.layers.*.self_attn",),
                inverse_transform_id="unpack-qkv",
                reason="Packed QKV is supported by the selected provider.",
                parameter_layout_change=True,
            ),
        ),
        warnings=("Attention uses the portable fallback.",),
    )


def test_execution_plan_round_trips_and_explains_every_decision():
    original = plan()
    restored = ExecutionPlan.from_json(original.to_json())
    explanation = restored.explain()

    assert restored == original
    assert restored.is_executable
    assert "attention.forward_backward: fallback" in explanation
    assert "requested=pallas_splash" in explanation
    assert "selected=torch_sdpa" in explanation
    assert "pack-qkv: trainlm.qkv_pack" in explanation
    assert "inverse=unpack-qkv" in explanation
    assert "Attention uses the portable fallback." in explanation


def test_schema_operations_do_not_mutate_a_model():
    model = nn.Linear(4, 4)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    current_plan = plan()
    current_plan.to_json()
    current_plan.explain()

    assert model.training
    assert all(
        torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_blocked_and_noop_plan_invariants_are_enforced():
    fingerprint = capabilities().fingerprint

    with pytest.raises(ValueError, match="explain their errors"):
        ExecutionPlan(
            schema_version=1,
            plan_id="blocked",
            status="blocked",
            policy="required",
            capability_fingerprint=fingerprint,
            backend="pytorch-xla",
            precision="bf16",
        )

    with pytest.raises(ValueError, match="No-op"):
        ExecutionPlan(
            schema_version=1,
            plan_id="noop",
            status="noop",
            policy="disabled",
            capability_fingerprint=fingerprint,
            backend="pytorch",
            precision="fp32",
            transformations=plan().transformations,
        )


@pytest.mark.parametrize("schema_version", (True, False))
def test_execution_plan_rejects_boolean_schema_versions(schema_version):
    values = plan().to_dict()
    values["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version=1 only"):
        ExecutionPlan.from_dict(values)


@pytest.mark.parametrize("field", ("decisions", "transformations", "warnings", "errors"))
def test_execution_plan_requires_complete_serialized_fields(field):
    values = plan().to_dict()
    del values[field]

    with pytest.raises(ValueError, match=rf"missing=.*{field}"):
        ExecutionPlan.from_dict(values)


def test_execution_plan_rejects_unknown_serialized_fields():
    values = plan().to_dict()
    values["typo"] = "ignored"

    with pytest.raises(ValueError, match="unknown=.*typo"):
        ExecutionPlan.from_dict(values)


@pytest.mark.parametrize("field", ("decisions", "transformations"))
def test_execution_plan_rejects_non_mapping_nested_entries(field):
    values = plan().to_dict()
    values[field] = ["not-a-mapping"]

    with pytest.raises(ValueError, match=rf"{field} must contain mappings"):
        ExecutionPlan.from_dict(values)


@pytest.mark.parametrize("field", ("decisions", "transformations"))
@pytest.mark.parametrize("change", ("missing", "unknown"))
def test_execution_plan_rejects_incomplete_nested_schemas(field, change):
    values = plan().to_dict()
    entry = values[field][0]
    if change == "missing":
        entry.pop(next(iter(entry)))
    else:
        entry["typo"] = "ignored"

    with pytest.raises(ValueError, match=rf"fields mismatch;.*{change}"):
        ExecutionPlan.from_dict(values)


@pytest.mark.parametrize("value", (0, 1, "false", None))
def test_model_transformation_requires_boolean_layout_flag(value):
    values = plan().to_dict()["transformations"][0]
    values["parameter_layout_change"] = value

    with pytest.raises(ValueError, match="parameter_layout_change must be a boolean"):
        ModelTransformation.from_dict(values)


def test_selected_decision_must_match_explicit_provider_request():
    with pytest.raises(ValueError, match="match their explicitly requested provider"):
        ProviderDecision(
            decision_id="mismatched-selection",
            component="attention",
            operation="forward_backward",
            status="selected",
            reason="Invalid selected-provider evidence.",
            selected_provider="torch-sdpa",
            requested_provider="pallas",
        )


def test_blocked_plan_cannot_retain_transformations():
    with pytest.raises(ValueError, match="Blocked execution plans cannot contain"):
        ExecutionPlan(
            schema_version=1,
            plan_id="blocked-with-transform",
            status="blocked",
            policy="required",
            capability_fingerprint=capabilities().fingerprint,
            backend="pytorch-xla",
            precision="bf16",
            transformations=plan().transformations,
            errors=("Required provider is unavailable.",),
        )


def test_ready_plan_must_select_at_least_one_provider():
    with pytest.raises(ValueError, match="Ready execution plans must select"):
        ExecutionPlan(
            schema_version=1,
            plan_id="empty-ready-plan",
            status="ready",
            policy="auto",
            capability_fingerprint=capabilities().fingerprint,
            backend="pytorch-xla",
            precision="bf16",
        )


@pytest.mark.parametrize(
    ("component", "provider"),
    (("attention", "trainlm.qkv_pack"), ("projections", "unselected-provider")),
)
def test_transformations_require_matching_selected_provider(component, provider):
    values = plan().to_dict()
    values["transformations"][0]["component"] = component
    values["transformations"][0]["provider"] = provider

    with pytest.raises(ValueError, match="matching selected provider decision"):
        ExecutionPlan.from_dict(values)
