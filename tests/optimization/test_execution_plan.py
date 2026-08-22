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
