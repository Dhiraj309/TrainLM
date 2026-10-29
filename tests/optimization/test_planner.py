"""Provider declarations below are test fixtures, not installed kernel claims."""

from dataclasses import replace
import json
from pathlib import Path

import jsonschema
import pytest
import torch

from trainlm.optimization import (
    CapabilityFact, CapabilityRequirement, ComponentCapability, ExecutionPlan, ModelTransformation,
    OperationRequest, PlanningContext, ProviderDefinition, ShapeConstraint, VersionRequirement,
    plan_optimizations,
)
from .test_capabilities import capabilities
from .test_adapters import FixtureModel, adapter


REQUEST = OperationRequest("attention", "forward_backward")


def provider(provider_id="fixture.attention", **changes):
    definition = ProviderDefinition(
        provider_id=provider_id, operation="forward_backward",
        capability=CapabilityRequirement("attention", ("gqa",), (CapabilityFact("heads", 8),)),
        backends=("cpu", "xla"), precisions=("bf16",), masks=("causal",), supports_backward=True,
        shapes=(ShapeConstraint("head_dim", minimum=64, maximum=256, multiple_of=64),),
        versions=(VersionRequirement("torch", ">=2.9,<2.10"),),
        transformations=(ModelTransformation(
            transform_id="fixture.pack", component="attention", provider=provider_id,
            target_paths=("model.layers.0.self_attn",), inverse_transform_id="fixture.unpack",
            reason="Fixture eligibility only; no implementation is applied.",
        ),),
    )
    return replace(definition, **changes)


def context(**changes):
    return replace(PlanningContext(backend="xla", precision="bf16", mask_kind="causal",
                                   shapes={"head_dim": 128}, versions={"torch": "2.9.0+cpu"}), **changes)


def make_plan(*, report=None, execution=None, providers=None, request=REQUEST, **options):
    return plan_optimizations(report or capabilities(), context=execution or context(),
        requests=(request,), providers=(provider(),) if providers is None else providers, **options)


def test_selected_plan_is_pure_stable_round_trippable_and_explains_inverse(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("hardware discovery")))
    rng = torch.get_rng_state().clone()
    plan = make_plan()
    assert plan.status == "ready" and plan.decisions[0].status == "selected"
    assert plan.decisions[0].selected_provider == "fixture.attention"
    assert "inverse=fixture.unpack" in plan.explain()
    assert "torch=2.9.0+cpu" in plan.explain()
    assert "head_dim=128" in plan.explain()
    assert ExecutionPlan.from_json(plan.to_json()) == plan
    assert plan.to_json() == make_plan().to_json()
    assert torch.equal(rng, torch.get_rng_state())
    schema = Path(__file__).parents[2] / "schemas/optimization/execution_plan_v1.schema.json"
    jsonschema.validate(json.loads(plan.to_json()), json.loads(schema.read_text()))


@pytest.mark.parametrize("execution, definition, report, reason", [
    (context(backend="cuda"), None, None, "backend cuda"),
    (context(precision="fp16"), None, None, "precision fp16"),
    (context(mask_kind="padding"), None, None, "mask padding"),
    (None, provider(supports_backward=False), None, "backward is unsupported"),
    (context(shapes={}), None, None, "missing shape dimension"),
    (context(shapes={"head_dim": 96}), None, None, "multiple_of=64"),
    (context(shapes={"head_dim": 512}), None, None, "maximum=256"),
    (context(versions={}), None, None, "missing version"),
    (context(versions={"torch": "2.10.0"}), None, None, "does not satisfy"),
    (None, None, replace(capabilities(), attention=ComponentCapability.unknown()), "status unknown"),
    (None, None, replace(capabilities(), attention=replace(capabilities().attention, kind="mha")), "kind mha"),
    (None, provider(capability=CapabilityRequirement("attention", ("gqa",), (CapabilityFact("heads", 4),))), None, "heads must equal 4"),
    (None, None, replace(capabilities(), attention=replace(capabilities().attention, status="inferred")), "status inferred"),
    (None, provider(adapter_ids=("fixture.adapter",)), None, "requires adapter"),
])
def test_each_eligibility_boundary_explains_reference_fallback(execution, definition, report, reason):
    plan = make_plan(execution=execution, providers=(definition or provider(),), report=report)
    decision = plan.decisions[0]
    assert decision.status == "fallback" and decision.selected_provider == "model_reference"
    assert reason in decision.reason
    assert not plan.transformations


def test_inference_and_adapter_requirements_need_explicit_opt_in():
    report = replace(capabilities(), attention=replace(capabilities().attention, status="inferred"))
    definition = provider(capability=replace(provider().capability, allow_inferred=True), adapter_ids=("fixture.adapter",))
    plan = make_plan(report=report, providers=(definition,), execution=context(adapter_id="fixture.adapter"))
    assert plan.decisions[0].status == "selected"


def test_explicit_requests_and_strict_policy_never_silently_substitute():
    request = replace(REQUEST, provider_id="missing")
    fallback = make_plan(request=request)
    assert fallback.decisions[0].status == "fallback"
    assert fallback.decisions[0].requested_provider == "missing"
    assert fallback.decisions[0].selected_provider == "fixture.attention"
    for options in ({"policy": "required"}, {"allow_fallbacks": False}):
        blocked = make_plan(request=request, **options)
        assert blocked.status == "blocked" and not blocked.is_executable
        assert blocked.transformations == ()
        assert "missing: not registered" in blocked.errors[0]
    assert make_plan(policy="required").is_executable
    assert make_plan(providers=(), policy="required").status == "blocked"
    with pytest.raises(ValueError, match="cannot allow fallbacks"):
        make_plan(policy="required", allow_fallbacks=True)


def test_disabled_policy_and_empty_registry_preserve_reference_behavior():
    disabled = make_plan(policy="disabled")
    assert disabled.status == "noop" and disabled.decisions[0].status == "skipped"
    assert disabled.transformations == ()
    generic = make_plan(providers=())
    assert generic.decisions[0].selected_provider == "model_reference"
    assert "No providers are registered" in generic.explain()


def test_selection_and_plan_ids_are_deterministic_across_registration_order():
    first = provider("fixture.a", transformations=(), priority=1)
    second = provider("fixture.b", transformations=(), priority=1)
    forward = make_plan(providers=(second, first))
    backward = make_plan(providers=(first, second))
    assert forward == backward
    assert forward.decisions[0].selected_provider == "fixture.a"
    higher = replace(second, priority=2)
    assert make_plan(providers=(first, higher)).decisions[0].selected_provider == "fixture.b"
    explicit = make_plan(providers=(first, higher), request=replace(REQUEST, provider_id="fixture.a"))
    assert explicit.decisions[0].selected_provider == "fixture.a"
    assert forward.plan_id != make_plan(providers=(first, second), execution=context(versions={"torch": "2.9.1"})).plan_id


def test_stale_capability_binding_and_conflicting_transforms_block_whole_plan():
    stale = make_plan(expected_capability_fingerprint="0" * 64)
    assert stale.status == "blocked" and not stale.transformations
    other = ProviderDefinition(
        provider_id="fixture.projection", operation="matmul",
        capability=CapabilityRequirement("projections", ("separate_qkv",)),
        backends=("xla",), precisions=("bf16",), masks=("causal",), supports_backward=True,
        transformations=(replace(provider().transformations[0], provider="fixture.projection", component="projections"),),
    )
    plan = plan_optimizations(capabilities(), context=context(), providers=(provider(), other),
                             requests=(REQUEST, OperationRequest("projections", "matmul")))
    assert plan.status == "blocked" and not plan.transformations
    assert "Conflicting transformation declarations" in plan.errors[0]


def test_context_copies_mutable_inputs_and_duplicate_requests_are_rejected():
    shapes, versions = {"head_dim": 128}, {"Torch": "2.9.0"}
    execution = PlanningContext("xla", "bf16", "causal", shapes=shapes, versions=versions)
    before = make_plan(execution=execution)
    shapes["head_dim"] = 1
    versions["Torch"] = "99.0"
    assert make_plan(execution=execution) == before
    with pytest.raises(ValueError, match="requests must be unique"):
        plan_optimizations(capabilities(), context=context(), requests=(REQUEST, REQUEST))


def test_optional_adapter_resolution_drives_planning_without_changing_model():
    from trainlm.optimization import AdapterRegistry
    model = FixtureModel().train()
    inputs = torch.tensor([[1, 2, 3]])
    expected = model(inputs).detach().clone()
    definition = ProviderDefinition(
        provider_id="fixture.residual", operation="forward_backward",
        capability=CapabilityRequirement("residual", ("serial",), (CapabilityFact("scale", 1.0),)),
        backends=("cpu",), precisions=("fp32",), masks=("none",), supports_backward=True,
        adapter_ids=("fixture.serial-v1",),
    )
    versions = {"torch": "2.9.0+cpu"}
    for registry, status in ((AdapterRegistry((adapter(),)), "selected"), (AdapterRegistry(), "fallback")):
        resolution = registry.resolve(model, versions=versions)
        plan = plan_optimizations(
            resolution.capabilities,
            context=PlanningContext("cpu", "fp32", "none", versions=versions, adapter_id=resolution.adapter_id),
            requests=(OperationRequest("residual", "forward_backward"),), providers=(definition,),
        )
        assert plan.decisions[0].status == status
        assert plan.capability_fingerprint == resolution.capabilities.fingerprint
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        assert model.training and model.head.weight is model.embedding.weight


@pytest.mark.parametrize("policy,providers", [("disabled", (provider(),)), ("auto", ()), ("required", ())])
def test_generated_noop_fallback_and_blocked_plans_obey_existing_json_schema(policy, providers):
    plan = make_plan(policy=policy, providers=providers)
    schema = Path(__file__).parents[2] / "schemas/optimization/execution_plan_v1.schema.json"
    jsonschema.validate(json.loads(plan.to_json()), json.loads(schema.read_text()))
    assert ExecutionPlan.from_json(plan.to_json()) == plan


def test_missing_semantic_evidence_and_boolean_fact_coercion_are_rejected():
    missing = replace(capabilities(), attention=replace(capabilities().attention, evidence=()))
    assert "no semantic evidence" in make_plan(report=missing).decisions[0].reason
    definition = ProviderDefinition(
        provider_id="fixture.head", operation="forward_backward",
        capability=CapabilityRequirement("lm_head", ("linear",), (CapabilityFact("tied", 1),)),
        backends=("xla",), precisions=("bf16",), masks=("causal",), supports_backward=True,
    )
    assert make_plan(providers=(definition,), request=OperationRequest("lm_head", "forward_backward")).decisions[0].status == "fallback"
    with pytest.raises(ValueError, match="positive integers"):
        context(shapes={"head_dim": True})


def test_runtime_shape_cannot_contradict_an_inspected_model_dimension():
    plan = make_plan(execution=context(shapes={"head_dim": 64}))
    assert plan.decisions[0].status == "fallback"
    assert "contradicts inspected fact 128" in plan.decisions[0].reason
