import pytest

from trainlm.optimization import (
    ModelTransformation,
    OperationRequest,
    OptimizationPlanner,
    ProviderSpec,
)

from .test_capabilities import capabilities


def _providers():
    transform = ModelTransformation(
        transform_id="pack-qkv",
        component="projections",
        provider="xla-qkv",
        target_paths=("model.layers.*.self_attn",),
        inverse_transform_id="unpack-qkv",
        reason="Provider consumes packed projections.",
        parameter_layout_change=True,
    )
    return (
        ProviderSpec(
            "xla-qkv", "projections", "forward_backward",
            ("pytorch-xla",), ("bf16",), ("separate_qkv",),
            ("backward", "causal"), (transform,), priority=10,
        ),
        ProviderSpec(
            "torch-reference", "projections", "forward_backward",
            ("pytorch", "pytorch-xla"), ("fp32", "bf16"),
            ("separate_qkv",), ("backward", "causal"), fallback=True,
        ),
    )


def _request(requested_provider=None):
    return OperationRequest(
        "projections", "forward_backward", ("backward", "causal"),
        requested_provider,
    )


def test_auto_plan_selects_highest_priority_eligible_provider_deterministically():
    planner = OptimizationPlanner(reversed(_providers()))
    first = planner.plan(
        capabilities(), backend="pytorch-xla", precision="bf16",
        policy="auto", requests=(_request(),),
    )
    second = planner.plan(
        capabilities(), backend="pytorch-xla", precision="bf16",
        policy="auto", requests=(_request(),),
    )

    assert first == second
    assert first.status == "ready"
    assert first.decisions[0].selected_provider == "xla-qkv"
    assert first.transformations[0].inverse_transform_id == "unpack-qkv"


def test_auto_plan_uses_explained_portable_fallback():
    plan = OptimizationPlanner(_providers()).plan(
        capabilities(), backend="pytorch", precision="fp32",
        policy="auto", requests=(_request(),),
    )

    assert plan.decisions[0].status == "fallback"
    assert plan.decisions[0].selected_provider == "torch-reference"
    assert "fallback" in plan.warnings[0]


def test_explicit_fallback_provider_is_a_successful_selection():
    plan = OptimizationPlanner(_providers()).plan(
        capabilities(),
        backend="pytorch",
        precision="fp32",
        policy="auto",
        requests=(_request("torch-reference"),),
    )

    assert plan.status == "ready"
    assert plan.decisions[0].status == "selected"
    assert plan.decisions[0].selected_provider == "torch-reference"
    assert plan.decisions[0].requested_provider == "torch-reference"
    assert plan.warnings == ()


def test_required_policy_rejects_automatic_fallback_only_provider():
    fallback = _providers()[1]
    plan = OptimizationPlanner((fallback,)).plan(
        capabilities(),
        backend="pytorch",
        precision="fp32",
        policy="required",
        requests=(_request(),),
    )

    assert plan.status == "blocked"
    assert plan.decisions[0].status == "blocked"
    assert plan.decisions[0].selected_provider is None
    assert "fallback-only provider" in plan.decisions[0].reason
    assert plan.transformations == ()


def test_required_or_explicit_unsupported_provider_blocks_before_mutation():
    planner = OptimizationPlanner(_providers())
    plan = planner.plan(
        capabilities(), backend="cuda", precision="fp16",
        policy="required", requests=(_request("xla-qkv"),),
    )

    assert plan.status == "blocked"
    assert not plan.is_executable
    assert plan.transformations == ()
    assert "backend 'cuda'" in plan.decisions[0].evidence[0]


def test_disabled_policy_is_a_noop_even_when_providers_match():
    plan = OptimizationPlanner(_providers()).plan(
        capabilities(), backend="pytorch-xla", precision="bf16",
        policy="disabled", requests=(_request(),),
    )

    assert plan.status == "noop"
    assert plan.decisions[0].status == "skipped"
    assert plan.transformations == ()


@pytest.mark.parametrize(
    ("argument", "value", "error", "message"),
    [
        ("capabilities", object(), TypeError, "must be ModelCapabilities"),
        ("backend", "", ValueError, "backend must be a non-empty string"),
        ("precision", " ", ValueError, "precision must be a non-empty string"),
        ("policy", "sometimes", ValueError, "Unsupported optimization policy"),
        ("requests", (object(),), TypeError, "must contain OperationRequest"),
        (
            "adapter_resolution",
            object(),
            TypeError,
            "must be AdapterResolution or None",
        ),
    ],
)
def test_planner_rejects_malformed_boundary_inputs(
    argument,
    value,
    error,
    message,
):
    arguments = {
        "capabilities": capabilities(),
        "backend": "pytorch-xla",
        "precision": "bf16",
        "policy": "auto",
        "requests": (_request(),),
    }
    arguments[argument] = value

    with pytest.raises(error, match=message):
        OptimizationPlanner(_providers()).plan(**arguments)
