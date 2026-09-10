import pytest
import torch
from torch import nn

from trainlm.optimization import (
    ExecutionPlan,
    ModelTransformation,
    ModelTransformRegistry,
    RematerializationMeasurement,
    RematerializationPolicy,
    module_rematerialization_transform_handler,
    select_rematerialization_policy,
)

from .test_capabilities import capabilities


def measurement(name, scopes, step, hbm, **values):
    defaults = dict(gradients_match=True, graph_stable=True)
    return RematerializationMeasurement(
        policy=RematerializationPolicy(
            policy_id=name,
            scopes=scopes,
            apply_before_fsdp=scopes != ("none",),
        ),
        step_seconds=step,
        peak_hbm_gib=hbm,
        **{**defaults, **values},
    )


def test_selects_lowest_hbm_policy_inside_slowdown_budget():
    result = select_rematerialization_policy(
        (
            measurement("none", ("none",), 1.0, 12.0),
            measurement("block", ("block",), 1.08, 8.0),
            measurement("attention", ("attention",), 1.04, 9.0),
        ),
        maximum_slowdown=0.10,
    )
    assert result.selected.policy.policy_id == "block"


def test_rejections_explain_parity_graph_and_step_failures():
    result = select_rematerialization_policy(
        (
            measurement("none", ("none",), 1.0, 12.0),
            measurement("bad-gradient", ("mlp",), 1.0, 4.0, gradients_match=False),
            measurement("bad-graph", ("attention",), 1.0, 4.0, graph_stable=False),
            measurement("too-slow", ("block",), 1.2, 4.0),
        )
    )
    assert result.selected.policy.policy_id == "none"
    assert result.rejected == (
        ("bad-gradient", "gradient parity failed"),
        ("bad-graph", "compiled graph is unstable"),
        ("too-slow", "step-time slowdown exceeds budget"),
    )


def test_selection_is_independent_of_measurement_order():
    values = (
        measurement("none", ("none",), 1.0, 12.0),
        measurement("mlp", ("mlp",), 1.05, 8.0),
        measurement("attention", ("attention",), 1.05, 8.0),
    )
    assert select_rematerialization_policy(values) == select_rematerialization_policy(
        tuple(reversed(values))
    )


def test_rematerialization_requires_pre_fsdp_application():
    with pytest.raises(ValueError, match="before FSDP"):
        RematerializationPolicy("block", ("block",), apply_before_fsdp=False)


def test_exactly_one_baseline_is_required():
    with pytest.raises(ValueError, match="Exactly one"):
        select_rematerialization_policy(
            (measurement("block", ("block",), 1.0, 8.0),)
        )


class _DecoderFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.block(inputs)


def _rematerialization_plan(*, component="block", layout_change=False):
    transformation = ModelTransformation(
        transform_id="rematerialize-module",
        component=component,
        provider="torch-checkpoint",
        target_paths=("block",),
        inverse_transform_id="restore-module-forward",
        reason="bound activation memory",
        parameter_layout_change=layout_change,
    )
    return ExecutionPlan(
        1,
        "rematerialization-plan",
        "ready",
        "auto",
        capabilities().fingerprint,
        "pytorch",
        "fp32",
        transformations=(transformation,),
    )


def test_module_rematerialization_preserves_output_gradient_and_state_keys():
    torch.manual_seed(19)
    reference = _DecoderFixture()
    transformed = _DecoderFixture()
    transformed.load_state_dict(reference.state_dict())
    reference_input = torch.randn(2, 4, requires_grad=True)
    transformed_input = reference_input.detach().clone().requires_grad_(True)
    original_keys = tuple(transformed.state_dict())

    registry = ModelTransformRegistry()
    policy = RematerializationPolicy("block", ("block",), True)
    registry.register(module_rematerialization_transform_handler(policy))
    transaction = registry.apply(transformed, _rematerialization_plan())

    expected = reference(reference_input)
    actual = transformed(transformed_input)
    assert torch.allclose(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert torch.allclose(transformed.block.weight.grad, reference.block.weight.grad)
    assert torch.allclose(transformed_input.grad, reference_input.grad)
    assert tuple(transformed.state_dict()) == original_keys
    transaction.commit()


def test_module_rematerialization_rollback_restores_original_forward():
    model = _DecoderFixture()
    original_forward = model.block.forward
    registry = ModelTransformRegistry()
    registry.register(
        module_rematerialization_transform_handler(
            RematerializationPolicy("block", ("block",), True)
        )
    )

    transaction = registry.apply(model, _rematerialization_plan())
    transaction.rollback()

    assert model.block.forward == original_forward


def test_module_rematerialization_rejects_scope_and_layout_mismatches():
    registry = ModelTransformRegistry()
    registry.register(
        module_rematerialization_transform_handler(
            RematerializationPolicy("attention", ("attention",), True)
        )
    )
    with pytest.raises(Exception, match="not enabled"):
        registry.apply(_DecoderFixture(), _rematerialization_plan())

    registry = ModelTransformRegistry()
    registry.register(
        module_rematerialization_transform_handler(
            RematerializationPolicy("block", ("block",), True)
        )
    )
    with pytest.raises(Exception, match="parameter layout"):
        registry.apply(
            _DecoderFixture(), _rematerialization_plan(layout_change=True)
        )


def test_loss_chunk_and_none_policies_do_not_create_module_handlers():
    with pytest.raises(ValueError, match="does not need"):
        module_rematerialization_transform_handler(
            RematerializationPolicy("none", ("none",), False)
        )
    with pytest.raises(ValueError, match="loss provider"):
        module_rematerialization_transform_handler(
            RematerializationPolicy("loss", ("loss_chunk",), True)
        )
