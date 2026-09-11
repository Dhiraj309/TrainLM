import pytest
import torch
from torch.optim import SGD

from trainlm.optimization import (
    GatedMLPProjectionSpec,
    PackedGatedMLPProjectionSpec,
)


def spec(**values):
    defaults = dict(
        prefix="layers.0.mlp",
        activation="swiglu",
        input_size=4,
        intermediate_size=8,
        gate_weight_key="gate_proj.weight",
        up_weight_key="up_proj.weight",
        packed_weight_key="gate_up_proj.weight",
        dtype="float32",
    )
    return GatedMLPProjectionSpec(**{**defaults, **values})


@pytest.mark.parametrize("activation", ["swiglu", "geglu"])
def test_gated_weight_round_trip(activation):
    canonical = {
        "gate_proj.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "up_proj.weight": torch.arange(32, 64, dtype=torch.float32).reshape(8, 4),
    }
    converter = spec(activation=activation).converter()
    packed = converter.to_transformed(canonical)
    restored = converter.to_canonical(packed)
    assert packed["gate_up_proj.weight"].shape == (16, 4)
    assert all(torch.equal(restored[key], value) for key, value in canonical.items())


def test_optional_bias_round_trip():
    layout = spec(
        gate_bias_key="gate_proj.bias",
        up_bias_key="up_proj.bias",
        packed_bias_key="gate_up_proj.bias",
    )
    canonical = {
        "gate_proj.weight": torch.zeros(8, 4),
        "up_proj.weight": torch.zeros(8, 4),
        "gate_proj.bias": torch.arange(8, dtype=torch.float32),
        "up_proj.bias": torch.arange(8, 16, dtype=torch.float32),
    }
    converter = layout.converter()
    restored = converter.to_canonical(converter.to_transformed(canonical))
    assert all(torch.equal(restored[key], value) for key, value in canonical.items())


def test_gelu_path_is_explicit_no_op():
    layout = GatedMLPProjectionSpec(
        prefix="layers.0.mlp",
        activation="gelu",
        input_size=4,
        intermediate_size=8,
    )
    assert layout.converter() is None


def test_gelu_rejects_accidental_gated_keys():
    with pytest.raises(ValueError, match="must remain unpacked"):
        spec(activation="gelu")


def test_partial_bias_layout_is_rejected():
    with pytest.raises(ValueError, match="bias keys must be configured together"):
        spec(gate_bias_key="gate_proj.bias")


@pytest.mark.parametrize("activation", ("swiglu", "geglu"))
def test_already_packed_gated_layout_is_an_explicit_no_op(activation):
    layout = PackedGatedMLPProjectionSpec(
        prefix="layers.0.mlp",
        activation=activation,
        input_size=4,
        intermediate_size=8,
        packed_weight_key="gate_up_proj.weight",
        packed_bias_key="gate_up_proj.bias",
        dtype="float32",
    )

    assert layout.weight_shape == (16, 4)
    assert layout.bias_shape == (16,)
    assert layout.converter() is None


def test_already_packed_gated_layout_rejects_plain_gelu():
    with pytest.raises(ValueError, match="require SwiGLU or GeGLU"):
        PackedGatedMLPProjectionSpec(
            prefix="layers.0.mlp",
            activation="gelu",
            input_size=4,
            intermediate_size=8,
            packed_weight_key="up_proj.weight",
        )


def test_already_packed_gated_layout_rejects_empty_weight_key():
    with pytest.raises(ValueError, match="packed_weight_key cannot be empty"):
        PackedGatedMLPProjectionSpec(
            prefix="layers.0.mlp",
            activation="swiglu",
            input_size=4,
            intermediate_size=8,
            packed_weight_key="",
        )

from copy import deepcopy
from torch import nn
from torch.nn import functional as F

from trainlm.optimization import (
    ExecutionPlan,
    ModelTransformation,
    ModelTransformRegistry,
    PackedGatedMLPProjection,
    gated_mlp_pack_transform_handler,
)
from .test_capabilities import capabilities


class _SeparateGatedMLP(nn.Module):
    def __init__(self, *, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=bias)
        self.up_proj = nn.Linear(4, 8, bias=bias)

    def forward(self, hidden_states):
        return F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)


class _MLPFixture(nn.Module):
    def __init__(self, *, bias=False):
        super().__init__()
        self.input_projection = _SeparateGatedMLP(bias=bias)

    def forward(self, hidden_states):
        return self.input_projection(hidden_states)


def _mlp_plan():
    transformation = ModelTransformation(
        transform_id="pack-gated-mlp",
        component="mlp",
        provider="fixture",
        target_paths=("input_projection",),
        inverse_transform_id="unpack-gated-mlp",
        reason="exercise one packed gate/up projection",
        parameter_layout_change=True,
    )
    return ExecutionPlan(
        1,
        "mlp-plan",
        "ready",
        "auto",
        capabilities().fingerprint,
        "pytorch",
        "fp32",
        transformations=(transformation,),
    )


def test_live_gated_mlp_transform_preserves_outputs_and_gradients():
    torch.manual_seed(11)
    original = _MLPFixture()
    transformed = deepcopy(original)
    original_input = torch.randn(2, 3, 4, requires_grad=True)
    transformed_input = original_input.detach().clone().requires_grad_(True)

    registry = ModelTransformRegistry()
    registry.register(gated_mlp_pack_transform_handler(spec(), F.silu))
    transaction = registry.apply(transformed, _mlp_plan())

    assert isinstance(transformed.input_projection, PackedGatedMLPProjection)
    expected = original(original_input)
    actual = transformed(transformed_input)
    assert torch.allclose(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    expected_weight_grad = torch.cat(
        [
            original.input_projection.gate_proj.weight.grad,
            original.input_projection.up_proj.weight.grad,
        ]
    )
    assert torch.allclose(transformed.input_projection.weight.grad, expected_weight_grad)
    assert torch.allclose(transformed_input.grad, original_input.grad)
    transaction.commit()


def test_live_gated_mlp_transform_rolls_back_original_wrapper():
    model = _MLPFixture()
    original = model.input_projection
    registry = ModelTransformRegistry()
    registry.register(gated_mlp_pack_transform_handler(spec(), F.silu))

    transaction = registry.apply(model, _mlp_plan())
    transaction.rollback()

    assert model.input_projection is original


def test_live_gated_mlp_transform_requires_explicit_non_gelu_semantics():
    gelu = GatedMLPProjectionSpec(
        prefix="layers.0.mlp",
        activation="gelu",
        input_size=4,
        intermediate_size=8,
    )
    with pytest.raises(ValueError, match="must remain"):
        gated_mlp_pack_transform_handler(gelu, F.gelu)


def test_live_gated_mlp_transform_preserves_frozen_weight_and_bias():
    model = _MLPFixture(bias=True)
    for parameter in model.input_projection.parameters():
        parameter.requires_grad_(False)
    layout = spec(
        gate_bias_key="gate_proj.bias",
        up_bias_key="up_proj.bias",
        packed_bias_key="gate_up_proj.bias",
    )
    registry = ModelTransformRegistry()
    registry.register(gated_mlp_pack_transform_handler(layout, F.silu))

    registry.apply(model, _mlp_plan()).commit()

    assert not model.input_projection.weight.requires_grad
    assert not model.input_projection.bias.requires_grad


def test_live_gated_mlp_transform_rejects_mixed_trainability():
    model = _MLPFixture()
    model.input_projection.up_proj.weight.requires_grad_(False)
    original = model.input_projection
    registry = ModelTransformRegistry()
    registry.register(gated_mlp_pack_transform_handler(spec(), F.silu))

    with pytest.raises(Exception, match="must share requires_grad"):
        registry.apply(model, _mlp_plan())

    assert model.input_projection is original


def test_live_gated_mlp_transform_rejects_parameter_aliases():
    model = _MLPFixture()
    model.input_projection.up_proj.weight = model.input_projection.gate_proj.weight
    original = model.input_projection
    registry = ModelTransformRegistry()
    registry.register(gated_mlp_pack_transform_handler(spec(), F.silu))

    with pytest.raises(Exception, match="parameter aliases that packing cannot preserve"):
        registry.apply(model, _mlp_plan())

    assert model.input_projection is original
    assert model.input_projection.up_proj.weight is model.input_projection.gate_proj.weight


@pytest.mark.parametrize("bias", (False, True))
def test_live_gated_mlp_transform_preserves_optimizer_update(bias):
    torch.manual_seed(23)
    original = _MLPFixture(bias=bias)
    transformed = deepcopy(original)
    layout = spec(
        gate_bias_key="gate_proj.bias" if bias else None,
        up_bias_key="up_proj.bias" if bias else None,
        packed_bias_key="gate_up_proj.bias" if bias else None,
    )
    registry = ModelTransformRegistry()
    registry.register(gated_mlp_pack_transform_handler(layout, F.silu))
    registry.apply(transformed, _mlp_plan()).commit()
    original_optimizer = SGD(original.parameters(), lr=0.05)
    transformed_optimizer = SGD(transformed.parameters(), lr=0.05)
    hidden_states = torch.randn(2, 3, 4)

    original(hidden_states).square().sum().backward()
    transformed(hidden_states).square().sum().backward()
    original_optimizer.step()
    transformed_optimizer.step()

    packed_gate_weight, packed_up_weight = transformed.input_projection.weight.chunk(
        2,
        dim=0,
    )
    torch.testing.assert_close(
        packed_gate_weight,
        original.input_projection.gate_proj.weight,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        packed_up_weight,
        original.input_projection.up_proj.weight,
        rtol=1e-5,
        atol=1e-6,
    )
    if bias:
        packed_gate_bias, packed_up_bias = transformed.input_projection.bias.chunk(2)
        torch.testing.assert_close(
            packed_gate_bias,
            original.input_projection.gate_proj.bias,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            packed_up_bias,
            original.input_projection.up_proj.bias,
            rtol=1e-5,
            atol=1e-6,
        )
    torch.testing.assert_close(
        transformed(hidden_states),
        original(hidden_states),
        rtol=1e-5,
        atol=1e-6,
    )
