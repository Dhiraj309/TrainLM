import pytest
import torch
from torch.optim import SGD

from trainlm.optimization import QKVProjectionSpec


def spec(**values):
    defaults = dict(
        prefix="layers.0.self_attn",
        query_heads=8,
        key_value_heads=4,
        head_dim=2,
        input_size=6,
        q_weight_key="q_proj.weight",
        k_weight_key="k_proj.weight",
        v_weight_key="v_proj.weight",
        packed_weight_key="qkv_proj.weight",
        dtype="float32",
    )
    return QKVProjectionSpec(**{**defaults, **values})


def test_gqa_weight_pack_and_canonical_export_round_trip():
    layout = spec()
    canonical = {
        "q_proj.weight": torch.arange(96, dtype=torch.float32).reshape(16, 6),
        "k_proj.weight": torch.arange(48, dtype=torch.float32).reshape(8, 6),
        "v_proj.weight": torch.arange(48, 96, dtype=torch.float32).reshape(8, 6),
    }
    converter = layout.converter()
    packed = converter.to_transformed(canonical)
    restored = converter.to_canonical(packed)
    assert packed["qkv_proj.weight"].shape == (32, 6)
    assert all(torch.equal(restored[key], value) for key, value in canonical.items())


def test_bias_layout_round_trip_uses_matching_qkv_geometry():
    layout = spec(
        q_bias_key="q_proj.bias",
        k_bias_key="k_proj.bias",
        v_bias_key="v_proj.bias",
        packed_bias_key="qkv_proj.bias",
    )
    canonical = {
        "q_proj.weight": torch.zeros(16, 6),
        "k_proj.weight": torch.zeros(8, 6),
        "v_proj.weight": torch.zeros(8, 6),
        "q_proj.bias": torch.arange(16, dtype=torch.float32),
        "k_proj.bias": torch.arange(8, dtype=torch.float32),
        "v_proj.bias": torch.arange(8, dtype=torch.float32),
    }
    converter = layout.converter()
    packed = converter.to_transformed(canonical)
    restored = converter.to_canonical(packed)
    assert packed["qkv_proj.bias"].shape == (32,)
    assert all(torch.equal(restored[key], value) for key, value in canonical.items())


def test_bias_keys_must_be_all_present_or_all_absent():
    with pytest.raises(ValueError, match="configured together"):
        spec(q_bias_key="q.bias")


def test_invalid_grouped_query_geometry_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        spec(query_heads=8, key_value_heads=3)


def test_manifest_reconstructs_identical_converter():
    converter = spec().converter()
    restored = type(converter).from_manifest(converter.manifest())
    assert restored.mappings == converter.mappings

from copy import deepcopy
from torch import nn

from trainlm.optimization import (
    ExecutionPlan,
    ModelTransformation,
    ModelTransformRegistry,
    PackedQKVProjection,
    ProviderDecision,
    qkv_pack_transform_handler,
)
from .test_capabilities import capabilities


class _SeparateQKV(nn.Module):
    def __init__(self, *, bias=False):
        super().__init__()
        self.q_proj = nn.Linear(6, 16, bias=bias)
        self.k_proj = nn.Linear(6, 8, bias=bias)
        self.v_proj = nn.Linear(6, 8, bias=bias)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)


class _AttentionFixture(nn.Module):
    def __init__(self, *, bias=False):
        super().__init__()
        self.projections = _SeparateQKV(bias=bias)

    def forward(self, hidden_states):
        return self.projections(hidden_states)


def _qkv_plan():
    transform = ModelTransformation(
        transform_id="pack-qkv",
        component="attention",
        provider="fixture",
        target_paths=("projections",),
        inverse_transform_id="unpack-qkv",
        reason="exercise one packed projection",
        parameter_layout_change=True,
    )
    return ExecutionPlan(
        1,
        "qkv-plan",
        "ready",
        "auto",
        capabilities().fingerprint,
        "pytorch",
        "fp32",
        decisions=(ProviderDecision(
            decision_id="attention.pack-qkv",
            component="attention",
            operation="transform",
            status="selected",
            reason="Fixture provider selected.",
            selected_provider="fixture",
        ),),
        transformations=(transform,),
    )


def test_live_qkv_transform_preserves_outputs_and_gradients():
    torch.manual_seed(7)
    original = _AttentionFixture()
    transformed = deepcopy(original)
    hidden_original = torch.randn(2, 3, 6, requires_grad=True)
    hidden_transformed = hidden_original.detach().clone().requires_grad_(True)

    registry = ModelTransformRegistry()
    registry.register(qkv_pack_transform_handler(spec()))
    transaction = registry.apply(transformed, _qkv_plan())

    assert isinstance(transformed.projections, PackedQKVProjection)
    expected = original(hidden_original)
    actual = transformed(hidden_transformed)
    for expected_projection, actual_projection in zip(expected, actual):
        torch.testing.assert_close(
            actual_projection,
            expected_projection,
            rtol=1e-5,
            atol=1e-6,
        )

    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    expected_weight_grad = torch.cat(
        [
            original.projections.q_proj.weight.grad,
            original.projections.k_proj.weight.grad,
            original.projections.v_proj.weight.grad,
        ]
    )
    torch.testing.assert_close(
        transformed.projections.weight.grad,
        expected_weight_grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        hidden_transformed.grad,
        hidden_original.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    transaction.commit()


def test_live_qkv_transform_rolls_back_to_original_wrapper():
    model = _AttentionFixture()
    original = model.projections
    registry = ModelTransformRegistry()
    registry.register(qkv_pack_transform_handler(spec()))

    transaction = registry.apply(model, _qkv_plan())
    transaction.rollback()

    assert model.projections is original


def test_live_qkv_transform_rejects_mismatched_projection_geometry():
    model = _AttentionFixture()
    model.projections.k_proj = nn.Linear(6, 7, bias=False)
    registry = ModelTransformRegistry()
    registry.register(qkv_pack_transform_handler(spec()))

    with pytest.raises(Exception, match="key projection geometry"):
        registry.apply(model, _qkv_plan())
    assert isinstance(model.projections, _SeparateQKV)


def test_live_qkv_transform_preserves_uniform_frozen_parameters():
    model = _AttentionFixture()
    for parameter in model.projections.parameters():
        parameter.requires_grad_(False)
    registry = ModelTransformRegistry()
    registry.register(qkv_pack_transform_handler(spec()))

    registry.apply(model, _qkv_plan()).commit()

    assert not model.projections.weight.requires_grad


def test_live_qkv_transform_rejects_mixed_frozen_parameters():
    model = _AttentionFixture()
    model.projections.k_proj.weight.requires_grad_(False)
    registry = ModelTransformRegistry()
    registry.register(qkv_pack_transform_handler(spec()))

    with pytest.raises(Exception, match="must share requires_grad"):
        registry.apply(model, _qkv_plan())

    assert isinstance(model.projections, _SeparateQKV)


def test_live_qkv_transform_rejects_external_parameter_alias():
    model = _AttentionFixture()
    model.shared_query_weight = model.projections.q_proj.weight
    original = model.projections
    registry = ModelTransformRegistry()
    registry.register(qkv_pack_transform_handler(spec()))

    with pytest.raises(Exception, match="parameter aliases that packing cannot preserve"):
        registry.apply(model, _qkv_plan())

    assert model.projections is original
    assert model.shared_query_weight is model.projections.q_proj.weight


@pytest.mark.parametrize("bias", (False, True))
def test_live_qkv_transform_preserves_optimizer_update(bias):
    torch.manual_seed(29)
    original = _AttentionFixture(bias=bias)
    transformed = deepcopy(original)
    layout = spec(
        q_bias_key="q_proj.bias" if bias else None,
        k_bias_key="k_proj.bias" if bias else None,
        v_bias_key="v_proj.bias" if bias else None,
        packed_bias_key="qkv_proj.bias" if bias else None,
    )
    registry = ModelTransformRegistry()
    registry.register(qkv_pack_transform_handler(layout))
    registry.apply(transformed, _qkv_plan()).commit()
    original_optimizer = SGD(original.parameters(), lr=0.05)
    transformed_optimizer = SGD(transformed.parameters(), lr=0.05)
    hidden_states = torch.randn(2, 3, 6)

    sum(value.square().sum() for value in original(hidden_states)).backward()
    sum(value.square().sum() for value in transformed(hidden_states)).backward()
    original_optimizer.step()
    transformed_optimizer.step()

    packed_weights = transformed.projections.weight.split((16, 8, 8), dim=0)
    original_weights = (
        original.projections.q_proj.weight,
        original.projections.k_proj.weight,
        original.projections.v_proj.weight,
    )
    for packed, separate in zip(packed_weights, original_weights):
        torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)
    if bias:
        packed_biases = transformed.projections.bias.split((16, 8, 8))
        original_biases = (
            original.projections.q_proj.bias,
            original.projections.k_proj.bias,
            original.projections.v_proj.bias,
        )
        for packed, separate in zip(packed_biases, original_biases):
            torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)
    for packed, separate in zip(transformed(hidden_states), original(hidden_states)):
        torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)

from trainlm.optimization import (
    PackedPartialQKVProjection,
    PartialQKVProjectionSpec,
    partial_qkv_pack_transform_handler,
)


def partial_spec(**values):
    defaults = dict(
        prefix="layers.0.self_attn",
        query_heads=8,
        key_value_heads=4,
        head_dim=2,
        input_size=6,
        q_weight_key="q_proj.weight",
        kv_weight_key="kv_proj.weight",
        packed_weight_key="qkv_proj.weight",
        dtype="float32",
    )
    return PartialQKVProjectionSpec(**{**defaults, **values})


def test_partial_q_kv_layout_round_trip_preserves_source_keys():
    canonical = {
        "q_proj.weight": torch.arange(96, dtype=torch.float32).reshape(16, 6),
        "kv_proj.weight": torch.arange(96, 192, dtype=torch.float32).reshape(16, 6),
    }
    converter = partial_spec().converter()

    transformed = converter.to_transformed(canonical)
    restored = converter.to_canonical(transformed)

    assert transformed["qkv_proj.weight"].shape == (32, 6)
    assert set(restored) == set(canonical)
    assert all(torch.equal(restored[key], value) for key, value in canonical.items())


def test_partial_q_kv_bias_layout_round_trip():
    layout = partial_spec(
        q_bias_key="q_proj.bias",
        kv_bias_key="kv_proj.bias",
        packed_bias_key="qkv_proj.bias",
    )
    canonical = {
        "q_proj.weight": torch.zeros(16, 6),
        "kv_proj.weight": torch.zeros(16, 6),
        "q_proj.bias": torch.arange(16, dtype=torch.float32),
        "kv_proj.bias": torch.arange(16, 32, dtype=torch.float32),
    }

    transformed = layout.converter().to_transformed(canonical)
    restored = layout.converter().to_canonical(transformed)

    assert transformed["qkv_proj.bias"].shape == (32,)
    assert all(torch.equal(restored[key], value) for key, value in canonical.items())


def test_partial_q_kv_layout_rejects_incomplete_bias_group():
    with pytest.raises(ValueError, match="bias keys must be configured together"):
        partial_spec(q_bias_key="q_proj.bias")


def test_partial_q_kv_manifest_reconstructs_identical_converter():
    converter = partial_spec().converter()
    restored = type(converter).from_manifest(converter.manifest())
    assert restored.mappings == converter.mappings


class _PartialQKV(nn.Module):
    def __init__(self, *, bias=False):
        super().__init__()
        self.q_proj = nn.Linear(6, 16, bias=bias)
        self.kv_proj = nn.Linear(6, 16, bias=bias)

    def forward(self, hidden_states):
        query = self.q_proj(hidden_states)
        key, value = self.kv_proj(hidden_states).split((8, 8), dim=-1)
        return query, key, value


class _PartialAttentionFixture(nn.Module):
    def __init__(self, *, bias=False):
        super().__init__()
        self.projections = _PartialQKV(bias=bias)

    def forward(self, hidden_states):
        return self.projections(hidden_states)


def _partial_qkv_plan():
    transform = ModelTransformation(
        transform_id="pack-partial-qkv",
        component="attention",
        provider="fixture",
        target_paths=("projections",),
        inverse_transform_id="unpack-partial-qkv",
        reason="exercise a query plus combined-KV projection",
        parameter_layout_change=True,
    )
    return ExecutionPlan(
        1,
        "partial-qkv-plan",
        "ready",
        "auto",
        capabilities().fingerprint,
        "pytorch",
        "fp32",
        decisions=(ProviderDecision(
            decision_id="attention.pack-partial-qkv",
            component="attention",
            operation="transform",
            status="selected",
            reason="Fixture provider selected.",
            selected_provider="fixture",
        ),),
        transformations=(transform,),
    )


@pytest.mark.parametrize("bias", (False, True))
def test_live_partial_qkv_transform_preserves_outputs_gradients_and_rollback(bias):
    torch.manual_seed(11)
    original = _PartialAttentionFixture(bias=bias)
    transformed = deepcopy(original)
    hidden_original = torch.randn(2, 3, 6, requires_grad=True)
    hidden_transformed = hidden_original.detach().clone().requires_grad_(True)
    layout = partial_spec(
        q_bias_key="q_proj.bias" if bias else None,
        kv_bias_key="kv_proj.bias" if bias else None,
        packed_bias_key="qkv_proj.bias" if bias else None,
    )
    registry = ModelTransformRegistry()
    registry.register(partial_qkv_pack_transform_handler(layout))

    transaction = registry.apply(transformed, _partial_qkv_plan())

    assert isinstance(transformed.projections, PackedPartialQKVProjection)
    expected = original(hidden_original)
    actual = transformed(hidden_transformed)
    for expected_projection, actual_projection in zip(expected, actual):
        torch.testing.assert_close(actual_projection, expected_projection)

    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    expected_weight_grad = torch.cat(
        (
            original.projections.q_proj.weight.grad,
            original.projections.kv_proj.weight.grad,
        )
    )
    torch.testing.assert_close(
        transformed.projections.weight.grad,
        expected_weight_grad,
    )
    torch.testing.assert_close(hidden_transformed.grad, hidden_original.grad)

    transaction.rollback()
    assert isinstance(transformed.projections, _PartialQKV)


def test_live_partial_qkv_transform_rejects_mismatched_combined_geometry():
    model = _PartialAttentionFixture()
    model.projections.kv_proj = nn.Linear(6, 15, bias=False)
    registry = ModelTransformRegistry()
    registry.register(partial_qkv_pack_transform_handler(partial_spec()))

    with pytest.raises(Exception, match="combined key/value projection geometry"):
        registry.apply(model, _partial_qkv_plan())

    assert isinstance(model.projections, _PartialQKV)


def test_live_partial_qkv_transform_preserves_frozen_weight_and_bias():
    model = _PartialAttentionFixture(bias=True)
    for parameter in model.projections.parameters():
        parameter.requires_grad_(False)
    layout = partial_spec(
        q_bias_key="q_proj.bias",
        kv_bias_key="kv_proj.bias",
        packed_bias_key="qkv_proj.bias",
    )
    registry = ModelTransformRegistry()
    registry.register(partial_qkv_pack_transform_handler(layout))

    registry.apply(model, _partial_qkv_plan()).commit()

    assert not model.projections.weight.requires_grad
    assert not model.projections.bias.requires_grad


def test_live_partial_qkv_transform_rejects_internal_parameter_alias():
    model = _PartialAttentionFixture()
    model.projections.kv_proj.weight = model.projections.q_proj.weight
    original = model.projections
    registry = ModelTransformRegistry()
    registry.register(partial_qkv_pack_transform_handler(partial_spec()))

    with pytest.raises(Exception, match="parameter aliases that packing cannot preserve"):
        registry.apply(model, _partial_qkv_plan())

    assert model.projections is original
    assert model.projections.kv_proj.weight is model.projections.q_proj.weight


@pytest.mark.parametrize("bias", (False, True))
def test_live_partial_qkv_transform_preserves_optimizer_update(bias):
    torch.manual_seed(31)
    original = _PartialAttentionFixture(bias=bias)
    transformed = deepcopy(original)
    layout = partial_spec(
        q_bias_key="q_proj.bias" if bias else None,
        kv_bias_key="kv_proj.bias" if bias else None,
        packed_bias_key="qkv_proj.bias" if bias else None,
    )
    registry = ModelTransformRegistry()
    registry.register(partial_qkv_pack_transform_handler(layout))
    registry.apply(transformed, _partial_qkv_plan()).commit()
    original_optimizer = SGD(original.parameters(), lr=0.05)
    transformed_optimizer = SGD(transformed.parameters(), lr=0.05)
    hidden_states = torch.randn(2, 3, 6)

    sum(value.square().sum() for value in original(hidden_states)).backward()
    sum(value.square().sum() for value in transformed(hidden_states)).backward()
    original_optimizer.step()
    transformed_optimizer.step()

    packed_query_weight, packed_kv_weight = transformed.projections.weight.split(
        (16, 16),
        dim=0,
    )
    torch.testing.assert_close(
        packed_query_weight,
        original.projections.q_proj.weight,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        packed_kv_weight,
        original.projections.kv_proj.weight,
        rtol=1e-5,
        atol=1e-6,
    )
    if bias:
        packed_query_bias, packed_kv_bias = transformed.projections.bias.split((16, 16))
        torch.testing.assert_close(
            packed_query_bias,
            original.projections.q_proj.bias,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            packed_kv_bias,
            original.projections.kv_proj.bias,
            rtol=1e-5,
            atol=1e-6,
        )
    for packed, separate in zip(transformed(hidden_states), original(hidden_states)):
        torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)

from trainlm.optimization import PackedQKVProjectionSpec


def test_already_packed_layout_is_an_explicit_no_op():
    layout = PackedQKVProjectionSpec(
        prefix="layers.0.self_attn",
        query_heads=8,
        key_value_heads=4,
        head_dim=2,
        input_size=6,
        packed_weight_key="query_key_value.weight",
        packed_bias_key="query_key_value.bias",
        dtype="float32",
    )

    assert layout.weight_shape == (32, 6)
    assert layout.bias_shape == (32,)
    assert layout.converter() is None


def test_already_packed_mqa_geometry_remains_compact():
    layout = PackedQKVProjectionSpec(
        prefix="layers.0.self_attn",
        query_heads=8,
        key_value_heads=1,
        head_dim=2,
        input_size=16,
        packed_weight_key="query_key_value.weight",
    )

    assert layout.q_size == 16
    assert layout.kv_size == 2
    assert layout.weight_shape == (20, 16)
    assert layout.bias_shape is None


def test_already_packed_layout_rejects_invalid_head_geometry():
    with pytest.raises(ValueError, match="divisible"):
        PackedQKVProjectionSpec(
            prefix="layers.0.self_attn",
            query_heads=8,
            key_value_heads=3,
            head_dim=2,
            input_size=6,
            packed_weight_key="query_key_value.weight",
        )


def test_already_packed_layout_rejects_empty_parameter_key():
    with pytest.raises(ValueError, match="packed_weight_key cannot be empty"):
        PackedQKVProjectionSpec(
            prefix="layers.0.self_attn",
            query_heads=8,
            key_value_heads=4,
            head_dim=2,
            input_size=6,
            packed_weight_key="",
        )
