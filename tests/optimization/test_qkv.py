import pytest
import torch

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
    def __init__(self):
        super().__init__()
        self.projections = _SeparateQKV()

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
    assert all(torch.allclose(left, right) for left, right in zip(expected, actual))

    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    expected_weight_grad = torch.cat(
        [
            original.projections.q_proj.weight.grad,
            original.projections.k_proj.weight.grad,
            original.projections.v_proj.weight.grad,
        ]
    )
    assert torch.allclose(transformed.projections.weight.grad, expected_weight_grad)
    assert torch.allclose(hidden_transformed.grad, hidden_original.grad)
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

from trainlm.optimization import PartialQKVProjectionSpec


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
