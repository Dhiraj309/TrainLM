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
