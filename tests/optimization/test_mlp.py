import pytest
import torch

from trainlm.optimization import GatedMLPProjectionSpec


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
