from types import SimpleNamespace

import pytest

from trainlm.model import detect_block_layout


@pytest.mark.parametrize(
    ("config", "normalization", "mlp", "residual"),
    [
        (
            SimpleNamespace(
                layer_norm_epsilon=1e-5,
                hidden_act="gelu",
                residual_type="serial",
            ),
            "layernorm",
            "gelu",
            "serial",
        ),
        (
            SimpleNamespace(
                rms_norm_eps=1e-6,
                ffn_type="swiglu",
                parallel_residual=True,
            ),
            "rmsnorm",
            "swiglu",
            "parallel",
        ),
        (
            SimpleNamespace(
                normalization="rms_norm",
                mlp_type="geglu",
                residual_connection="parallel",
            ),
            "rmsnorm",
            "geglu",
            "parallel",
        ),
    ],
)
def test_explicit_block_layouts_are_classified(
    config, normalization, mlp, residual
):
    result = detect_block_layout(config)

    assert result.normalization == normalization
    assert result.mlp == mlp
    assert result.residual == residual
    assert result.evidence


def test_unknown_or_ambiguous_fields_are_not_guessed():
    result = detect_block_layout(
        SimpleNamespace(
            normalization="rmsnorm",
            norm_type="layernorm",
            mlp_type="swiglu",
            hidden_act="gelu",
        )
    )

    assert result.normalization == "unknown"
    assert result.mlp == "unknown"
    assert result.residual == "unknown"
    assert all("unknown or contradictory" in item for item in result.evidence[-3:])


def test_module_structure_is_an_optional_fallback():
    class RMSNorm:
        pass

    class SwiGLU:
        pass

    class ParallelResidual:
        pass

    class Model:
        def modules(self):
            return iter((RMSNorm(), SwiGLU(), ParallelResidual()))

    result = detect_block_layout(SimpleNamespace(), Model())

    assert result.normalization == "rmsnorm"
    assert result.mlp == "swiglu"
    assert result.residual == "parallel"


def test_model_without_modules_is_safe():
    result = detect_block_layout(SimpleNamespace(), object())

    assert result.normalization == "unknown"
    assert result.mlp == "unknown"
    assert result.residual == "unknown"


def test_config_is_required():
    with pytest.raises(TypeError, match="config is required"):
        detect_block_layout(None)
