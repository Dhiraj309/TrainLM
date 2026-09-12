from types import SimpleNamespace

import pytest

from trainlm.optimization import (
    AttentionMaskSpec,
    CanonicalAttentionSpec,
    HFAttentionProvider,
    expected_causal_visibility,
    install_hf_attention_provider,
)


class Registry:
    def __init__(self):
        self.values = {}

    def register(self, name, implementation):
        self.values[name] = implementation


class Model:
    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation="sdpa")

    def set_attn_implementation(self, name):
        self.config._attn_implementation = name


def spec(**values):
    defaults = dict(
        schema_version=1,
        layout="gqa",
        query_heads=8,
        key_value_heads=4,
        head_dim=64,
        position_encoding="rope",
        mask=AttentionMaskSpec(),
    )
    return CanonicalAttentionSpec(**{**defaults, **values})


def provider(**values):
    defaults = dict(
        provider_id="trainlm.test_attention",
        attention_forward=lambda *args, **kwargs: None,
        mask_factory=lambda *args, **kwargs: None,
        layouts=("gqa",),
        mask_layouts=("causal",),
        position_encodings=("rope",),
    )
    return HFAttentionProvider(**{**defaults, **values})


def test_installs_matching_attention_and_mask_under_the_same_hf_key():
    attention, masks, model = Registry(), Registry(), Model()

    result = install_hf_attention_provider(
        model,
        spec(),
        provider(),
        attention_interface=attention,
        mask_interface=masks,
    )

    assert set(attention.values) == {"trainlm.test_attention"}
    assert set(masks.values) == {"trainlm.test_attention"}
    assert model.config._attn_implementation == "trainlm.test_attention"
    assert result.registered_attention and result.registered_mask


def test_incompatible_semantics_fail_before_interface_registration():
    attention, masks = Registry(), Registry()
    with pytest.raises(ValueError, match="sliding"):
        install_hf_attention_provider(
            Model(),
            spec(mask=AttentionMaskSpec(
                layout="causal_sliding_window", sliding_window=4
            )),
            provider(),
            attention_interface=attention,
            mask_interface=masks,
        )
    assert attention.values == masks.values == {}


def test_reference_visibility_detects_future_window_and_segment_leakage():
    full = spec(mask=AttentionMaskSpec(segment_ids=True))
    sliding = spec(mask=AttentionMaskSpec(
        layout="causal_sliding_window", sliding_window=3
    ))

    assert expected_causal_visibility(full, query_position=3, key_position=3)
    assert not expected_causal_visibility(full, query_position=3, key_position=4)
    assert not expected_causal_visibility(
        full, query_position=3, key_position=2, same_segment=False
    )
    assert expected_causal_visibility(sliding, query_position=5, key_position=3)
    assert not expected_causal_visibility(sliding, query_position=5, key_position=2)


def test_missing_mask_registration_interface_is_rejected():
    with pytest.raises(TypeError, match="must support register"):
        install_hf_attention_provider(
            Model(), spec(), provider(),
            attention_interface=Registry(), mask_interface=object(),
        )


def test_missing_model_selection_boundary_fails_before_registration():
    attention, masks = Registry(), Registry()
    with pytest.raises(TypeError, match="selection boundary"):
        install_hf_attention_provider(
            object(), spec(), provider(),
            attention_interface=attention, mask_interface=masks,
        )
    assert attention.values == masks.values == {}


def test_provider_support_flags_must_be_boolean():
    with pytest.raises(TypeError, match="supports_dropout"):
        provider(supports_dropout=1)
