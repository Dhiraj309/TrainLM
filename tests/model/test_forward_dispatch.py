import pytest
import torch
from torch import nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
)

from trainlm.model import (
    ForwardBatchDispatcher,
    ForwardSignatureError,
    dispatch_model_batch,
)


class ExplicitFamilyForward(nn.Module):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        family_field=None,
    ):
        return input_ids


class OpenFamilyForward(nn.Module):
    def forward(self, input_ids, **kwargs):
        return input_ids


def test_dispatch_preserves_declared_fields_and_reports_unknown_fields():
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "cache_position": torch.arange(4),
        "family_field": torch.tensor(1),
        "dataset_document_id": torch.tensor(7),
    }

    dispatched = dispatch_model_batch(ExplicitFamilyForward(), batch)

    assert tuple(dispatched.inputs) == (
        "input_ids",
        "attention_mask",
        "position_ids",
        "cache_position",
        "family_field",
    )
    assert dispatched.dropped_fields == ("dataset_document_id",)
    assert dispatched.signature.accepts_var_kwargs is False


def test_dynamic_extensions_require_kwargs_and_explicit_passthrough():
    batch = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "new_family_field": torch.tensor(3),
    }

    default_dispatch = dispatch_model_batch(OpenFamilyForward(), batch)
    dispatcher = ForwardBatchDispatcher.from_model(
        OpenFamilyForward(),
        passthrough_fields=("new_family_field",),
    )
    dispatched = dispatcher.dispatch(batch)

    assert tuple(default_dispatch.inputs) == ("input_ids",)
    assert default_dispatch.dropped_fields == ("new_family_field",)
    assert dict(dispatched.inputs) == batch
    assert dispatched.dropped_fields == ()
    assert dispatched.signature.accepts_var_kwargs is True


def test_dispatch_fails_early_when_required_forward_input_is_missing():
    dispatcher = ForwardBatchDispatcher.from_model(ExplicitFamilyForward())

    with pytest.raises(ForwardSignatureError, match="input_ids"):
        dispatcher.dispatch({"attention_mask": torch.ones(1, 2)})


@pytest.mark.parametrize(
    "model",
    (
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=8,
                n_embd=8,
                n_layer=1,
                n_head=2,
            )
        ),
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=8,
            )
        ),
    ),
)
def test_tiny_hf_models_receive_only_their_supported_forward_fields(
    model,
):
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "cache_position": torch.arange(4),
        "token_type_ids": torch.zeros(1, 4, dtype=torch.long),
        "dataset_document_id": torch.tensor(0),
    }

    dispatched = dispatch_model_batch(model, batch)

    declared = set(dispatched.signature.keyword_parameters)
    expected = set(batch) & declared
    assert set(dispatched.inputs) == expected
    assert set(dispatched.dropped_fields) == set(batch) - expected
    assert {"input_ids", "attention_mask", "position_ids"} <= expected
    assert "dataset_document_id" in dispatched.dropped_fields
