from types import SimpleNamespace

import pytest
import torch

from trainlm.model import normalize_causal_lm_output


def test_normalizes_attribute_mapping_and_tuple_outputs():
    logits = torch.randn(1, 3, 8)
    loss = torch.tensor(2.0)

    attribute = normalize_causal_lm_output(
        SimpleNamespace(logits=logits, loss=loss)
    )
    mapping = normalize_causal_lm_output({"logits": logits, "loss": loss})
    tuple_with_loss = normalize_causal_lm_output((loss, logits, "cache"))
    tuple_without_loss = normalize_causal_lm_output((logits, "cache"))

    assert attribute.logits is mapping.logits is tuple_with_loss.logits
    assert attribute.loss is mapping.loss is tuple_with_loss.loss
    assert tuple_without_loss.logits is logits
    assert tuple_without_loss.loss is None


def test_rejects_missing_logits_and_non_scalar_model_loss():
    with pytest.raises(TypeError, match="logits"):
        normalize_causal_lm_output({"loss": torch.tensor(1.0)})

    with pytest.raises(TypeError, match="scalar"):
        normalize_causal_lm_output(
            {"logits": torch.randn(1, 2, 4), "loss": torch.ones(2)}
        )
