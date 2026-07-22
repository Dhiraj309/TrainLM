import torch.nn as nn

from transformers import PreTrainedModel

from trainlm.config import TrainLMConfig
from trainlm.model import TrainLMPreTrainedModel

import torch

from transformers.modeling_outputs import BaseModelOutputWithPast

from trainlm.model import TrainLMModel



class DummyModel(TrainLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        pass


def test_pretrained_model_inheritance():
    model = DummyModel(TrainLMConfig())

    assert isinstance(model, PreTrainedModel)


def test_config_class():
    assert TrainLMPreTrainedModel.config_class is TrainLMConfig


def test_base_model_prefix():
    assert TrainLMPreTrainedModel.base_model_prefix == "model"


def test_linear_initialization():
    model = DummyModel(TrainLMConfig())

    linear = nn.Linear(16, 16)

    model._init_weights(linear)

    assert linear.bias is not None


def test_model_construction():
    config = TrainLMConfig()

    model = TrainLMModel(config)

    assert model.config is config


def test_input_embeddings():
    config = TrainLMConfig()

    model = TrainLMModel(config)

    embeddings = model.get_input_embeddings()

    assert embeddings.num_embeddings == config.vocab_size
    assert embeddings.embedding_dim == config.hidden_size


def test_forward_returns_model_output():
    config = TrainLMConfig()

    model = TrainLMModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 16),
    )

    outputs = model(input_ids=input_ids)

    assert isinstance(outputs, BaseModelOutputWithPast)

    assert outputs.last_hidden_state.shape == (
        2,
        16,
        config.hidden_size,
    )


def test_inputs_embeds():
    config = TrainLMConfig()

    model = TrainLMModel(config)

    inputs_embeds = torch.randn(
        2,
        8,
        config.hidden_size,
    )

    outputs = model(
        inputs_embeds=inputs_embeds,
    )

    assert outputs.last_hidden_state.shape == (
        2,
        8,
        config.hidden_size,
    )


def test_input_validation():
    config = TrainLMConfig()

    model = TrainLMModel(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 4),
    )

    inputs_embeds = torch.randn(
        1,
        4,
        config.hidden_size,
    )

    import pytest

    with pytest.raises(ValueError):
        model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

    with pytest.raises(ValueError):
        model()
