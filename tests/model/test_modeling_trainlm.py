
import tempfile

import torch

from trainlm.config import TrainLMConfig
from trainlm.model import TrainLMForCausalLM


def test_save_and_load_pretrained():
    config = TrainLMConfig()

    model = TrainLMForCausalLM(config)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        loaded = TrainLMForCausalLM.from_pretrained(tmpdir)

        assert isinstance(
            loaded,
            TrainLMForCausalLM,
        )

        #
        # Hugging Face automatically updates `_name_or_path`
        # when loading from a checkpoint.
        #
        original_config = model.config.to_dict()
        loaded_config = loaded.config.to_dict()

        original_config.pop("_name_or_path", None)
        loaded_config.pop("_name_or_path", None)

        assert loaded_config == original_config


def test_state_dict_roundtrip():
    config = TrainLMConfig()

    model = TrainLMForCausalLM(config)

    state_dict = model.state_dict()

    new_model = TrainLMForCausalLM(config)

    missing, unexpected = new_model.load_state_dict(state_dict)

    assert missing == []

    assert unexpected == []


def test_forward_consistency_after_reload():
    config = TrainLMConfig()

    model = TrainLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        loaded = TrainLMForCausalLM.from_pretrained(tmpdir)
        loaded.eval()

        with torch.no_grad():
            logits1 = model(input_ids=input_ids).logits
            logits2 = loaded(input_ids=input_ids).logits

        assert torch.allclose(
            logits1,
            logits2,
        )


def test_weight_tying():
    config = TrainLMConfig()

    model = TrainLMForCausalLM(config)

    assert (
        model.get_input_embeddings().weight
        is model.get_output_embeddings().weight
    )


def test_config_serialization():
    config = TrainLMConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_pretrained(tmpdir)

        loaded = TrainLMConfig.from_pretrained(tmpdir)

        original = config.to_dict()
        restored = loaded.to_dict()

        original.pop("_name_or_path", None)
        restored.pop("_name_or_path", None)

        assert restored == original
