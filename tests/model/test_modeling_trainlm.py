
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

        #
        # ------------------------------------------------------------------
        # Configuration diagnostics
        # ------------------------------------------------------------------
        #
        print("\n=== CONFIG ===")
        print("Original head_dim:       ", model.config.head_dim)
        print("Loaded head_dim:         ", loaded.config.head_dim)

        print("Original kv groups:      ", model.config.num_key_value_groups)
        print("Loaded kv groups:        ", loaded.config.num_key_value_groups)

        print("Original rope_theta:     ", model.config.rope_theta)
        print("Loaded rope_theta:       ", loaded.config.rope_theta)

        print("Original hidden_size:    ", model.config.hidden_size)
        print("Loaded hidden_size:      ", loaded.config.hidden_size)

        print("Original num_heads:      ", model.config.num_attention_heads)
        print("Loaded num_heads:        ", loaded.config.num_attention_heads)

        print("Original num_kv_heads:   ", model.config.num_key_value_heads)
        print("Loaded num_kv_heads:     ", loaded.config.num_key_value_heads)

        #
        # ------------------------------------------------------------------
        # Parameter diagnostics
        # ------------------------------------------------------------------
        #
        state1 = model.state_dict()
        state2 = loaded.state_dict()

        mismatch_found = False

        for name in state1:
            if not torch.allclose(state1[name], state2[name]):
                mismatch_found = True

                diff = (state1[name] - state2[name]).abs()

                print("\n=== PARAMETER MISMATCH ===")
                print(name)
                print("Shape:", tuple(state1[name].shape))
                print("Max abs diff:", diff.max().item())
                print("Mean abs diff:", diff.mean().item())

                break

        if not mismatch_found:
            print("\nAll parameters and persistent buffers are identical.")

        #
        # ------------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------------
        #
        with torch.no_grad():
            logits1 = model(input_ids=input_ids).logits
            logits2 = loaded(input_ids=input_ids).logits

        if not torch.allclose(logits1, logits2):
            diff = (logits1 - logits2).abs()

            print("\n=== LOGITS ===")
            print("Max abs diff:", diff.max().item())
            print("Mean abs diff:", diff.mean().item())

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
