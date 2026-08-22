import torch
from transformers import PreTrainedModel

from trainlm.config import ModelSourceConfig
from trainlm.model import load_huggingface_causal_lm


def test_generic_provider_builds_tiny_hf_causal_lm_from_model_type():
    loaded = load_huggingface_causal_lm(
        ModelSourceConfig(
            provider="huggingface",
            initialization="config",
            model_type="gpt2",
            dtype="float32",
            config_overrides={
                "vocab_size": 32,
                "n_positions": 16,
                "n_embd": 8,
                "n_layer": 1,
                "n_head": 2,
                "tie_word_embeddings": True,
            },
        )
    )

    assert isinstance(loaded.model, PreTrainedModel)
    assert loaded.model.training is True
    assert loaded.config is loaded.model.config
    assert loaded.metadata.model_type == "gpt2"
    assert loaded.metadata.requested_source == "model_type:gpt2"
    assert loaded.metadata.requested_dtype == "float32"
    assert loaded.metadata.resolved_dtype == "float32"
    assert any(
        {"transformer.wte.weight", "lm_head.weight"}.issubset(group)
        for group in map(set, loaded.metadata.tied_parameter_groups)
    )
    assert type(loaded.model).__module__.startswith("transformers.models.gpt2")


def test_generic_provider_forwards_local_pretrained_acquisition_controls(
    monkeypatch,
    tmp_path,
):
    from trainlm.model import huggingface as provider_module

    calls = {}

    class FakeConfig:
        model_type = "fixture"
        architectures = ("FixtureForCausalLM",)
        _commit_hash = "resolved-commit"

    class FakeModel(torch.nn.Module):
        dtype = torch.bfloat16

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["config"] = (name, kwargs)
            return FakeConfig()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["model"] = (name, kwargs)
            return FakeModel()

    monkeypatch.setattr(provider_module, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(provider_module, "AutoModelForCausalLM", FakeAutoModel)

    loaded = load_huggingface_causal_lm(
        ModelSourceConfig(
            provider="huggingface",
            initialization="pretrained",
            name_or_path=str(tmp_path),
            revision="requested-commit",
            dtype="bfloat16",
            cache_dir="cache/models",
            local_files_only=True,
            subfolder="checkpoint",
            use_safetensors=True,
            config_overrides={"use_cache": False},
        )
    )

    config_name, config_kwargs = calls["config"]
    model_name, model_kwargs = calls["model"]
    assert config_name == model_name == str(tmp_path)
    assert config_kwargs == {
        "trust_remote_code": False,
        "local_files_only": True,
        "revision": "requested-commit",
        "cache_dir": "cache/models",
        "subfolder": "checkpoint",
        "dtype": "bfloat16",
        "use_cache": False,
    }
    assert model_kwargs["config"] is loaded.config
    assert model_kwargs["revision"] == "requested-commit"
    assert model_kwargs["dtype"] == "bfloat16"
    assert model_kwargs["use_safetensors"] is True
    assert loaded.metadata.local_source is True
    assert loaded.metadata.requested_revision == "requested-commit"
    assert loaded.metadata.resolved_revision == "resolved-commit"


def test_generic_provider_rejects_non_huggingface_source():
    source = ModelSourceConfig()

    try:
        load_huggingface_causal_lm(source)
    except ValueError as error:
        assert "provider: huggingface" in str(error)
    else:
        raise AssertionError("Expected an incompatible-provider error.")


def test_pretrained_auto_dtype_is_forwarded_only_to_weight_loading(
    monkeypatch,
    tmp_path,
):
    from trainlm.model import huggingface as provider_module

    calls = {}

    class FakeConfig:
        model_type = "fixture"
        architectures = ()
        _commit_hash = None

    class FakeModel(torch.nn.Module):
        dtype = torch.float32

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["config"] = kwargs
            return FakeConfig()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["model"] = kwargs
            return FakeModel()

    monkeypatch.setattr(provider_module, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(provider_module, "AutoModelForCausalLM", FakeAutoModel)

    load_huggingface_causal_lm(
        ModelSourceConfig(
            provider="huggingface",
            initialization="pretrained",
            name_or_path=str(tmp_path),
            dtype="auto",
            local_files_only=True,
        )
    )

    assert "dtype" not in calls["config"]
    assert calls["model"]["dtype"] == "auto"
