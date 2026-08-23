# Generic Hugging Face causal-model provider

TrainLM acquires ordinary Transformers models without copying or subclassing
their implementations. `HuggingFaceCausalLMProvider` selects the concrete
architecture through `AutoConfig` and `AutoModelForCausalLM`, then returns the
unchanged model plus immutable acquisition metadata.

## Pretraining from a configuration

```python
from trainlm.config import ModelSourceConfig
from trainlm.model import load_huggingface_causal_lm

loaded = load_huggingface_causal_lm(ModelSourceConfig(
    provider="huggingface",
    initialization="config",
    model_type="llama",
    dtype="bfloat16",
    config_overrides={
        "vocab_size": 32064,
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 2048,
    },
))

model = loaded.model
```

The architecture fields are passed directly to the HF config class. TrainLM
does not translate them into a Llama-shaped internal configuration.

## Continuing from pretrained weights

```python
loaded = load_huggingface_causal_lm(ModelSourceConfig(
    provider="huggingface",
    initialization="pretrained",
    name_or_path="organization/model",
    revision="immutable-commit-sha",
    dtype="bfloat16",
    use_safetensors=True,
))
```

`name_or_path` may also be a local `save_pretrained` directory. Set
`local_files_only=True` for an offline-only load and `cache_dir` to control the
Hub cache. Authentication remains with the standard Hugging Face environment;
tokens are deliberately not stored in TrainLM configuration.

The provider explicitly forwards dtype at both the config and model boundary,
keeps the requested and resolved Hub revisions, records the concrete HF model
and config classes, and observes tied parameter aliases without modifying them.
It normalizes both initialization paths to training mode. Forward dispatch,
loss handling, optimization analysis, and model transformations belong to
later roadmap stories.

Call `explain_huggingface_compatibility(loaded)` to obtain the current generic
path, conservative capability state, explicit optimization fallback, and
support-level boundary. See
[`COMPATIBILITY_EXPLANATION.md`](COMPATIBILITY_EXPLANATION.md).
