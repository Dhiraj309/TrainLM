# Configuration ownership

TrainLM separates model architecture from training policy. Hugging Face
`PretrainedConfig` is the source of truth for every model-family field, while
TrainLM owns the settings needed to train that model.

| Section | Owner | Examples |
|---|---|---|
| `model` | Model acquisition | provider, Hub ID, revision, initialization |
| `model.config_overrides` | Selected HF model config | hidden size, layers, heads, vocabulary |
| `dataset` | Data pipeline | `.bin` sources, sequence length, prefetch |
| `loss` | Task loss policy | ignored index, normalization, z-loss |
| `optimizer`, `scheduler`, `trainer` | Training engine | AdamW, schedule, accumulation, stop criteria |
| `runtime` | Backend execution | device, precision, execution strategy |
| `parallelism` | Logical topology | data, FSDP, tensor, sequence, pipeline axes |
| `optimizations` | Optimization request | compile, fallback policy, requested passes |
| `checkpoint`, `logging`, `monitoring`, `evaluation` | Run services | persistence, reporting, diagnostics, evaluation |

## Supplying an already constructed model

The default provider is `external`. This is the zero-magic path used when an
application constructs an arbitrary `PreTrainedModel` and passes it to the
trainer. No architecture config is created by TrainLM.

```yaml
trainer:
  max_steps: 1000
runtime:
  device: xla
  precision: bf16
```

## Selecting a Hugging Face architecture

Use a Hub/local config or an AutoConfig `model_type`. Architecture overrides
remain opaque to TrainLM's training config and will be validated by the HF
configuration selected by the model provider.

```yaml
model:
  provider: huggingface
  model_type: llama
  initialization: config
  config_overrides:
    hidden_size: 1024
    num_hidden_layers: 8
    num_attention_heads: 8
    vocab_size: 32064
```

For pretrained weights, select a revision explicitly when reproducibility is
required:

```yaml
model:
  provider: huggingface
  name_or_path: org/model
  revision: immutable-commit-sha
  initialization: pretrained
```

`trust_remote_code` defaults to `false` and must be consciously enabled. Its
security and support implications are defined in [the scope contract](../SCOPE.md).

## Selecting TrainLM's reference architecture

The reference model is opt-in. This descriptor does not instantiate
`TrainLMConfig`; the reference-model provider will do that at the model
construction boundary.

```yaml
model:
  provider: trainlm
  initialization: config
  config_overrides:
    hidden_size: 768
    num_hidden_layers: 12
```

Legacy YAML that placed architecture keys directly under `model` is rejected
with a migration message. Move those keys under `config_overrides` and choose
the provider explicitly. Legacy `runtime.compile` is migrated to
`optimizations.compile` only when the new field is absent.

