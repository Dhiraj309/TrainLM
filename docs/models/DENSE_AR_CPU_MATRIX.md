# Dense-AR CPU Conformance Matrix

This matrix is the V1 compatibility gate for representative decoder-only
Hugging Face architectures. It demonstrates that TrainLM's generic provider,
batch dispatcher, causal task, and plain-Transformers export path do not depend
on a Llama-shaped model implementation.

## Covered capability clusters

| Cluster | Matrix families |
|---|---|
| Learned positions, LayerNorm, GELU | GPT-2, OPT |
| RoPE or ALiBi, parallel residuals, packed projections | GPT-NeoX, BLOOM |
| MQA/GQA and non-Llama dense blocks | Falcon, Phi |
| RoPE, RMSNorm, gated MLPs | Llama, Mistral, Qwen2, Gemma |

All cases use tiny official Transformers configurations. The test constructs
each model through `ModelSourceConfig` and `AutoModelForCausalLM`; TrainLM does
not copy, subclass, or patch the family implementation.

## Per-family assertions

Each case must prove that:

1. the generic Hugging Face provider resolves the requested architecture;
2. every model module remains outside the `trainlm` model namespace;
3. the forward-aware dispatcher passes model inputs and drops dataset metadata;
4. logits follow the expected batch, sequence, and vocabulary shape;
5. repeated causal-language-model updates produce finite loss and gradients,
   change parameters, and materially reduce the fixed-batch loss; and
6. the updated model saves and reloads with plain `AutoModelForCausalLM` while
   preserving deterministic logits.

The executable matrix lives in
`tests/model/test_dense_ar_conformance.py`. The adjacent round-trip suite also
checks tied and untied embedding/head layouts for every family.

## What this gate does not claim

CPU conformance establishes the **Compatible** support level only. It does not
certify PyTorch/XLA compilation, TPU numerical parity, graph stability, HBM,
throughput, MFU, or an optimized kernel. Those claims require the later TPU
milestones and saved hardware evidence defined in the roadmap.

Run this matrix in a compatible development environment with:

```text
pytest tests/model/test_dense_ar_conformance.py tests/model/test_plain_hf_roundtrip.py
```
