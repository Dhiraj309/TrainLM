# TrainLM Dense Autoregressive Support Contract

- **Status:** Normative for dense-AR V1
- **Applies to:** TrainLM `0.x` development toward dense-AR V1

## Purpose

This document defines what TrainLM means by supporting a Hugging Face model.
It is authoritative for scope decisions, conformance tests, issue
classification, release notes, and public performance claims.

## Product promise

TrainLM will accept an ordinary in-scope Hugging Face
`AutoModelForCausalLM`, train it through TrainLM's performance-sensitive
engine, and export an ordinary Hugging Face checkpoint.

TrainLM must preserve two independent properties:

1. **Generic compatibility:** an in-scope dense causal model can train without
   a family-specific optimization adapter.
2. **Measured optimization:** an advertised optimized model uses validated
   structural transformations or TPU kernel providers and has evidence for
   correctness and efficiency.

TrainLM does not fork or modify installed Transformers model source. Optional
family adapters may describe model structure, but trainer, task, data,
checkpoint, and runtime layers remain model-family independent.

## Meaning of “any autoregressive model”

For dense-AR V1, “any autoregressive model” means a model satisfying all of the
following conditions:

- it is a decoder-only causal language model exposed through
  `AutoModelForCausalLM` or an equivalent local `PreTrainedModel` contract;
- its training objective is next-token prediction over a dense vocabulary;
- it accepts token IDs and has a statically representable training forward pass;
- it returns a causal-LM loss when given labels, or logits from which standard
  shifted causal loss can be computed;
- its trainable blocks are dense Transformer blocks, not routed experts;
- its required operations have PyTorch semantics supported by the selected
  TrainLM backend;
- it does not require a CUDA-only operation for its normal training path;
- it can be saved to and reconstructed from a canonical Hugging Face
  configuration and state dictionary.

This is a structural contract, not a fixed model-name allowlist. A new model
family satisfying this contract is eligible for generic training before it has
a dedicated optimization adapter or certification record.

## Dense-AR V1 capability envelope

### Model and checkpoint intake

- `AutoConfig.from_pretrained` and compatible local configurations;
- `AutoModelForCausalLM.from_config` for pretraining from initialization;
- `AutoModelForCausalLM.from_pretrained` for continued pretraining;
- tied and untied input/output embeddings;
- canonical Hugging Face export that does not require TrainLM to reload;
- `trust_remote_code=False` by default.

### Transformer structure

- multi-head, grouped-query, and multi-query attention;
- full causal attention and an explicitly selected sliding-window path;
- learned absolute positions, RoPE, and ALiBi;
- LayerNorm and RMSNorm;
- GELU, SwiGLU, GeGLU, and related dense gated MLPs;
- serial and parallel residual blocks;
- separate Q/K/V, packed QKV, and partially packed projections;
- bias and bias-free projections where defined by the source model;
- fixed-shape data-parallel training on TPU v5e-8;
- FSDP as the larger-model extension within dense-AR V1.

### Training lifecycle

An in-scope model reaches generic compatibility only if it completes:

1. configuration and model construction;
2. forward pass with correct causal masking;
3. finite loss and backward pass;
4. at least one optimizer update;
5. evaluation with token-weighted loss;
6. internal checkpoint save and exact compatible resume;
7. canonical Hugging Face export and clean reload.

Initialization alone, inference alone, or a forward-only TPU graph does not
meet the training support contract.

## Representative conformance matrix

TrainLM tests structural diversity rather than treating Llama descendants as
the complete autoregressive ecosystem.

| Capability cluster | Required representatives | Distinguishing behavior |
|---|---|---|
| Learned positions + LayerNorm + GELU | GPT-2, OPT | Learned position tables and non-gated MLPs |
| RoPE/parallel residual + ALiBi/packed QKV | GPT-NeoX/Pythia, BLOOM | Parallel residuals, ALiBi, and packed projections |
| MQA/GQA + non-Llama dense blocks | Falcon, dense Phi | Unequal Q/KV heads and different projection/block layouts |
| RoPE + RMSNorm + gated MLP | Llama, Mistral, dense Qwen, dense Gemma | Gated MLPs plus family-specific masks, scaling, or normalization |

Tiny configurations represent these families in per-change conformance tests.
Approximately 135M-class configurations are used for TPU performance evidence.
Passing one representative does not automatically certify every model in its
row; certification is versioned and model-specific.

## Support levels

### Compatible

A model is **Compatible** when it completes the generic training lifecycle and
plain-HF round trip without a model-family optimization adapter.

Compatible means correct and usable. It does not imply that TrainLM selected a
fused kernel, reduced memory traffic, reached an MFU target, or matched LaughLM.

### Optimized

A model is **Optimized** when it is Compatible and TrainLM safely activates one
or more validated structural optimizations or backend kernel providers for its
actual capabilities.

The optimization plan must be explainable. Unsupported operations are reported
as fallbacks; partial optimization is not full certification.

### Certified

A model is **Certified** when it is Optimized and has a current hardware record
covering:

- forward, loss, gradient, and update correctness;
- mask and positional semantics;
- state-dict conversion, tied aliases, checkpoint/resume, and plain-HF export;
- stable post-warmup graph count and no unexpected CPU fallback;
- peak HBM, throughput, and MFU on named hardware and software versions;
- a model-specific performance target defined before certification.

Only Certified models may be described as “fully supported,” “optimally
trainable,” or meeting a published TPU performance target.

## Fallback and strict-mode policy

- Generic fallback is a required compatibility mechanism, not evidence of
  optimization.
- `auto` mode may select a safe generic provider, but records why a faster
  provider was not selected.
- `required` or certification mode fails before TPU allocation when a requested
  capability cannot be provided safely.
- No fallback may silently change causal masking, position handling, label
  shift, loss reduction, parameter tying, or checkpoint layout.
- `trainer.explain()` is the public source for providers, transformations,
  fallbacks, and certification state.

## Remote-code policy

`trust_remote_code=False` is the supported default because remote model code
can execute arbitrary Python and may not obey standard structural contracts.

A remote-code model requires explicit user opt-in. It is best-effort and
uncertified until TrainLM records the exact repository revision, audits its
training behavior, supplies an adapter only where needed, and passes the same
conformance and certification gates as built-in models.

## Explicitly outside dense-AR V1

The following are not dense-AR V1 blockers:

- BERT-style encoder-only models;
- encoder-decoder sequence-to-sequence models;
- mixture-of-experts models and expert parallelism;
- diffusion language models and denoising objectives;
- state-space, recurrent, and hybrid attention/SSM architectures;
- multimodal causal models;
- quantized pretraining;
- models whose normal training path requires unsupported CUDA-only operations.

MoE and diffusion language models are future extensions. Supporting them must
extend task and capability contracts without weakening dense-AR gates.

## Issue classification

Issues changing or questioning this contract use the `scope` label plus an
appropriate outcome:

| Label | Meaning |
|---|---|
| `support:compatible` | Generic dense-AR lifecycle is expected or being validated |
| `support:optimized` | Structural optimization exists but hardware certification is incomplete |
| `support:certified` | Current model/hardware certification evidence exists |
| `scope:future` | Valid future extension, not a dense-AR V1 blocker |
| `scope:out-of-scope` | Does not fit the accepted product direction |
| `backend:xla` | Specific to the current PyTorch/XLA backend |
| `backend:torchtpu` | Reserved for the future native TorchTPU backend |

A model request identifies the HF model/config revision, architecture type,
desired support level, backend/hardware, and any custom-code requirement.

## Release-note requirements

Every release changing model support states:

- the affected family and configuration/revision range;
- the old and new support level;
- enabled providers and known fallbacks;
- tested Transformers, backend, and hardware versions;
- whether checkpoint/export compatibility changed;
- benchmark evidence for every performance claim.

Release notes must not use Compatible, Optimized, and Certified
interchangeably. Generated release categories are defined in
`.github/release.yml`.

## Changing this contract

A scope change requires one focused PR that:

1. explains the user need and structural capability change;
2. updates this contract and the implementation roadmap;
3. updates conformance fixtures and issue/release classification if affected;
4. states whether the change blocks dense-AR V1;
5. receives review before implementation is advertised as supported.

Performance experiments do not change product scope by themselves.
