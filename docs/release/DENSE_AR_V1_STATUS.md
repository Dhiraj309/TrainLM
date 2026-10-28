# Dense-AR V1 development support status

> **Historical release snapshot:** This page and its machine-readable manifest
> describe the `0.1.0-dev` PyTorch/XLA-era support state. The active target is
> native TorchTPU under T0–T11; this page does not establish TorchTPU support.

The machine-readable source for this page is
[`support/dense_ar_v1.json`](../../support/dense_ar_v1.json). It describes the
current development release, not a completed hardware certification.

## Recorded `0.1.0-dev` support snapshot (PyTorch/XLA era)

- **CPU:** compatible portable execution; not performance certified.
- **CUDA:** compatible portable execution; Tier 1 release evidence is pending.
- **TPU v5e-8 through PyTorch/XLA:** experimental coordinator path; correctness,
  lifecycle, canonical export, and performance certification remain pending.
- **TorchTPU:** not present in this release snapshot. Native TorchTPU is the
  current architecture target, pending T0/T1 review and hardware validation.

The portable Hugging Face implementation remains the semantic fallback.
Chunked linear causal cross-entropy is the portable full-logits bypass.
Pallas attention and Tokamax linear-CE integrations remain unverified and are
not advertised as certified providers.

## Model-family mappings

Explicit, version-guarded mappings exist for GPT-2/OPT, GPT-NeoX/BLOOM,
Falcon/Phi, and Llama/Mistral/Qwen2/Gemma variants. A mapping is an eligibility
contract, not proof of optimized or certified execution. Each family still
needs the correctness, graph, export, HBM, and target-hardware evidence recorded
in the roadmap.

## Release caveats

- Dense decoder-only autoregressive models are the current scope; MoE and DLLM
  are future extensions.
- TPU rank-local training checkpoints are not plain Hugging Face exports.
- No family or TPU provider is Certified in version `0.1.0-dev`.
- Release certification remains blocked until current Tier 0-3 evidence exists
  for one exact release commit.
