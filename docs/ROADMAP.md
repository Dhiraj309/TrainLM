# TrainLM: HF-Native, TPU-Optimized Pretraining Roadmap

- Status: implementation plan
- Roadmap date: 2026-08-21
- Roadmap branch: `codex/hf-tpu-parity-roadmap`
- Implementation base: `m4/training-framework` at `1a00b44a96e6`
- Reference baseline: LaughLM at `0705d255faab`

## 1. Objective

TrainLM will be a Hugging Face-native pretraining framework that accepts an
ordinary dense autoregressive `AutoModelForCausalLM`, trains it efficiently on
Google TPUs, applies safe TPU-specific optimizations without editing the
installed Transformers source, and exports an ordinary Hugging Face checkpoint.

The first release must satisfy two independent promises:

1. **Universal dense-AR compatibility:** every in-scope dense Hugging Face
   causal LM can initialize, train, evaluate, checkpoint, resume, and export
   through the generic TrainLM path without a model-family adapter.
2. **Certified TPU optimization:** every model TrainLM calls fully supported has
   a measured, semantically correct optimized path. The exact 135M LaughLM
   geometry must run within 10% of the validated LaughLM v5e-8 result, with a
   preferred target within 5%.

The product is not complete if it only makes Llama-style models fast. It is also
not complete if arbitrary dense causal models merely execute but cannot receive
effective structural optimization.

## 2. Non-negotiable product contracts

### 2.1 Hugging Face remains the model and checkpoint contract

- Users load models with `AutoConfig` and `AutoModelForCausalLM`.
- Users may initialize from config or continue from `from_pretrained` weights.
- TrainLM does not fork or edit Hugging Face model source files on disk.
- In-memory transformations occur before optimizer construction and are
  reversible.
- Final exports load with plain `AutoModelForCausalLM.from_pretrained` without
  TrainLM installed.
- `trust_remote_code=False` is the supported default. Remote-code models are
  best-effort until explicitly certified.

### 2.2 TrainLM owns the performance-sensitive training engine

- `TrainLMTrainer`, not Hugging Face `Trainer`, owns the optimized training
  loop.
- The public API remains familiar: `train`, `evaluate`, `save_model`, resume,
  callbacks, state, and metrics.
- The compiled region can include forward, optimized loss, backward, gradient
  reduction, clipping, AdamW, and scheduler update.
- No tensor-to-host synchronization is allowed in the hot path.
- Hugging Face `Trainer` interoperability may remain as a compatibility mode,
  but it does not carry the TPU parity guarantee.

### 2.3 Optimization is capability-driven

- The trainer and runtime operate on generic `torch.nn.Module` and Hugging Face
  contracts.
- Model-family adapters only map family-specific module locations and semantics
  onto reusable structural capabilities.
- Adapters are never required for generic training.
- Adapters may be required for certified fused loss, projection packing, or an
  unusual attention semantic.
- Unsupported optimizations fall back explicitly in compatibility mode and fail
  explicitly in strict certification mode.
- `trainer.explain()` reports every capability, selected provider, fallback,
  transformation, checkpoint conversion, and certification status.

### 2.4 Runtime backends are replaceable

TrainLM targets PyTorch semantics, not PyTorch/XLA internals:

```text
TrainLM trainer and optimization engine
                 |
          ExecutionBackend
          /       |       \
       CPU      CUDA    PyTorch/XLA now
                           |
                       TorchTPU later
```

- `torch_xla` imports stay inside the PyTorch/XLA backend and its kernel
  providers.
- Synchronization, compilation, sharding, device loading, distributed
  checkpointing, and profiling are backend interfaces.
- TorchTPU becomes a new backend rather than a rewrite of TrainLM.
- PyTorch/XLA is not removed until TorchTPU matches correctness, checkpoint,
  and performance gates.

### 2.5 Support means more than execution

TrainLM exposes three explicit support levels:

| Level | Meaning |
|---|---|
| Compatible | Correct generic training and HF round trip |
| Optimized | Compatible structural optimizations are active |
| Certified | Correctness, HBM, graph stability, throughput, and MFU passed on real hardware |

Only `Certified` may be described as fully supported or optimally trainable.

## 3. Scope

### 3.1 Dense autoregressive V1

V1 covers dense decoder-only Hugging Face causal Transformers with:

- MHA, GQA, and MQA;
- causal full attention and a safe sliding-window fallback;
- learned positional embeddings, RoPE, and ALiBi;
- LayerNorm and RMSNorm;
- GELU, SwiGLU, GeGLU, and related dense gated MLPs;
- serial and parallel residual layouts;
- separate, partially packed, and packed QKV layouts;
- tied and untied input/output embeddings;
- initialization from config and continuation from pretrained weights;
- packed fixed-length `.bin` pretraining data;
- DP on v5e-8 and FSDP for larger models.

The conformance and certification matrix must contain structurally different
representatives, not only closely related Llama descendants:

| Capability cluster | Representative families |
|---|---|
| Learned position + LayerNorm + GELU | GPT-2, OPT |
| RoPE/parallel residual and ALiBi/packed QKV | GPT-NeoX/Pythia, BLOOM |
| MQA/GQA and non-Llama residual/projection layouts | Falcon, dense Phi |
| RoPE + RMSNorm + gated MLP | Llama, Mistral, dense Qwen, dense Gemma |

Tiny configs are used for per-commit correctness CI. Approximately 135M-class
configs are used for TPU efficiency certification.

### 3.2 Explicitly deferred

- BERT-style encoder-only or encoder-decoder training;
- MoE and expert parallelism;
- diffusion language models (DLLMs);
- state-space, recurrent, and hybrid attention/SSM architectures;
- multimodal causal models;
- arbitrary CUDA-only custom operations;
- quantized pretraining.

MoE and DLLM are future milestones in this roadmap; they are not allowed to
weaken or delay the dense-AR V1 acceptance gates.

## 4. Baselines and measurable targets

### 4.1 Starting TrainLM state

At the implementation base:

- [`pyproject.toml`](../pyproject.toml) depends on PyTorch and Transformers but
  not PyTorch/XLA.
- [`src/trainlm/training/trainer.py`](../src/trainlm/training/trainer.py) is an
  orchestration skeleton with unimplemented training methods.
- [`src/trainlm/runtime`](../src/trainlm/runtime) contains conflicting runtime
  abstractions.
- [`src/trainlm/config/loader.py`](../src/trainlm/config/loader.py) hard-codes
  the custom `TrainLMConfig` model configuration.
- [`src/trainlm/model/trainlm`](../src/trainlm/model/trainlm) provides a useful
  HF-compatible reference architecture but is currently the primary model path.
- The current causal-LM head materializes full logits, upcasts them to FP32, and
  applies standard cross-entropy.
- Production data, evaluation, checkpoint, XLA, SPMD, and kernel paths are not
  implemented.

### 4.2 LaughLM 135M v5e-8 reference

The parity manifest is fixed to:

- vocabulary `32,064`;
- hidden size `1,024`;
- 8 layers, 8 Q heads, 8 KV heads;
- intermediate size `2,816`;
- sequence length `2,048`;
- RoPE, pre-RMSNorm, SwiGLU, tied embeddings, no bias;
- fused QKV and SplashAttention;
- BF16 compute, FP32 parameters/output;
- microbatch/device `2`, gradient accumulation `32`, DP `8`;
- `1,048,576` tokens per optimizer update;
- chunked logits `4,096`, z-loss `1e-4`;
- AdamW `2e-4`, betas `0.9/0.95`, epsilon `1e-8`, weight decay `0.1`, clip `1.0`;
- WSD over 20B tokens, 1% warmup, 95% stable, 5% minimum LR ratio;
- native memmap data, prefetch `16`, persistent cache, asynchronous checkpoint;
- unscanned layer stack for the production throughput result.

Validated reference result:

- `1.014M` global tokens/sec;
- `1.023M` device tokens/sec;
- `53.1%` non-embedding MFU;
- `65.9%` logits-inclusive MFU;
- `5.73 GB` peak HBM;
- `1.034 s` total step and `1.025 s` device step.

### 4.3 Parity gates

| Gate | Global tok/s | Total step | Non-embedding MFU |
|---|---:|---:|---:|
| Hard V1 acceptance (90%) | `>= 912,600` | `<= 1.149 s` | `>= 47.8%` |
| Preferred release (95%) | `>= 963,300` | `<= 1.089 s` | `>= 50.4%` |
| Parity/stretch | about `1.014M` | about `1.034 s` | about `53.1%` |

Numbers are steady-state medians, not the fastest sample. A release result
requires three matched runs plus a 200-update real-shard stability run.

### 4.4 Intermediate go/no-go gates

| Stage | Gate | Required response on failure |
|---|---:|---|
| Generic HF + stable XLA runtime | `>= 600K tok/s` | Stop feature expansion; diagnose graph/sync/runtime overhead |
| Optimized loss + TPU attention | `>= 850K tok/s` | Stop family expansion; inspect HLO, kernels, accumulation, and memory traffic |
| Final 135M optimized path | `>= 912.6K tok/s` | V1 performance gate not met |

Intermediate values are engineering gates, not public performance claims.

## 5. Target architecture

```text
AutoConfig / AutoModelForCausalLM
                |
         HFModelProvider
                |
       ModelCapabilityInspector
                |
       OptimizationPlanner  <---- Certification registry
          /             \
 Reversible transforms   Kernel providers
          \             /
       PreparedTrainingModel
                |
          TrainLMTrainer
                |
         ExecutionBackend
      /        |          \
    CPU      CUDA     PyTorch/XLA ----> TorchTPU
                |
     Internal resume checkpoint
                |
       Reversible HF exporter
```

### 5.1 Mandatory interfaces

- `HFModelProvider`: construct from config or pretrained weights and preserve HF
  metadata.
- `CausalLMTask`: batch dispatch, label semantics, loss normalization, and
  evaluation contract.
- `ModelCapabilityInspector`: immutable description of attention, positions,
  norms, MLP, residual, projections, head, embeddings, and forward signature.
- `ModelAdapter`: optional family-specific mapping to structural capabilities.
- `OptimizationPlanner`: pure capability/provider selection with reasons.
- `ModelTransformation`: validate, apply, record, reverse, and export.
- `KernelProvider`: backend-specific implementation with support predicates,
  forward/backward correctness, fake/meta behavior, and benchmark metadata.
- `ExecutionBackend`: device, compile, mesh, shard, synchronize, data loader,
  checkpoint, and diagnostics.
- `PreparedTrainingModel`: optimized training view plus canonical HF export view.
- `TrainLMTrainer`: backend-neutral training orchestration.
- `CheckpointCodec`: internal resume format and standard HF export format.

### 5.2 Preparation lifecycle

The order is mandatory:

```text
load HF model on CPU/meta device
  -> validate task contract
  -> inspect capabilities
  -> create and explain plan
  -> apply reversible transformations
  -> validate state-dict mapping
  -> construct optimizer and scheduler
  -> initialize backend/cache/mesh
  -> place and shard model/state
  -> compile train/eval operations
  -> warm up
  -> train
```

Creating the optimizer before parameter-layout transformations is a correctness
bug and must be prevented by API design.

## 6. Branch, PR, and commit policy

- This roadmap is delivered as one documentation commit on
  `codex/hf-tpu-parity-roadmap`.
- Implementation begins on `codex/hf-tpu-parity`, based on the accepted
  `m4/training-framework` state (or updated `main` if M4 is merged first).
- Open one draft PR immediately so CI and review follow every feature.
- Every numbered feature below is one future atomic commit.
- Do not squash the implementation PR: commit boundaries are the rollback and
  review boundaries.
- Each feature commit includes its tests and narrowly required documentation.
- Benchmark-only commits contain machine-readable results and analysis, never
  product-code changes.
- Experimental providers stay behind explicit flags until certified.
- No feature may weaken a completed milestone gate without a recorded decision.

Commit subjects use `type(scope): imperative summary`, as listed below.

## 7. Milestone overview

| Milestone | Outcome |
|---|---|
| M0 | Reproducible scope, parity manifest, metrics, and dependency contract |
| M1 | Backend-neutral, task-neutral framework contracts |
| M2 | Universal HF dense-causal intake and CPU conformance |
| M3 | Production packed-binary data pipeline |
| M4 | Correct TrainLM trainer on CPU/CUDA |
| M5 | PyTorch/XLA DP8 runtime and accumulation feasibility |
| M6 | Universal dense-AR TPU compatibility gate |
| M7 | Production checkpointing, observability, and integrity |
| M8 | Capability planner and reversible optimization engine |
| M9 | Memory-efficient causal-LM loss |
| M10 | TPU attention provider family |
| M11 | Projection, optimizer, remat, and HLO tuning |
| M12 | Exact LaughLM 135M parity certification |
| M13 | Optimized dense-AR family certification and 1.3B scaling |
| M14 | V1 production release |
| M15 | TorchTPU backend migration |
| M16 | MoE extension |
| M17 | DLLM extension |

## 8. Milestones and atomic feature commits

### M0 - Reproducibility and release contract

Goal: eliminate ambiguous scope, geometry, metrics, and dependency versions
before implementation work begins.

#### M0-F1 - Dense-AR product scope

Commit: `docs(scope): define dense causal LM support contract`

- Record V1 inclusions, exclusions, support levels, and the rule that fallback
  is not certification.
- Define `trust_remote_code` policy and deferred architecture policy.
- Include the representative family/capability matrix.

Acceptance: one normative scope document is referenced by tests, issue labels,
and release notes.

#### M0-F2 - LaughLM parity manifest

Commit: `test(benchmark): lock LaughLM 135M parity manifest`

- Encode exact model, initialization, loss, optimizer, scheduler, batch,
  precision, data, and hardware geometry in a versioned manifest.
- Record LaughLM reference commit and saved evidence.
- Validate parameter count (`135,611,392`) and tokens/update (`1,048,576`).

Acceptance: CI fails if a parity field changes without updating the manifest
version and decision record.

#### M0-F3 - Benchmark schema and MFU calculator

Commit: `feat(benchmark): add throughput and MFU result schema`

- Define global/device tokens/sec, total/device step time, compile time, HBM,
  FLOPs, non-embedding MFU, logits-inclusive MFU, input idle, and collectives.
- Count actual non-ignored tokens and all data-parallel replicas.
- Require device completion before timed windows end.

Acceptance: calculator reproduces the recorded LaughLM metrics within rounding
tolerance from the locked manifest.

#### M0-F4 - Supported dependency matrix

Commit: `build(deps): define reproducible framework compatibility matrix`

- Separate core, CUDA, PyTorch/XLA, Pallas, profiling, and development extras.
- Pin matched PyTorch/PyTorch-XLA/JAX/libtpu combinations for TPU profiles.
- Define tested Transformers v5 range and compatibility CI policy.
- Record that experimental APIs require wrappers and version gates.

Acceptance: clean CPU and pinned TPU environments resolve without dependency
ambiguity.

M0 exit: scope, parity data, metric formulas, and dependency profiles are
reviewed and immutable enough to support matched benchmarks.

### M1 - Framework contracts and backend boundary

Goal: make the core independent of custom TrainLM models, PyTorch/XLA, and the
future task type.

#### M1-F1 - Configuration ownership boundaries

Commit: `refactor(config): separate model training runtime and optimization config`

- Keep HF `PretrainedConfig` as the model architecture contract.
- Separate training, optimizer, scheduler, loss, data, checkpoint, runtime,
  parallelism, optimization, and monitoring configuration.
- Preserve backward parsing only where it is unambiguous.

Acceptance: no runtime/trainer config loader constructs `TrainLMConfig` unless
the user explicitly selected the TrainLM reference model.

#### M1-F2 - Execution backend protocol

Commit: `feat(runtime): define replaceable execution backend protocol`

- Define device, precision, compile, mesh, shard, synchronize, loader,
  checkpoint, diagnostics, and lifecycle hooks.
- Consolidate the conflicting runtime classes into one public contract.
- Provide a CPU backend for contract testing.

Acceptance: trainer-facing packages contain no `torch_xla` import.

#### M1-F3 - Language-model task protocol

Commit: `feat(tasks): define causal language modeling task protocol`

- Own batch dispatch, label shifting, token counting, loss normalization,
  evaluation outputs, and ignored-token semantics.
- Leave an extension point for future `DiffusionLMTask`.

Acceptance: trainer code does not hard-code causal labels outside the task.

#### M1-F4 - Capability and plan schemas

Commit: `feat(optimization): define capability and execution plan schemas`

- Define immutable specifications for attention, position, normalization, MLP,
  residual, projection, output head, embeddings, and checkpoint layout.
- Define provider decisions: enabled, fallback, or strict error.
- Include stable serialization for reports and checkpoints.

Acceptance: plans can be generated and rendered without mutating a model.

#### M1-F5 - Checkpoint and export contracts

Commit: `feat(checkpoint): define internal resume and HF export contracts`

- Specify canonical state, sharded internal state, transformation metadata,
  optimizer priming, RNG, scheduler, token counter, and data cursor.
- Define atomicity and incomplete-checkpoint detection.

Acceptance: formats are versioned before implementations depend on them.

M1 exit: CPU contract tests demonstrate that trainer, task, optimization, and
runtime interfaces have no model-family or XLA dependency.

### M2 - Universal Hugging Face dense-causal intake

Goal: every representative dense `AutoModelForCausalLM` works through an
unchanged, unoptimized generic path.

#### M2-F1 - Generic HF model provider

Commit: `feat(models): add generic HF causal LM provider`

- Support `AutoConfig.from_pretrained`, `AutoModelForCausalLM.from_config`, and
  `AutoModelForCausalLM.from_pretrained`.
- Preserve revisions, dtype intent, config metadata, tied weights, and local
  loading.
- Disable generation cache during training without changing checkpoint intent.

Acceptance: construction requires no TrainLM model class.

#### M2-F2 - Forward-signature-aware batch dispatch

Commit: `feat(models): add forward signature aware batch dispatch`

- Filter optional batch fields against the model forward signature.
- Preserve `attention_mask`, `position_ids`, cache position, and family-specific
  supported inputs without guessing internal module names.
- Reject missing required inputs with actionable errors.

Acceptance: generic dispatch passes the representative tiny-model matrix.

#### M2-F3 - Generic causal-LM output and loss path

Commit: `feat(models): add generic causal LM output and loss protocol`

- Prefer a model-provided finite loss when labels are supported.
- Otherwise compute correctly shifted standard CE from logits.
- Normalize by actual supervised token count for accumulation correctness.

Acceptance: loss and gradients match direct HF execution on CPU.

#### M2-F4 - Plain HF checkpoint round trip

Commit: `test(models): certify generic HF save and reload round trip`

- Train one update, use canonical HF save, reload without TrainLM-specific model
  code, and compare state/output.
- Cover tied and untied heads.

Acceptance: every representative family passes.

#### M2-F5 - Dense-AR CPU conformance matrix

Commit: `test(models): add dense AR architecture conformance matrix`

- Tiny GPT-2, OPT, GPT-NeoX, BLOOM, Falcon, Phi, Llama, Mistral, Qwen, and Gemma
  configs.
- Check construction, forward, finite loss, backward, one optimizer update,
  save, reload, and deterministic fixed-batch overfit.
- Record unsupported semantic features rather than silently skipping models.

Acceptance: adapters are not used by this matrix.

#### M2-F6 - Explain-compatible baseline

Commit: `feat(models): report generic model capabilities and fallbacks`

- Report architecture, support level, known capabilities, unknowns, and generic
  path selection.
- Avoid asserting optimization readiness before M8.

Acceptance: every model has a stable machine-readable compatibility report.

M2 exit: all representative dense AR families satisfy Level 1 compatibility on
CPU without an optimization adapter.

### M3 - Production packed-binary data pipeline

Goal: replace notebook-local data code with a validated, restartable,
distributed, fixed-shape pipeline for Hugging Face-hosted `.bin` shards.

#### M3-F1 - Versioned binary format and manifest

Commit: `feat(data): define packed binary shard manifest`

- Specify header bytes, token dtype, endianness, token count, vocabulary bounds,
  checksum, document-boundary policy, and shard metadata.
- Reject token IDs `>= vocab_size`.
- Support the existing 1,024-byte-header `uint16` format.

Acceptance: corrupt header, size, checksum, and token-range fixtures fail early.

#### M3-F2 - Hugging Face shard source

Commit: `feat(data): add revision-pinned HF binary shard source`

- Resolve repo, revision, directory, prefix, shard IDs, cache directory, and
  offline reuse.
- Never embed credentials; rely on standard `HF_TOKEN` resolution.
- Separate train and validation manifests.

Acceptance: local mocked Hub tests verify deterministic filenames and revisions.

#### M3-F3 - Batched memmap reader

Commit: `feat(data): add contiguous packed memmap batch reader`

- Read complete `[batch, sequence]` regions rather than Python-building every
  token sample.
- Produce `input_ids`, labels, and optional attention/loss masks.
- Enforce fixed shapes and deterministic incomplete-tail policy.

Acceptance: byte-for-byte output matches the existing notebook reader for valid
samples while fixing the range check.

#### M3-F4 - Deterministic shard mixing and distributed partitioning

Commit: `feat(data): add deterministic shard shuffle and host partitioning`

- Seeded epoch/token-order policy, no accidental duplication between ranks, and
  explicit cross-shard behavior.
- Preserve held-out validation shards.

Acceptance: simulated multi-host tests cover every sample exactly as specified.

#### M3-F5 - Backend-aware prefetch interface

Commit: `feat(data): add backend-aware asynchronous prefetch queue`

- Keep host reader independent of TPU APIs.
- Let runtime backends wrap device transfer and prefetch.
- Make queue depth configurable and measurable.

Acceptance: CPU tests prove ordering/backpressure; TPU tuning starts with depth
`16` but does not assume it is universally optimal.

#### M3-F6 - Checkpointable data cursor

Commit: `feat(data): make packed data iteration exactly resumable`

- Save source revision, shard permutation, shard ID, offset, epoch/token count,
  and RNG state.
- Refuse incompatible resume manifests unless explicitly restarted.

Acceptance: interrupted and uninterrupted runs yield the same next batches.

M3 exit: the 28/2 LaughLM shard split can stream deterministic fixed-shape
batches, validate data, and resume exactly.

### M4 - Correct backend-neutral TrainLM trainer

Goal: complete the trainer independently of TPU optimization.

#### M4-F1 - Training lifecycle and state machine

Commit: `feat(training): implement trainer lifecycle and control state`

- Prepare, train, evaluate, checkpoint, resume, stop, and finalize states.
- Keep orchestration backend-neutral and task-driven.

Acceptance: lifecycle and callback ordering tests cover normal, error, and resume
flows.

#### M4-F2 - Token-correct gradient accumulation

Commit: `feat(training): implement token-normalized gradient accumulation`

- Correctly handle variable ignored-token counts without double division.
- Separate microstep and optimizer-update counters.
- Define scheduler/token accounting in global supervised tokens.

Acceptance: accumulated updates match an equivalent large batch on CPU.

#### M4-F3 - Optimizer and dtype-state factory

Commit: `feat(optim): add backend-neutral AdamW state policy`

- Standard AdamW semantics, weight-decay exclusions if configured, gradient
  clipping, and independently configurable parameter/first/second-moment dtype.
- Reject CUDA `fused`/`foreach` options on incompatible backends.

Acceptance: reference update matches PyTorch AdamW where policies are identical.

#### M4-F4 - WSD and token-based scheduler

Commit: `feat(scheduler): add token-based WSD schedule`

- Warmup, stable, decay, minimum LR ratio, and horizon-token semantics.
- Resume by consumed global tokens.

Acceptance: boundary and resume tests reproduce the locked schedule.

#### M4-F5 - Streaming evaluation without logits retention

Commit: `feat(evaluation): add token-weighted causal LM evaluation`

- Accumulate loss/perplexity without retaining full prediction tensors.
- Limit batches deterministically and isolate the evaluation graph/state.

Acceptance: evaluation matches a direct reference and does not mutate training
state.

#### M4-F6 - Synchronization-safe callback contract

Commit: `feat(training): separate host callbacks from compiled metrics`

- Callbacks consume already-materialized sparse metrics, not live device
  tensors.
- Document which callbacks may cause synchronization.

Acceptance: backend mock proves no hidden scalar extraction in microsteps.

#### M4-F7 - Multi-family fixed-batch overfit

Commit: `test(training): overfit dense AR conformance models`

- Run the generic trainer against representative tiny families.
- Check falling loss, finite gradients, resume equivalence, and HF export.

Acceptance: no family-specific branch exists in the trainer.

M4 exit: the trainer is correct on CPU and has a CUDA smoke path, while the
existing TrainLM model remains only one valid HF reference model.

### M5 - PyTorch/XLA runtime and accumulation feasibility

Goal: establish a stable, backend-contained DP8 execution path before writing
specialized kernels.

#### M5-F1 - Optional PyTorch/XLA backend package

Commit: `feat(runtime): add pinned PyTorch XLA backend`

- Initialize runtime, BF16 policy, device, ordinals, and version checks inside
  the backend.
- Keep Pallas/JAX imports optional and guarded.

Acceptance: importing core TrainLM without TPU extras remains clean.

#### M5-F2 - SPMD mesh and sharding policy

Commit: `feat(runtime): add SPMD data parallel mesh`

- Support DP8 replicated parameters and sharded batch first.
- Translate logical `data/fsdp/tensor` config into backend sharding.
- Reject inconsistent mesh products.

Acceptance: v5e-8 smoke confirms correct global/local shapes and gradient
reduction.

#### M5-F3 - Persistent compilation cache and static-shape guard

Commit: `feat(runtime): add XLA cache and recompile guard`

- Initialize cache before computation.
- Fingerprint train/eval graphs and report unexpected compiles.
- Enforce fixed batch, sequence, mask, and accumulation structures.

Acceptance: no recompilation occurs after the allowed warmup graph set.

#### M5-F4 - Compiled train microstep and optimizer update

Commit: `feat(runtime): compile XLA training operations`

- Compile supported boundaries through the backend rather than trainer imports.
- Keep logging, data resolution, and checkpoint I/O outside compiled regions.

Acceptance: forward, backward, reduction, clip, optimizer, and scheduler are
correct on v5e-8.

#### M5-F5 - Gradient-accumulation strategy spike

Commit: `perf(runtime): select v5e gradient accumulation strategy`

- A/B compiled microstep + compiled update, full unroll, XLA loop/scan where
  compatible, and any current backend-native option.
- Measure compile time, dispatches, step time, HBM, and interaction with custom
  Pallas calls.
- Do not adopt `scan_layers` while its custom-Pallas limitation applies.

Acceptance: record a decision for microbatch `2`, accumulation `32`, sequence
`2048`, including a fallback for unsupported loop capture.

#### M5-F6 - XLA diagnostics bundle

Commit: `feat(runtime): expose XLA metrics HLO and profile diagnostics`

- Metrics report, compile/execute counts, `aten::` fallback counters, HLO dump
  controls, sync-safe timing, and XProf capture metadata.
- Redact credentials and bound artifact size.

Acceptance: a benchmark artifact proves graph count and absence of unexpected
CPU fallbacks.

#### M5-F7 - Generic 135M XLA benchmark

Commit: `perf(benchmark): record generic HF 135M XLA baseline`

- Exact parity geometry, generic HF loss/attention, fake data and real data.
- Report cold compile separately from steady state.

Acceptance: stable graph, finite updates, and at least `600K tok/s`; otherwise
the roadmap pauses for runtime diagnosis.

M5 exit: PyTorch/XLA is a functional replaceable backend, DP8 is correct, and
the generic exact-geometry run clears the first go/no-go gate.

### M6 - Universal dense-AR TPU compatibility gate

Goal: prove the framework is not Llama-specific before optimized Llama parity
work dominates the implementation.

#### M6-F1 - TPU smoke for positional semantics

Commit: `test(tpu): cover learned RoPE and ALiBi causal models`

- Tiny learned-position, RoPE, and ALiBi models run five updates.
- Check finite state and stable graph after warmup.

Acceptance: all three positional classes complete five finite updates with only
the expected warmup compilations.

#### M6-F2 - TPU smoke for attention layouts

Commit: `test(tpu): cover dense MHA GQA and MQA layouts`

- Separate and packed projection representatives.
- Generic path only; no optimized adapter required.

Acceptance: MHA, GQA, and MQA representatives compile and update correctly
without an optimization adapter or CPU fallback.

#### M6-F3 - TPU smoke for block and MLP layouts

Commit: `test(tpu): cover norm MLP and residual layouts`

- LayerNorm/RMSNorm, GELU/gated MLP, serial/parallel residual.

Acceptance: every block/MLP class completes the same generic TPU smoke suite
with stable output structures.

#### M6-F4 - TPU generic checkpoint round trip

Commit: `test(tpu): certify generic dense AR save resume and export`

- Internal resume and plain HF export/reload across the representative matrix.

Acceptance: resumed next-update state and clean Transformers-only reload match
the uninterrupted/canonical references within defined tolerances.

M6 exit: every V1 capability cluster trains generically on TPU, does not
recompile after warmup, has no unexpected CPU fallback, and round-trips through
HF without an optimization adapter.

### M7 - Production checkpointing, observability, and integrity

Goal: make long TPU pretraining recoverable and measurable before aggressive
model mutation.

#### M7-F1 - Distributed internal resume checkpoints

Commit: `feat(checkpoint): add sharded distributed resume checkpoints`

- Model, optimizer, scheduler, trainer, RNG, token counter, data cursor, plan,
  and manifest versions.
- Direct shard I/O where supported.

Acceptance: interrupted DP8 run resumes to an equivalent next update.

#### M7-F2 - Asynchronous checkpoint manager

Commit: `feat(checkpoint): add asynchronous checkpoint lifecycle`

- Backend implementation may use XLA/DCP planners and async save.
- Surface in-flight, completion, failure, retention, and shutdown semantics.
- Never report success before durable completion.

Acceptance: training resumes while host persistence runs; finalization waits or
fails clearly.

#### M7-F3 - Canonical HF export

Commit: `feat(checkpoint): export canonical Hugging Face checkpoints`

- `config.json`, safetensors, generation config, tokenizer passthrough, tied
  weights, and shard index.
- Initially identity layout; M8 adds transformed layouts.

Acceptance: plain Transformers reload matches model output.

#### M7-F4 - Synchronization-safe telemetry

Commit: `feat(monitoring): add TPU throughput memory and MFU telemetry`

- Sparse logging, correct device waits at benchmark boundaries, compile
  accounting, HBM, input idle, and collective metrics.
- No `.item()` or tensor printing inside compiled/microstep paths.

Acceptance: metric overhead is measured and bounded at production intervals.

#### M7-F5 - Sparse training-integrity gates

Commit: `feat(monitoring): add configurable training integrity checks`

- Loss/gradient/parameter finiteness, update magnitude, token-range, and data
  continuity checks at a configurable interval.
- Checks are off or sparse in throughput runs.

Acceptance: injected corruption is detected without silently changing normal
throughput configuration.

M7 exit: 200 updates can survive checkpoint/resume, produce trustworthy metrics,
and detect injected integrity failures.

### M8 - Capability planner and reversible optimization engine

Goal: safely transform loaded HF models in memory without coupling the trainer
to family internals.

#### M8-F1 - Structural model inspector

Commit: `feat(optimization): inspect dense causal LM capabilities`

- Prefer public HF config/module interfaces and registered metadata.
- Describe unknown semantics explicitly.
- Never infer causality, mask behavior, or weight tying from names alone.

Acceptance: reports match hand-authored fixtures for every representative
family.

#### M8-F2 - Adapter registry and version guards

Commit: `feat(optimization): add optional model adapter registry`

- Resolve by explicit class/model-type support plus semantic validation.
- Keep family-specific paths out of trainer/runtime/data/checkpoint modules.
- Include supported Transformers version ranges.

Acceptance: removing all adapters still preserves M6 compatibility.

#### M8-F3 - Pure optimization planner

Commit: `feat(optimization): select kernels transforms and fallbacks`

- Match capability specs against backend providers, shapes, dtypes, masks, and
  training/backward support.
- Implement `auto`, `required`, `disabled`, and explicit provider policies.

Acceptance: deterministic plan snapshots explain every decision.

#### M8-F4 - Transactional transformation lifecycle

Commit: `feat(optimization): apply validated reversible model transforms`

- Plan first, apply before optimizer creation, validate parameters and aliases,
  and retain reverse metadata.
- On unsupported plans, fall back before mutation; never leave partial patches.
- Avoid global import-time monkey patches and hot-path forward hooks.

Acceptance: injected transformation failure leaves a usable original model.

#### M8-F5 - State-dict conversion and alias preservation

Commit: `feat(optimization): add reversible parameter layout mappings`

- Pack/split mappings, tied-parameter alias handling, dtype/shape checks, and
  transformed internal checkpoint metadata.
- Load pretrained HF layouts and export canonical HF layouts.

Acceptance: transformed train/save/reload output and weights match the expected
canonical model.

#### M8-F6 - TrainLM explain report

Commit: `feat(optimization): expose model optimization explanation`

- Human and JSON output for capabilities, transforms, providers, fallbacks,
  backend, graph policy, and certification.

Acceptance: strict mode fails before TPU allocation when a required capability
is unsupported.

M8 exit: no-op and fixture transformations are transactional, explainable,
checkpoint-safe, and independent of the execution backend.

### M9 - Memory-efficient causal-LM loss

Goal: remove the full FP32 logits bottleneck while preserving exact causal and
token-normalization semantics.

#### M9-F1 - Reference chunked linear CE

Commit: `feat(loss): add reference chunked linear causal cross entropy`

- Operate on final hidden states, output weight, labels, and optional bias.
- Correct shift, ignore index, token count, tied/untied head, and FP32-stable
  reductions.
- Configurable chunk size and z-loss.

Acceptance: FP32 loss and hidden/head gradients match full-logits reference
within defined tolerances across edge cases.

#### M9-F2 - Optimized training-view wrapper

Commit: `feat(loss): bypass full HF logits during optimized training`

- Use adapter/capability mapping to obtain backbone final hidden state and
  canonical output weight.
- Preserve the original HF model for inference/export.
- Fall back to standard HF loss when separation is not proven safe.

Acceptance: no full `[B,S,V]` tensor appears in the optimized HLO.

#### M9-F3 - Rematerialized chunk backward

Commit: `feat(loss): add rematerialized chunked loss backward`

- Bound activation/HBM use and preserve accumulation semantics.
- Benchmark chunk sizes `2048`, `4096`, and `8192`; start from LaughLM's `4096`.

Acceptance: select based on matched throughput/HBM evidence.

#### M9-F4 - TPU loss kernel provider evaluation

Commit: `perf(loss): evaluate native XLA Pallas and Tokamax loss providers`

- Compare pure XLA chunking, PyTorch/XLA-wrapped Pallas, and Tokamax's
  memory-efficient linear CE where integration/backward/version support permit.
- Tokamax remains optional because its API is under active development.

Acceptance: record correctness, HBM, throughput, compatibility, and selected
provider; do not depend on an unstable provider without a fallback.

#### M9-F5 - Multi-family fused-loss adapters

Commit: `test(loss): certify chunked loss across dense output-head layouts`

- Tied/untied linear heads and representative family backbone outputs.
- HF save/reload after optimized loss training.

Acceptance: every covered output-head layout passes loss/gradient parity and
canonical HF round trip without materializing full logits.

M9 exit: the reference geometry no longer materializes full logits, gradient
parity passes, and the selected v5e loss provider is benchmarked.

### M10 - TPU attention provider family

Goal: provide semantically correct memory-efficient causal attention across V1
attention classes.

#### M10-F1 - Canonical attention specification

Commit: `feat(attention): define canonical causal attention specification`

- Q/K/V layouts, Q/KV head counts, scale, causal/window mask, segments, ALiBi,
  dropout, soft-cap, QK norm, and output layout.
- Separate positional transformation from the attention kernel where possible.

Acceptance: specs represent every V1 family without model-name conditionals.

#### M10-F2 - HF attention and mask registration

Commit: `feat(attention): integrate TrainLM attention with HF interfaces`

- Use `AttentionInterface` and `AttentionMaskInterface` where the model supports
  them.
- Always register matching mask semantics; never allow an implicit `None` mask.
- Preserve a module-adapter path when public interfaces are insufficient.

Acceptance: causal leakage tests fail if mask registration is removed.

#### M10-F3 - PyTorch/XLA Pallas attention provider

Commit: `feat(attention): add XLA Pallas causal attention provider`

- Wrap supported built-in/custom Pallas attention behind `KernelProvider`.
- Guard JAX import order and matched dependency versions.
- Register/validate forward and backward behavior.

Acceptance: MHA forward/backward matches reference and appears as the expected
custom call/HLO.

#### M10-F4 - GQA and MQA without physical KV repeat

Commit: `feat(attention): support grouped and multi query TPU attention`

- Support unequal Q/KV head counts in kernel/layout or use an explicitly
  benchmarked fallback.
- Do not silently expand K/V in HBM.

Acceptance: MHA `8/8`, GQA `8/4`, and MQA `8/1` correctness and HBM tests pass.

#### M10-F5 - ALiBi and sliding-window attention

Commit: `feat(attention): add ALiBi and sliding window capabilities`

- Provider support where safe; efficient XLA fallback otherwise.
- Packed segment masks remain separately gated until proven.

Acceptance: position/mask semantic tests cover boundary tokens and gradients.

#### M10-F6 - Attention autotune and provider cache

Commit: `perf(attention): tune v5e attention provider configurations`

- Tune forward/backward tiles for locked shapes and cache by hardware, shape,
  dtype, mask, and provider version.
- Record cold tuning separately from steady state.

Acceptance: deterministic selected configurations and reproducible benchmark
artifact.

#### M10-F7 - Combined loss and attention benchmark

Commit: `perf(benchmark): certify optimized loss and attention stage`

- Exact 135M geometry, fake and real data, correct device timing and HLO.

Acceptance: at least `850K tok/s`, no unexpected recompile/fallback, and lower
peak HBM than the full-logits generic baseline.

M10 exit: MHA/GQA/MQA have correct optimized providers, positional/mask variants
have explicit provider or fallback decisions, and the second go/no-go gate is
cleared.

### M11 - Projection, optimizer, rematerialization, and HLO tuning

Goal: close the remaining parity gap with reversible structural transforms and
measured compiler-level tuning.

#### M11-F1 - Reversible QKV packing

Commit: `feat(optimization): pack compatible QKV projections`

- Separate Q/K/V, packed QKV, and separate-Q/packed-KV layouts.
- Preserve biases, head dimensions, tensor aliases, and canonical export keys.
- Replace the complete projection/attention path where three separate matmuls
  would otherwise remain.

Acceptance: output/gradient/update parity and pretrained/export round trip pass.

#### M11-F2 - Reversible gated-MLP packing

Commit: `feat(optimization): pack compatible gate and up projections`

- SwiGLU/GeGLU-compatible gate/up packing with exact activation semantics.
- Preserve non-gated GELU paths.

Acceptance: output/gradient/update parity and canonical export pass.

#### M11-F3 - XLA-native elementwise fusion audit

Commit: `perf(hlo): audit norm RoPE residual and MLP fusion`

- Inspect optimized HLO before adding custom kernels.
- Identify copies, transposes, materializations, custom-call boundaries, and
  poor layouts.
- Add a custom norm/RoPE/activation provider only in a later isolated commit if
  the audit proves a benefit.

Acceptance: decision record for each candidate; no speculative custom kernel.

#### M11-F4 - Decoder rematerialization policies

Commit: `feat(optimization): add structural decoder rematerialization policies`

- None, block, attention, MLP, and loss-chunk policies.
- Apply before FSDP wrapping where required.
- Keep scan-layer experiments separate because of Pallas/AOTAutograd limits.

Acceptance: selected 135M policy is based on step/HBM data, not LaughLM parity by
assumption.

#### M11-F5 - XLA optimizer-state precision path

Commit: `feat(optim): optimize XLA AdamW state and update graph`

- BF16 first moment option, FP32 second moment baseline, global clipping, weight
  decay, and reduction ordering.
- Verify state precision and resume compatibility.

Acceptance: matched update correctness and measured HBM/step impact.

#### M11-F6 - Matched batch and prefetch tuning

Commit: `perf(runtime): tune microbatch accumulation and prefetch geometry`

- Recheck `MB2/GA32`, `MB1/GA64`, and safe alternatives after optimized kernels.
- Recheck prefetch depth around `16`.
- Reject changes that improve an incomplete timing window only.

Acceptance: one selected production geometry with full device wait and real data.

#### M11-F7 - Final HLO and host-overhead closure

Commit: `perf(hlo): remove residual parity path bottlenecks`

- Address demonstrated transposes, graph breaks, CPU fallbacks, excessive
  collectives, host sync, or input stalls one measured issue at a time.
- Any code fix should be its own preceding feature commit; this commit records
  the final evidence only.

Acceptance: candidate reaches at least the hard parity thresholds before M12.

M11 exit: transformations are reversible and correct, the final graph clears
the hard 90% parity thresholds, and remaining differences are explained.

### M12 - Exact LaughLM 135M parity certification

Goal: convert a promising benchmark into release-quality evidence.

#### M12-F1 - Matched initialization and numerical contract

Commit: `test(parity): align 135M initialization and loss semantics`

- Match initialization standard deviations, residual scaling, norm epsilon,
  RoPE theta, label shift, z-loss, optimizer, scheduler, and dtype policy.
- Compare early updates on deterministic data where framework ordering permits.

Acceptance: all semantic differences are eliminated or explicitly justified.

#### M12-F2 - Three-run steady-state benchmark

Commit: `perf(parity): certify repeated v5e 135M throughput`

- Three matched runs with warmup excluded, medians and dispersion reported.
- Include compile cache cold/warm behavior, HBM, HLO fingerprints, compile
  counts, input idle, and MFU.

Acceptance: every run clears `912.6K tok/s` and `47.8%` MFU; preferred release
clears `963.3K tok/s` and `50.4%` median MFU.

#### M12-F3 - Real-shard 200-update stability run

Commit: `test(parity): validate 200 update real data stability`

- Use revision-pinned diverse training shards, sparse integrity checks,
  evaluation, checkpoint, and resume.
- Verify finite state, expected loss trend, data continuity, and stable graph.

Acceptance: no recompile after warmup, no unexpected fallback, and successful
resume/export.

#### M12-F4 - Plain-HF export certification

Commit: `test(parity): certify optimized HF checkpoint interoperability`

- Export packed internal layouts back to canonical HF keys.
- Reload in a clean Transformers-only environment and compare logits/loss.

Acceptance: no TrainLM code is required to consume the final checkpoint.

#### M12-F5 - Parity report

Commit: `docs(benchmark): publish LaughLM TrainLM parity report`

- Repro commands, environment, configs, metrics, limitations, profile/HLO
  summaries, and exact comparison.

Acceptance: a reviewer can reproduce the locked result from the report without
undocumented environment or configuration choices.

M12 exit: exact 135M parity is independently reproducible and clears the hard
release gate. No performance conclusion is generalized to other families yet.

### M13 - Optimized dense-AR certification and larger-model scaling

Goal: prove TrainLM is a framework, not one optimized Llama configuration.

#### M13-F1 - GPT-2 and OPT optimized capability mapping

Commit: `feat(adapters): optimize learned-position dense causal models`

- Reuse existing packed QKV where present, GELU/LayerNorm capabilities, chunked
  loss, and attention provider.
- No Llama field names outside adapters.

Acceptance: GPT-2 and OPT optimized paths pass output/gradient/update parity,
stable TPU graph, and canonical HF export.

#### M13-F2 - GPT-NeoX and BLOOM optimized capability mapping

Commit: `feat(adapters): optimize parallel residual and ALiBi models`

- Parallel residual, RoPE or ALiBi, packed/separate projection semantics.

Acceptance: GPT-NeoX and BLOOM optimized paths preserve residual, position, and
mask semantics and pass the shared TPU certification suite.

#### M13-F3 - Falcon and Phi optimized capability mapping

Commit: `feat(adapters): optimize MQA GQA and nonstandard dense blocks`

- MQA/GQA, parallel blocks, fused or partial projection layouts, and family
  activation semantics.

Acceptance: Falcon and Phi optimized paths pass MQA/GQA correctness, HBM,
stable-graph, and canonical export gates.

#### M13-F4 - Llama Mistral Qwen and Gemma capability mapping

Commit: `feat(adapters): certify RoPE gated dense model families`

- Share structural capabilities while preserving sliding window, GeGLU/SwiGLU,
  embedding scale, soft-cap, QK norm, and model-specific differences.

Acceptance: each advertised family passes semantic fixtures for its differences
and the common optimized train/resume/export suite.

#### M13-F5 - 135M-class cross-family performance matrix

Commit: `perf(certification): benchmark dense AR capability families`

- Approximately matched parameter/sequence/batch configurations on v5e-8.
- Report raw throughput and architecture-adjusted MFU.
- Require stable graph, no unexpected fallback, and a model-specific target
  recorded before granting Level 3 certification.
- Dense full-attention 135M-class models target at least 45% non-embedding MFU
  unless a reviewed architectural roofline explains otherwise.

Acceptance: every advertised V1 family has a published capability/provider and
performance record; generic-only families are labeled Compatible, not Certified.

#### M13-F6 - SPMD FSDP runtime

Commit: `feat(runtime): add backend-neutral FSDP mesh policy`

- Start with logical `data=4, fsdp=2` and compatible block wrapping.
- Shard optimizer/model state, apply remat in correct order, and checkpoint
  directly from shards.

Acceptance: correct 1.3B-class train/resume/export smoke on v5e-8.

#### M13-F7 - 1.3B LaughLM-class benchmark

Commit: `perf(benchmark): certify 1.3B FSDP scaling path`

- Compare against the recorded LaughLM result of about `41.9K tok/s` and `42.6%`
  median MFU at the matched sequence/mesh geometry.
- Separate architecture and runtime differences.

Acceptance: target is set from a fully matched manifest before certification;
results include collective and HBM analysis.

M13 exit: all advertised dense-AR capability clusters are Level 3 or explicitly
listed as lower support; the framework has a working larger-model FSDP path.

### M14 - Dense-AR V1 production release

Goal: turn certified components into a safe, documented, reproducible user
release.

#### M14-F1 - Stable public training API

Commit: `feat(api): finalize TrainLM dense AR pretraining interface`

- Code-first and YAML examples for from-config and from-pretrained workflows.
- Backward-compatible deprecation policy for public names.

Acceptance: public API tests execute both workflows without importing internal
packages or custom TrainLM model classes.

#### M14-F2 - Secure HF binary training example

Commit: `docs(tutorial): add HF model and binary shard TPU pretraining guide`

- `HF_TOKEN` secret handling, revision pinning, train/validation shards,
  `trainer.explain`, resume, and HF export.
- Never include a real credential.

Acceptance: the documented example passes secret scanning and a revision-pinned
smoke run from a clean environment.

#### M14-F3 - CI and hardware certification tiers

Commit: `ci(test): add dense AR release certification tiers`

- Tier 0 per-commit CPU correctness.
- Tier 1 CUDA portability smoke where available.
- Tier 2 scheduled v5e correctness/compile tests.
- Tier 3 release/manual performance and 200-update runs.

Acceptance: branch protection/release procedure identifies the required tier for
every test and prevents a release without current Tier 3 evidence.

#### M14-F4 - Preemption and failure recovery test

Commit: `test(checkpoint): validate preemption and incomplete save recovery`

- Kill during compute, host staging, and persistence; resume only from durable
  checkpoints.

Acceptance: every injected failure resumes from the latest complete checkpoint
without accepting partial state or skipping data silently.

#### M14-F5 - V1 support manifest and release notes

Commit: `docs(release): publish dense AR support and certification matrix`

- Exact versions, hardware, providers, caveats, generic fallback status, and
  TorchTPU migration status.

Acceptance: the published manifest is machine-readable and agrees with
`trainer.explain()` for each advertised family/configuration.

M14 exit: a new user can securely reproduce a certified v5e-8 run and export a
plain HF model using documented APIs.

### M15 - TorchTPU backend migration

Trigger: TorchTPU has a public, installable, documented training stack with the
required APIs. Do not invent APIs from the announcement.

#### M15-F1 - TorchTPU compatibility ADR

Commit: `docs(runtime): map TorchTPU APIs to TrainLM backend contracts`

- Device, `torch.compile`, synchronization, DDP/FSDPv2/DTensor, Pallas/JAX
  custom kernels, checkpointing, profiler, and cache.
- Record gaps and version gates.

Acceptance: every `ExecutionBackend` and `KernelProvider` method maps to a
public TorchTPU API or is explicitly recorded as a blocking gap.

#### M15-F2 - Native TorchTPU execution backend

Commit: `feat(runtime): add native TorchTPU backend`

- Implement existing `ExecutionBackend`; do not branch the trainer.
- Prefer standard PyTorch device/distributed APIs promised by TorchTPU.

Acceptance: unchanged CPU contract tests plus one dense causal training smoke
run execute through backend selection alone.

#### M15-F3 - TorchTPU Pallas kernel providers

Commit: `feat(kernels): register TrainLM kernels with TorchTPU`

- Reuse validated kernel mathematics where possible; replace XLA bridge and
  registration details.

Acceptance: attention and loss provider forward/backward parity passes through
TorchTPU capture with no backend calls in core optimization code.

#### M15-F4 - TorchTPU correctness and checkpoint parity

Commit: `test(runtime): certify TorchTPU dense AR compatibility`

- Run M6 and checkpoint/export suites unchanged against the new backend.

Acceptance: the unmodified M6 and M7 certification suites pass on TorchTPU.

#### M15-F5 - TorchTPU performance parity

Commit: `perf(runtime): compare TorchTPU and PyTorch XLA backends`

- Exact 135M and 1.3B manifests, compilation modes, kernels, HBM, and MFU.

Acceptance: exact matched reports demonstrate that TorchTPU meets the active
release thresholds before it becomes the preferred TPU backend.

M15 exit: TorchTPU meets or beats the accepted PyTorch/XLA correctness and
performance gates. Deprecation of PyTorch/XLA is a separate decision, not an
automatic consequence.

### M16 - MoE extension

Goal: add efficient autoregressive MoE without weakening dense paths.

#### M16-F1 - MoE capability and task contract

Commit: `feat(moe): define router expert and auxiliary loss capabilities`

- Top-k routing, capacity, load balance, shared experts, grouped weights, and
  checkpoint layout.

Acceptance: capability fixtures represent the selected HF MoE families without
router/model names in trainer or runtime code.

#### M16-F2 - Grouped/ragged expert kernels

Commit: `feat(moe): add backend grouped expert kernel providers`

- Evaluate Pallas/Tokamax ragged dot and native backend alternatives.
- Forward/backward, empty groups, determinism, and shape gates.

Acceptance: the selected provider passes expert-output/gradient parity including
empty and imbalanced routing cases, with an explicit fallback.

#### M16-F3 - Expert parallel mesh

Commit: `feat(moe): add expert parallel routing and collectives`

- Logical expert axis, all-to-all/token dispatch, overlap, and load telemetry.

Acceptance: multi-device token dispatch returns every token to its source order,
preserves gradients, and reports expert imbalance/communication cost.

#### M16-F4 - MoE checkpoint and resume

Commit: `feat(moe): add expert-sharded checkpoint conversion`

- Internal shards and canonical HF MoE export.

Acceptance: expert-sharded resume matches uninterrupted training and exports a
checkpoint loadable by plain Transformers.

#### M16-F5 - MoE certification matrix

Commit: `perf(moe): certify representative HF MoE families`

- Correctness, router convergence, expert balance, communication, HBM, MFU,
  resume, and HF round trip.

Acceptance: every advertised MoE family has a model-specific target and current
Level 3 certification record; dense V1 results do not regress.

M16 exit: representative HF MoE causal LMs are Certified; dense regression
suite remains unchanged.

### M17 - Diffusion language model extension

Goal: add DLLM-specific training semantics on the shared runtime and
optimization infrastructure.

#### M17-F1 - Diffusion LM task protocol

Commit: `feat(dllm): add diffusion language modeling task`

- Timestep/noise sampling, corruption/masking, target construction, token
  weighting, and evaluation contract.

Acceptance: deterministic fixtures verify schedule, corruption, targets, token
weights, and resume state independently of causal-LM semantics.

#### M17-F2 - Diffusion attention and mask capabilities

Commit: `feat(dllm): add noncausal and mixed mask specifications`

- Keep causal and diffusion mask semantics impossible to confuse.

Acceptance: bidirectional/mixed-mask leakage and boundary fixtures pass, while a
causal mask cannot be selected accidentally for a DLLM task.

#### M17-F3 - Diffusion loss providers

Commit: `feat(dllm): add optimized diffusion token losses`

- Chunked/fused loss where mathematically appropriate, with gradient parity.

Acceptance: reference and optimized diffusion loss/gradient results match across
timesteps, ignored tokens, and supported prediction parameterizations.

#### M17-F4 - Compiled denoising training step

Commit: `feat(dllm): compile diffusion training updates`

- Keep host randomness and schedule changes from causing unintended
  recompilation.

Acceptance: the selected denoising update has a bounded warmup graph set and no
step-dependent recompilation in a representative run.

#### M17-F5 - DLLM certification matrix

Commit: `perf(dllm): certify representative HF diffusion language models`

- Quality/correctness, HBM, graph stability, throughput, checkpoint/resume, and
  export contracts.

Acceptance: every advertised DLLM has a model/task-specific target and current
Level 3 record, with causal dense-AR regression gates unchanged.

M17 exit: representative DLLMs are Certified under a distinct task contract;
the causal trainer semantics remain unchanged.

## 9. Cross-cutting validation requirements

Every optimization feature must pass all applicable gates before defaulting on.

### 9.1 Semantic correctness

- Forward outputs versus the original HF operation/model.
- Loss, label shift, masks, z-loss, and token reduction.
- Gradients for inputs and every transformed parameter.
- One-update and multi-update optimizer parity.
- Tied weight aliasing and state-dict key/shape/dtype preservation.
- Pretrained load and canonical HF export/reload.
- Determinism differences documented where operation reordering changes bits.

### 9.2 Compilation health

- Fixed expected graph count after warmup.
- No unexpected `aten::` CPU fallback.
- No tensor scalar extraction in the hot path.
- No data-dependent recompilation.
- Compile cache keyed by dependency/provider/shape/config fingerprint.
- Custom providers support backward and the active capture mechanism.

### 9.3 Performance evidence

- Cold compile, warm compile-cache, and steady-state timings separated.
- Correct final device synchronization.
- Median and dispersion, not best-case only.
- Fake-data compute ceiling and real-data end-to-end result.
- HBM, input idle, collectives, HLO fingerprint, and profiler summary.
- Optimization A/B uses identical model, data, precision, batch, and seed.

### 9.4 Regression policy

- A provider does not become `auto` default without a certification record.
- A Transformers or backend version bump reruns affected capability suites.
- A changed HLO fingerprint at the parity geometry triggers performance review.
- The generic path is never removed merely because all current adapters pass.

## 10. Risks and mandatory mitigations

| Risk | Mitigation |
|---|---|
| HF model internals change | Public interfaces first, thin versioned adapters, multi-version conformance CI |
| A patch changes semantics | Transactional plans, reference/gradient tests, strict mask tests, reversible export |
| Full logits remain hidden in graph | HLO shape assertion and peak-HBM gate |
| Pallas/JAX version mismatch | Optional pinned extra, import guard, provider version gate, native fallback |
| Gradient accumulation creates huge graph | M5 strategy spike before kernel investment |
| `scan_layers` conflicts with Pallas | Keep experimental; do not combine until backend supports it |
| Logging/checkpoint forces sync | Sparse materialized metrics and backend async checkpoint lifecycle |
| "Any AR" becomes Llama-only | M6 before parity work; M13 required for V1 framework certification |
| Generic fallback is marketed as optimized | Three support levels and public certification manifest |
| TorchTPU obsoletes XLA-specific code | Backend/kernel interfaces; no `torch_xla` in trainer/core |
| Remote model runs arbitrary code | `trust_remote_code=False` default and explicit uncertified policy |
| Hub credential leakage | Standard `HF_TOKEN`, secret scanning, no notebook literals |

## 11. Critical path and stop conditions

Critical path:

```text
M0 -> M1 -> M2 -> M4 -> M5 -> M6
       \-> M3 ---------^   |
                           M7 -> M8 -> M9 -> M10 -> M11 -> M12 -> M13 -> M14
```

- M3 can proceed alongside M2-M4 after contracts land.
- M6 must pass before model-family-specific optimization dominates work.
- M9 and M10 may prototype in parallel only after M8 contracts are stable.
- M13 begins only after the exact parity path clears the M12 hard gate, except
  for small adapter correctness work needed to keep M6 healthy.
- M16 and M17 begin only after dense-AR V1 is released.
- If M5 is below `600K tok/s`, pause kernels and fix the runtime.
- If M10 is below `850K tok/s`, pause family expansion and inspect graph/kernel
  quality.
- If M12 is below `912.6K tok/s`, do not claim LaughLM-class performance.

## 12. Reference implementations and API decisions

These are references, not blanket dependencies.

### Hugging Face

- [AttentionInterface and AttentionMaskInterface](https://huggingface.co/docs/transformers/attention_interface): use public attention replacement where supported; always pair attention and mask registration.
- [Transformers kernels](https://huggingface.co/docs/transformers/kernels): mirror device/provider selection and explicit fallback concepts.
- [Writing HF kernels](https://huggingface.co/docs/transformers/main/kernel_doc/writing_kernels): reference parameter conversion and module-fusion layout patterns.
- [Loading/training kernels](https://github.com/huggingface/transformers/blob/main/docs/source/en/kernel_doc/loading_kernels.md): reference training/compile mode awareness.
- [Transformers causal loss implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py): baseline semantics; replacing only this loss is insufficient to avoid full logits.

Decision: integrate public HF extension points when they express the required TPU
operation, but keep a TrainLM capability/transform layer because current Kernel
Hub device coverage does not include a complete TPU provider and algorithmic
linear-CE replacement needs final hidden states plus output weights.

### PyTorch/XLA and OpenXLA

- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/): current TPU backend.
- [Compile API](https://docs.pytorch.org/xla/master/eager_mode.html): compile the performance-sensitive training operation.
- [SPMD](https://docs.pytorch.org/xla/master/spmd.html): DP/FSDP implementation reference.
- [Pallas custom kernels](https://docs.pytorch.org/xla/release/r2.6/features/pallas.html): attention and custom TPU provider bridge.
- [Persistent compilation cache](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html): initialize before computation.
- [XLA diagnostics](https://docs.pytorch.org/xla/master/debug.html): compile counts, fallbacks, HLO, and metrics.
- [Distributed checkpointing](https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html): DCP planners, async persistence, and preemption patterns.
- [Scan and scan_layers](https://docs.pytorch.org/xla/master/features/scan.html): optional compile-time experiment with documented Pallas/AOTAutograd limitations.
- [OpenXLA HLO dumps](https://openxla.org/xla/hlo_dumps) and [XProf](https://openxla.org/xprof): compiler and device evidence.
- [Tokamax](https://github.com/openxla/tokamax): evaluate TPU linear CE, Splash/Pallas work, and autotuning; keep optional while APIs remain under development.

### TorchTPU

- [TorchTPU technical announcement](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/): future native PyTorch TPU tensors, `torch.compile` to StableHLO/XLA, Pallas/JAX custom kernels, DDP/FSDPv2/DTensor.
- [Native TPU backend RFC](https://github.com/pytorch/xla/issues/9684): direction away from XLA-specific user APIs.
- [PyTorch/XLA repository notice](https://github.com/pytorch/xla): TorchTPU is intended to replace PyTorch/XLA once public.

Decision: do not wait for TorchTPU and do not invent its API. Build reusable
model/data/trainer/optimization contracts now, isolate PyTorch/XLA, and add the
native backend when its public contract exists.

### Open-source training and patching patterns

- [Unsloth](https://github.com/unslothai/unsloth) and [Unsloth compiler](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/compiler.py): reference seamless HF-facing acceleration and fused loss; avoid global/source-string patching as the primary mechanism.
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel): reference family patching, fused linear CE, convergence tests, and benchmark discipline; its Triton kernels are not TPU implementations.
- [VeOmni kernel selection](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/design/kernel_selection.md): reference config-driven operation slots and explicit provider selection.
- [TorchTitan](https://github.com/pytorch/torchtitan): reference native training loop, composable parallelism, WSD, BF16 optimizer state, distributed checkpointing, and MFU reporting; do not copy its model-specific model ownership.
- [TorchPrime archive](https://github.com/AI-Hypercomputer/torchprime): mine PyTorch/XLA TPU patterns and Splash integration, but do not depend on it; it was archived in favor of the native TorchTPU direction.
- [MaxText](https://github.com/AI-Hypercomputer/maxtext): reference mature TPU performance, parallelism, and profiling practices; TrainLM retains the PyTorch/HF model contract.

## 13. Definition of done

Dense-AR V1 is done only when all of the following are true:

- A user can load or initialize any in-scope representative HF causal LM without
  changing TrainLM code.
- All representative capability clusters pass CPU and TPU compatibility,
  training, resume, and plain-HF export.
- `trainer.explain()` accurately reports the plan and never hides a fallback.
- Exact 135M TrainLM clears `912.6K tok/s` and `47.8%` non-embedding MFU on
  v5e-8 in three matched runs, with the preferred release target at `963.3K`
  tok/s and `50.4%` MFU.
- A 200-update real-shard run is finite, stable, checkpointable, resumable, and
  free of unexpected recompilation/CPU fallback.
- Advertised optimized families have model-specific certification records;
  generic-only models are labeled Compatible.
- Internal transformed checkpoints and canonical HF exports are both tested.
- Core TrainLM has no dependency on PyTorch/XLA internals, so TorchTPU can be
  implemented as another backend.
- Documentation contains secure, reproducible from-config and from-pretrained
  workflows using packed HF-hosted `.bin` shards.

That release establishes the intended product: Hugging Face model compatibility
at the surface, TrainLM's capability-driven optimizations in memory, and a
replaceable high-performance TPU execution backend underneath.
