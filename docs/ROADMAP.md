# TrainLM Roadmap

- **Status:** Active implementation plan
- **Product:** Hugging Face-native, TPU-optimized autoregressive pretraining
- **Primary hardware:** Google TPU v5e-8, single VM
- **Current TPU backend:** PyTorch/XLA
- **Future TPU backend:** TorchTPU, after public training APIs are available
- **Dataset path:** Hugging Face-hosted packed `.bin` shards
- **Roadmap branch:** `roadmap/hf-tpu-parity`
- **Performance reference:** LaughLM at `0705d255faab`

## Product goal

TrainLM will load an ordinary dense autoregressive Hugging Face
`AutoModelForCausalLM`, train it efficiently on Google TPUs through TrainLM's
own engine, and export a checkpoint loadable by plain Transformers.

V1 makes two independent promises:

1. **Universal dense-AR compatibility:** every in-scope model trains,
   evaluates, checkpoints, resumes, and exports through the generic path
   without a family adapter.
2. **Certified TPU optimization:** advertised optimized models pass semantic,
   graph, HBM, throughput, MFU, resume, and export gates. The exact 135M
   reference must reach at least 90% of LaughLM throughput; 95% is preferred.

The normative model and support definition is in [`SCOPE.md`](SCOPE.md).
Merely executing is not optimization, and making only Llama-style models fast
is not a successful framework.

## Operating rules

Status flags: `[ ]` not started, `[~]` in progress or awaiting validation,
`[x]` complete, and `[d]` intentionally deferred.

- Only one milestone should be active at a time.
- Every feature/story below is one atomic commit with its tests and narrowly
  required documentation.
- Do not squash implementation PRs; feature commits are review and rollback
  boundaries.
- TPU-dependent validation is performed on the target TPU environment, not the
  local Windows development machine.
- An optimization defaults on only after semantic and hardware evidence.
- A completed milestone gate cannot be weakened without a decision record.

Every implementation change includes a focused diff, static review, relevant
non-TPU tests, an exact TPU command when required, saved evidence, and a keep,
revise, or revert decision.

## Non-negotiable contracts

### Hugging Face remains the public model contract

- Load through `AutoConfig` and `AutoModelForCausalLM` from config or pretrained
  weights.
- Do not edit installed Transformers source.
- Apply reversible in-memory transformations before optimizer construction.
- Export canonical HF state loadable without TrainLM.
- Default to `trust_remote_code=False`; remote code is best-effort until
  explicitly certified.

### TrainLM owns the performance-sensitive engine

- `TrainLMTrainer`, not HF `Trainer`, owns optimized training.
- TrainLM owns task semantics, data, loss, accumulation, runtime, checkpointing,
  telemetry, and optimization planning.
- No tensor-to-host synchronization is allowed in the hot path.
- HF Trainer interoperability may exist but carries no TPU parity guarantee.

### Optimization is capability-driven

- The core accepts generic `torch.nn.Module` and HF contracts.
- Optional adapters map family structure to reusable capabilities; adapters are
  never required for generic training.
- Unsupported optimizations fall back explicitly in compatibility mode and fail
  before allocation in strict mode.
- `trainer.explain()` reports capabilities, providers, transformations,
  fallbacks, checkpoint conversions, and certification.

### Runtime backends are replaceable

```text
HF model -> capability plan -> prepared model -> TrainLMTrainer
                                                   |
                                            ExecutionBackend
                                         /        |         \
                                       CPU      CUDA    PyTorch/XLA
                                                             |
                                                         TorchTPU later
```

Core trainer, task, data, and planner code must not import `torch_xla`.
Compilation, device placement, sharding, synchronization, checkpoint I/O,
profiling, and backend kernels live behind interfaces.

### Support levels are public

| Level | Meaning |
|---|---|
| Compatible | Correct generic training and plain-HF round trip |
| Optimized | Compatible structural optimizations are active |
| Certified | Correctness, HBM, graph stability, throughput, and MFU passed on hardware |

Only Certified may be called fully supported or optimally trainable.

## Dense-AR V1 scope

V1 covers decoder-only causal Transformers with MHA/GQA/MQA, full causal and
safe sliding-window attention, learned positions/RoPE/ALiBi, LayerNorm/RMSNorm,
GELU/SwiGLU/GeGLU, serial/parallel residuals, separate/partial/packed QKV,
tied/untied heads, packed binary data, DP, and a larger-model FSDP path.

Required representative clusters:

| Capability cluster | Families |
|---|---|
| Learned positions + LayerNorm + GELU | GPT-2, OPT |
| RoPE/parallel residual + ALiBi/packed QKV | GPT-NeoX/Pythia, BLOOM |
| MQA/GQA + non-Llama dense blocks | Falcon, dense Phi |
| RoPE + RMSNorm + gated MLP | Llama, Mistral, dense Qwen, dense Gemma |

Deferred: encoder/encoder-decoder models, MoE, DLLMs, SSM/recurrent/hybrid
models, multimodal models, quantized pretraining, and CUDA-only training paths.

## Performance contract

The locked LaughLM geometry is vocab 32,064; hidden 1,024; 8 layers; 8 Q/8 KV
heads; MLP 2,816; sequence 2,048; RoPE; pre-RMSNorm; SwiGLU; tied embeddings;
fused QKV; SplashAttention; BF16 compute; FP32 parameters/output; microbatch 2;
GA 32; DP 8; 1,048,576 tokens/update; chunk size 4,096; z-loss `1e-4`; AdamW;
WSD; native memmap; prefetch 16; persistent cache; async checkpoint; unscanned
layers.

Validated reference: `1.014M` global tok/s, `53.1%` non-embedding MFU,
`5.73 GiB` peak HBM, and `1.034 s` total step.

| Stage | Required result | Failure response |
|---|---:|---|
| Generic HF + stable XLA | `>= 600K tok/s` | Pause expansion; diagnose runtime/graph/sync |
| Optimized loss + attention | `>= 850K tok/s` | Pause adapters; inspect HLO/kernels/GA/HBM |
| Hard V1 parity | `>= 912.6K tok/s`, `>= 47.8%` MFU | Do not claim LaughLM-class performance |
| Preferred release | `>= 963.3K tok/s`, `>= 50.4%` MFU | Document hard-gate release if used |

Release evidence uses steady-state medians from three matched runs plus one
200-update real-shard stability run.

## Branch and PR policy

Branch names describe repository work, not the tool or contributor:

- `roadmap/<initiative>` for planning;
- `milestone/<range>-<outcome>` for implementation PRs;
- `feature/<milestone>-<feature>-<outcome>` for isolated experiments;
- `benchmark/<milestone>-<feature>-<hardware>` for evidence-only work.

### Dense-AR V1 PR sequence

| PR | Branch | Milestones | Commits | Merge gate |
|---|---|---|---:|---|
| PR1 | `milestone/m0-m2-foundation` | M0-M2 | 15 | Contracts and generic HF CPU conformance |
| PR2 | `milestone/m3-m4-data-trainer` | M3-M4 | 13 | Packed data and backend-neutral trainer |
| PR3 | `milestone/m5-m7-xla-compatibility` | M5-M7 | 16 | 600K, generic TPU, checkpoint/resume |
| PR4 | `milestone/m8-m9-optimization-core` | M8-M9 | 11 | Reversible planner and optimized loss |
| PR5 | `milestone/m10-m12-kernels-parity` | M10-M12 | 19 | 850K and hard LaughLM parity |
| PR6 | `milestone/m13-m14-family-release` | M13-M14 | 12 | Cross-family certification and V1 release |

Future tracks: `milestone/m15-torchtpu`, `milestone/m16-moe`, and
`milestone/m17-dllm`, five commits each. Before activating a later PR branch,
rebase that untouched branch onto the latest merged predecessor.

## Milestone overview

| Status | Milestone | Outcome |
|---|---|---|
| [~] | M0 | Scope, parity manifest, metrics, and dependencies |
| [ ] | M1 | Backend-neutral framework contracts |
| [ ] | M2 | Universal HF dense-causal CPU path |
| [ ] | M3 | Packed-binary data pipeline |
| [ ] | M4 | Correct backend-neutral trainer |
| [ ] | M5 | Stable PyTorch/XLA DP8 runtime |
| [ ] | M6 | Universal dense-AR TPU compatibility |
| [ ] | M7 | Checkpointing, telemetry, and integrity |
| [ ] | M8 | Reversible capability optimization engine |
| [ ] | M9 | Memory-efficient causal loss |
| [ ] | M10 | TPU attention and 850K gate |
| [ ] | M11 | Projection, optimizer, remat, and HLO tuning |
| [ ] | M12 | Exact 135M parity certification |
| [ ] | M13 | Cross-family certification and 1.3B scaling |
| [ ] | M14 | Dense-AR V1 release |
| [d] | M15 | TorchTPU migration |
| [d] | M16 | MoE extension |
| [d] | M17 | DLLM extension |

## M0 — Reproducibility and release contract

**Status:** [x] Complete

**Goal:** Freeze scope, geometry, metrics, and dependency expectations before
runtime or kernel work.

- [x] **M0-F1 — Dense-AR product scope**
  `docs(scope): define dense causal LM support contract`
  Define inclusions, exclusions, support levels, remote-code policy, issue
  labels, release terminology, and representative families.
  **Acceptance:** One normative scope document is referenced by tests, issue
  configuration, release notes, and README.

- [x] **M0-F2 — LaughLM parity manifest**
  `test(benchmark): lock LaughLM 135M parity manifest`
  Lock model, initialization, loss, optimizer, schedule, batch, precision,
  data, runtime, hardware, reference commit, and evidence.
  **Acceptance:** Validate `135,611,392` parameters and `1,048,576`
  tokens/update; a same-version field change fails the lock test.

- [x] **M0-F3 — Benchmark schema and MFU calculator**
  `feat(benchmark): add throughput and MFU result schema`
  Standardize global/device throughput, step/compile time, HBM, FLOPs, MFU,
  input idle, and collectives with real token counts and device waits.
  **Acceptance:** Reproduce saved LaughLM metrics within rounding tolerance.

- [x] **M0-F4 — Dependency matrix**
  `build(deps): define reproducible framework compatibility matrix`
  Separate core, CUDA, XLA, Pallas, profiling, and dev extras; pin TPU stacks
  and tested Transformers v5 ranges.
  **Acceptance:** Clean CPU and pinned TPU environments resolve reproducibly.

### Exit gate

- [x] Scope, parity data, metric formulas, and dependency profiles are reviewed
  and versioned.

## M1 — Framework contracts and backend boundary

**Status:** [~] M1-F1-M1-F4 complete; M1-F5 implemented and awaiting validation

**Goal:** Remove custom-model, XLA, and task assumptions from core TrainLM.

- [x] **M1-F1 — Configuration ownership**
  `refactor(config): separate model training runtime and optimization config`
  HF `PretrainedConfig` owns architecture; TrainLM owns training/runtime policy.
  **Acceptance:** `TrainLMConfig` is built only when explicitly selected.

- [x] **M1-F2 — Execution backend protocol**
  `feat(runtime): define replaceable execution backend protocol`
  Define device, precision, compile, mesh, shard, sync, data, checkpoint,
  diagnostics, and lifecycle hooks.
  **Acceptance:** Trainer-facing packages contain no `torch_xla` import.

- [x] **M1-F3 — Language-model task protocol**
  `feat(tasks): define causal language modeling task protocol`
  Own dispatch, label shift, ignored tokens, normalization, counting, and eval.
  **Acceptance:** Trainer code contains no causal-label logic.

- [x] **M1-F4 — Capability and plan schemas**
  `feat(optimization): define capability and execution plan schemas`
  Model attention, position, norm, MLP, residual, projections, head, embedding,
  checkpoint, and provider decisions.
  **Acceptance:** Plans serialize and explain without mutating a model.

- [~] **M1-F5 — Checkpoint/export contracts**
  `feat(checkpoint): define internal resume and HF export contracts`
  Version canonical/sharded state, transforms, optimizer, RNG, scheduler,
  tokens, data cursor, atomicity, and incomplete-save behavior.
  **Acceptance:** Both formats are fixed before implementations depend on them.

### Exit gate

- [~] CPU boundary contracts are implemented and awaiting validation; they prove
  trainer, task, optimization, checkpoint, and runtime interfaces have no family
  or XLA dependency.

## M2 — Universal Hugging Face dense-causal intake

**Status:** [ ] Not started

**Goal:** Every representative HF causal LM works through the unchanged generic
path before optimization adapters exist.

- [ ] **M2-F1 — Generic HF model provider**
  `feat(models): add generic HF causal LM provider`
  Support `AutoConfig`, `from_config`, and `from_pretrained`; preserve revision,
  dtype, metadata, ties, and local loading.
  **Acceptance:** No TrainLM model class is required.

- [ ] **M2-F2 — Forward-aware batch dispatch**
  `feat(models): add forward signature aware batch dispatch`
  Filter optional inputs while preserving masks, positions, cache position, and
  supported family fields.
  **Acceptance:** Tiny representative models receive only valid inputs.

- [ ] **M2-F3 — Generic output/loss path**
  `feat(models): add generic causal LM output and loss protocol`
  Use valid model loss or correctly shifted CE normalized by supervised tokens.
  **Acceptance:** CPU loss and gradients match direct HF execution.

- [ ] **M2-F4 — Plain-HF round trip**
  `test(models): certify generic HF save and reload round trip`
  Train one update, save, reload without TrainLM model code, compare tied and
  untied state/output.
  **Acceptance:** Every representative family passes.

- [ ] **M2-F5 — Dense-AR CPU matrix**
  `test(models): add dense AR architecture conformance matrix`
  Cover tiny GPT-2, OPT, GPT-NeoX, BLOOM, Falcon, Phi, Llama, Mistral, Qwen,
  and Gemma through construct/forward/loss/backward/update/export/overfit.
  **Acceptance:** No optimization adapter is used.

- [ ] **M2-F6 — Compatibility explanation**
  `feat(models): report generic model capabilities and fallbacks`
  Report known/unknown capabilities, support level, and selected generic path.
  **Acceptance:** Every model has stable human and machine-readable output.

### Exit gate

- [ ] All representative clusters are Compatible on CPU without an adapter.

## M3 — Production packed-binary data

**Status:** [ ] Not started

**Goal:** Provide safe, distributed, fixed-shape, exactly resumable `.bin` data.

- [ ] **M3-F1 — Binary manifest**
  `feat(data): define packed binary shard manifest`
  Define header, dtype, endian, count, vocab bounds, checksum, documents, and
  existing 1,024-byte-header `uint16` compatibility.
  **Acceptance:** Corrupt metadata, size, checksum, and token range fail early.

- [ ] **M3-F2 — HF shard source**
  `feat(data): add revision-pinned HF binary shard source`
  Resolve repo/revision/path/IDs/cache/offline reuse with standard `HF_TOKEN`.
  **Acceptance:** Mocked source resolution is deterministic.

- [ ] **M3-F3 — Batched memmap reader**
  `feat(data): add contiguous packed memmap batch reader`
  Read complete `[batch, sequence]` regions and fixed-shape inputs/labels/masks.
  **Acceptance:** Valid bytes match the reference reader.

- [ ] **M3-F4 — Deterministic partitioning**
  `feat(data): add deterministic shard shuffle and host partitioning`
  Define seeded order, rank ownership, cross-shard, and validation behavior.
  **Acceptance:** Simulated hosts cover intended samples exactly once.

- [ ] **M3-F5 — Backend-aware prefetch**
  `feat(data): add backend-aware asynchronous prefetch queue`
  Keep reading backend-neutral; expose transfer wrappers, depth, and timing.
  **Acceptance:** Ordering/backpressure pass; TPU begins tuning at depth 16.

- [ ] **M3-F6 — Resumable cursor**
  `feat(data): make packed data iteration exactly resumable`
  Save revision, permutation, shard, offset, epoch/tokens, and RNG.
  **Acceptance:** Interrupted and uninterrupted next batches match.

### Exit gate

- [ ] The 28/2 reference split validates, streams, partitions, and resumes.

## M4 — Correct backend-neutral trainer

**Status:** [ ] Not started

**Goal:** Complete correct training independently of TPU optimization.

- [ ] **M4-F1 — Lifecycle/state machine**
  `feat(training): implement trainer lifecycle and control state`
  Prepare, train, evaluate, save, resume, stop, finalize, and callback order.
  **Acceptance:** Normal, failure, stop, and resume flows pass.

- [ ] **M4-F2 — Token-correct accumulation**
  `feat(training): implement token-normalized gradient accumulation`
  Handle ignored-token variation and separate microstep/update/token counters.
  **Acceptance:** Accumulation matches an equivalent large CPU batch.

- [ ] **M4-F3 — Optimizer/state dtype factory**
  `feat(optim): add backend-neutral AdamW state policy`
  AdamW, decay policy, clipping, independent parameter/moment dtype, and
  backend option validation.
  **Acceptance:** Match PyTorch AdamW under identical policy.

- [ ] **M4-F4 — Token-based WSD**
  `feat(scheduler): add token-based WSD schedule`
  Warmup/stable/decay/min-LR/horizon with token-based resume.
  **Acceptance:** Boundaries and resume match the locked schedule.

- [ ] **M4-F5 — Streaming evaluation**
  `feat(evaluation): add token-weighted causal LM evaluation`
  Compute loss/perplexity without retaining predictions; isolate eval state.
  **Acceptance:** Match reference without mutating training.

- [ ] **M4-F6 — Sync-safe callbacks**
  `feat(training): separate host callbacks from compiled metrics`
  Callbacks consume sparse materialized metrics, never live hot-path tensors.
  **Acceptance:** Backend mock detects no hidden scalar extraction.

- [ ] **M4-F7 — Multi-family overfit**
  `test(training): overfit dense AR conformance models`
  Check falling loss, finite gradients, resume, and export across tiny families.
  **Acceptance:** Trainer contains no family branch.

### Exit gate

- [ ] CPU training and CUDA smoke are correct; the custom model is only one HF
  reference model.

## M5 — PyTorch/XLA runtime and accumulation feasibility

**Status:** [ ] Not started

**Goal:** Establish stable DP8 execution before specialized TPU kernels.

- [ ] **M5-F1 — Optional XLA backend**
  `feat(runtime): add pinned PyTorch XLA backend`
  Contain initialization, BF16 policy, device/ordinal, versions, and guarded
  Pallas/JAX imports.
  **Acceptance:** Core TrainLM imports without TPU extras.

- [ ] **M5-F2 — SPMD data-parallel mesh**
  `feat(runtime): add SPMD data parallel mesh`
  Implement DP8 replicated parameters/sharded batch and validate logical axes.
  **Acceptance:** v5e-8 shapes and gradient reduction are correct.

- [ ] **M5-F3 — Cache and recompile guard**
  `feat(runtime): add XLA cache and recompile guard`
  Initialize persistent cache, fingerprint graphs, and enforce static batch,
  sequence, mask, and accumulation structures.
  **Acceptance:** No compile beyond the allowed warmup graph set.

- [ ] **M5-F4 — Compiled training operations**
  `feat(runtime): compile XLA training operations`
  Compile forward/backward/reduction/clip/update/schedule boundaries while
  keeping I/O and logging outside.
  **Acceptance:** A v5e-8 update matches reference semantics.

- [ ] **M5-F5 — Accumulation strategy spike**
  `perf(runtime): select v5e gradient accumulation strategy`
  Compare compiled microstep/update, unroll, XLA loop/scan, and available native
  paths at MB2/GA32/S2048.
  **Acceptance:** Select one strategy and fallback from compile/dispatch/HBM data.

- [ ] **M5-F6 — XLA diagnostics**
  `feat(runtime): expose XLA metrics HLO and profile diagnostics`
  Capture compile/execute counts, `aten::` fallbacks, HLO, synchronized timing,
  and XProf metadata with bounded artifacts.
  **Acceptance:** Evidence proves graph count and fallback status.

- [ ] **M5-F7 — Generic 135M baseline**
  `perf(benchmark): record generic HF 135M XLA baseline`
  Run exact geometry with generic HF loss/attention on fake and real data.
  **Acceptance:** Stable finite updates and `>= 600K tok/s`; otherwise pause.

### Exit gate

- [ ] PyTorch/XLA is replaceable/correct and the first go/no-go gate passes.

## M6 — Universal dense-AR TPU compatibility

**Status:** [ ] Not started

**Goal:** Prove the framework is not Llama-specific before parity optimization.

- [ ] **M6-F1 — Positional semantics**
  `test(tpu): cover learned RoPE and ALiBi causal models`
  Run learned-position, RoPE, and ALiBi models for five finite updates.
  **Acceptance:** Stable post-warmup graph for all three.

- [ ] **M6-F2 — Attention layouts**
  `test(tpu): cover dense MHA GQA and MQA layouts`
  Cover MHA/GQA/MQA and separate/packed projections on the generic path.
  **Acceptance:** No adapter or unexpected CPU fallback.

- [ ] **M6-F3 — Block/MLP layouts**
  `test(tpu): cover norm MLP and residual layouts`
  Cover LayerNorm/RMSNorm, GELU/gated MLP, serial/parallel residual.
  **Acceptance:** All variants pass the same finite-update/graph gate.

- [ ] **M6-F4 — Generic TPU round trip**
  `test(tpu): certify generic dense AR save resume and export`
  Verify equivalent next-update resume and clean HF reload across the matrix.
  **Acceptance:** Internal and canonical state match defined tolerances.

### Exit gate

- [ ] Every V1 cluster trains generically on TPU without adapter, recompile, or
  fallback and round-trips through HF.

## M7 — Checkpointing, observability, and integrity

**Status:** [ ] Not started

**Goal:** Make long TPU training recoverable and measurable before mutation.

- [ ] **M7-F1 — Distributed resume state**
  `feat(checkpoint): add sharded distributed resume checkpoints`
  Save model, optimizer, scheduler, trainer, RNG, tokens, cursor, plan, and
  manifest versions with direct shard I/O where supported.
  **Acceptance:** Interrupted DP8 resumes to an equivalent next update.

- [ ] **M7-F2 — Async checkpoint lifecycle**
  `feat(checkpoint): add asynchronous checkpoint lifecycle`
  Surface in-flight, completion, failure, retention, and shutdown semantics.
  **Acceptance:** Compute overlaps persistence; durability is never reported early.

- [ ] **M7-F3 — Canonical HF export**
  `feat(checkpoint): export canonical Hugging Face checkpoints`
  Export config, safetensors, generation config, tokenizer, ties, and shard index.
  **Acceptance:** Plain Transformers reload matches outputs.

- [ ] **M7-F4 — Sync-safe telemetry**
  `feat(monitoring): add TPU throughput memory and MFU telemetry`
  Measure synchronized throughput, compile, HBM, idle, and collectives sparsely.
  **Acceptance:** No `.item()` or tensor print in the hot path; overhead bounded.

- [ ] **M7-F5 — Integrity gates**
  `feat(monitoring): add configurable training integrity checks`
  Check finite loss/grad/parameters, update size, tokens, and data continuity.
  **Acceptance:** Sparse checks detect injected corruption.

### Exit gate

- [ ] A 200-update run resumes, reports trusted metrics, and detects corruption.

## M8 — Capability planner and reversible optimization

**Status:** [ ] Not started

**Goal:** Transform loaded HF models safely without family logic in core.

- [ ] **M8-F1 — Structural inspector**
  `feat(optimization): inspect dense causal LM capabilities`
  Prefer public HF contracts and expose unknown semantics explicitly.
  **Acceptance:** Reports match family fixtures; no name-only semantic guesses.

- [ ] **M8-F2 — Adapter registry**
  `feat(optimization): add optional model adapter registry`
  Resolve explicit model/class adapters with semantic and version guards.
  **Acceptance:** Removing adapters preserves M6 compatibility.

- [ ] **M8-F3 — Pure planner**
  `feat(optimization): select kernels transforms and fallbacks`
  Match capability to provider by backend, shape, dtype, mask, and backward
  support under auto/required/disabled/explicit policies.
  **Acceptance:** Deterministic snapshots explain every decision.

- [ ] **M8-F4 — Transactional transforms**
  `feat(optimization): apply validated reversible model transforms`
  Plan before mutation, transform before optimizer, preserve aliases, and roll
  back failure without global patches or hot-path hooks.
  **Acceptance:** Injected failure leaves the original model usable.

- [ ] **M8-F5 — State-dict conversion**
  `feat(optimization): add reversible parameter layout mappings`
  Pack/split maps, aliases, dtype/shape validation, transformed resume, and
  canonical import/export.
  **Acceptance:** Train/save/reload matches canonical HF state.

- [ ] **M8-F6 — Explain report**
  `feat(optimization): expose model optimization explanation`
  Human/JSON capabilities, providers, transforms, fallbacks, backend, graph,
  and certification.
  **Acceptance:** Strict mode fails before TPU allocation when unsupported.

### Exit gate

- [ ] No-op and fixture transforms are transactional, explainable,
  checkpoint-safe, and backend-independent.

## M9 — Memory-efficient causal-LM loss

**Status:** [ ] Not started

**Goal:** Remove the full FP32 logits bottleneck without changing semantics.

- [ ] **M9-F1 — Reference chunked linear CE**
  `feat(loss): add reference chunked linear causal cross entropy`
  Consume hidden state, output weight, labels, optional bias; support shift,
  ignore index, tied/untied, FP32 reduction, chunks, and z-loss.
  **Acceptance:** Loss and hidden/head gradients match full-logits reference.

- [ ] **M9-F2 — Optimized training view**
  `feat(loss): bypass full HF logits during optimized training`
  Safely obtain final hidden state/output weight while preserving HF export.
  **Acceptance:** HLO contains no full `[B,S,V]` tensor.

- [ ] **M9-F3 — Rematerialized chunks**
  `feat(loss): add rematerialized chunked loss backward`
  Compare chunks 2,048/4,096/8,192 and bound HBM.
  **Acceptance:** Select size from matched throughput/HBM evidence.

- [ ] **M9-F4 — TPU loss providers**
  `perf(loss): evaluate native XLA Pallas and Tokamax loss providers`
  Compare pure XLA, Pallas bridge, and Tokamax where versions/backward permit.
  **Acceptance:** Record correctness, HBM, speed, compatibility, and fallback.

- [ ] **M9-F5 — Multi-family loss adapters**
  `test(loss): certify chunked loss across dense output-head layouts`
  Cover tied/untied heads and representative backbone outputs.
  **Acceptance:** Loss/gradient/export parity passes without full logits.

### Exit gate

- [ ] Reference geometry avoids full logits and the selected provider has TPU
  correctness and performance evidence.

## M10 — TPU attention provider family

**Status:** [ ] Not started

**Goal:** Correct memory-efficient attention across the full V1 semantic surface.

- [ ] **M10-F1 — Canonical attention spec**
  `feat(attention): define canonical causal attention specification`
  Represent layouts, head geometry, scale, masks, segments, ALiBi, dropout,
  soft-cap, QK norm, and output without model-name conditionals.
  **Acceptance:** Every V1 family maps to the specification.

- [ ] **M10-F2 — HF attention/mask integration**
  `feat(attention): integrate TrainLM attention with HF interfaces`
  Use public attention/mask interfaces when sufficient and adapters otherwise.
  **Acceptance:** Causal leakage tests detect missing/wrong mask registration.

- [ ] **M10-F3 — XLA Pallas provider**
  `feat(attention): add XLA Pallas causal attention provider`
  Wrap supported Pallas attention with import/version guards and backward.
  **Acceptance:** MHA matches reference and emits expected HLO/custom call.

- [ ] **M10-F4 — GQA/MQA without KV repeat**
  `feat(attention): support grouped and multi query TPU attention`
  Support 8/8, 8/4, and 8/1 without silent K/V HBM expansion.
  **Acceptance:** Correctness and HBM tests pass or explicit fallback is used.

- [ ] **M10-F5 — ALiBi/sliding window**
  `feat(attention): add ALiBi and sliding window capabilities`
  Provide optimized support or efficient explicit XLA fallback.
  **Acceptance:** Boundary-token, mask, and gradient fixtures pass.

- [ ] **M10-F6 — Attention autotuning**
  `perf(attention): tune v5e attention provider configurations`
  Tune/cache tiles by hardware, shape, dtype, mask, and provider version.
  **Acceptance:** Selection is deterministic and reproducible.

- [ ] **M10-F7 — Loss/attention benchmark**
  `perf(benchmark): certify optimized loss and attention stage`
  Benchmark exact 135M geometry on fake and real data with HLO evidence.
  **Acceptance:** `>= 850K tok/s`, lower HBM, stable graph, no fallback.

### Exit gate

- [ ] MHA/GQA/MQA paths are correct, semantic fallbacks are explicit, and the
  second go/no-go gate passes.

## M11 — Projection, optimizer, rematerialization, and HLO tuning

**Status:** [ ] Not started

**Goal:** Close the parity gap one measured bottleneck at a time.

- [ ] **M11-F1 — Reversible QKV packing**
  `feat(optimization): pack compatible QKV projections`
  Handle separate/partial/packed layouts, bias, heads, aliases, load/export.
  **Acceptance:** Output/gradient/update and round-trip parity pass.

- [ ] **M11-F2 — Reversible gated-MLP packing**
  `feat(optimization): pack compatible gate and up projections`
  Pack compatible SwiGLU/GeGLU paths and preserve GELU paths.
  **Acceptance:** Output/gradient/update and export parity pass.

- [ ] **M11-F3 — Native fusion audit**
  `perf(hlo): audit norm RoPE residual and MLP fusion`
  Inspect copies, transposes, materialization, custom calls, and layout before
  writing kernels.
  **Acceptance:** Every candidate has evidence-backed native/custom decision.

- [ ] **M11-F4 — Decoder rematerialization**
  `feat(optimization): add structural decoder rematerialization policies`
  None/block/attention/MLP/loss-chunk with correct FSDP ordering.
  **Acceptance:** Selected policy is justified by step/HBM evidence.

- [ ] **M11-F5 — XLA optimizer-state path**
  `feat(optim): optimize XLA AdamW state and update graph`
  Evaluate BF16 first moment, FP32 second, clip, decay, reduction, and resume.
  **Acceptance:** Correct update with measured HBM/step impact.

- [ ] **M11-F6 — Batch/prefetch tuning**
  `perf(runtime): tune microbatch accumulation and prefetch geometry`
  Re-test MB2/GA32, MB1/GA64, safe alternatives, and prefetch near 16.
  **Acceptance:** One synchronized real-data production geometry is selected.

- [ ] **M11-F7 — Final HLO/host closure**
  `perf(hlo): remove residual parity path bottlenecks`
  Resolve proven transpose, graph, fallback, collective, sync, or input issues.
  **Acceptance:** Exact reference reaches the hard 90% thresholds.

### Exit gate

- [ ] Transforms are reversible/correct and the graph reaches `912.6K tok/s`
  plus `47.8%` MFU.

## M12 — Exact LaughLM 135M parity certification

**Status:** [ ] Not started

**Goal:** Convert the fast path into repeatable release evidence.

- [ ] **M12-F1 — Numerical alignment**
  `test(parity): align 135M initialization and loss semantics`
  Match initialization, residual scale, eps, RoPE, shift, z-loss, optimizer,
  schedule, and dtype; compare early deterministic updates where possible.
  **Acceptance:** Every semantic difference is removed or justified.

- [ ] **M12-F2 — Three-run benchmark**
  `perf(parity): certify repeated v5e 135M throughput`
  Report matched runs, dispersion, cache, HBM, HLO, compiles, idle, and MFU.
  **Acceptance:** Every run clears hard thresholds; preferred median clears 95%.

- [ ] **M12-F3 — Real-shard stability**
  `test(parity): validate 200 update real data stability`
  Run pinned diverse shards with eval, integrity, checkpoint, resume, and graph.
  **Acceptance:** Finite/stable, continuous data, no recompile/fallback, export.

- [ ] **M12-F4 — Plain-HF export certification**
  `test(parity): certify optimized HF checkpoint interoperability`
  Reverse internal layouts and reload in clean Transformers-only environment.
  **Acceptance:** Logits/loss match without TrainLM installed.

- [ ] **M12-F5 — Parity report**
  `docs(benchmark): publish LaughLM TrainLM parity report`
  Publish environment, commands, configs, metrics, limits, profile, and HLO.
  **Acceptance:** Reviewer can reproduce without hidden configuration.

### Exit gate

- [ ] Exact 135M parity is reproducible and clears the hard gate; no result is
  generalized to other families yet.

## M13 — Cross-family certification and larger-model scaling

**Status:** [ ] Not started

**Goal:** Prove TrainLM is a framework, not one optimized Llama geometry.

- [ ] **M13-F1 — GPT-2/OPT mapping**
  `feat(adapters): optimize learned-position dense causal models`
  Reuse QKV, GELU/LayerNorm, loss, and attention capabilities.
  **Acceptance:** Both pass output/gradient/update, graph, and export.

- [ ] **M13-F2 — GPT-NeoX/BLOOM mapping**
  `feat(adapters): optimize parallel residual and ALiBi models`
  Preserve parallel residual, RoPE/ALiBi, and projection semantics.
  **Acceptance:** Shared TPU certification passes.

- [ ] **M13-F3 — Falcon/Phi mapping**
  `feat(adapters): optimize MQA GQA and nonstandard dense blocks`
  Preserve head geometry, parallel blocks, projections, and activations.
  **Acceptance:** Correctness, HBM, graph, and export pass.

- [ ] **M13-F4 — Llama/Mistral/Qwen/Gemma mapping**
  `feat(adapters): certify RoPE gated dense model families`
  Share structure while preserving windows, activations, scaling, soft-cap,
  QK norm, and family differences.
  **Acceptance:** Difference fixtures and common certification pass.

- [ ] **M13-F5 — 135M cross-family matrix**
  `perf(certification): benchmark dense AR capability families`
  Report matched raw throughput and architecture-adjusted MFU; dense full
  attention targets 45% MFU absent a reviewed roofline.
  **Acceptance:** Every advertised family has a support/performance record.

- [ ] **M13-F6 — SPMD FSDP**
  `feat(runtime): add backend-neutral FSDP mesh policy`
  Begin data=4/FSDP=2 with correct wrapping, remat, state sharding, checkpoint.
  **Acceptance:** Correct 1.3B train/resume/export smoke on v5e-8.

- [ ] **M13-F7 — 1.3B benchmark**
  `perf(benchmark): certify 1.3B FSDP scaling path`
  Compare matched LaughLM approximately `41.9K tok/s`, `42.6%` MFU with
  collective/HBM analysis.
  **Acceptance:** A matched target is locked before certification.

### Exit gate

- [ ] Advertised clusters are Certified or explicitly lower-level and the 1.3B
  FSDP path works.

## M14 — Dense-AR V1 release

**Status:** [ ] Not started

**Goal:** Deliver a safe, documented, reproducible user release.

- [ ] **M14-F1 — Stable public API**
  `feat(api): finalize TrainLM dense AR pretraining interface`
  Code/YAML from-config and from-pretrained workflows with deprecation policy.
  **Acceptance:** Examples use no internal packages/custom model class.

- [ ] **M14-F2 — Secure `.bin` TPU guide**
  `docs(tutorial): add HF model and binary shard TPU pretraining guide`
  Document token secret, revisions, splits, explain, resume, and export.
  **Acceptance:** Secret scan and clean-environment smoke pass.

- [ ] **M14-F3 — CI/hardware tiers**
  `ci(test): add dense AR release certification tiers`
  Tier 0 CPU, Tier 1 CUDA, Tier 2 scheduled v5e correctness, Tier 3 release
  performance/stability.
  **Acceptance:** Release requires current Tier 3 evidence.

- [ ] **M14-F4 — Preemption recovery**
  `test(checkpoint): validate preemption and incomplete save recovery`
  Kill during compute, staging, and persistence; accept only durable state.
  **Acceptance:** No partial checkpoint or silently skipped data.

- [ ] **M14-F5 — Support manifest/release notes**
  `docs(release): publish dense AR support and certification matrix`
  Publish versions, hardware, providers, caveats, fallbacks, TorchTPU status.
  **Acceptance:** Machine manifest agrees with `trainer.explain()`.

### Exit gate

- [ ] A new user reproduces a certified v5e-8 run and plain-HF export through
  documented public APIs.

## M15 — TorchTPU backend migration

**Status:** [d] Future; starts only when required public APIs exist

**Goal:** Add native TorchTPU through existing contracts, without trainer rewrite.

- [d] **M15-F1 — Compatibility ADR**
  `docs(runtime): map TorchTPU APIs to TrainLM backend contracts`
  Map device, compile, sync, distributed, kernels, checkpoint, profiler, cache.
  **Acceptance:** Every backend method maps to public API or blocking gap.

- [d] **M15-F2 — Native backend**
  `feat(runtime): add native TorchTPU backend`
  Implement existing `ExecutionBackend`, not a second trainer.
  **Acceptance:** Existing contracts and dense causal smoke pass by selection.

- [d] **M15-F3 — Native kernel providers**
  `feat(kernels): register TrainLM kernels with TorchTPU`
  Reuse kernel mathematics and replace bridge/registration details.
  **Acceptance:** Attention/loss forward-backward parity passes capture.

- [d] **M15-F4 — Correctness/checkpoint parity**
  `test(runtime): certify TorchTPU dense AR compatibility`
  Run unchanged M6/M7 suites on the backend.
  **Acceptance:** Both suites pass.

- [d] **M15-F5 — Performance parity**
  `perf(runtime): compare TorchTPU and PyTorch XLA backends`
  Compare exact 135M/1.3B manifests, compile, kernels, HBM, and MFU.
  **Acceptance:** TorchTPU meets active thresholds before becoming preferred.

### Exit gate

- [d] TorchTPU meets or beats XLA correctness/performance. XLA deprecation is a
  separate decision.

## M16 — MoE extension

**Status:** [d] Future; starts after dense-AR V1 release

**Goal:** Add efficient autoregressive MoE without weakening dense paths.

- [d] **M16-F1 — MoE capabilities**
  `feat(moe): define router expert and auxiliary loss capabilities`
  Top-k, capacity, balancing, shared experts, grouped weights, checkpoints.
  **Acceptance:** Selected HF MoE structures require no family names in core.

- [d] **M16-F2 — Expert kernels**
  `feat(moe): add backend grouped expert kernel providers`
  Evaluate Pallas/Tokamax/native including empty/imbalanced groups and backward.
  **Acceptance:** Selected provider and fallback pass parity.

- [d] **M16-F3 — Expert parallel mesh**
  `feat(moe): add expert parallel routing and collectives`
  Expert axis, all-to-all, overlap, source order, load/communication telemetry.
  **Acceptance:** Tokens and gradients return correctly across devices.

- [d] **M16-F4 — MoE checkpoint/export**
  `feat(moe): add expert-sharded checkpoint conversion`
  Equivalent sharded resume and canonical HF MoE export.
  **Acceptance:** Resume matches uninterrupted and plain HF reloads.

- [d] **M16-F5 — MoE certification**
  `perf(moe): certify representative HF MoE families`
  Correctness, routing, balance, communication, HBM, MFU, resume, export.
  **Acceptance:** Each advertised family has a current model-specific target.

### Exit gate

- [d] Representative HF MoE causal LMs are Certified without dense regression.

## M17 — Diffusion language model extension

**Status:** [d] Future; starts after dense-AR V1 release

**Goal:** Add DLLM semantics on shared infrastructure without contaminating
causal contracts.

- [d] **M17-F1 — Diffusion task**
  `feat(dllm): add diffusion language modeling task`
  Timestep/noise, corruption, targets, token weights, eval, resumable RNG.
  **Acceptance:** Deterministic task fixtures pass independently of causal LM.

- [d] **M17-F2 — Diffusion masks**
  `feat(dllm): add noncausal and mixed mask specifications`
  Keep causal and bidirectional/mixed semantics impossible to confuse.
  **Acceptance:** Leakage/boundary fixtures pass; causal cannot be accidental.

- [d] **M17-F3 — Diffusion loss providers**
  `feat(dllm): add optimized diffusion token losses`
  Chunk/fuse only where mathematically valid across timesteps/parameterizations.
  **Acceptance:** Reference/optimized loss and gradients match.

- [d] **M17-F4 — Compiled denoising update**
  `feat(dllm): compile diffusion training updates`
  Keep host randomness/timestep variation from causing unbounded graphs.
  **Acceptance:** Bounded warmup graph set and no step-dependent compile.

- [d] **M17-F5 — DLLM certification**
  `perf(dllm): certify representative HF diffusion language models`
  Quality, correctness, HBM, graph, throughput, resume, and export.
  **Acceptance:** Each advertised DLLM has a task-specific certification target.

### Exit gate

- [d] Representative DLLMs are Certified under a distinct task contract and
  dense causal gates remain unchanged.

## Cross-cutting validation

Every optimization passes applicable gates before becoming an `auto` default.

### Semantic correctness

- Original HF forward/output reference.
- Loss, label shift, mask, z-loss, ignore index, and token reduction.
- Input and transformed-parameter gradients.
- One-update and multi-update parity.
- Tied aliases and state-dict key/shape/dtype preservation.
- Pretrained load and canonical HF export/reload.
- Documented numerical differences from operation reordering.

### Compilation health

- Fixed expected graph count after warmup.
- No unexpected `aten::` CPU fallback or scalar extraction in the hot path.
- No data-dependent recompilation.
- Cache keyed by dependency/provider/shape/config fingerprint.
- Custom providers support backward and active capture.

### Performance evidence

- Separate cold compile, warm cache, and steady-state timing.
- Device synchronization before timed windows end.
- Median and dispersion, never best sample only.
- Fake-data compute ceiling and real-data end-to-end result.
- HBM, input idle, collectives, HLO fingerprint, and profile summary.
- Identical model, data, precision, batch, seed, and hardware for A/B.

### Regression policy

- No provider becomes `auto` without a certification record.
- Transformers/backend upgrades rerun affected capability suites.
- A changed parity HLO fingerprint triggers performance review.
- The generic path is never removed because current adapters pass.

## Critical path and stop conditions

```text
M0 -> M1 -> M2 -> M4 -> M5 -> M6 -> M7 -> M8 -> M9 -> M10 -> M11 -> M12 -> M13 -> M14
       \-> M3 ---------^

After V1: M15 TorchTPU | M16 MoE | M17 DLLM
```

- M3 may proceed alongside M2-M4 after M1 contracts land.
- M6 passes before family-specific optimization dominates work.
- M9/M10 may prototype in parallel only after M8 stabilizes.
- M13 starts after M12 hard parity except small M6 correctness work.
- M16/M17 start only after dense-AR V1 release.
- Below 600K at M5: stop kernels and fix runtime/graph behavior.
- Below 850K at M10: stop adapters and diagnose loss/attention/HLO/GA.
- Below 912.6K at M12: do not claim LaughLM-class performance.

## Primary risks

| Risk | Required mitigation |
|---|---|
| HF internals change | Public interfaces, thin versioned adapters, multi-version CI |
| Transform changes semantics | Transactional plan, forward/gradient/mask tests, reversible export |
| Full logits remain hidden | HLO shape assertion and HBM gate |
| Pallas/JAX mismatch | Optional pinned extra, import/version guard, native fallback |
| GA creates oversized graph | Decide strategy in M5 before kernel investment |
| Layer scan conflicts with Pallas | Isolate until support is proven |
| Metrics/checkpoints synchronize | Sparse metrics and async backend lifecycle |
| “Any AR” becomes Llama-only | M6 before parity; M13 framework certification |
| Fallback marketed as fast | Public Compatible/Optimized/Certified levels |
| TorchTPU obsoletes XLA code | Backend/provider boundary; no XLA in core |
| Remote code executes Python | `trust_remote_code=False` default |
| Hub credentials leak | Standard `HF_TOKEN`, secret scans, no literals |

## Reference implementations

References guide design; they are not blanket dependencies.

- [Hugging Face attention interfaces](https://huggingface.co/docs/transformers/attention_interface)
- [Hugging Face kernels](https://huggingface.co/docs/transformers/kernels)
- [PyTorch/XLA](https://docs.pytorch.org/xla/)
- [PyTorch/XLA SPMD](https://docs.pytorch.org/xla/master/spmd.html)
- [Pallas custom kernels](https://docs.pytorch.org/xla/release/r2.6/features/pallas.html)
- [OpenXLA HLO dumps](https://openxla.org/xla/hlo_dumps)
- [XProf](https://openxla.org/xprof)
- [Tokamax](https://github.com/openxla/tokamax)
- [TorchTPU announcement](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)

Use public HF extension points where sufficient, but retain TrainLM's
capability/transformation layer for complete TPU coverage and hidden-state-based
linear CE. Use PyTorch/XLA now behind interfaces. Do not invent TorchTPU APIs.

## Dense-AR V1 definition of done

V1 is complete only when:

- every representative HF causal LM loads from config/pretrained without
  TrainLM code changes;
- every cluster passes CPU/TPU generic training, resume, and plain-HF export;
- `trainer.explain()` never hides a fallback;
- exact 135M reaches `>= 912.6K tok/s` and `>= 47.8%` MFU in three matched runs,
  preferably `>= 963.3K tok/s` and `>= 50.4%` MFU;
- a 200-update real-shard run is stable, resumable, and graph-clean;
- advertised families have current certification records;
- transformed resume and canonical HF export both pass;
- core has no PyTorch/XLA dependency and accepts a future TorchTPU backend;
- secure `.bin` workflows are documented.

The finished product keeps Hugging Face compatibility at the surface, applies
explainable reversible optimization in memory, and runs on a replaceable
high-performance TPU backend.
