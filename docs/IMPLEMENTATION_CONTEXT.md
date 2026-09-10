# TrainLM implementation context

This document is the handoff for continuing TrainLM work in another Codex
session or in Codex Cloud. It records the agreed product direction and the
current implementation boundary; update it whenever a milestone changes.

## Product intent

TrainLM is a Hugging Face-like trainer for decoder-only autoregressive models.
Users should provide a Hugging Face model ID/path (or a model object), a
tokenizer/processing class, a dataset or packed `.bin` source, and a small set
of familiar training arguments. TrainLM owns TPU/PJRT process launch, data
validation, model capability inspection, safe optimization transforms, XLA
compilation, checkpointing, logging, and export behind that surface.

The model implementation in Transformers remains the source of semantic
truth. TrainLM must not fork or edit every HF model family. Optimizations are
selected through capability inspection and optional, version-guarded adapters;
unsupported features remain on a correct fallback path and are reported by
`trainer.explain()`.

Initial scope is dense decoder-only AR language models. MoE and diffusion/
DLLM architectures are later extensions, not hidden promises of the current
API.

## User experience target

The documented workflow must look like a normal HF trainer, not like a TPU
debug notebook:

```python
from trainlm import TrainLMTrainer, TrainLMTrainingArguments

trainer = TrainLMTrainer(
    model="org/model-or-local-path",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=TrainLMTrainingArguments(
        output_dir="runs/example",
        max_steps=1000,
        sequence_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        accelerator="auto",
    ),
)
trainer.train()
```

The worker script, rank probe, preflight stage, cache directory, manifest
layout, and log parsing are internal implementation details. An equivalent
`trainlm train` CLI/config path is required for unattended jobs.

## Current repository state

- PR1 and PR2 are merged.
- PR3 (`milestone/m5-m7-xla-compatibility`) was rebase-merged into `main`,
  preserving its individual feature commits in linear mainline history.
- PR4 is the active implementation branch: `milestone/m8-m9-optimization-core`.
- PR4 has been synchronized with the merged PR3 mainline; remote backup branches
  preserve the pre-synchronization PR3 and PR4 tips.
- Do not run git commands in Codex sessions; the repository owner performs
  fetch, rebase, commit, push, and merge manually.
- Do not claim TPU validation from a local run. TPU validation is performed by
  the owner on Kaggle/Cloud TPU.

## Measured TPU evidence

The current generic PyTorch/XLA path has completed a v5e-8 two-update smoke
and a 100-step baseline. The latest sparse-loss-materialization run reports
approximately **319,302 global supervised tokens/s** and **319,458 scheduled
tokens/s** over **311.824 s**. This is far below LaughLM's approximately 1M
tokens/s result and is a baseline, not a parity claim.

Current known bottlenecks are full-logit materialization, generic HF SDPA,
host-unrolled gradient accumulation, lack of fused QKV/MLP/RMSNorm kernels,
and incomplete TPU-side data/worker orchestration behind the public API.
Optimization targets (850K gate and LaughLM-class parity) must be measured in
matched runs; planning ranges are not guarantees.

## Implemented public API slice (M8-F0, in progress)

`src/trainlm/api.py` currently provides:

- `TrainLMTrainingArguments` with familiar batch, sequence, optimizer,
  scheduler, precision, logging, checkpoint, and accelerator fields;
- `TrainLMTrainer` accepting a model object, HF model ID/path, or
  `ModelSourceConfig`;
- map-style/iterable torch datasets and prebuilt `DataLoader` inputs;
- `train()`, `evaluate()`, `save_model()`, `save_state()`, `explain()`, and a
  lightweight `log_metrics()` hook;
- lazy HF model resolution so importing `trainlm` does not import
  Transformers/XLA before a model ID is actually used;
- delegation to the existing backend-neutral `training.Trainer` rather than
  a second training loop;
- private TPU coordination that defers model/runtime/optimizer construction to
  spawned workers, runs collective and model preflight stages, preserves the
  supported public training arguments in the worker request, and returns a
  structured coordinator summary;
- public `PackedBinDataset.from_directory()` and `.from_hub()` constructors
  that validate manifests and payload integrity eagerly, yield fixed-length
  causal-LM examples, and apply deterministic rank partitioning;
- local CPU/CUDA save and evaluation cadence, callback metric delivery, and
  versioned model/optimizer/scheduler/runtime/RNG/trainer-state restoration;
- structured TPU metric artifacts that the coordinator validates and delivers
  to public callbacks together with a normalized trainer-state snapshot;
- a family-neutral structural inspector used by `trainer.explain()` that
  reports only config/module/alias/signature-backed facts and leaves residual
  or custom semantics explicitly unknown.
- a deterministic optional model-adapter registry whose entries require exact
  model/config classes, inspected semantic capabilities, source providers, and
  explicit tested package versions; resolution is pure and records every
  rejection reason without importing or mutating a model.
- a pure optimization planner that selects declarative providers by backend,
  precision, inspected component kind, runtime requirements, and explicit
  request; it records portable fallbacks and blocks required unsupported paths
  before any model mutation.
- a transactional transform registry that captures rollback state before each
  mutation, validates inverse mappings and parameter aliases, and reverses the
  failing transform plus all prior transforms when application or downstream
  optimizer construction fails.
- a versioned state-dict layout converter for transformed resume and canonical
  Hugging Face export, with reversible concatenation/splitting plus strict key,
  shape, dtype, device, collision, and tied-alias validation.
- a versioned optimization explanation report with stable dictionary, JSON,
  and human-readable views covering capabilities, adapters, providers,
  transforms, fallbacks, backend/precision, graph evidence, limitations, and
  certification state; strict mode rejects unproven semantics before launch.

This slice is intentionally not complete. `accelerator="tpu"` now reaches the
private single-VM coordinator for reconstructible pretrained HF model sources
and a validated `PackedBinDataset`. Local lifecycle cadence and resume now use
the backend-neutral engine hooks. TPU callback metrics and return state no
longer require stage-log parsing. TPU worker checkpoint/evaluation parity and
transactional optimization application are next stories. Do not mark M8-F0
complete until TPU lifecycle behavior is validated on target hardware.

## Required next sequence

Implement one small commit/story at a time and update `docs/ROADMAP.md` and
this file in every turn:

1. **M8-F0 TPU checkpoint parity:** carry safe resume and save/evaluation
   cadence through worker coordination.
   The coordinator and worker now carry save cadence and resume paths. Each TPU
   rank atomically persists model, optimizer, scheduler, runtime, host/device RNG,
   and deterministic packed-data progress, and validates the committed topology
   before restoring. The tensor-aware implementation remains in the private
   `_tpu_checkpoint` boundary so `trainlm.checkpoint` contracts stay framework
   independent. TPU evaluation datasets now use the same validated packed-data
   boundary and scheduled engine evaluation, with structured callback metrics.
   Every replica currently evaluates the same stream to preserve exact global
   semantics until backend-level distributed evaluation reductions are added.
   Target-TPU lifecycle validation remains.
   Coordinator contract coverage constructs mismatched evaluation requests by
   replacing serialized optional fields, matching real request round trips.
2. **M9-F3 rematerialized chunks:** an explicit non-reentrant per-chunk
   checkpointing policy now preserves reference gradients. Compare chunk sizes
   2,048/4,096/8,192 and collect target-XLA HLO/HBM/throughput evidence before
   selecting a default or marking the story complete.
3. **M9-F4 TPU loss providers:** the pure catalog gates Pallas and Tokamax on
   explicit runtime and backward evidence, with portable chunked loss as the
   fallback. Implement and benchmark eligible providers before default use.
4. **M10-F3 XLA Pallas provider:** guarded HF attention/mask registration now
   requires canonical semantic compatibility and registers both interfaces
   under one key. Add the first version-guarded Pallas MHA provider with backward
   and target-HLO evidence while retaining the registered portable fallback.
   A dependency-free MHA bridge now gates construction on exact torch_xla
   versions and explicit backward evidence, uses an injected stable kernel
   adapter, and rejects unsupported dense masks/dropout. Keep M10-F3 open until
   numerical, gradient, and HLO/custom-call checks pass on target TPU hardware.
5. **M10-F4 GQA/MQA without KV repeat:** a logical head-owner mapping and
   guarded provider now preserve compact K/V inputs for 8/8, 8/4, and 8/1 and
   pass head geometry directly to the kernel adapter. Keep the story open until
   target-TPU correctness, gradient, and HBM evidence proves no hidden repeat.
6. **M10-F5 ALiBi/sliding window:** the guarded provider now passes an integer
   window to the compact kernel without materializing a dense mask and requires
   explicit model-supplied ALiBi slopes rather than deriving family semantics.
   Both paths remain gated until target-TPU boundary, gradient, and HBM evidence.
7. **M10-F6 attention autotuning:** a deterministic exact-key tuning cache now
   keys hardware, provider/version, dtype, batch/sequence/head geometry, mask,
   and window and uses stable candidate-ID tie breaking. Populate candidate
   ranges and measurements on v5e before selecting production defaults.
8. **M10-F7 loss/attention benchmark:** a matched-result gate now requires
   v5e-8 geometry, at least 850K global tokens/s, lower HBM, stable compilation,
   zero CPU fallback, eliminated full logits, named providers, and HLO evidence.
   Run synchronized fake- and real-data measurements on target hardware.
9. **M11-F1 reversible QKV packing:** a family-neutral descriptor now creates
   reversible state-dict mappings for separate MHA/GQA/MQA weights and optional
   biases using explicit head geometry and key names. Add partial-layout and
   transactional live-module transforms, then validate output/gradient/update.
10. **M11-F2 reversible gated-MLP packing:** explicit SwiGLU/GeGLU descriptors
    now create reversible gate/up weight and optional-bias mappings, while GELU
    remains an explicit no-transform path. Add transactional live transforms
    and validate output/gradient/update parity before completing the story.
11. **M11-F3 native fusion audit:** a deterministic audit now turns explicit
    HLO fingerprints and copy/transpose/materialization/custom-call/fallback
    observations into native, custom, or blocked decisions for norm, RoPE,
    residual, and MLP. Populate it from target-XLA captures and benchmark every
    custom candidate before implementing another kernel.
12. **M11-F4 decoder rematerialization:** explicit none/block/attention/MLP/
    loss-chunk policies now require pre-FSDP application, and selection uses
    measured gradient parity, graph stability, step slowdown, and peak HBM with
    deterministic ties. Populate target-XLA measurements before defaulting.
13. **M11-F5 XLA optimizer state:** a versioned AdamW policy now locks first/
    second moment precision, clipping, decoupled decay, and gradient reduction.
    Its gate requires update/resume parity, stable graphs, no CPU fallback,
    bounded step regression, and lower HBM. Populate target-XLA evidence.
14. **M11-F6 batch/prefetch tuning:** a deterministic production selector now
    requires matched token/update geometry, real data, stable graphs, zero CPU
    fallback, and HBM/input-idle budgets before ranking throughput. Measure
    MB2/GA32, MB1/GA64, safe alternatives, and prefetch near 16 on target TPU.
15. **M11-F7 final HLO/host closure:** a final gate now combines the 912.6K
    throughput and 47.8% MFU thresholds with graph, fallback, transpose/layout,
    host-sync, full-logits, input-idle, and collective evidence. Run the exact
    reference on v5e and resolve every reported reason before closing M11.
16. **M12-F1 numerical alignment:** a locked semantic comparator now covers
    initialization, residual/norm/position, shifted loss/z-loss, optimizer,
    schedule, and dtype fields and requires deterministic early-update errors
    within tolerance. Populate it from the exact reference and justify or remove
    every difference before certification.
17. **M12-F2 three-run benchmark:** a repeated-run evaluator now requires three
    matched warm-cache v5e-8 results with identical HLO, stable compilation,
    zero fallback, and per-run hard gates, then reports medians, spread, and max
    HBM. Populate it with three synchronized target runs.
18. **M12-F3 real-shard stability:** a structured gate now requires 200 updates
    on revision-pinned diverse shards with evaluation, resume, integrity,
    cursor continuity, finite loss/gradient ranges, stable compilation, zero
    fallback, and canonical export. Populate it from the target stability run.
19. **M12-F4 plain-HF export:** a structured gate now requires a clean
    Transformers-only environment, canonical keys, preserved aliases, no
    missing/unexpected keys, and logits/loss within tolerance. Run it against
    the optimized checkpoint without TrainLM installed.
20. **M12-F5 parity report:** a versioned report now records environment,
    reproduction commands, configuration/metric/profile/HLO artifacts,
    limitations, and all four M12 gate outcomes in stable JSON and Markdown.
    Populate it only with real target evidence before declaring certification.
21. **M13-F1 GPT-2/OPT mapping:** an explicit, version-guarded catalog now
    distinguishes GPT-2 fused QKV and learned absolute positions from OPT
    separate QKV and offset learned positions while mapping both to shared
    attention, QKV-layout, GELU-MLP, and linear causal-loss operations. Run
    output/gradient/update, graph, export, and target-TPU certification before
    advertising either family as optimized.
22. **M13-F2 GPT-NeoX/BLOOM mapping:** explicit adapters now preserve
    GPT-NeoX RoPE plus parallel residual semantics and BLOOM ALiBi plus serial
    residual semantics while sharing guarded MHA, fused-QKV, LayerNorm, GELU,
    and linear causal-loss operations. Complete shared target-TPU correctness,
    graph, export, and performance certification before enabling either path.
23. **M13-F3 Falcon/Phi mapping:** explicit variants now prevent Falcon MQA
    and GQA head ownership from being interchanged and preserve its fused-QKV
    parallel block. Phi separately requires MHA, partial RoPE, separate QKV,
    LayerNorm/GELU, and parallel residual semantics. Complete correctness,
    HBM, graph, export, and target-TPU certification before enabling them.
24. **M13-F4 RoPE gated families:** guarded mappings now distinguish Llama,
    Mistral sliding-window attention, Qwen2 biased QKV, Gemma scaled embeddings
    and GeGLU, and Gemma2 alternating windows plus attention soft-capping.
    Add QK-normalized variants where supported and complete shared correctness,
    graph, export, HBM, and target-TPU certification before advertising them.
25. **M13-F5 cross-family matrix:** a structured evaluator now rejects missing
    or duplicate advertised-family records, mismatched benchmark geometry,
    failed correctness/graph/export evidence, and full-attention results below
    45% architecture-adjusted MFU. Populate it with synchronized 135M target
    runs using family-specific FLOPs/token before completing the matrix.
26. **M13-F6 SPMD FSDP:** a backend-neutral policy now locks data/FSDP mesh
    dimensions, explicit decoder wrap classes, adapter-provided parameter
    partitions, sharded optimizer state, pre-wrap rematerialization, and
    topology-matched distributed checkpoints. Wire it into the XLA runtime and
    complete the 1.3B train/resume/export v5e-8 smoke before enabling it.
27. **M13-F7 1.3B benchmark:** a structured gate now compares measured FSDP
    evidence only against a review-locked target with exact workload, parameter,
    hardware, and mesh geometry. It also requires throughput/MFU parity, stable
    compilation, zero CPU fallback, correctness/resume/export, and collective
    plus HBM artifacts. Lock the comparison and populate it on target hardware.
28. **M14-F1 stable public API:** `TrainLMTrainer.from_pretrained()` and the
    versioned mapping/YAML `from_config()` path now construct explicit HF model
    sources and public training arguments while rejecting internal keys. The
    first deprecated YAML alias warns before removal. Add clean-environment and
    compatibility checks before declaring the release interface stable.
29. **M14-F2 secure packed-bin TPU guide:** the public tutorial now covers
    secret-managed `HF_TOKEN`, immutable model/data revisions, separate
    train/eval manifests, `explain()`, committed-checkpoint resume, private
    artifacts, and the honest current TPU export limitation. Run the guide in a
    clean target environment and implement canonical TPU export before closure.
30. **M14-F3 CI/hardware tiers:** a release gate now requires current passing
    CPU, CUDA, scheduled v5e correctness, and v5e performance/stability evidence
    for the exact release commit, with Tier 3 mandatory. Provision explicit
    hardware workflows and populate real Tier 1-3 artifacts next.
31. **M14-F4 preemption recovery:** TPU checkpoint shards now carry deterministic
    step/micro-step generations, committed destinations cannot be overwritten,
    manifest/shard progress must agree, and recovery discovers only complete,
    safe, committed topologies. Add real compute/staging/persistence process-kill
    tests and target-TPU cursor-continuity evidence before closure.
32. **M14-F5 support manifest:** the versioned machine manifest and development
    release notes now publish dependency ranges, hardware support, execution
    paths, providers, fallbacks, caveats, and deferred TorchTPU status. A pure
    validator prevents `trainer.explain()` from exceeding published backend or
    path support. Refresh the manifest from final certification evidence.
33. **CI contract regressions:** the core dependency assertion now includes the
    directly imported PyYAML runtime, and the FSDP mismatch fixture uses a valid
    16-device data=4/FSDP=4 mesh so mismatch rejection is exercised by the
    evaluator rather than failing during evidence construction.
34. **M14-F1 compatibility contract:** `support/public_api_v1.json` now records
    the versioned package-root symbols, accepted trainer configuration keys,
    and deprecated aliases. The release evaluator fails when an installed
    surface changes without an API-version update. The CI test suite now also
    builds and installs a wheel into an isolated environment, runs outside the
    source checkout, and verifies the documented package-root construction
    surface; M14-F1 is complete.
35. **M14-F4 process-kill recovery:** the private rank-local saver exposes a
    test-only stage hook at shard staging, shard publication, and manifest
    staging/publication boundaries. Spawned child processes are terminated at
    each pre-commit persistence boundary, and recovery continues to select the
    prior committed generation. Target-TPU cursor-continuity evidence remains.
36. **M14-F4 durable boundary:** spawned process termination now also covers
    compute before checkpointing and the instant after atomic manifest publish.
    Recovery rejects the former attempt and accepts the latter, establishing
    the manifest rename as the host-side durability boundary.
37. **M14-F2 immutable TPU model intake:** remote Hugging Face pretrained models
    now require lowercase 40-character commit revisions before the coordinator
    stages data or launches workers. Existing local model directories remain a
    supported offline snapshot path. Target tutorial smoke remains outstanding.
38. **M11-F1 live QKV transform:** an explicit adapter-selected wrapper path can
    now be replaced before optimizer construction by one packed linear QKV
    projection. The transform validates MHA/GQA/MQA geometry, bias layout,
    dtype/device agreement, returns compact query/key/value views, and restores
    the original wrapper on rollback. Partial source layouts and target-XLA
    update evidence remain outstanding.
   M9-F5 software conformance now covers representative hidden-output forms and
   tied/untied, biased/bias-free heads; M9-F3/F4 still require TPU measurements.

## Definition of done for the public surface

The public API is ready for optimization benchmarking only when all of these
are true:

- a user never writes `subprocess`, PJRT environment variables, rank probes,
  or worker command arguments;
- the same concise code works for CPU smoke and TPU execution;
- HF model output/labels/tokenizer conventions are preserved;
- `.bin` data is validated before workers start and partitioned without
  duplicate or missing tokens;
- `trainer.explain()` identifies selected kernels and every fallback;
- checkpoint/resume and export behavior are documented and tested;
- performance is reported as global supervised tokens/s with warm-up excluded
  and matched against LaughLM geometry.

## Commit convention for this branch

Use one feature/story per commit. Suggested next messages:

- `feat(api): hide TPU worker orchestration behind TrainLMTrainer`
- `feat(data): expose validated packed-bin dataset source`
- `feat(training): complete HF-like lifecycle and resume facade`
- `feat(optimization): add model capability inspector and plan report`
- `feat(optimization): add reversible kernel transform registry`
