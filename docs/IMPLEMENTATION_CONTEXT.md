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
- PR3 (`milestone/m5-m7-xla-compatibility`) is merge-ready after mainline
  integration and CI fixes; one approving write-access review is still required.
- PR4 is the active implementation branch: `milestone/m8-m9-optimization-core`.
- PR3 has been integrated with the current mainline and passes CI; repository
  policy still requires one approving write-access review before PR13 can be
  squash-merged. Remote backup branches preserve the pre-integration PR3 and
  PR4 tips.
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
2. **M9 loss path:** add chunked/rematerialized causal loss and benchmark it
   against the measured 319K baseline before enabling it by default.
3. **M10+ kernels:** integrate TPU attention/projection/norm/optimizer/remat
   providers only after shape, dtype, mask, and backward correctness tests.

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
