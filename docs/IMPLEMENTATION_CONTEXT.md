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
  integration and CI fixes; its required write-access approval must be renewed after
  the history rewrite.
- PR4 is the active implementation branch: `milestone/m8-m9-optimization-core`.
- PR3's branch is integrated with the current `main` history so its pull
  request can merge without discarding either line of development; remote
  backup branches preserve both PR3 and PR4 tips from before that integration.
- PR3 history is linearized onto the current `main` tip so GitHub can retain
  its individual feature commits through the repository's rebase-merge path.
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
- synchronous replicated-worker checkpoints at `save_steps`, with per-rank
  model/optimizer/scheduler/runtime/RNG state, canonical optimizer names,
  consumption-based packed-data cursors, and an all-rank commit gate;
- TPU `train(resume_from_checkpoint=...)` and periodic packed-data evaluation,
  including cross-rank token-weighted loss reduction and public metric delivery;
- standalone TPU evaluation, plain HF model export, and full state saving by
  reopening a retained final worker checkpoint; global supervised-token stopping
  rounds up to complete optimizer updates without changing accumulation shapes;
- a family-neutral structural inspector used by `trainer.explain()` that
  reports only config/module/alias/signature-backed facts and leaves residual
  or custom semantics explicitly unknown.

This slice is intentionally not complete. `accelerator="tpu"` now reaches the
private single-VM coordinator for reconstructible pretrained HF model sources
and a validated `PackedBinDataset`. Local lifecycle cadence and resume now use
the backend-neutral engine hooks. TPU callback metrics and return state no
longer require stage-log parsing. The worker now honors save/evaluation cadence
and resumes same-topology replicated checkpoints before starting prefetch. The
private format is distinct from the M7 sharded manifest and cannot reshard or
resume CPU/CUDA checkpoints. Incomplete/corrupt or incompatible checkpoints are
rejected. See [worker lifecycle details](training/TPU_WORKER_LIFECYCLE.md).
TPU `evaluate()`, `save_model()`, and `save_state()` now launch private worker
operations against the latest retained state, or the initial pretrained model
before training. A final checkpoint is saved before training runtime teardown,
even without periodic saves. `save_state()` returns a committed checkpoint
directory suitable for explicit resume; operation logs are kept separately from
training artifacts. Token-only runs use `max_steps=None` and `max_tokens=N`,
where N is a global supervised-token budget rounded up to a full optimizer
update. Target-hardware lifecycle validation, public CLI/config alignment, and
capability-based optimization planning remain outstanding. Do not mark M8-F0
complete or claim CUDA/TPU certification from CPU tests.

The CPU/CUDA public path remains supported, including explicit
`accelerator="cuda"` and automatic CUDA selection when available. This is a
single-process path; multi-GPU distributed training is not established here.

Checkpoint lifecycle validation uses CPU doubles for the worker's XLA APIs,
exact interrupted/uninterrupted dropout-training comparisons, mixed moment
precision, corrupt/incompatible cursor rejection, and two-rank publication and
failure coordination. Real TPU validation belongs to the owner's environment.
The full CPU regression run after standalone lifecycle implementation executed
357 tests: 351 passed and
6 failed. The failures match onboarding: five Phi cases hit Transformers 5.0.0's
missing `PhiConfig.pad_token_id`, and the existing TrainLM save/reload test
reports changed embedding weights. These defects are tracked separately from
the lifecycle implementation; no assertions were disabled or source patched.
The focused facade/worker/lifecycle group passed all 44 tests, including
standalone operations against trained state, initial-state resume, token-only
limits, canonical HF export/reload, and failed-export propagation.

## Required next sequence

Implement one small commit/story at a time and update `docs/ROADMAP.md` and
this file in every turn:

1. **M8-F6 public plan integration:** connect inspection/planning/application and
   canonical state I/O to trainer setup before optimizer creation and expose
   selections/fallbacks. No providers are enabled by default.
2. **M8-F0 integration gate:** align the public CLI/config entry point with the
   facade and validate the implemented TPU lifecycle on owner hardware without
   blocking independent planner implementation.
3. **M9 loss path:** add chunked/rematerialized causal loss and benchmark it
   against the measured 319K baseline before enabling it by default.
4. **M10+ kernels:** integrate TPU attention/projection/norm/optimizer/remat
   providers only after shape, dtype, mask, and backward correctness tests.

## Implemented adapter registry and planner (M8-F2/F3)

`optimization.AdapterRegistry` resolves exact qualified model/config classes,
explicit package-version guards, semantic guards, and evidence-backed capability
overrides without changing model weights, aliases, mode, or generic execution.
No family-name or subclass guessing is used. Missing guards preserve the generic
report with rejection reasons; ambiguous or contradictory adapters fail explicitly.

`optimization.plan_optimizations()` selects declarative providers against an
immutable context of backend, precision, mask, shapes, backward intent, versions,
and adapter ID. It supports disabled/auto/required policies and explicit provider
requests, deterministic priority/ID selection, reference fallback, stale-report
rejection, conflicting-transform blocking, and content-addressed plan IDs. Every
transform description includes its existing inverse ID. See
[planner contract](optimization/PLANNER.md) for policies and integration boundaries.

No optimized provider is registered by default. Selection neither executes a
kernel nor certifies an optimization. M8-F6 public explanation/runtime integration
remains open;
the public CPU/CUDA/TPU execution paths are unchanged by this slice.

The full CPU regression run for this slice executed 392 tests: 386 passed and
the same 6 pre-existing onboarding failures remained. All 35 new adapter/planner
tests passed. Coverage includes registry removal and non-mutation, class/config/
version/semantic guards, backend/precision/mask/shape/backward eligibility,
strict and explicit fallbacks, canonical plan ordering/IDs, immutable contexts,
stale capability bindings, conflicting transforms, and version-one JSON schema
round trips. No accelerator kernels or hardware were exercised.

## Implemented transactional application (M8-F4)

`optimization.apply_execution_plan()` replays inspection/adapter resolution and
planning inputs to reject stale plans, resolves registered forward/inverse
handlers, and preflights target selectors/classes/aliases/ownership. All module
work happens on isolated copies before live graph edits. Canonical names,
metadata, values, hooks, modes, alias groups, and inverse class/topology are
validated. Only then are original Parameter/buffer objects rebound into staged
replacements and committed together. Failed commits restore the original modules
without requiring a provider inverse to succeed.

The returned undo handle preserves trained values, gradients, optimizer references,
ties, and current mode. Failed/stale undo leaves replacements installed rather
than losing live tensors. Call application before optimizer construction;
external optimizer references cannot be discovered automatically. See the
[application contract](optimization/APPLICATION.md) for the trusted-handler,
single-writer, dense-state, and no-global-side-effect boundaries. Layout-changing
transforms now require M8-F5 mappings; public trainer integration remains M8-F6.

The full CPU regression run for application executed 410 tests: 404 passed and
the same 6 pre-existing onboarding failures remained. All 18 new transaction
tests passed. They cover output/gradient/optimizer equivalence, trained undo,
forward/inverse/validator/commit failure injection, stale plan/context/ownership
guards, module/tensor aliases, hook-removal handles, buffers, modes, RNG state,
and no-op/reference plans. No GPU/TPU execution was exercised.

## Implemented canonical state conversion (M8-F5)

`optimization.StateLayout` declares versioned canonical tensor metadata, alias
owners, and disjoint pack/split mappings. Layout factories bind actual target
shapes; source/candidate/inverse validation rejects implicit layout or value
changes. Application supports registered mapped transforms at initialization,
keeps unmapped original tensors/aliases, and blocks tied packed sources that
would require a separate view/gradient implementation.

Application handles expose canonical model state, canonical model copies/HF
export, and named model/optimizer/scheduler/Torch-RNG state I/O. Nonpersistent
runtime buffers are included in training state. Standard/compact AdamW and SGD
moment layouts preserve slot precision, group membership, and scalar steps;
inconsistent or unsupported state fails rather than being cast or discarded.
Late restore failures roll back tensors, optimizer, scheduler, and RNG. Mapped
undo can convert an existing optimizer back to canonical parameters while
preserving trained values and gradients. See [state conversion](optimization/LAYOUTS.md).

This is a local conversion codec, not the M7 distributed ResumeManifest or an
automatic public trainer checkpoint change. M8-F6 must wire the codec into
public setup/checkpoint/export paths. No optimized providers are enabled, and
GPU/TPU kernels or hardware are not certified by CPU fixture evidence.

The full CPU regression run for state conversion executed 432 tests: 426 passed
and the same 6 pre-existing onboarding failures remained. All 22 new layout tests
passed, covering pack/split metadata, independent moment precision, exact next
update after dropout/runtime-buffer resume, trained optimizer-aware undo, unsafe
group/step/alias/dtype rejection, late restore rollback, and plain Transformers
export/reload of a trained mapped projection. No accelerator hardware was used.

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
