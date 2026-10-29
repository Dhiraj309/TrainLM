# Public optimization integration

`TrainLMTrainer` owns inspection, planning, model application, and canonical
state conversion. No optimized provider is registered by default: ordinary
training keeps the model-reference implementation and reports that fallback.
Selection/application is never presented as hardware certification.

## Controls and profiles

`TrainLMTrainingArguments` adds `optimization_policy="auto"` and
`allow_optimization_fallbacks=None`. Auto permits eligible alternatives or the
model reference. Required mode forbids fallbacks and raises `OptimizationError`
with a structured `report` when requested optimization is unavailable. Disabled
mode skips transforms; a disabled factory profile is not imported.

The optional `optimization=` trainer argument accepts `OptimizationOptions` for
local CPU/CUDA runs or an importable `"package.module:factory"` for all backends.
Factories return `OptimizationOptions`: requests, provider declarations, handler
and adapter registries, shape constraints, and mask intent. Profile factories
and handlers are trusted code, must be host-safe/deterministic, and must not
initialize accelerators or mutate global state. TPU uses importable factories,
not pickled callbacks; the same installed module must be importable in the host
preflight and every worker.

```python
trainer = TrainLMTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=TrainLMTrainingArguments(
        output_dir="runs/example", accelerator="cpu", max_steps=100,
        optimization_policy="required", save_steps=10,
    ),
    optimization="my_installed_providers:training_profile",
)
trainer.train()
print(trainer.explain(format="text"))
```

This example requires a real installed provider profile; TrainLM does not ship
or certify the fixture providers used by tests. Providers without executable
model transforms are rejected by public profile validation rather than silently
claimed as selected optimizations.

Observed package versions are collected outside the pure planner. Context
includes target backend/precision, declared batch/sequence/accumulation geometry,
world/global batch sizes, adapter selection, and optional evaluation geometry.
Profiles cannot override dimensions fixed by training arguments. Providers must
also qualify for declared evaluation geometry when validation data is supplied;
unavailable candidates and their evaluation rejection reasons appear in reports.
Mask intent defaults to `unknown`. Additional profile shapes/mask declarations
are trusted provider inputs, not measurements of arbitrary user datasets;
kernels must validate actual batches and implement safe dispatch/fallbacks for
variable lengths or masks. This integration does not reshape arbitrary datasets.

## Local initialization and failures

The facade resolves the model, inspects/resolves adapters, plans, and validates
handlers/targets/mappings on an isolated host copy before runtime placement.
This catches strict unsupported cases without constructing a runtime/optimizer
or changing the caller's graph. Host validation can temporarily copy model
weights for configured transforms; default reference training avoids that work.
The runtime then prepares the model once, transforms are applied on its final
device, and only then are optimizer and scheduler constructed. The engine accepts
that prepared model without a second placement/compilation pass.

Transforming profiles require trainer-owned optimizer creation. Existing
optimizers remain supported on reference paths, with ownership validation after
placement. A supplied compiled runtime cannot be combined with model transforms.
Pending gradients prevent layout changes. If later setup fails, installed
transforms are reverted. GPU execution uses the same ordering and contracts,
but no CUDA performance/correctness certification follows from CPU tests.

## Checkpoint, resume, and export

Reference local checkpoints retain schema version one. Transformed checkpoints
use schema version two, nesting canonical model/named optimizer/scheduler/RNG and
runtime-buffer state alongside the existing engine progress/runtime/data-loader
state. Resume requires a matching optimization profile/plan/layout and retains
the deterministic loader position. Legacy raw optimizer checkpoints cannot be
restored into a transformed optimizer without canonical conversion. Missing
profiles or incompatible state fail rather than loading packed names as HF keys.

For transformed local models, `save_model()` writes a canonical model copy;
`save_state()` writes a full resumable checkpoint and returns `trainer_state.pt`.
Neither export nor inspection edits the active transformed graph. Reference
save methods retain their prior behavior.

## TPU host/worker boundary

Default TPU construction still defers model/runtime allocation. Required mode
without registered providers fails immediately. A configured profile runs a
CPU-only `optimization_preflight` subprocess before the collective probe: it loads
the model, checks plan/handlers/layouts, and writes a structured artifact. Failed
preflight cannot proceed to probe/model-preflight/training. Host validation may
load/copy CPU weights once extra; those objects are released before worker launch.
The verified plan ID is passed to each stage. Workers reconstruct the profile,
require the same plan, prepare/place once, apply transforms, and construct their
optimizers afterwards. Configured private CLI runs perform the same host check
before importing XLA when no verified plan was supplied.
Worker application preserves the XLA RNG seed around temporary transform
construction, in addition to the application layer's CPU/CUDA/Python RNG guards.

Per-rank transformed checkpoints use canonical state conversion inside the
existing all-rank checksum/commit/cursor protocol. Default schema-one worker
checkpoints remain supported. Profile/context changes are rejected on resume.
Standalone TPU evaluation, export, and state saving reopen the same profile;
exports use a canonical model copy. Worker summaries deliver optimization reports
to the parent without stage-log parsing. The existing XLA attention dtype adapter
is recorded separately as a compatibility adapter, without performance claims.

## Explain and validation boundary

`explain()` retains its dictionary return type and existing source-model fields,
adding `optimization`. `format="json"` returns JSON; `format="text"` renders the
human plan. Reports include original semantic capabilities, adapter guards,
provider decisions/fallbacks/exclusions, targets/inverses, actual applied IDs,
backend/precision, compilation request/verification flags, and explicit lack of
hardware certification. Before default TPU workers run, the report is honestly
`deferred`; completed workers supply actual inspection/application results.

CPU tests cover public mapped training/evaluation/checkpoint/resume/export,
runtime/optimizer ordering, strict rejection, fallback/disabled behavior,
evaluation shape exclusion, canonical plain Transformers reload, and TPU host/
worker/commit integration through controlled CPU doubles. Real TPU/CUDA
validation and the public CLI/config entry point remain separate open gates.

The full CPU regression command
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q`
executed 446 tests: 440 passed and 6 failed. All 14 new integration tests passed.
The failures match onboarding: five Phi padding-configuration errors with
Transformers 5.0.0 and the existing TrainLM save/reload weight mismatch.
