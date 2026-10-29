# Transactional module application

`apply_execution_plan()` is the M8-F4 application boundary. It stages module
replacements without executing optimizer steps, compilers, or accelerator
initialization. No transforms are registered or enabled by default. The existing
public trainer continues its current execution path until M8-F6 integration.

## Plan and handler validation

Application takes the original planning context, operation requests, provider
declarations, policy fallback setting, and optional adapter registry. It repeats
generic inspection/adapter resolution and pure planning, then requires the
complete regenerated plan to equal the supplied plan. Changed model capabilities,
versions, shapes, provider metadata, requests, or adapter resolution therefore
cannot silently reuse an old plan. No JSON Schema change is needed.

A `TransformRegistry` maps forward IDs to `TransformHandler` records. Each
handler binds its provider/component, inverse ID, exact source module classes,
forward/inverse callables, and an optional provider-specific validator. Every
transform must belong to a selected provider. Missing/mismatched handlers,
blocked plans, unsupported target classes, and layout-changing declarations
fail before any transform callback runs.

Targets are registered child-module paths; a whole `*` path segment expands
deterministically to one segment, as in `model.layers.*.self_attn`. Root
replacement, unmatched selectors, partial module-alias selection, and overlapping
targets are rejected. All aliases of a shared module must be selected by the
same transform and receive the same replacement object.

## Stage, validate, commit

Forward handlers receive deep-copied targets. An optional validator receives a
separate source copy and the staged replacement. Inverse handlers run on another
isolated copy to validate canonical restoration. The application checks:

- canonical parameter/buffer names, shapes, dtypes, devices, and registrations;
- tensor values, `requires_grad`, alias groups, and buffer persistence;
- existing hook registrations/dispatch flags and per-module train/eval modes;
- canonical module classes/topology after inverse conversion.

Handlers cannot add hot-path hooks or silently change weights/layouts. CPU Torch
and Python RNG state, plus CUDA RNG state when already initialized, are preserved
around staging and inverse validation. Materialized dense tensors are supported;
meta/sparse tensors and modules that cannot be deep-copied are not accepted by
this application path. Extra state or numerical semantics beyond these checks
remain the provider's audited responsibility; CPU fixture tests are not kernel
certification.

After every target passes, staged replacements reuse the model's **original
Parameter and buffer objects**. Commit installs all replacements through ordinary
PyTorch registration dictionaries and verifies global canonical ownership.
Cross-target and embedding/head ties therefore remain intact. A staging failure
leaves the live graph untouched. A commit failure restores installed original
modules in reverse order without depending on a provider inverse succeeding.
Shared child-module aliases are checked across target boundaries, and existing
hook dictionaries are reused so the caller's removal handles remain functional.

## Optimizer ordering and undo

Call application before optimizer construction, compilation, and preferably
device placement. Passing an existing optimizer is rejected; the application
cannot discover optimizers or references held elsewhere by the caller.

```python
# model, context, requests, providers, and handlers are caller-owned inputs.
from trainlm.optimization import (
    apply_execution_plan, inspect_dense_causal_lm, plan_optimizations,
)

planning = dict(context=context, requests=requests, providers=providers)
plan = plan_optimizations(inspect_dense_causal_lm(model), **planning)
applied = apply_execution_plan(model, plan, registry=handlers, **planning)
optimizer = make_optimizer(model.parameters())  # Construct only after success.

# Revert modules later while retaining their current, possibly trained tensors.
applied.revert()
```

`AppliedExecutionPlan.revert()` validates all inverses on copies before restoring
original modules. It preserves current parameter values, gradients, optimizer
references, tied aliases, buffers, and train/eval modes. Repeating undo is a
no-op. If another operation replaced a target or changed tensor ownership, undo
is refused rather than restoring stale tensors. If inverse validation fails,
the live optimized modules remain installed; no partial undo is attempted.
Use one writer per model, and do not apply/undo concurrently with training or
external graph edits. Trusted handlers may only mutate their isolated argument;
global patches, external resources, and other process-wide side effects are
outside the provider contract.

## Remaining work

`parameter_layout_change=True` is intentionally blocked until M8-F5 adds
canonical state conversion, alias-aware pack/split mappings, transformed resume,
and HF export. M8-F6 must connect plans/application to public trainer setup and
explanations. Real GPU/TPU provider correctness and throughput remain separate
validation gates.

The full CPU regression command
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q`
executed 410 tests: 404 passed and 6 failed. All 18 new transaction tests passed.
The failures match onboarding: five Phi padding-configuration errors under
Transformers 5.0.0 and the existing TrainLM save/reload weight mismatch.
