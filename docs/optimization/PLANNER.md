# Optional adapters and pure optimization planning

Inspection, adapter resolution, provider selection, and model application are
separate operations. The registry and planner produce immutable metadata; they
never replace a module, change a parameter, initialize an accelerator, import a
kernel, or query installed package versions. No optimized providers or adapters
are enabled by default. A selected plan is an eligibility result, not evidence
that a kernel ran or passed hardware certification.

## Adapter registry

`AdapterDefinition` names an exact, fully qualified model class and config class,
mandatory package `VersionRequirement` guards, optional semantic
`CapabilityRequirement` guards, and evidence-backed component overrides.
`AdapterRegistry.resolve(model, versions=...)` first performs generic inspection.
It considers only exact class matches, checks versions/semantics, and enriches
the report without editing the model. Family metadata such as `model_type` and
unregistered subclasses never select an adapter.

Missing/incompatible versions or failed semantic guards leave generic inspection
intact and appear in `AdapterResolution.rejections`. Multiple eligible adapters
are an error rather than an order-dependent choice. An override cannot contradict
inspected kinds/facts or replace unsupported semantics. Every override and fact
requires evidence; the enriched report records adapter/class provenance and has
its own capability fingerprint. Removing a registry restores generic inspection
and leaves model outputs, parameter aliases, and training mode unchanged.

## Planning inputs

`plan_optimizations()` consumes a `ModelCapabilities` report, a `PlanningContext`,
explicit `OperationRequest` entries, and optional `ProviderDefinition` metadata.
The context snapshots backend, precision, mask kind, training/backward intent,
runtime shape dimensions, observed package versions, and resolved adapter ID.
Mappings are copied into immutable sorted tuples. Capture versions outside the
planner, for example with `importlib.metadata.version()`.

Each provider declares its operation/component, supported capability kinds and
facts, backends, precisions, masks, backward support, shape bounds/multiples,
package-version guards, optional adapter IDs, priority, and reversible transform
descriptions. Unknown/unsupported capabilities and missing evidence do not grant
eligibility. Inferred semantics require explicit opt-in. Facts compare without
boolean/numeric coercion. Runtime shape constraints cannot contradict inspected
facts with the same dimension name. Backend/dtype/mask strings match exactly;
the caller must describe the intended execution target, without name guessing.

```python
from trainlm.optimization import (
    AdapterRegistry, OperationRequest, PlanningContext, plan_optimizations,
)

# model is an existing, caller-owned model. No provider is installed by this code.
versions = {"torch": "2.9.0"}  # Replace with externally observed versions.
resolved = AdapterRegistry().resolve(model, versions=versions)
context = PlanningContext(
    backend="cpu", precision="fp32", mask_kind="causal",
    versions=versions, adapter_id=resolved.adapter_id,
)
plan = plan_optimizations(
    resolved.capabilities, context=context,
    requests=(OperationRequest("attention", "forward_backward"),),
)
# Empty provider registry: keep the model's existing attention implementation.
assert plan.decisions[0].selected_provider == "model_reference"
print(plan.explain())
```

Provider metadata must describe an actual implementation and its audited
constraints. The planner does not install/discover providers or test their
numerics. Fixture providers in CPU tests are explicitly test-only.

## Policy, determinism, and failure

- `disabled` skips optimization and produces a no-op plan.
- `auto` selects an eligible provider by descending priority, then provider ID.
  If none qualifies, it records the unchanged `model_reference` fallback.
- An explicit provider ID on an operation takes precedence over priority.
  When it is unavailable, auto may choose another eligible provider or the
  model reference and records the requested/selected IDs and rejection reasons.
- `required` or `allow_fallbacks=False` blocks unavailable requested optimizations.
  Required mode cannot enable fallbacks. Callers must reject a blocked plan
  before device allocation/application; no public runtime hook is added here.

Requests, registry entries, guards, and context mappings have canonical ordering.
Transform sequences preserve declaration order. JSON and human explanations
record eligibility/rejection reasons, requirements, evidence, targets, inverse
IDs, and layout declarations. Plan IDs hash the capability fingerprint, full
context, requests, provider declarations, policy, and resulting decisions.
Equivalent registration order gives identical output; changing versions or
requirements changes the ID. Expected capability fingerprints detect stale
snapshots. Conflicting transform IDs block the whole plan, clearing transforms
so a partial selection cannot be mistaken for an actionable plan.

## Remaining integration

The existing version-one capability/plan JSON Schemas are unchanged. This slice
does not apply transforms or supply inverse handlers, convert checkpoint layouts,
or change `TrainLMTrainer` execution. Transactional application is M8-F4,
canonical state conversion is M8-F5, and public plan-aware explanation/strict
runtime integration is M8-F6. CPU tests establish selection and non-mutation
behavior; GPU/TPU kernel correctness and performance remain separate gates.

The full CPU regression command
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q`
executed 392 tests: 386 passed and 6 failed. All 35 new adapter/planner tests
passed. The six failures match onboarding: five Phi padding-configuration errors
under Transformers 5.0.0 and the existing TrainLM save/reload weight mismatch.
