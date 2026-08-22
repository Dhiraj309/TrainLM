# Capability and execution-plan schemas

TrainLM separates three operations that must never be conflated:

1. **Inspection** reads a model and produces `ModelCapabilities`.
2. **Planning** selects providers and produces an `ExecutionPlan`.
3. **Application** performs validated reversible changes from that plan.

This milestone defines only the first two operations' data contracts. The
schemas contain no model references, modules, tensors, hooks, or callables, so
serializing or explaining them cannot mutate a model.

## Model capabilities

`ModelCapabilities` records model/config identity and nine required semantic
components:

- attention;
- positional encoding;
- normalization;
- MLP;
- residual layout;
- projection layout;
- token embeddings;
- LM head; and
- gradient checkpointing.

Every component has an explicit status: `known`, `inferred`, `unknown`, or
`unsupported`. Known and inferred capabilities name their semantic `kind` and
record unique scalar facts, evidence paths, and notes. Unknown is a first-class
result; absence of evidence must never be interpreted as Llama-style behavior.

Reports have a deterministic SHA-256 fingerprint derived from their canonical
serialized content. Execution plans bind to that fingerprint so a stale plan
cannot silently describe a different capability report.

## Execution plans

An `ExecutionPlan` records:

- backend, precision, and optimization policy;
- one explicit status and reason per provider decision;
- requested and selected providers, including fallbacks;
- provider requirements and supporting evidence;
- declarative model transformations and target paths;
- a mandatory inverse ID for every transformation;
- parameter-layout change declarations; and
- warnings or blocking errors.

Provider decisions use `selected`, `fallback`, `skipped`, or `blocked`.
Plans use `ready`, `noop`, or `blocked`. Invalid combinations are rejected,
such as a blocked decision in a ready plan, a transformation in a no-op plan,
or a fallback that does not record the requested provider.

`ExecutionPlan.explain()` emits deterministic human-readable text containing
the provider, fallback reason, transformation targets, inverse operations,
warnings, and errors. `to_json()` provides the corresponding machine-readable
form.

Language-neutral JSON Schemas are versioned at
`schemas/optimization/model_capabilities_v1.schema.json` and
`schemas/optimization/execution_plan_v1.schema.json`.

## Deliberate exclusions

This commit does not inspect models, select providers, resolve adapters, apply
transforms, or change parameter layouts. Those behaviors are introduced in M8
after generic HF compatibility and TPU correctness gates. These schemas are the
stable contract those later implementations must obey.
