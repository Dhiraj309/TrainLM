# Model compatibility explanation

`explain_huggingface_compatibility(loaded)` reports what TrainLM can honestly
claim immediately after acquiring a causal LM through the generic Hugging Face
provider. It produces the same information as deterministic human-readable
text and versioned JSON.

## Generic-path result

The M2 explanation selects:

- the unchanged `AutoModelForCausalLM` implementation for forward/backward;
- forward-signature-aware TrainLM batch dispatch;
- the backend-neutral TrainLM causal-language-model task; and
- no architecture adapter or model transformation.

The result has support level **Compatible**. It explicitly records the
architecture-optimized provider as a fallback to `huggingface.generic` because
M8 capability inspection and reversible planning have not run. Compatible is
not presented as TPU Optimized or hardware Certified.

## Conservative capability reporting

Model and configuration identity come from immutable provider metadata. The
nine structural components remain `unknown` on this generic path: attention,
position, normalization, MLP, residual layout, projections, embeddings, LM
head, and checkpointing.

This is intentional. Model-type names and familiar config fields are not
sufficient evidence for a safe kernel or module replacement. M8 will inspect
structure and attach evidence before it may produce known or inferred
capabilities. Until then, absence of evidence must never be interpreted as a
Llama-style architecture.

## Example

```python
from trainlm.model import (
    explain_huggingface_compatibility,
    load_huggingface_causal_lm,
)

loaded = load_huggingface_causal_lm(source)
report = explain_huggingface_compatibility(loaded)

print(report.explain())
json_payload = report.to_json()
fallbacks = report.fallbacks
```

The report embeds `ModelCapabilities` and an `ExecutionPlan`. The plan carries
the capability fingerprint, so deserialization rejects a plan that describes a
different report. Explanation performs no forward pass, tensor movement,
module replacement, hook installation, or parameter mutation.

## Interpretation boundary

The report explains the selected compatibility path; it does not create TPU
evidence. CPU conformance comes from the representative M2 architecture matrix.
TPU graph, numerical, HBM, throughput, and MFU claims remain gated by later
hardware milestones.
