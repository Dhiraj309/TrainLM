# M5 generic 135M XLA baseline

M5-F7 is the first end-to-end TPU gate for TrainLM. It measures the unchanged
generic Hugging Face causal-LM path against the locked LaughLM workload; it
does not enable family-specific attention or loss transforms.

## Locked workload

Use [`laughlm_135m_v5e8_v1.json`](../manifests/laughlm_135m_v5e8_v1.json):

- TPU v5e-8, one host, eight data-parallel replicas;
- sequence length 2,048, microbatch 2 per device, GA32;
- 1,048,576 scheduled tokens per optimizer update;
- BF16 compute with FP32 parameters/output;
- persistent cache, fixed-shape batches, and the generic HF loss/attention path.

Load and validate the manifest before constructing a runner:

```python
from trainlm.benchmark import load_baseline_workload

workload = load_baseline_workload(
    "benchmarks/manifests/laughlm_135m_v5e8_v1.json"
)
```

## Measurement protocol

Run cold-cache compilation separately, then clear metrics and run a warm-cache
window. After the graph stabilizes, record three steady-state windows with
synchronized device timing and actual supervised-token counts from all eight
replicas. Save compile/execute counts, unexpected compiles, `aten::` counters,
HBM, input idle, collectives, HLO fingerprint, and the XProf artifact metadata.

Evaluate each steady-state result with `evaluate_baseline()`. The M5 go/no-go
gate is at least 600K global supervised tokens/second with no unexpected
post-warmup compilation. CPU fallbacks are reported as warnings for diagnosis;
they must be eliminated before any optimized or parity claim.

This module intentionally does not manufacture a result when no TPU evidence
exists. A failed gate pauses kernel/adaptor expansion until the graph, input,
HBM, or synchronization bottleneck is explained.
