# Benchmark Contracts

TrainLM benchmarks use versioned manifests so performance comparisons cannot
silently change model geometry, numerical policy, batch size, data, or hardware.

## Locked LaughLM reference

- Manifest: [`laughlm_135m_v5e8_v1.json`](../../benchmarks/manifests/laughlm_135m_v5e8_v1.json)
- Lock: [`laughlm_135m_v5e8_v1.lock.json`](../../benchmarks/manifests/laughlm_135m_v5e8_v1.lock.json)
- Decision: [`0001-laughlm-135m-reference.md`](decisions/0001-laughlm-135m-reference.md)

The manifest is the matched workload for TrainLM's 135M TPU parity work. Its
reference metrics are evidence from LaughLM; they are not claims about current
TrainLM performance.

## Change rules

Never edit a locked manifest in place. A changed workload requires:

1. a new versioned manifest filename;
2. a new SHA-256 lock record;
3. a new decision record explaining every changed field;
4. a new known-lock entry in the contract test;
5. preservation of the previous version.

The `.gitattributes` rule forces manifest JSON to LF so the content digest is
stable across Windows and Linux checkouts.

## Result schema and MFU

Benchmark results use
[`benchmark_result_v1.schema.json`](../../benchmarks/schemas/benchmark_result_v1.schema.json)
and the matching `trainlm.benchmark.BenchmarkResult` API.

- Token throughput uses actual supervised-token counts from every
  data-parallel replica.
- Global throughput divides those tokens by the synchronized wall window.
- Device throughput divides the same tokens by the device execution window.
- MFU uses device throughput and the aggregate theoretical peak of all devices.
- Non-embedding MFU excludes embedding/output projection parameter work and
  includes quadratic attention work.
- Logits-inclusive MFU adds the output projection computation even when its
  weights are tied.

Cold compilation, warm-cache execution, and steady-state measurement must be
reported separately. The schema also records HBM, input idle, collectives,
compile/fallback counts, and workload identity.
