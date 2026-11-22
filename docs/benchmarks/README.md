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

The M5 generic execution gate and measurement procedure are documented in
[`M5_BASELINE.md`](M5_BASELINE.md). Its evaluator requires an actual
synchronized `BenchmarkResult`; it never substitutes the LaughLM reference
numbers.

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

## User-run measured training performance

The public `TrainLMTrainer` records bounded, synchronized wall-clock windows at
the configured logging cadence. Progress metrics include observed supervised
tokens/second, wall time, and the fraction spent waiting for the training
dataloader. `TrainLMTrainer.performance_summary()` and from-scratch run reports
expose the aggregate for the current process run; fine-tuning result summaries
carry the same report.

For a single native TPU v5e using BF16, TrainLM reports non-embedding and
logits-inclusive MFU using measured supervised-token throughput, the dense
causal-LM FLOPs formula in `trainlm.benchmark.calculate_causal_lm_flops`, and
Google's documented 197 TFLOPs BF16 peak per v5e chip
([hardware specification](https://docs.cloud.google.com/tpu/docs/v5e)). This is
the standard model-FLOPs utilization definition; MFU is a model-based ratio,
not a hardware FLOP counter. No MFU is reported when hardware, precision, model
geometry, or measurement scope is unknown.

Elevated dataloader wait is surfaced as an input-pipeline hint. A low wait
fraction does not identify a particular slow kernel: detailed operator and
device timeline profiling remains a separate opt-in diagnostic. A window
includes startup/compilation work if it occurs during that window, so compare
later logging windows for steady-state behavior.
