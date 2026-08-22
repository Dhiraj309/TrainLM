# Decision 0001: Lock the LaughLM 135M v5e-8 Reference

- **Status:** Accepted
- **Manifest:** `benchmarks/manifests/laughlm_135m_v5e8_v1.json`
- **Manifest version:** 1
- **Manifest SHA-256:** `a7a78b4b3fd2da14b4314944d67e8a9576237769a557cc0805e98c71697bbfe1`
- **Reference revision:** LaughLM `0705d255faab`

## Context

TrainLM needs one immutable workload before TPU runtime and kernel work begins.
Without a locked model, optimizer, batch, precision, data, and hardware
contract, throughput differences could come from geometry changes rather than
framework efficiency.

The selected reference is the validated LaughLM dense 135M production path on
a single TPU v5e-8. It is intentionally MHA with eight Q heads and eight KV
heads. GQA is a separate architecture experiment and must not silently replace
this parity workload.

## Decision

Version 1 is the normative TrainLM parity workload. It locks:

- model architecture and parameter count;
- initialization and numerical policy;
- causal loss and z-loss behavior;
- AdamW and WSD configuration;
- microbatch, accumulation, sequence length, and DP topology;
- parameter, compute, output, and optimizer-state dtypes;
- data source and shard split;
- compilation, rematerialization, prefetch, and checkpoint behavior;
- saved LaughLM evidence and TrainLM acceptance thresholds.

The expected parameter count is `135,611,392`. The expected global tokens per
optimizer update are `1,048,576`.

## Change policy

Changing any manifest byte requires a new manifest version and filename, a new
lock, a new decision record, and preservation of this version. Formatting-only
changes also create a new digest and therefore require a new version.

## Consequences

- TrainLM and LaughLM results can use an explicit matched workload.
- Results with different attention geometry, token counts, precision, or data
  are separate experiments rather than parity claims.
- Dependency and backend manifests may reference this workload while recording
  their own environment versions.
