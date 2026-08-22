# Changelog

Notable TrainLM changes will be recorded here. Support terminology and
performance claims follow the normative
[`docs/SCOPE.md`](docs/SCOPE.md) contract.

## Unreleased

### Product contract

- Defined the dense autoregressive V1 scope, representative Hugging Face model
  matrix, and distinct Compatible, Optimized, and Certified support levels.
- Locked the versioned LaughLM 135M TPU v5e-8 parity workload and release
  thresholds.
- Added the versioned benchmark result schema and dense causal-LM MFU
  calculator.
- Defined portable core dependencies and exact CPU, PyTorch/XLA, libtpu, and
  Pallas compatibility profiles.
