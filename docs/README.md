# TrainLM documentation guide

This page separates the live development plan from the project’s earlier
PyTorch/XLA work. Read the status labels before treating an item as current.

## Current direction

TrainLM is rebaselining toward native TorchTPU on Kaggle TPU v5e-8. The
current backend in the code is PyTorch/XLA; native TorchTPU is a target, not an
implemented or certified backend. The active planning work is T0: inventory
existing code, map old proposals to native gates, and decide the next work only
after owner review.

- [T0–T11 roadmap and gates](TORCHTPU_REBASELINE.md)
- [Code salvage inventory](TORCHTPU_SALVAGE_INVENTORY.md)
- [Candidate work mapped to T0–T11](PR_DELIVERY_PLAN.md#t02-candidate-mapping-to-native-torchtpu-gates)
- [Current handoff and verified baseline](HANDOFF.md)
- [Current implementation context](IMPLEMENTATION_CONTEXT.md)

## Status meanings

- **Current:** the approved work sequence or observed repository state.
- **Implemented:** present in code; hardware support still depends on the
  evidence listed for that backend.
- **Partial:** code or tests exist, with required validation or integration
  still outstanding.
- **Historical XLA:** describes the PyTorch/XLA implementation or a past
  measurement. It is useful evidence and source material, not TorchTPU support.
- **Deferred:** deliberately outside the active sequence until its stated
  dependency or approval is met.
- **Deprecated:** no longer a supported user path. A historical page is not
  deprecated unless it says so explicitly.

## Reference by topic

| Topic | Start here | Status boundary |
|---|---|---|
| Product support | [Dense-AR scope](SCOPE.md) | Support claims require matching evidence. |
| Dependencies | [Compatibility policy](DEPENDENCIES.md) | XLA profiles are documented; no TorchTPU profile is verified. |
| Runtime | [Backend protocol](runtime/BACKEND_PROTOCOL.md) | Generic contract; backend guides describe their named backend. |
| PyTorch/XLA | [XLA runtime guide](runtime/PYTORCH_XLA.md) | Existing implementation/reference, not native TorchTPU. |
| Data | [Data contracts](data/PACKED_BINARY_MANIFEST.md) | Generic format and reader behavior, with backend checks still required. |
| Checkpoint/export | [Checkpoint contracts](checkpoint/CONTRACTS.md) | Portable contract; exact distributed resume requires target validation. |
| Performance | [Benchmark index](benchmarks/README.md) | Historical measurements must be labeled by backend and workload. |
| Release snapshot | [Dense-AR V1 status](release/DENSE_AR_V1_STATUS.md) | Version 0.1-dev XLA-era snapshot, not TorchTPU certification. |

## Historical roadmap

[`ROADMAP.md`](ROADMAP.md) preserves the detailed M0–M17 specification and
status from the XLA era. It is retained for traceability. Its milestone labels
and XLA-specific PR cards are not the active TorchTPU queue; use the T0–T11
roadmap and owner-reviewed T0.2 mapping for current sequencing.
