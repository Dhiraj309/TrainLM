# TrainLM documentation guide

This page separates the live development plan from the project’s earlier
PyTorch/XLA work. Read the status labels before treating an item as current.

## Current direction

TrainLM is rebaselining toward native TorchTPU on Kaggle TPU v5e-8. The
current backend in the code is PyTorch/XLA; native TorchTPU is a target, not an
implemented or certified backend. PR #20 closed T0.2. T1 feasibility starts on
v5e-1 for single-device checks; use v5e-8 when testing multi-device collectives,
DP8, sharding, distributed checkpoints, or scaling. The owner reported a
passing standalone eager TPU smoke. T1 now prioritizes shared hardware and
capability discovery, a verified install profile and process lifecycle, then
distributed entry checks; native trainer integration remains planned.
Trainer preflight now exposes a non-allocating discovery snapshot; native TPU
training and automatic TPU routing remain unverified and disabled.

- [TorchTPU utility inventory, automatic discovery and UX](TORCHTPU_UTILITIES.md)
- [T0–T11 roadmap and gates](TORCHTPU_REBASELINE.md)
- [Code salvage inventory](TORCHTPU_SALVAGE_INVENTORY.md)
- [Accepted mapping of legacy work to T0–T11](PR_DELIVERY_PLAN.md#t02-accepted-mapping-to-native-torchtpu-gates)
- [Current handoff and verified baseline](HANDOFF.md)
- [Current implementation context](IMPLEMENTATION_CONTEXT.md)
- [T1 v5e-1 single-device probe](T1_V5E1_PROBE.md)
- [Kaggle/Colab-style T1 probe notebook](../notebooks/TorchTPU_T1_v5e1_Probe.ipynb)
- [Optional single-host v5e-8 distributed probe notebook](../notebooks/TorchTPU_T1_v5e8_Distributed_Probe.ipynb)

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
