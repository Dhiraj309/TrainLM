# TrainLM documentation guide

This page separates the live development plan from the project’s earlier
PyTorch/XLA work. Read the status labels before treating an item as current.

## Current direction

For a task-oriented map of the notebook, implementation, tests, and evidence
limits, start with the [coding-agent guide](AI_AGENT_GUIDE.md).

TrainLM is rebaselining toward native TorchTPU on Kaggle TPU v5e-8. The
established backend is PyTorch/XLA. An experimental, explicitly selected
single-device native TorchTPU path and a from-scratch starter workflow are
implemented in PR #28. PR #29 merged the streamed uint32 packed-data path into
the scratch workflow. PR #30 added passing CI coverage for packed-data update
parity, PR #31 added the random-init GPT-2 train/eval/checkpoint/export CPU
smoke, PR #32 added exact packed-data checkpoint recovery, and PR #33 fixed
active notebooks to load merged `main`; PR #34 added a BPE tokenizer trained
from the training split; PR #35 builds GPT-2 from an explicit local config;
PR #36 adds a pinned user-selected HF corpus and split/text-column settings.
PR #37 adds opt-in BF16 autocast and precision reporting; PR #38 adds learning
rate, weight decay, warmup, and seed controls; PR #39 adds opt-in gradient
checkpointing; PR #40 adds per-device batch sizes; PR #41 adds supervised-token
budgets and actual token counts; PR #42 adds configurable sparse logging and
final loss snapshots; PR #43 packages the exported model/tokenizer and JSON
evidence into one download for Colab or Kaggle. The current notebook check
reloads that saved artifact on CPU and confirms it generates tokens. It does
not certify TPU generation or training quality. PR #44 merged this check at
`fad1df41278d4ea904b4d26968d75ee97ae17923`; it adds no TPU-generation or
convergence evidence. PR #45 rebase-merged at
`d983408221b92d631aea817bee7539e07de1bbf5`; it adds agent-facing documentation
and package metadata, not runtime support or new TPU evidence. These checks do
not certify TPU support or end-to-end convergence. PR #20 closed T0.2.
T1 software probes cover shared hardware and capability discovery, dependency
metadata and process lifecycle, and a distributed entry check; owner-run
compile, lifecycle and v5e-8 collective evidence remain pending. Use v5e-1 for
single-device checks and v5e-8 for collectives, DP8, sharding, distributed
checkpoints, or scaling. The owner reported a passing standalone eager TPU
smoke. Automatic native TPU routing remains disabled until compatibility and
hardware gates pass. Trainer preflight exposes a non-allocating discovery
snapshot; `accelerator="native_tpu"` is the opt-in single-device prototype.

- [TorchTPU utility inventory, automatic discovery and UX](TORCHTPU_UTILITIES.md)
- [T0–T11 roadmap and gates](TORCHTPU_REBASELINE.md)
- [Code salvage inventory](TORCHTPU_SALVAGE_INVENTORY.md)
- [Accepted mapping of legacy work to T0–T11](PR_DELIVERY_PLAN.md#t02-accepted-mapping-to-native-torchtpu-gates)
- [Current handoff and verified baseline](HANDOFF.md)
- [Current implementation context](IMPLEMENTATION_CONTEXT.md)
- [T1 v5e-1 single-device probe](T1_V5E1_PROBE.md)
- [Open T1 single-device probe directly in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TorchTPU_T1_v5e1_Probe.ipynb) · [view notebook source](../notebooks/TorchTPU_T1_v5e1_Probe.ipynb)
- [Open optional v5e-8 distributed probe directly in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TorchTPU_T1_v5e8_Distributed_Probe.ipynb) · [view notebook source](../notebooks/TorchTPU_T1_v5e8_Distributed_Probe.ipynb)
- [Open experimental single-device training directly in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb) · [view notebook source](../notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb)
- [Open native TorchTPU pretraining from scratch directly in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) · [view notebook source](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)

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
