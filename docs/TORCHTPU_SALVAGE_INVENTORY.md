# Native TorchTPU salvage inventory

**Snapshot:** `main` at `23959d29434fd2ae3f78b20469e1d61c6b1e18c9`, after PR #18 merged.
**Purpose:** T0.1 read-only classification to guide the T0 architecture decision. This is a source review, not a compatibility claim or approval to remove code.

## Reusable candidates

| Area | Paths | Why it may carry forward | Required check before reuse |
|---|---|---|---|
| Backend-neutral engine | `src/trainlm/runtime/base.py`, `runtime/runtime.py`, `training/`, `tasks/base.py`, `tasks/adapter.py` | Defines execution hooks and keeps the training loop separate from accelerator implementations. | Compare every hook to TorchTPU's public device, distributed, compile, synchronization, and optimizer APIs. Keep accelerator initialization out of core imports. |
| HF model contract | `src/trainlm/model/huggingface.py`, `model/compatibility.py`, `model/outputs.py`, public API in `api.py` | Preserves Transformers model semantics and generic model loading. | Validate against the supported Transformers versions and real TorchTPU tensor/device behavior. Avoid model-family rewrites. |
| Data | `src/trainlm/data/`, `scripts/trainlm_tpu_data.py` | Manifest validation, packed token readers, cursors, sharding, and host-side data preparation are largely framework independent. | Verify distributed partitioning, token dtype/range, worker ownership, and data-position resume under the target topology. |
| Checkpoint and export contracts | `src/trainlm/checkpoint/`, `src/trainlm/_tpu_checkpoint.py` | Artifact validation, atomic commit records, portable HF export, and state contracts can inform backend-independent behavior. | Separate generic serialization from XLA device RNG, rank topology, collectives, and filesystem assumptions; prove clean-process resume equivalence. |
| Capability/planning core | `optimization/capabilities.py`, `inspection.py`, `planner.py`, `plan.py`, `adapters.py`, `explanation.py`, `state_dict.py`, `transforms.py` | Structural inspection, explicit eligibility, reversible transforms, and user explanations do not inherently require XLA. | Audit provider requirements and ensure unknown capabilities fall back safely. |
| Generic model operations | `tasks/causal_lm.py`, `tasks/chunked_loss.py`, `tasks/training_view.py`, selected QKV/MLP transforms | PyTorch reference implementations and semantic tests can remain useful. | Re-run output, gradient, mask, tied-weight, export, and memory tests on the selected TorchTPU software stack before enabling providers. |

## XLA-specific components

| Area | Paths / behavior | T0 direction |
|---|---|---|
| Runtime | `src/trainlm/runtime/xla.py`, `runtime/diagnostics.py`, `runtime/shape_guard.py` | Keep behind the existing runtime boundary. Map behavior to TorchTPU only where its public API has a demonstrated equivalent. XLA `mark_step`, SPMD mesh/sharding, cache, RNG, and metrics are not portable contracts. |
| Worker launch | `scripts/trainlm_tpu_worker.py`, `scripts/trainlm_tpu_training.py`, `_tpu_coordinator.py` | The worker's PJRT setup, `torch_xla.launch`, rendezvous, XLA collectives, and step synchronization are XLA-specific. Preserve the user-facing coordinator contract where practical; design a separate TorchTPU worker/runtime only after T1 establishes launch requirements. |
| Optimizer and kernels | `optimization/xla_optimizer.py`, `optimization/pallas_attention.py`, XLA-gated providers in `loss_providers.py` and attention modules | Preserve as optional XLA providers. Pallas bridges and XLA optimizer semantics cannot be assumed to work with native TorchTPU. |
| Dependencies | `tpu-xla`, `tpu-pallas` extras and `constraints/tpu-*.txt` | Keep as reproducible legacy profiles while supported. Do not rename them as TorchTPU or add guessed versions. |

## Uncertain / needs evidence

- `ExecutionBackend` is useful, but compile, mesh, sharding, synchronization, checkpoint hooks, and gradient reduction may need capability-specific contracts rather than one-for-one API translation.
- `_tpu_coordinator.py` contains reusable process supervision and structured result handling alongside XLA worker assumptions; split only after a concrete TorchTPU launch model is known.
- `runtime/fsdp.py`, optimizer sharding, and the distributed checkpoint code need topology and ownership review. No TorchTPU FSDP or multi-device support is established by these abstractions.
- `optimization/attention.py`, `hf_attention.py`, `mlp.py`, `qkv.py`, rematerialization, and compiled-step policies contain a mix of generic transforms and backend/provider assumptions; classify each provider and gate separately.
- `src/trainlm/model/trainlm/` is a custom model implementation. The target product contract remains Hugging Face semantics; determine whether this model is a test/reference fixture or supported product API before carrying it into the native backend scope.
- Tests under `tests/runtime`, `tests/optimization`, `tests/checkpoint`, and TPU scripts include CPU fakes and XLA-specific assertions. Preserve generic contract tests; label and retain XLA tests with the XLA backend.

## Dependency and evidence boundary

At this snapshot, `pyproject.toml` defines core CPU dependencies and PyTorch/XLA extras only. `docs/DEPENDENCIES.md` records XLA/Pallas constraint profiles; the checked-in CI workflow runs Python 3.11 and installs the broad development extra. The environment used for the available CPU baseline had Python 3.12.14, torch 2.9.0+cpu, Transformers 5.0.0, Hub 1.3.5, pytest 9.1.1. No TorchTPU distribution or compatibility profile is declared here. T1 must discover the supported install and runtime from the owner-approved Kaggle image and verify imports, device operations, autograd, optimizer updates, collectives, and compile behavior before changing dependency metadata.

## T0 conclusion and next gate

Carry the generic training, Hugging Face, data, serialization, and planning contracts forward as candidates. Retain the XLA backend and its tests while the native path is proved. Do not mass-delete XLA or claim native support from CPU tests, an API name, or a package install. T1 should establish the real TorchTPU environment; T2 can then decide which public configuration and backend-boundary changes are needed.
