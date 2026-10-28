# TorchTPU utility inventory and integration plan

Status: researched design, 2026-10-28; implementation and hardware validation
remain separate gates. This inventory covers TrainLM's dense autoregressive
training lifecycle, not every operator in the upstream library.

## Evidence and source policy

The owner supplied a successful Colab-run eager probe: Python 3.13.18,
torch 2.14.1+cpu, torch_tpu 0.2.0, libtpu 0.0.50, tpu-info 0.14.2.
Allocation, three forward/backward steps and a reference AdamW update passed.
Hardware model/topology, torch.compile, distributed operation, resume and
performance are unverified. This is owner-provided output, not a local rerun.
The preceding process failure's exact cause remains unconfirmed.

Use versioned documentation plus the public API policy, then inspect the
installed package's supported interfaces. An upstream API listing is not
proof that a particular wheel, hardware generation or TrainLM path supports it.
The 0.2.0 reference and current main differ in some names; resolve those
differences in a compatibility adapter, never by blindly trying internal APIs.

Sources checked:

- [A: TorchTPU 0.2.0 API reference](https://torchtpu.readthedocs.io/en/0.2.0/api_reference/)
- [B: current public API policy/list](https://github.com/google-pytorch/torch_tpu/blob/main/docs/public_apis.md)
- [C: distributed support](https://torchtpu.readthedocs.io/en/0.2.0/distributed_support/)
- [D: execution model](https://torchtpu.readthedocs.io/en/0.2.0/torchtpu_execution_model/)
- [E: precision](https://torchtpu.readthedocs.io/en/0.2.0/numerical_precision/)
- [F: custom kernels](https://github.com/google-pytorch/torch_tpu/blob/main/docs/custom_kernels.md)
- [G: profiling](https://torchtpu.readthedocs.io/en/0.2.0/performance_profiling_with_xprof/)
- [H: LibTPU monitoring](https://docs.cloud.google.com/tpu/docs/tpu-monitoring-library)
- [I: tpu-info](https://docs.cloud.google.com/tpu/docs/tpu-info-cli)
- [J: dynamic shapes](https://torchtpu.readthedocs.io/en/0.2.0/dynamic_shapes/)
- [K: upstream packaging](https://github.com/google-pytorch/torch_tpu/blob/main/pyproject.toml)
- [L: PyTorch distributed contracts](https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.md)
- [M: FSDP guidance](https://torchtpu.readthedocs.io/en/0.2.0/case_studies_fsdp/)

## Capability inventory

The integration and acceptance columns are TrainLM design decisions.
A/B API stability labels describe current upstream policy; installed-version
validation is required for all entries. Optional tools must remain optional.

| Area | Upstream facility / evidence | TrainLM responsibility and acceptance | Gate |
|---|---|---|---|
| Package/runtime provenance | torch_tpu registration, package metadata; K, H | Record Python, torch, TorchTPU, libtpu and source revision; validate resolver compatibility without upgrading the notebook implicitly. | T1 |
| Availability and placement | `torch.accelerator.current_accelerator`, `is_available`, `device_count`; A/B | Confirm backend type, then prove allocation in the owning process. Classify absent, incompatible, busy and initialization-error states independently. | T1–T2 |
| Hardware identity and capacity | `torch.tpu.get_device_properties`: name, topology, per-device/per-chip HBM and FLOPs; A/B, experimental | Normalize reported facts with units and source. Unknown fields remain unknown. Distinguish chips, visible devices, participating ranks and host count; one rank does not prove one-chip hardware. | T1 |
| Hardware diagnostics | `tpu-info` CLI; I | Optional bounded diagnostic snapshot when available; retain tool version/status. No user copy/paste requirement or parsing stdout as the primary runtime contract. | T1–T2 |
| Launch and topology | `torch.tpu.distributed.set_tpu_launch_env`; `torchrun`; C | Prepare topology before initialization, keep tensors out of the parent launcher, and own startup, timeouts, cancellation and cleanup. Preserve explicit user launch configuration. | T1, T4 |
| Collective execution | `torch.distributed` with `tpu_dist`; C | Implement reductions/barriers through backend protocol; validate global valid-token weighting, uneven masks, accumulation and rank failures. | T4 |
| Mesh and scaling | `torch.tpu.distributed.topology_aware_mesh`, DeviceMesh, DTensor, DDP/FSDPv2; C/M | Keep logical mesh independent of physical identifiers. Verify each parallel strategy separately; no automatic FSDP/TP enablement from device count. | T4, T11 |
| Precision | `torch.amp.autocast("tpu")`, capability query; A. MXU precision controls; E | Select a validated dtype and accumulation policy; test numerical parity. A listed dtype is not operator-wide support. Configure precision outside compiled regions when required. | T1, T3, T6 |
| RNG and deterministic boundaries | `torch.accelerator.random` state queries; TPU seed/state restore APIs; A/B | Capture Python, CPU, device and data-loader state; test clean-process continuation. Do not promise deterministic kernels merely from a seed. | T3, T5 |
| Execution and synchronization | `torch.accelerator.synchronize`; eager fusion modes, streams/events; B/D | Put synchronization at reporting, checkpoint and timing boundaries. Use the runtime scheduler before adding manual stream orchestration. | T3, T6 |
| Compilation | `torch.compile`; D | Separate eager operation compilation from the explicit compile API. Prove loss/gradient parity, graph behavior and explain fallback. Strict mode must fail visibly. | T1, T6 |
| Shapes and cache | Tiered caches; D. Bounded dynamic-shape limitations; J | Own cache location/provenance and shape policy. Test cold/warm/restart behavior for the exact version; do not infer persistent reuse from a cache counter. | T3, T6, T9 |
| Memory and telemetry | Accelerator memory APIs; B. `libtpu.sdk.tpumonitoring` metric discovery; H | Report allocated/reserved/peak/capacity distinctly, with scope and units. Unsupported metrics are unavailable, never zero. Bound polling overhead. | T1, T10 |
| Profiling and diagnosis | `torch.profiler` CPU/TPU activities, XProf, TensorBoard, `TpuProfilerConfig`; G | Offer opt-in bounded traces and a downloadable report. Preserve child errors and compiler diagnostics; never dump credentials or the whole environment. | T1, T6, T10 |
| Data movement | Existing PyTorch loader/batch-transfer contracts; repository runtime protocol | Reuse packed readers, cursors and validation. Benchmark transfer/prefetch behavior; do not assume CUDA pin-memory behavior transfers to TPU. | T3 |
| Attention and custom kernels | Native operators, Splash SDPA, Pallas bridge, Tokamax; B/F | Audit upstream providers before implementing a new kernel. Require mask/gradient/HF-export parity and measured benefit. Helion remains an experimental candidate. | T7 |
| Optimizers and compiled steps | Traceable AdamW, Muon and SingleTraceTrainer; B | Keep reference AdamW first; adapt experimental providers behind capability guards. Validate state conversion/resume before enabling fused or full-step paths. | T8–T9 |
| Checkpoint/export | PyTorch state/distributed checkpoint building blocks; L, repository contracts | TrainLM owns atomicity, data position, RNG, topology and HF export. TorchTPU availability does not establish exact resume. | T5, T11 |
| Evidence and UX | Upstream facts plus TrainLM policy | Produce one structured report consumed by notebook, CLI and trainer. Statuses: observed, documented, tested, unavailable, failed; attach version and hardware scope. | T1–T11 |

Current B prefers generic accelerator APIs over redundant backend calls.
For example, `torch.tpu.device_count/is_available/synchronize` are internal
on the modern PyTorch versions documented there. `get_device_name` is
deprecated in 0.2.1; hardware-property, optimizer, mesh and profiler extensions
include experimental APIs. Do not enable the legacy-API escape hatch.
The older topology alias in A conflicts with B/C; use the version-reviewed
public interface, not a tutorial spelling copied without verification.

MoE/SparseCore workloads, serving engines, Ray clusters and multi-slice
orchestration remain outside the immediate dense-AR scope. They are useful
ecosystem references, not prerequisites for the basic trainer.

## Existing TrainLM code to reuse or adapt

| Repository area | Observed state | Disposition |
|---|---|---|
| `src/trainlm/api.py` | `auto` selects CUDA or CPU; explicit TPU enters the existing coordinator | Add native discovery/routing behind the public API after compatibility tests; preserve intentional CPU/CUDA selection. |
| `src/trainlm/runtime/base.py` | ExecutionBackend, LogicalMesh, BackendDiagnostics already exist | Extend portable contracts only where needed; implement a native adapter. |
| `src/trainlm/runtime/runtime.py` | Portable TorchRuntime targets CPU/CUDA | Reuse lifecycle/batch-transfer design; native TPU still needs its own validated behavior. |
| `src/trainlm/runtime/diagnostics.py`, `runtime/xla.py` | XLA collectors/runtime | Preserve for XLA; replace collectors behind the adapter for native TPU. |
| `src/trainlm/_tpu_coordinator.py`, `_tpu_checkpoint.py` | Private launch and checkpoint infrastructure | Retain error/lifecycle contracts; audit XLA-specific worker assumptions before reuse. |
| `src/trainlm/checkpoint/`, `training/` | Resume, export, trainer and metric contracts | Reuse and re-prove device RNG, distributed state and clean-process resume. |
| `src/trainlm/optimization/`, `monitoring/telemetry.py` | Capability plans, explanations, provider gates and metric schema | Extend the existing report instead of a second independent planner. |
| `examples/torchtpu_t1_v5e1_probe.py` and T1 notebook | Shared device-owned discovery, bounded JSON artifact and labeled CPU fallback; no production native routing | Keep T1 evidence separate from trainer support; verify hardware identity and install profile before enabling a native worker. |
| `pyproject.toml` | `torch>=2.9,<2.14` | Owner's 2.14.1 smoke is outside the package contract. Test a separate native profile before changing ranges; the script pass is not a TrainLM install test. |

## Desired user experience and internal discovery contract

Users select an accelerator in Colab/Kaggle or attach a Cloud TPU VM, supply
model/data/tokenizer and training arguments, then call `trainer.train()`.
TrainLM discovers the hardware, chooses a validated backend, manages worker
lifetime, explains the selected precision/compile/parallel plan, and saves
diagnostics automatically. It cannot provision unavailable hardware or infer
a TPU model from the notebook provider's name.

The public design remains `accelerator="auto"` or an explicit backend choice.
Proposed precedence is validated native TPU, then CUDA, then CPU, with clear
selection reporting and an explicit override. Preserve existing CPU/CUDA
behavior on machines without TPU. Explicit TPU requests fail if unusable.
An absent accelerator can follow the documented auto policy; a detected TPU
with initialization, compatibility or ownership failure must produce a
targeted error rather than silently training on CPU. The development smoke's
current permissive CPU fallback is not the production contract.

Discovery must be staged: package/launch inspection before worker startup;
hardware properties and allocation in the process that will own the device;
rank/topology reconciliation after distributed initialization. Do not initialize
JAX/TensorFlow merely to identify TPU hardware. The notebook must not allocate
a tensor and then ask another long-lived process to acquire the same device.
Repeated execution, cancellation and restart are acceptance cases.

Proposed internal report fields: schema/source revision, package versions,
host context, detected backend, device name, visibility scope, device/chip
counts, topology, rank/world size, HBM capacity/use, precision capabilities,
execution mode, cache/profile availability, probe results and failure details.
Each optional fact carries its source and unavailable/error reason. An unknown
generation can permit a generic smoke; it cannot satisfy a v5e-specific gate.

`trainer.preflight()` now returns a source-level, non-allocating TPU snapshot
with the selected existing backend; `trainer.explain(include_hardware=True)`
adds it to text, JSON or dict output. Neither method initializes a native
training worker, checks device properties or certifies hardware. `auto` blocks
on detected TPU until the native route is validated; explicit `tpu` still
means the historical XLA coordinator. A CLI diagnostic command and notebook
integration with the trainer remain planned.
The T1 single-device probe can save a bounded discovery JSON from its owning
process. Trainer result handling can consume that schema from a future native
worker summary and show its status beside parent preflight. Today's XLA worker
does not emit native TorchTPU evidence.
Code cells remain directly copyable; normal users should not type device
aliases, rank variables or parse logs. Advanced configuration stays explicit.

## Revised delivery order and acceptance

These are T1 work packages, not new branches or renumbered GitHub PRs:

1. **T1.1 — capability inventory (this document):** record public facilities,
   version policy, reuse boundaries and owner evidence.
2. **T1.2 — shared discovery/reporting (partial):** a source-level report
   records API availability, TPU allocation, process-visible count, device
   properties and package versions. The standalone probe prints it; a
   non-allocating subset now feeds trainer preflight and opt-in explanation.
   A bounded device-owning artifact and worker-result diagnostic bridge exist,
   but native worker production is pending. Next: verify owner-run reports,
   incomplete properties and version mismatches. Explicit compile evidence
   remains pending.
3. **T1.3 — installation and lifecycle feasibility (partial):** the notebook
   reports installed versions and runs read-only pip dependency checking; its
   eager/compile child processes have timeouts and surface partial output.
   Resolve the native dependency profile and verify clean install/import,
   repeated runs, cleanup and CPU regression gates on the chosen profile.
4. **T1.4 — distributed entry feasibility (partial):** the opt-in
   [single-host v5e-8 notebook](../notebooks/TorchTPU_T1_v5e8_Distributed_Probe.ipynb)
   uses the [probe script](../examples/torchtpu_t1_distributed_probe.py).
   Its parent process calls `set_tpu_launch_env` before spawning workers;
   source-level checks reject inconsistent topology, while each worker
   initializes `tpu_dist`, reports rank/world and performs one sum all-reduce.
   The launcher reconciles every rank and bounds execution/cleanup. CPU
   contracts pass; actual v5e-8 launch and collective evidence are pending.
   This probe assumes one host and does not establish gradient-correct DP (T4).
5. **T2–T3:** connect discovery and the native backend to the familiar trainer;
   prove causal-LM/data correctness. T4–T11 retain their original semantic
   gates, with the utilities above assigned to them.

Prefer v5e-1 for single-device evidence when available; v5e-8 for genuinely
multi-device checks. A one-device run on a larger slice remains labeled with
that hardware. No need to wait for eight devices to implement/test pure
discovery contracts, and no distributed certification from a single rank.

Before dependencies change or a provider defaults on, require explicit
version/hardware evidence. Next, run the T1.2/T1.3 owner hardware checks and
the opt-in T1.4 collective on an isolated v5e-8 host; use the results to
decide the native dependency profile and T2 integration scope. Current work
does not implement native training or change dependency pins.
