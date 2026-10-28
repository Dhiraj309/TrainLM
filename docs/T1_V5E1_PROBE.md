# T1 single-device probe on v5e-1

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/docs/01-t1-v5e1-probe/notebooks/TorchTPU_T1_v5e1_Probe.ipynb)

Open the notebook in Colab to run or copy cells directly, which avoids selecting code from this guide on a mobile device.

The setup cell clones a fresh snapshot of the T1 branch, records its full
commit SHA, and preserves existing checkouts. Later cells reject changed
source so results cannot silently come from an older or edited checkout.
The last cell downloads `trainlm_t1_evidence.json` in Colab or provides a
file link on other notebook hosts. Run it even when a diagnostic cell fails.

Use this probe on a single-chip Cloud TPU v5e VM with the owner-approved
TorchTPU runtime. By default, it detects TPU through a real allocation and
runs the smoke in the same process, or selects CPU with an explicit CPU-only
label.

Hosted [Kaggle notebooks](https://www.kaggle.com/product-announcements/607202)
currently document TPU v5e-8; the [public Colab TPU notebook](https://colab.research.google.com/notebooks/tpu.ipynb)
documents TPU v2. Neither hosted option establishes v5e-1 behavior. Colab can
be used as a frontend connected to a one-chip v5e VM; that VM provides the
hardware. See Google's [PyTorch on Cloud TPU guide](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)
for the TorchTPU package and device setup.

## Run

Start a fresh one-chip v5e VM with the intended image. Google lists
ct5lp-hightpu-1t as a one-chip v5e VM shape; confirm regional availability
and record the selected machine type, runtime image, and accelerator before
running the probe. In a notebook cell,
run:

The [Kaggle/Colab-style notebook](../notebooks/TorchTPU_T1_v5e1_Probe.ipynb)
provides a guided version and runs detection and the smoke in one process.
CPU success is not TPU evidence.
Its first diagnostic cell runs the read-only
[install-profile check](../examples/torchtpu_t1_install_preflight.py) and
prints a structured result. An `incompatible` or `incomplete` profile is not
silently treated as an install pass; the standalone smoke remains a separate
environment experiment.

```bash
python examples/torchtpu_t1_v5e1_probe.py
```

To save only the bounded discovery result observed inside the probe process,
use `--report-json` with a local artifact path. The report is written after
device discovery, before the model smoke; a `ready` report alone does not mean
the forward, backward or optimizer checks passed.

```bash
python examples/torchtpu_t1_v5e1_probe.py --report-json t1_worker_discovery.json
```

The script refuses `cpu` unless `--allow-cpu` is explicitly supplied, so an
unnoticed CPU fallback cannot be reported as TPU success.

The first run checks eager forward execution, finite loss and gradients, and a
real AdamW parameter update using a tiny synthetic model. To test the public
PyTorch compile entry point separately, run:

```bash
python examples/torchtpu_t1_v5e1_probe.py \
  --compile
```

The notebook reports installed versions and read-only pip dependency conflicts
before the smoke; it bounds eager runs to 300 seconds and compile runs to 900
seconds. A timeout is a failure, not TPU support evidence.

### Latest owner result (2026-10-28)

The Colab report at source `f122b7e1` passed native TPU allocation, forward,
autograd and AdamW update. Runtime properties returned device name `TPU v5e`
and one process-visible device; the owner confirmed this was v5e-1. The shared
Colab output did not include topology or VM image details, which Colab may
hide. Python was 3.13.18 with
torch 2.14.1+cpu, torch_tpu 0.2.0 and libtpu 0.0.50. The read-only package
profile was `incompatible`: TrainLM was absent from installed metadata, torch
2.14.1+cpu is outside TrainLM's declared `<2.14` range, and `pip check`
reported conflicts. TorchTPU's own torch requirement matched this version.
No packages were installed or changed by the notebook. Explicit compile and
repeat-process lifecycle checks were skipped.

Before resolving the dependency profile, run the check in a clean environment
with TrainLM installed from this source revision. Record the actual accelerator
and VM image, then repeat on a torch version inside TrainLM's declared range
that is supported by the installed TorchTPU wheel. Keep compatibility and
hardware results separate; do not widen dependency bounds from this Colab
runtime alone.

Compilation may take longer than eager execution. Preserve the complete output
and any runtime/compiler error; do not silently fall back to eager mode. TPU
detection and training share one process to avoid starting two TPU runtimes.

## Optional restart and cleanup check

The [lifecycle harness](../examples/torchtpu_t1_lifecycle_probe.py) prints its
plan without launching anything by default:

```bash
python examples/torchtpu_t1_lifecycle_probe.py
```

On an isolated single-device TPU runtime, opt in with `--run`. It launches two
fresh processes sequentially, requires each native allocation and one-step
smoke to pass, compares device/version observations, and saves `summary.json`
plus each discovery report in a new session directory. Each child is bounded
to 300 seconds; timeout or interruption terminates its process group. The
notebook has a copyable cell with `RUN_LIFECYCLE = False` by default.

```bash
python examples/torchtpu_t1_lifecycle_probe.py --run --repeats 2 \
  --timeout 300 --output-dir t1_lifecycle_reports
```

Do not infer cleanup or repeatability from a dry run or CPU test. If the first
attempt fails, the harness stops; preserve the saved result and investigate
before another TPU launch. Record the actual accelerator identity separately.

## Evidence to record

Share the downloaded JSON and the actual VM machine type, image/runtime
identifier and accelerator type. The report records the source SHA, Python
and package versions, normalized discovery, and package/eager/compile/
lifecycle/collective outcomes. CPU runs retain a non-TPU label and skipped
checks stay `not_run`. It does not export raw logs, environment variables,
coordinator addresses or local artifact paths. Keep raw output locally for
investigation; review any additional logs before sharing them.

Package incompatibility remains visible even when the standalone TPU smoke
passes. Fatal core-dump or shutdown-hang messages fail a probe despite exit
code zero. These are feasibility observations requiring owner review; the
report never marks the entire T1 gate or native training as certified.

This is intended as a v5e-1 single-device smoke only. It cannot establish multi-device
collectives, DP8, sharding/FSDP, distributed checkpoint behavior, v5e-8
performance, or production support. Run those checks on v5e-8 when the
operation requires multiple devices. A failure on v5e-1 is useful T1 evidence;
do not add dependency pins or claim support until the runtime owner and package
provenance are understood.
