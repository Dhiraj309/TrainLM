# T1 single-device probe on v5e-1

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

Compilation may take longer than eager execution. Preserve the complete output
and any runtime/compiler error; do not silently fall back to eager mode. TPU
detection and training share one process to avoid starting two TPU runtimes.

## Evidence to record

Save the VM machine type and image/runtime identifier, accelerator type,
Python and PyTorch versions, printed Torch/TPU/XLA-related distribution
versions, exact command,
and complete output. Record whether import, device placement, forward,
autograd, optimizer update, and optional compilation passed. Do not include
tokens, credentials, or unrelated environment variables in the evidence.

This is intended as a v5e-1 single-device smoke only. It cannot establish multi-device
collectives, DP8, sharding/FSDP, distributed checkpoint behavior, v5e-8
performance, or production support. Run those checks on v5e-8 when the
operation requires multiple devices. A failure on v5e-1 is useful T1 evidence;
do not add dependency pins or claim support until the runtime owner and package
provenance are understood.
