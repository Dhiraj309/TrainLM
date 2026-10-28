# T1 single-device probe on v5e-1

Use this probe on a single-chip Cloud TPU v5e VM with the owner-approved
TorchTPU runtime. The notebook detects the official PyTorch device by testing
a real allocation and otherwise runs an explicitly labeled CPU smoke.

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
provides a guided version and selects tpu when a real tensor allocation
succeeds, otherwise running an explicitly labeled CPU smoke. CPU success is
not TPU evidence.

```bash
python examples/torchtpu_t1_v5e1_probe.py --device tpu
```

The script refuses `cpu` unless `--allow-cpu` is explicitly supplied, so an
unnoticed CPU fallback cannot be reported as TPU success.

The first run checks eager forward execution, finite loss and gradients, and a
real AdamW parameter update using a tiny synthetic model. To test the public
PyTorch compile entry point separately, run:

```bash
python examples/torchtpu_t1_v5e1_probe.py \
  --device tpu --compile
```

Compilation may take longer than eager execution. Preserve the complete output
and any runtime/compiler error; do not silently fall back to eager mode.

## Evidence to record

Save the VM machine type and image/runtime identifier, accelerator type,
Python and PyTorch versions, printed Torch/TPU/XLA-related distribution
versions, exact command,
and complete output. Record whether import, device placement, forward,
autograd, optimizer update, and optional compilation passed. Do not include
tokens, credentials, or unrelated environment variables in the evidence.

This is a v5e-1 single-device smoke only. It cannot establish multi-device
collectives, DP8, sharding/FSDP, distributed checkpoint behavior, v5e-8
performance, or production support. Run those checks on v5e-8 when the
operation requires multiple devices. A failure on v5e-1 is useful T1 evidence;
do not add dependency pins or claim support until the runtime owner and package
provenance are understood.
