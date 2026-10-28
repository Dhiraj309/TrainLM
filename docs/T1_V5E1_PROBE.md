# T1 single-device probe on Kaggle v5e-1

Use this probe to gather initial TorchTPU environment evidence on v5e-1. It
does not install or pin packages and deliberately does not assume a TorchTPU
module name or device alias. Use the device string and package/runtime setup
documented by the owner-approved Kaggle image.

## Run

Start a fresh v5e-1 session with the intended image. Record the image/runtime
name before running the probe. In a notebook cell, run:

```bash
python examples/torchtpu_t1_v5e1_probe.py --device '<runtime-documented-device>'
```

Replace the placeholder with the exact device string from that runtime's
public documentation or setup output. The script refuses `cpu` unless
`--allow-cpu` is explicitly supplied, so an unnoticed CPU fallback cannot be
reported as TPU success.

The first run checks eager forward execution, finite loss and gradients, and a
real AdamW parameter update using a tiny synthetic model. To test the public
PyTorch compile entry point separately, run:

```bash
python examples/torchtpu_t1_v5e1_probe.py \
  --device '<runtime-documented-device>' --compile
```

Compilation may take longer than eager execution. Preserve the complete output
and any runtime/compiler error; do not silently fall back to eager mode.

## Evidence to record

Save the Kaggle image/runtime identifier, accelerator type, Python and PyTorch
versions, printed Torch/TPU/XLA-related distribution versions, exact command,
and complete output. Record whether import, device placement, forward,
autograd, optimizer update, and optional compilation passed. Do not include
tokens, credentials, or unrelated environment variables in the evidence.

This is a v5e-1 single-device smoke only. It cannot establish multi-device
collectives, DP8, sharding/FSDP, distributed checkpoint behavior, v5e-8
performance, or production support. Run those checks on v5e-8 when the
operation requires multiple devices. A failure on v5e-1 is useful T1 evidence;
do not add dependency pins or claim support until the runtime owner and package
provenance are understood.
