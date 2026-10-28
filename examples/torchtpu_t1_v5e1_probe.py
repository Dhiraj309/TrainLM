#!/usr/bin/env python3
"""Single-device PyTorch smoke probe for the owner-provided TorchTPU runtime.

This script intentionally does not guess a TorchTPU package or device API.
Pass the device string documented by the installed Kaggle runtime. The probe
checks eager tensor/model execution, autograd, and an optimizer update. Pass
``--compile`` separately to probe the runtime's ``torch.compile`` support.

It does not test collectives, sharding, distributed checkpoints, or scaling.
Those checks require v5e-8 and remain separate acceptance evidence.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import platform
import re
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        required=True,
        help="Device string from the installed runtime's public documentation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of tiny synthetic optimizer steps (default: 3).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with the public torch.compile API before running.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU for a local script check; this is not TPU evidence.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    return args


def _print_environment(torch: object) -> None:
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch={getattr(torch, '__version__', 'unknown')}")

    pattern = re.compile(r"torch|tpu|xla|libtpu", re.IGNORECASE)
    packages = sorted(
        (dist.metadata.get("Name", ""), dist.version)
        for dist in metadata.distributions()
        if pattern.search(dist.metadata.get("Name", ""))
    )
    print("related_distributions:")
    if packages:
        for name, version in packages:
            print(f"  {name}=={version}")
    else:
        print("  (none found)")


def _run(torch: object, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "Refusing CPU: pass --allow-cpu only for a local script check. "
            "For T1, use the device string documented by the Kaggle runtime."
        )
    if device.type == "cpu":
        print("scope=CPU script check only; this is not TPU evidence")

    print(f"device={device}")
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 2),
    ).to(device)
    if args.compile:
        model = torch.compile(model)
        print("compile=torch.compile enabled")
    else:
        print("compile=eager")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_parameters = [p.detach().clone() for p in model.parameters()]

    for step in range(1, args.steps + 1):
        inputs = torch.randn(8, 4, device=device)
        targets = torch.randn(8, 2, device=device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        if not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError(f"step {step}: loss is not finite")
        loss.backward()

        gradients = [p.grad for p in model.parameters()]
        if not gradients or any(grad is None for grad in gradients):
            raise RuntimeError(f"step {step}: one or more gradients are missing")
        if any(not bool(torch.isfinite(grad).all().item()) for grad in gradients):
            raise RuntimeError(f"step {step}: one or more gradients are not finite")

        optimizer.step()
        print(f"step={step} loss={loss.detach().cpu().item():.6f}")

    changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_parameters, model.parameters(), strict=True)
    )
    if not changed:
        raise RuntimeError("optimizer step did not change any model parameter")

    print("forward=PASS")
    print("autograd=PASS")
    print("optimizer_update=PASS")
    if args.compile:
        print("compile=PASS")
    print("result=single-device smoke passed; no distributed or performance claim")


def main() -> int:
    args = _parse_args()
    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch import failed: {exc}", file=sys.stderr)
        return 2

    _print_environment(torch)
    _run(torch, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
