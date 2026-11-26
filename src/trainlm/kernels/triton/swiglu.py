"""Fused Triton SwiGLU kernel (silu(gate) * up) with backward pass."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    _HAS_TRITON = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(
        GATE_ptr,
        UP_ptr,
        OUT_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        g = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        silu_g = g * sig
        tl.store(OUT_ptr + offs, silu_g * u, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        DOUT_ptr,
        GATE_ptr,
        UP_ptr,
        DGATE_ptr,
        DUP_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        dout = tl.load(DOUT_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        silu_g = g * sig
        du = dout * silu_g
        dsilu = sig * (1.0 + g * (1.0 - sig))
        dg = dout * u * dsilu
        tl.store(DGATE_ptr + offs, dg, mask=mask)
        tl.store(DUP_ptr + offs, du, mask=mask)


class _TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        g = gate.contiguous()
        u = up.contiguous()
        out = torch.empty_like(g)
        n_elements = g.numel()
        block = 1024
        grid = (triton.cdiv(n_elements, block),)
        _swiglu_fwd_kernel[grid](g, u, out, n_elements, BLOCK_SIZE=block)
        ctx.save_for_backward(g, u)
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors
        dout_c = dout.contiguous()
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        n_elements = g.numel()
        block = 1024
        grid = (triton.cdiv(n_elements, block),)
        _swiglu_bwd_kernel[grid](dout_c, g, u, dg, du, n_elements, BLOCK_SIZE=block)
        return dg, du


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute ``F.silu(gate) * up`` using fused Triton on CUDA, else eager PyTorch."""
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)()
    if _HAS_TRITON and gate.is_cuda and up.is_cuda and gate.shape == up.shape and not is_compiling:
        return _TritonSwiGLUFn.apply(gate, up)
    return F.silu(gate) * up
