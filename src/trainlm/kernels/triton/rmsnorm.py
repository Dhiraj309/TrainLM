"""Fused Triton RMSNorm kernel with autograd and Inductor/triton_op compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    _HAS_TRITON = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


@dataclass(frozen=True, slots=True)
class TritonRMSNormPatchResult:
    """Summary of reversible Triton RMSNorm patching on a model."""

    applied: bool
    patched_modules: int
    reason: str | None = None


def is_triton_available() -> bool:
    """Return True when Triton is importable and CUDA is available."""
    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:

    @triton.jit
    def _rms_norm_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        RSTD_ptr,
        stride_x_row,
        stride_y_row,
        N: tl.constexpr,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        X_row = X_ptr + row * stride_x_row
        Y_row = Y_ptr + row * stride_y_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = tl.math.rsqrt(mean_sq + eps)
        tl.store(RSTD_ptr + row, rstd)

        y = x * rstd * w
        tl.store(Y_row + cols, y, mask=mask)

    @triton.jit
    def _rms_norm_bwd_dx_kernel(
        DX_ptr,
        DY_ptr,
        DW_ptr,
        X_ptr,
        W_ptr,
        RSTD_ptr,
        stride_dx_row,
        stride_dy_row,
        stride_x_row,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        DX_row = DX_ptr + row * stride_dx_row
        DY_row = DY_ptr + row * stride_dy_row
        X_row = X_ptr + row * stride_x_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY_row + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row)

        x_hat = x * rstd
        dy_w = dy * w
        c1 = tl.sum(dy_w * x_hat, axis=0) / N
        dx = rstd * (dy_w - x_hat * c1)

        tl.store(DX_row + cols, dx, mask=mask)
        tl.atomic_add(DW_ptr + cols, dy * x_hat, mask=mask)


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        shape = x.shape
        n_cols = shape[-1]
        x_2d = x.reshape(-1, n_cols).contiguous()
        n_rows = x_2d.shape[0]
        block_size = triton.next_power_of_2(n_cols)

        y_2d = torch.empty_like(x_2d)
        rstd = torch.empty((n_rows,), dtype=torch.float32, device=x.device)
        w = weight.contiguous()

        _rms_norm_fwd_kernel[(n_rows,)](
            x_2d,
            w,
            y_2d,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            N=n_cols,
            eps=eps,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(x_2d, w, rstd)
        ctx.n_cols = n_cols
        ctx.block_size = block_size
        return y_2d.reshape(shape)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x_2d, w, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, ctx.n_cols).contiguous()
        n_rows = dy_2d.shape[0]

        dx_2d = torch.empty_like(x_2d)
        dw = torch.zeros((ctx.n_cols,), dtype=torch.float32, device=w.device)

        _rms_norm_bwd_dx_kernel[(n_rows,)](
            dx_2d,
            dy_2d,
            dw,
            x_2d,
            w,
            rstd,
            dx_2d.stride(0),
            dy_2d.stride(0),
            x_2d.stride(0),
            N=ctx.n_cols,
            BLOCK_SIZE=ctx.block_size,
        )
        return dx_2d.reshape(dy.shape), dw.to(dtype=w.dtype), None


def triton_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute fused RMSNorm via Triton on CUDA when available, else F.rms_norm."""
    if (
        _HAS_TRITON
        and hidden_states.is_cuda
        and hidden_states.shape[-1] <= 16384
    ):
        w = weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)
        return _TritonRMSNormFn.apply(hidden_states, w, float(eps))
    rms_norm_fn = getattr(F, "rms_norm", None)
    if callable(rms_norm_fn):
        w = weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)
        return rms_norm_fn(hidden_states, (hidden_states.shape[-1],), w, float(eps))
    x_fp32 = hidden_states.to(torch.float32)
    var = x_fp32.pow(2).mean(-1, keepdim=True)
    return weight * (x_fp32 * torch.rsqrt(var + eps)).to(dtype=hidden_states.dtype)


def _is_standard_rms_norm_module(module: nn.Module) -> bool:
    class_name = type(module).__name__
    if "Liger" in class_name or "Gemma" in class_name:
        return False
    if not class_name.endswith("RMSNorm"):
        return False
    if getattr(module, "add_unit_offset", False):
        return False
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    return hasattr(module, "variance_epsilon") or hasattr(module, "eps")


def apply_triton_rms_norm(model: nn.Module) -> TritonRMSNormPatchResult:
    """Reversibly replace HF RMSNorm forward methods with ``triton_rms_norm``."""
    if not _HAS_TRITON:
        return TritonRMSNormPatchResult(
            applied=False,
            patched_modules=0,
            reason="triton package unavailable",
        )
    patched = 0
    for module in model.modules():
        if getattr(module, "_trainlm_triton_rms_norm_patched", False):
            continue
        if not _is_standard_rms_norm_module(module):
            continue
        orig_forward = module.forward
        eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))

        def _triton_forward(
            hidden_states: torch.Tensor,
            _mod: nn.Module = module,
            _eps: float = eps,
        ) -> torch.Tensor:
            return triton_rms_norm(hidden_states, _mod.weight, _eps)

        module._trainlm_orig_rms_forward = orig_forward  # type: ignore[attr-defined]
        module._trainlm_triton_rms_norm_patched = True  # type: ignore[attr-defined]
        module.forward = _triton_forward  # type: ignore[method-assign]
        patched += 1

    if patched == 0:
        return TritonRMSNormPatchResult(
            applied=False,
            patched_modules=0,
            reason="no compatible RMSNorm modules found",
        )
    return TritonRMSNormPatchResult(applied=True, patched_modules=patched)


def restore_triton_rms_norm(model: nn.Module) -> int:
    """Restore original ``forward`` methods patched by ``apply_triton_rms_norm``."""
    restored = 0
    for module in model.modules():
        if not getattr(module, "_trainlm_triton_rms_norm_patched", False):
            continue
        orig_forward: Any = getattr(module, "_trainlm_orig_rms_forward", None)
        if orig_forward is not None:
            module.forward = orig_forward  # type: ignore[method-assign]
        if hasattr(module, "_trainlm_orig_rms_forward"):
            delattr(module, "_trainlm_orig_rms_forward")
        if hasattr(module, "_trainlm_triton_rms_norm_patched"):
            delattr(module, "_trainlm_triton_rms_norm_patched")
        restored += 1
    return restored
