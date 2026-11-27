"""TPU Pallas fused recurrent-injection RMSNorm kernel with PyTorch fallback.

Fuses:
    z = h_t + sigmoid(gate) * h_0
    y = RMSNorm(z, weight, eps)
in FP32 accumulation with BF16/FP16/FP32 inputs and exact analytical backward
through TorchTPU's JAX-to-PyTorch Pallas bridge (``torch.tpu.pallas.jax_op`` or
``torch_tpu._internal.pallas.custom_jax_kernel``).
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from .splash_attention import (
    _register_pallas_autograd,
    _resolve_pallas_op,
    probe_pallas_bridge,
    register_pallas_jax_op,
    verify_pallas_bridge_runtime,
)

_RECURRENT_NORM_OP_CACHE: dict[tuple[Any, ...], tuple[str, str]] = {}
_PALLAS_REC_NORM_DISABLED = False
_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None), "assume_constant_result", lambda fn: fn
)


def is_pallas_recurrent_norm_available() -> bool:
    """Return True when JAX Pallas (``pl.pallas_call``) and TorchTPU bridge exist."""
    if _PALLAS_REC_NORM_DISABLED:
        return False
    try:
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
        from jax.experimental import pallas as pl  # type: ignore[import-not-found]
    except Exception:
        return False
    if jax is None or jnp is None or not callable(getattr(pl, "pallas_call", None)):
        return False
    return verify_pallas_bridge_runtime(compiled=False)


def _pytorch_recurrent_inject_rmsnorm(
    h_t: torch.Tensor,
    h_0: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact FP32-accumulated PyTorch reference for ``RMSNorm(h_t + sigmoid(gate) * h_0)``."""
    out_dtype = h_t.dtype
    s = torch.sigmoid(gate.float())
    z = h_t.float() + s * h_0.float()
    inv_rms = torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    y = (z * inv_rms) * weight.float()
    return y.to(dtype=out_dtype)


def _resolve_block_m(rows: int) -> int:
    for candidate in (128, 64, 32, 16, 8):
        if rows >= candidate:
            return candidate
    return 8


def _jax_recurrent_norm_fwd_ref(
    h_t: Any,
    h_0: Any,
    gate: Any,
    weight: Any,
    *,
    eps: float,
) -> Any:
    import jax
    import jax.numpy as jnp

    out_dtype = h_t.dtype
    s = jax.nn.sigmoid(gate.astype(jnp.float32))
    z = h_t.astype(jnp.float32) + s * h_0.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(z * z, axis=-1, keepdims=True) + eps)
    y = (z * inv_rms) * weight.astype(jnp.float32)
    return y.astype(out_dtype)


def _jax_recurrent_norm_bwd_ref(
    h_t: Any,
    h_0: Any,
    gate: Any,
    weight: Any,
    dy: Any,
    *,
    eps: float,
) -> tuple[Any, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    h_t_f = h_t.astype(jnp.float32)
    h_0_f = h_0.astype(jnp.float32)
    s_f = jax.nn.sigmoid(gate.astype(jnp.float32))
    w_f = weight.astype(jnp.float32)
    dy_f = dy.astype(jnp.float32)

    z = h_t_f + s_f * h_0_f
    inv_rms = jax.lax.rsqrt(jnp.mean(z * z, axis=-1, keepdims=True) + eps)
    x_hat = z * inv_rms
    dx_hat = dy_f * w_f
    c_m = jnp.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    dz = inv_rms * (dx_hat - x_hat * c_m)

    reduce_axes = tuple(range(h_t_f.ndim - 1))
    grad_weight = jnp.sum(dy_f * x_hat, axis=reduce_axes).astype(weight.dtype)
    grad_h_t = dz.astype(h_t.dtype)
    grad_h_0 = (dz * s_f).astype(h_0.dtype)
    grad_s = jnp.sum(dz * h_0_f, axis=reduce_axes)
    grad_gate = (grad_s * s_f * (1.0 - s_f)).astype(gate.dtype)
    return grad_h_t, grad_h_0, grad_gate, grad_weight


def _pallas_recurrent_norm_fwd_2d(
    h_t_2d: Any,
    h_0_2d: Any,
    gate_1d: Any,
    weight_1d: Any,
    *,
    eps: float,
    interpret: bool = False,
) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    rows, hidden = h_t_2d.shape
    out_dtype = h_t_2d.dtype
    block_m = _resolve_block_m(rows)
    pad_m = (-rows) % block_m
    pad_h = (-hidden) % 128
    padded_m = rows + pad_m
    padded_h = hidden + pad_h
    grid_m = padded_m // block_m
    inv_hidden = 1.0 / float(hidden)

    h_t_in = h_t_2d if (pad_m == 0 and pad_h == 0) else jnp.pad(h_t_2d, ((0, pad_m), (0, pad_h)))
    h_0_in = h_0_2d if (pad_m == 0 and pad_h == 0) else jnp.pad(h_0_2d, ((0, pad_m), (0, pad_h)))
    g_in = gate_1d if pad_h == 0 else jnp.pad(gate_1d, (0, pad_h))
    w_in = weight_1d if pad_h == 0 else jnp.pad(weight_1d, (0, pad_h))
    g_2d = jnp.broadcast_to(g_in[None, :], (8, padded_h))
    w_2d = jnp.broadcast_to(w_in[None, :], (8, padded_h))

    def _fwd_kernel(h_t_ref: Any, h_0_ref: Any, g_ref: Any, w_ref: Any, y_ref: Any) -> None:
        ht = h_t_ref[:, :].astype(jnp.float32)
        h0 = h_0_ref[:, :].astype(jnp.float32)
        s = jax.nn.sigmoid(g_ref[0:1, :].astype(jnp.float32))
        w = w_ref[0:1, :].astype(jnp.float32)
        z = ht + s * h0
        mean_sq = jnp.sum(z * z, axis=-1, keepdims=True) * inv_hidden
        inv_rms = jax.lax.rsqrt(mean_sq + eps)
        y = (z * inv_rms) * w
        y_ref[:, :] = y.astype(out_dtype)

    call_kwargs: dict[str, Any] = {}
    if interpret:
        call_kwargs["interpret"] = True
    y_padded = pl.pallas_call(
        _fwd_kernel,
        out_shape=jax.ShapeDtypeStruct((padded_m, padded_h), out_dtype),
        grid=(grid_m,),
        in_specs=[
            pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
            pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
            pl.BlockSpec((8, padded_h), lambda i: (0, 0)),
            pl.BlockSpec((8, padded_h), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
        **call_kwargs,
    )(h_t_in, h_0_in, g_2d, w_2d)

    return y_padded[:rows, :hidden]


def _pallas_recurrent_norm_bwd_2d(
    h_t_2d: Any,
    h_0_2d: Any,
    gate_1d: Any,
    weight_1d: Any,
    dy_2d: Any,
    *,
    eps: float,
    interpret: bool = False,
) -> tuple[Any, Any, Any, Any]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    rows, hidden = h_t_2d.shape
    block_m = _resolve_block_m(rows)
    pad_m = (-rows) % block_m
    pad_h = (-hidden) % 128
    padded_m = rows + pad_m
    padded_h = hidden + pad_h
    grid_m = padded_m // block_m
    inv_hidden = 1.0 / float(hidden)

    h_t_in = h_t_2d if (pad_m == 0 and pad_h == 0) else jnp.pad(h_t_2d, ((0, pad_m), (0, pad_h)))
    h_0_in = h_0_2d if (pad_m == 0 and pad_h == 0) else jnp.pad(h_0_2d, ((0, pad_m), (0, pad_h)))
    dy_in = dy_2d if (pad_m == 0 and pad_h == 0) else jnp.pad(dy_2d, ((0, pad_m), (0, pad_h)))
    g_in = gate_1d if pad_h == 0 else jnp.pad(gate_1d, (0, pad_h))
    w_in = weight_1d if pad_h == 0 else jnp.pad(weight_1d, (0, pad_h))
    g_2d = jnp.broadcast_to(g_in[None, :], (8, padded_h))
    w_2d = jnp.broadcast_to(w_in[None, :], (8, padded_h))

    def _bwd_kernel(
        h_t_ref: Any,
        h_0_ref: Any,
        g_ref: Any,
        w_ref: Any,
        dy_ref: Any,
        dh_t_ref: Any,
        dh_0_ref: Any,
        dgate_part_ref: Any,
        dweight_part_ref: Any,
    ) -> None:
        ht = h_t_ref[:, :].astype(jnp.float32)
        h0 = h_0_ref[:, :].astype(jnp.float32)
        dy = dy_ref[:, :].astype(jnp.float32)
        s = jax.nn.sigmoid(g_ref[0:1, :].astype(jnp.float32))
        w = w_ref[0:1, :].astype(jnp.float32)

        z = ht + s * h0
        mean_sq = jnp.sum(z * z, axis=-1, keepdims=True) * inv_hidden
        inv_rms = jax.lax.rsqrt(mean_sq + eps)
        x_hat = z * inv_rms
        dx_hat = dy * w
        c_m = jnp.sum(dx_hat * x_hat, axis=-1, keepdims=True) * inv_hidden
        dz = inv_rms * (dx_hat - x_hat * c_m)

        dh_t_ref[:, :] = dz.astype(h_t_2d.dtype)
        dh_0_ref[:, :] = (dz * s).astype(h_0_2d.dtype)

        dw_vec = jnp.sum(dy * x_hat, axis=0, keepdims=True)
        dg_vec = jnp.sum(dz * h0, axis=0, keepdims=True) * s * (1.0 - s)
        dweight_part_ref[:, :] = jnp.broadcast_to(dw_vec, (8, padded_h))
        dgate_part_ref[:, :] = jnp.broadcast_to(dg_vec, (8, padded_h))

    call_kwargs: dict[str, Any] = {}
    if interpret:
        call_kwargs["interpret"] = True
    dh_t_pad, dh_0_pad, dg_parts, dw_parts = pl.pallas_call(
        _bwd_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded_m, padded_h), h_t_2d.dtype),
            jax.ShapeDtypeStruct((padded_m, padded_h), h_0_2d.dtype),
            jax.ShapeDtypeStruct((grid_m * 8, padded_h), jnp.float32),
            jax.ShapeDtypeStruct((grid_m * 8, padded_h), jnp.float32),
        ),
        grid=(grid_m,),
        in_specs=[
            pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
            pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
            pl.BlockSpec((8, padded_h), lambda i: (0, 0)),
            pl.BlockSpec((8, padded_h), lambda i: (0, 0)),
            pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
            pl.BlockSpec((block_m, padded_h), lambda i: (i, 0)),
            pl.BlockSpec((8, padded_h), lambda i: (i, 0)),
            pl.BlockSpec((8, padded_h), lambda i: (i, 0)),
        ],
        **call_kwargs,
    )(h_t_in, h_0_in, g_2d, w_2d, dy_in)

    dh_t = dh_t_pad[:rows, :hidden]
    dh_0 = dh_0_pad[:rows, :hidden]
    dgate = jnp.sum(dg_parts.reshape(grid_m, 8, padded_h)[:, 0, :hidden], axis=0).astype(
        gate_1d.dtype
    )
    dweight = jnp.sum(dw_parts.reshape(grid_m, 8, padded_h)[:, 0, :hidden], axis=0).astype(
        weight_1d.dtype
    )
    return dh_t, dh_0, dgate, dweight


@_assume_constant_result
def _build_pallas_recurrent_norm_ops(
    *,
    rows: int,
    hidden_size: int,
    eps: float,
) -> tuple[str, str]:
    cache_key = (rows, hidden_size, float(eps))
    cached = _RECURRENT_NORM_OP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if probe_pallas_bridge() is None:
        raise RuntimeError("TorchTPU Pallas bridge is unavailable.")

    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]
    from jax.experimental import pallas as pl  # type: ignore[import-not-found]

    jax_array_cls = getattr(jax, "Array", Any)
    use_pallas_call = callable(getattr(pl, "pallas_call", None))

    def _rec_norm_fwd_jax(h_t: Any, h_0: Any, gate: Any, weight: Any) -> Any:
        if use_pallas_call and h_t.ndim >= 2:
            flat_h_t = h_t.reshape(-1, h_t.shape[-1])
            flat_h_0 = h_0.reshape(-1, h_0.shape[-1])
            y_2d = _pallas_recurrent_norm_fwd_2d(
                flat_h_t,
                flat_h_0,
                gate.reshape(-1),
                weight.reshape(-1),
                eps=float(eps),
            )
            return y_2d.reshape(h_t.shape)
        return _jax_recurrent_norm_fwd_ref(h_t, h_0, gate, weight, eps=float(eps))

    def _rec_norm_bwd_jax(
        h_t: Any, h_0: Any, gate: Any, weight: Any, dy: Any
    ) -> tuple[Any, Any, Any, Any]:
        if use_pallas_call and h_t.ndim >= 2:
            flat_h_t = h_t.reshape(-1, h_t.shape[-1])
            flat_h_0 = h_0.reshape(-1, h_0.shape[-1])
            flat_dy = dy.reshape(-1, dy.shape[-1])
            dh_t_2d, dh_0_2d, dgate, dweight = _pallas_recurrent_norm_bwd_2d(
                flat_h_t,
                flat_h_0,
                gate.reshape(-1),
                weight.reshape(-1),
                flat_dy,
                eps=float(eps),
            )
            return (
                dh_t_2d.reshape(h_t.shape),
                dh_0_2d.reshape(h_0.shape),
                dgate.reshape(gate.shape),
                dweight.reshape(weight.shape),
            )
        return _jax_recurrent_norm_bwd_ref(h_t, h_0, gate, weight, dy, eps=float(eps))

    sig = hashlib.sha1(repr(cache_key).encode()).hexdigest()[:12]
    fwd_name = f"rec_norm_fwd_{sig}"
    bwd_name = f"rec_norm_bwd_{sig}"
    fwd_qualname = f"trainlm_pallas::{fwd_name}"
    bwd_qualname = f"trainlm_pallas::{bwd_name}"

    def _fwd_fake(
        h_t: torch.Tensor,
        h_0: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        del h_0, gate, weight
        return torch.empty_like(h_t)

    def _bwd_fake(
        h_t: torch.Tensor,
        h_0: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del dy
        return (
            torch.empty_like(h_t),
            torch.empty_like(h_0),
            torch.empty_like(gate),
            torch.empty_like(weight),
        )

    fwd_op_def = register_pallas_jax_op(
        fwd_qualname,
        _rec_norm_fwd_jax,
        schema="(Tensor h_t, Tensor h_0, Tensor gate, Tensor weight) -> Tensor",
        fake_impl=_fwd_fake,
        jax_annotations={
            "h_t": jax_array_cls,
            "h_0": jax_array_cls,
            "gate": jax_array_cls,
            "weight": jax_array_cls,
            "return": jax_array_cls,
        },
    )
    register_pallas_jax_op(
        bwd_qualname,
        _rec_norm_bwd_jax,
        schema=(
            "(Tensor h_t, Tensor h_0, Tensor gate, Tensor weight, Tensor dy) "
            "-> (Tensor, Tensor, Tensor, Tensor)"
        ),
        fake_impl=_bwd_fake,
        jax_annotations={
            "h_t": jax_array_cls,
            "h_0": jax_array_cls,
            "gate": jax_array_cls,
            "weight": jax_array_cls,
            "dy": jax_array_cls,
            "return": tuple[jax_array_cls, jax_array_cls, jax_array_cls, jax_array_cls],
        },
    )

    def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        h_t, h_0, gate, weight = inputs
        ctx.save_for_backward(h_t, h_0, gate, weight)

    def _backward(ctx: Any, grad_y: torch.Tensor) -> tuple[Any, Any, Any, Any]:
        h_t, h_0, gate, weight = ctx.saved_tensors
        bwd_op = _resolve_pallas_op(bwd_name)
        dh_t, dh_0, dgate, dweight = bwd_op(
            h_t, h_0, gate, weight, grad_y.contiguous()
        )
        return (
            dh_t if ctx.needs_input_grad[0] else None,
            dh_0 if ctx.needs_input_grad[1] else None,
            dgate if ctx.needs_input_grad[2] else None,
            dweight if ctx.needs_input_grad[3] else None,
        )

    _register_pallas_autograd(
        fwd_qualname, fwd_op_def, _backward, setup_context=_setup_context
    )
    _RECURRENT_NORM_OP_CACHE[cache_key] = (fwd_name, bwd_name)
    return fwd_name, bwd_name


def fused_recurrent_inject_rmsnorm(
    h_t: torch.Tensor,
    h_0: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_pallas: bool = True,
) -> torch.Tensor:
    """Compute ``RMSNorm(h_t + sigmoid(gate) * h_0, weight, eps)``.

    Dispatches to the VMEM-fused Pallas kernel on TPU when enabled and
    available; otherwise uses the exact FP32-accumulated PyTorch reference.
    """
    if h_t.shape != h_0.shape:
        raise ValueError(f"h_t shape {tuple(h_t.shape)} must match h_0 shape {tuple(h_0.shape)}.")
    if h_t.ndim < 2:
        raise ValueError(f"Expected at least 2D tensors (..., hidden), got shape {tuple(h_t.shape)}.")

    hidden_size = int(h_t.shape[-1])
    rows = int(h_t.numel() // hidden_size)
    if (
        not use_pallas
        or h_t.device.type != "tpu"
        or gate.numel() != hidden_size
        or weight.numel() != hidden_size
        or not is_pallas_recurrent_norm_available()
    ):
        return _pytorch_recurrent_inject_rmsnorm(h_t, h_0, gate, weight, eps=eps)

    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    compiling = bool(callable(is_compiling) and is_compiling())
    try:
        fwd_name, _ = _build_pallas_recurrent_norm_ops(
            rows=rows,
            hidden_size=hidden_size,
            eps=float(eps),
        )
        fwd_op = _resolve_pallas_op(fwd_name)
        return fwd_op(
            h_t.contiguous(),
            h_0.contiguous(),
            gate.contiguous(),
            weight.contiguous(),
        )
    except Exception:
        if compiling:
            raise
        return _pytorch_recurrent_inject_rmsnorm(h_t, h_0, gate, weight, eps=eps)


__all__ = [
    "_jax_recurrent_norm_bwd_ref",
    "_jax_recurrent_norm_fwd_ref",
    "_pallas_recurrent_norm_bwd_2d",
    "_pallas_recurrent_norm_fwd_2d",
    "_pytorch_recurrent_inject_rmsnorm",
    "fused_recurrent_inject_rmsnorm",
    "is_pallas_recurrent_norm_available",
]
