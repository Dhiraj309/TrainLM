"""TPU Pallas/JAX scan-based fused chunked linear cross-entropy kernel.

Bridges a JAX ``lax.scan`` forward+backward chunked cross-entropy program via
``torch_tpu._internal.pallas.custom_jax_kernel`` (or ``torch.tpu.pallas.jax_op``)
and ``torch.library.custom_op``, ensuring logits never materialize in PyTorch
autograd HBM and backward recomputes chunks sequentially inside a single fused
XLA program.
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

_CE_OP_CACHE: dict[tuple[Any, ...], tuple[str, str]] = {}
_PALLAS_CE_DISABLED = False
_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None), "assume_constant_result", lambda fn: fn
)


def is_pallas_ce_available() -> bool:
    """Return True when a working TorchTPU Pallas bridge and JAX exist."""
    if _PALLAS_CE_DISABLED:
        return False
    try:
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
    except Exception:
        return False
    if jax is None or jnp is None:
        return False
    return verify_pallas_bridge_runtime(compiled=False)


@_assume_constant_result
def _build_pallas_ce_ops(
    *,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    chunk_size: int,
    ignore_index: int,
    z_loss: float,
    has_bias: bool,
) -> tuple[str, str]:
    cache_key = (num_tokens, hidden_size, vocab_size, chunk_size, ignore_index, z_loss, has_bias)
    cached = _CE_OP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if probe_pallas_bridge() is None:
        raise RuntimeError("TorchTPU Pallas bridge is unavailable.")

    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]

    jax_array_cls = getattr(jax, "Array", Any)
    eff_chunk = min(chunk_size, num_tokens)
    pad_tokens = (-num_tokens) % eff_chunk
    num_chunks = (num_tokens + pad_tokens) // eff_chunk

    def _chunk_fwd_terms(h_c: Any, w_c: Any, b_f32: Any, tgt_c: Any) -> tuple[Any, Any, Any]:
        logits = jnp.matmul(h_c, w_c.T, preferred_element_type=jnp.float32)
        if b_f32 is not None:
            logits = logits + b_f32
        logits_fp32 = logits.astype(jnp.float32)
        max_l = jnp.max(logits_fp32, axis=-1, keepdims=True)
        shifted = logits_fp32 - max_l
        logsumexp = max_l[..., 0] + jnp.log(jnp.sum(jnp.exp(shifted), axis=-1))

        active = tgt_c != ignore_index
        safe_tgt = jnp.where(active, tgt_c, 0)
        target_logits = jnp.take_along_axis(logits_fp32, safe_tgt[:, None], axis=-1)[:, 0]
        nll = jnp.where(active, logsumexp - target_logits, 0.0)
        loss_sum = jnp.sum(nll)
        if z_loss > 0.0:
            z_sum = jnp.sum(jnp.where(active, jnp.square(logsumexp), 0.0))
        else:
            z_sum = jnp.array(0.0, dtype=jnp.float32)
        return loss_sum, z_sum, logsumexp.astype(jnp.float32)

    def _run_fwd(h: Any, w: Any, b: Any, tgt: Any) -> tuple[Any, Any, Any]:
        if pad_tokens > 0:
            h_pad = jnp.pad(h, ((0, pad_tokens), (0, 0)))
            tgt_pad = jnp.pad(tgt, (0, pad_tokens), constant_values=ignore_index)
        else:
            h_pad = h
            tgt_pad = tgt
        h_chunks = h_pad.reshape(num_chunks, eff_chunk, hidden_size)
        tgt_chunks = tgt_pad.reshape(num_chunks, eff_chunk)
        w_c = w.astype(h.dtype)
        b_f32 = b.astype(jnp.float32) if b is not None else None
        denom = jnp.maximum(jnp.sum(tgt != ignore_index).astype(jnp.float32), 1.0)

        def _scan_body(carry: tuple[Any, Any], inputs: tuple[Any, Any]) -> tuple[tuple[Any, Any], Any]:
            l_acc, z_acc = carry
            h_c, t_c = inputs
            l_c, z_c, lse_c = _chunk_fwd_terms(h_c, w_c, b_f32, t_c)
            return (l_acc + l_c, z_acc + z_c), lse_c

        (total_loss, total_z), lse_chunks = jax.lax.scan(
            _scan_body,
            (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
            (h_chunks, tgt_chunks),
        )
        mean_loss = total_loss / denom
        mean_z = total_z / denom
        if z_loss > 0.0:
            mean_loss = mean_loss + z_loss * mean_z
        lse_out = lse_chunks.reshape(-1)[:num_tokens]
        return jnp.reshape(mean_loss, (1,)), jnp.reshape(mean_z, (1,)), lse_out

    def _run_bwd(
        h: Any, w: Any, b: Any, tgt: Any, lse: Any, grad_loss: Any, grad_z: Any
    ) -> tuple[Any, Any, Any]:
        if pad_tokens > 0:
            h_pad = jnp.pad(h, ((0, pad_tokens), (0, 0)))
            tgt_pad = jnp.pad(tgt, (0, pad_tokens), constant_values=ignore_index)
            lse_pad = jnp.pad(lse, (0, pad_tokens), constant_values=0.0)
        else:
            h_pad = h
            tgt_pad = tgt
            lse_pad = lse
        h_chunks = h_pad.reshape(num_chunks, eff_chunk, hidden_size)
        tgt_chunks = tgt_pad.reshape(num_chunks, eff_chunk)
        lse_chunks = lse_pad.reshape(num_chunks, eff_chunk).astype(jnp.float32)

        w_c = w.astype(h.dtype)
        b_f32 = b.astype(jnp.float32) if b is not None else None
        denom = jnp.maximum(jnp.sum(tgt != ignore_index).astype(jnp.float32), 1.0)
        g_loss = jnp.reshape(grad_loss, ()).astype(jnp.float32)
        g_z = jnp.reshape(grad_z, ()).astype(jnp.float32)

        s_loss = g_loss / denom
        s_z = (
            (g_z + z_loss * g_loss) / denom
            if z_loss > 0.0
            else jnp.array(0.0, dtype=jnp.float32)
        )
        # Keep per-token logit cotangents at O(1) dynamic range before casting to
        # bfloat16 so both backward vocabulary GEMMs run on the 197 TFLOP/s BF16 MXU
        # without 1/denom mantissa underflow, then apply scale_out in FP32 afterward.
        scale_out = jnp.abs(s_loss) + jnp.abs(s_z)
        safe_scale = jnp.where(scale_out > 0.0, scale_out, 1.0)
        w_loss = jnp.where(scale_out > 0.0, s_loss / safe_scale, 0.0)
        w_z = jnp.where(scale_out > 0.0, s_z / safe_scale, 0.0)
        vocab_ids = jnp.arange(vocab_size, dtype=jnp.int32)[None, :]

        def _bwd_body(
            carry: tuple[Any, Any], inputs: tuple[Any, Any, Any]
        ) -> tuple[tuple[Any, Any], Any]:
            dw_acc, db_acc = carry
            h_c, t_c, lse_c = inputs
            logits = jnp.matmul(h_c, w_c.T, preferred_element_type=jnp.float32)
            if b_f32 is not None:
                logits = logits + b_f32
            probs = jnp.exp(logits.astype(jnp.float32) - lse_c[:, None])
            active = t_c != ignore_index
            safe_tgt = jnp.where(active, t_c, 0)
            is_target = vocab_ids == safe_tgt[:, None]
            ce_grad = jnp.where(is_target, probs - 1.0, probs)
            if z_loss > 0.0:
                u = w_loss * ce_grad + w_z * (2.0 * lse_c[:, None] * probs)
            else:
                u = w_loss * ce_grad
            u = jnp.where(active[:, None], u, 0.0)
            if b is not None:
                db_acc = db_acc + jnp.sum(u, axis=0)
            u_compute = u.astype(h.dtype)
            dh_c = (
                jnp.matmul(u_compute, w_c, preferred_element_type=jnp.float32) * scale_out
            ).astype(h.dtype)
            dw_c = jnp.matmul(u_compute.T, h_c, preferred_element_type=jnp.float32)
            return (dw_acc + dw_c, db_acc), dh_c

        init_dw = jnp.zeros((vocab_size, hidden_size), dtype=jnp.float32)
        init_db = (
            jnp.zeros((vocab_size,), dtype=jnp.float32)
            if b is not None
            else jnp.zeros((1,), dtype=jnp.float32)
        )
        (dw_tot, db_tot), dh_chunks = jax.lax.scan(
            _bwd_body, (init_dw, init_db), (h_chunks, tgt_chunks, lse_chunks)
        )
        dh_flat = dh_chunks.reshape(-1, hidden_size)[:num_tokens].astype(h.dtype)
        dw_out = (dw_tot * scale_out).astype(w.dtype)
        db_out = (db_tot * scale_out).astype(b.dtype) if b is not None else None
        return dh_flat, dw_out, db_out

    digest = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
    fwd_name = f"trainlm_ce_fwd_{digest}"
    bwd_name = f"trainlm_ce_bwd_{digest}"
    fwd_qualname = f"trainlm_pallas::{fwd_name}"
    bwd_qualname = f"trainlm_pallas::{bwd_name}"

    if has_bias:
        def ce_fwd_bias(h: Any, w: Any, b: Any, tgt: Any) -> tuple[Any, Any, Any]:
            return _run_fwd(h, w, b, tgt)

        def ce_bwd_bias(
            h: Any, w: Any, b: Any, tgt: Any, lse: Any, grad_loss: Any, grad_z: Any
        ) -> tuple[Any, Any, Any]:
            dh, dw, db = _run_bwd(h, w, b, tgt, lse, grad_loss, grad_z)
            assert db is not None
            return dh, dw, db

        fwd_op = register_pallas_jax_op(
            fwd_qualname,
            ce_fwd_bias,
            schema="(Tensor h, Tensor w, Tensor b, Tensor tgt) -> (Tensor, Tensor, Tensor)",
            fake_impl=lambda h, w, b, tgt: (
                torch.empty((1,), dtype=torch.float32, device=h.device),
                torch.empty((1,), dtype=torch.float32, device=h.device),
                torch.empty((h.shape[0],), dtype=torch.float32, device=h.device),
            ),
            jax_annotations={
                "h": jax_array_cls,
                "w": jax_array_cls,
                "b": jax_array_cls,
                "tgt": jax_array_cls,
                "return": tuple[jax_array_cls, jax_array_cls, jax_array_cls],
            },
        )
        bwd_op = register_pallas_jax_op(
            bwd_qualname,
            ce_bwd_bias,
            schema="(Tensor h, Tensor w, Tensor b, Tensor tgt, Tensor lse, Tensor grad_loss, Tensor grad_z) -> (Tensor, Tensor, Tensor)",
            fake_impl=lambda h, w, b, tgt, lse, grad_loss, grad_z: (
                torch.empty(h.shape, dtype=h.dtype, device=h.device),
                torch.empty(w.shape, dtype=w.dtype, device=w.device),
                torch.empty(b.shape, dtype=b.dtype, device=b.device),
            ),
            jax_annotations={
                "h": jax_array_cls,
                "w": jax_array_cls,
                "b": jax_array_cls,
                "tgt": jax_array_cls,
                "lse": jax_array_cls,
                "grad_loss": jax_array_cls,
                "grad_z": jax_array_cls,
                "return": tuple[jax_array_cls, jax_array_cls, jax_array_cls],
            },
        )

        def _setup_ctx_bias(ctx: Any, inputs: Any, output: Any) -> None:
            h, w, b, tgt = inputs
            _, _, lse = output
            ctx.save_for_backward(h, w, b, tgt, lse)

        def _backward_bias(
            ctx: Any,
            grad_loss: torch.Tensor | None,
            grad_z: torch.Tensor | None,
            grad_lse: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
            del grad_lse
            h, w, b, tgt, lse = ctx.saved_tensors
            safe_grad_loss = (
                torch.zeros((1,), dtype=torch.float32, device=h.device)
                if grad_loss is None
                else grad_loss.reshape(1).to(dtype=torch.float32).contiguous()
            )
            safe_grad_z = (
                torch.zeros((1,), dtype=torch.float32, device=h.device)
                if grad_z is None
                else grad_z.reshape(1).to(dtype=torch.float32).contiguous()
            )
            bwd_caller = _resolve_pallas_op(bwd_name) if bwd_op is None else bwd_op
            dh, dw, db = bwd_caller(h, w, b, tgt, lse, safe_grad_loss, safe_grad_z)
            return dh, dw, db, None

        _register_pallas_autograd(
            fwd_qualname, fwd_op, _backward_bias, setup_context=_setup_ctx_bias
        )
    else:
        def ce_fwd_nobias(h: Any, w: Any, tgt: Any) -> tuple[Any, Any, Any]:
            return _run_fwd(h, w, None, tgt)

        def ce_bwd_nobias(
            h: Any, w: Any, tgt: Any, lse: Any, grad_loss: Any, grad_z: Any
        ) -> tuple[Any, Any]:
            dh, dw, _ = _run_bwd(h, w, None, tgt, lse, grad_loss, grad_z)
            return dh, dw

        fwd_op = register_pallas_jax_op(
            fwd_qualname,
            ce_fwd_nobias,
            schema="(Tensor h, Tensor w, Tensor tgt) -> (Tensor, Tensor, Tensor)",
            fake_impl=lambda h, w, tgt: (
                torch.empty((1,), dtype=torch.float32, device=h.device),
                torch.empty((1,), dtype=torch.float32, device=h.device),
                torch.empty((h.shape[0],), dtype=torch.float32, device=h.device),
            ),
            jax_annotations={
                "h": jax_array_cls,
                "w": jax_array_cls,
                "tgt": jax_array_cls,
                "return": tuple[jax_array_cls, jax_array_cls, jax_array_cls],
            },
        )
        bwd_op = register_pallas_jax_op(
            bwd_qualname,
            ce_bwd_nobias,
            schema="(Tensor h, Tensor w, Tensor tgt, Tensor lse, Tensor grad_loss, Tensor grad_z) -> (Tensor, Tensor)",
            fake_impl=lambda h, w, tgt, lse, grad_loss, grad_z: (
                torch.empty(h.shape, dtype=h.dtype, device=h.device),
                torch.empty(w.shape, dtype=w.dtype, device=w.device),
            ),
            jax_annotations={
                "h": jax_array_cls,
                "w": jax_array_cls,
                "tgt": jax_array_cls,
                "lse": jax_array_cls,
                "grad_loss": jax_array_cls,
                "grad_z": jax_array_cls,
                "return": tuple[jax_array_cls, jax_array_cls],
            },
        )

        def _setup_ctx_nobias(ctx: Any, inputs: Any, output: Any) -> None:
            h, w, tgt = inputs
            _, _, lse = output
            ctx.save_for_backward(h, w, tgt, lse)

        def _backward_nobias(
            ctx: Any,
            grad_loss: torch.Tensor | None,
            grad_z: torch.Tensor | None,
            grad_lse: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, None]:
            del grad_lse
            h, w, tgt, lse = ctx.saved_tensors
            safe_grad_loss = (
                torch.zeros((1,), dtype=torch.float32, device=h.device)
                if grad_loss is None
                else grad_loss.reshape(1).to(dtype=torch.float32).contiguous()
            )
            safe_grad_z = (
                torch.zeros((1,), dtype=torch.float32, device=h.device)
                if grad_z is None
                else grad_z.reshape(1).to(dtype=torch.float32).contiguous()
            )
            bwd_caller = _resolve_pallas_op(bwd_name) if bwd_op is None else bwd_op
            dh, dw = bwd_caller(h, w, tgt, lse, safe_grad_loss, safe_grad_z)
            return dh, dw, None

        _register_pallas_autograd(
            fwd_qualname, fwd_op, _backward_nobias, setup_context=_setup_ctx_nobias
        )

    _CE_OP_CACHE[cache_key] = (fwd_name, bwd_name)
    return fwd_name, bwd_name


def pallas_chunked_linear_cross_entropy(
    shifted_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    shifted_labels: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    z_loss: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute Pallas/JAX scan chunked linear cross-entropy on TPU tensors."""
    fwd_name, _ = _build_pallas_ce_ops(
        num_tokens=int(shifted_hidden.shape[0]),
        hidden_size=int(shifted_hidden.shape[1]),
        vocab_size=int(output_weight.shape[0]),
        chunk_size=int(chunk_size),
        ignore_index=int(ignore_index),
        z_loss=float(z_loss),
        has_bias=bias is not None,
    )
    fwd_op = _resolve_pallas_op(fwd_name)
    targets = (
        shifted_labels
        if shifted_labels.dtype == torch.int32
        else shifted_labels.to(dtype=torch.int32)
    )
    if bias is not None:
        mean_loss_1d, mean_z_1d, _ = fwd_op(
            shifted_hidden.contiguous(),
            output_weight.contiguous(),
            bias.contiguous(),
            targets.contiguous(),
        )
    else:
        mean_loss_1d, mean_z_1d, _ = fwd_op(
            shifted_hidden.contiguous(),
            output_weight.contiguous(),
            targets.contiguous(),
        )
    return mean_loss_1d.reshape(()), (mean_z_1d.reshape(()) if z_loss else None)
