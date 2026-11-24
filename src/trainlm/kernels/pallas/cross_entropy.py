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

from .splash_attention import probe_pallas_bridge

_CE_OP_CACHE: dict[tuple[Any, ...], tuple[str, str]] = {}
_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None), "assume_constant_result", lambda fn: fn
)


def is_pallas_ce_available() -> bool:
    """Return True when TorchTPU's Pallas/JAX bridge and JAX are both importable."""
    if probe_pallas_bridge() is None:
        return False
    try:
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
    except Exception:
        return False
    return jax is not None and jnp is not None


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

    custom_jax_kernel = probe_pallas_bridge()
    if custom_jax_kernel is None:
        raise RuntimeError("TorchTPU Pallas custom_jax_kernel bridge is unavailable.")

    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]

    eff_chunk = min(chunk_size, num_tokens)
    pad_tokens = (-num_tokens) % eff_chunk
    num_chunks = (num_tokens + pad_tokens) // eff_chunk

    def _chunk_ce_terms(h_c: Any, w: Any, b: Any, tgt_c: Any) -> tuple[Any, Any]:
        logits = jnp.matmul(h_c, w.T, preferred_element_type=jnp.float32)
        if has_bias and b is not None:
            logits = logits + b.astype(jnp.float32)
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
        return loss_sum, z_sum

    @jax.jit
    def ce_fwd(h: Any, w: Any, b: Any, tgt: Any) -> tuple[Any, Any, Any]:
        if pad_tokens > 0:
            h_pad = jnp.pad(h, ((0, pad_tokens), (0, 0)))
            tgt_pad = jnp.pad(tgt, (0, pad_tokens), constant_values=ignore_index)
        else:
            h_pad = h
            tgt_pad = tgt
        h_chunks = h_pad.reshape(num_chunks, eff_chunk, hidden_size)
        tgt_chunks = tgt_pad.reshape(num_chunks, eff_chunk)
        denom = jnp.maximum(jnp.sum(tgt != ignore_index).astype(jnp.float32), 1.0)

        def _scan_body(carry: tuple[Any, Any], inputs: tuple[Any, Any]) -> tuple[tuple[Any, Any], None]:
            l_acc, z_acc = carry
            h_c, t_c = inputs
            l_c, z_c = _chunk_ce_terms(h_c, w, b, t_c)
            return (l_acc + l_c, z_acc + z_c), None

        (total_loss, total_z), _ = jax.lax.scan(
            _scan_body,
            (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
            (h_chunks, tgt_chunks),
        )
        mean_loss = total_loss / denom
        mean_z = total_z / denom
        if z_loss > 0.0:
            mean_loss = mean_loss + z_loss * mean_z
        return mean_loss, mean_z, denom

    @jax.jit
    def ce_bwd(h: Any, w: Any, b: Any, tgt: Any, grad_loss: Any, grad_z: Any) -> tuple[Any, Any, Any]:
        if pad_tokens > 0:
            h_pad = jnp.pad(h, ((0, pad_tokens), (0, 0)))
            tgt_pad = jnp.pad(tgt, (0, pad_tokens), constant_values=ignore_index)
        else:
            h_pad = h
            tgt_pad = tgt
        h_chunks = h_pad.reshape(num_chunks, eff_chunk, hidden_size)
        tgt_chunks = tgt_pad.reshape(num_chunks, eff_chunk)
        denom = jnp.maximum(jnp.sum(tgt != ignore_index).astype(jnp.float32), 1.0)
        eff_grad_loss = grad_loss.astype(jnp.float32) / denom
        eff_grad_z = (grad_z.astype(jnp.float32) + z_loss * grad_loss.astype(jnp.float32)) / denom

        def _chunk_loss_for_vjp(h_c: Any, w_in: Any, b_in: Any, t_c: Any) -> Any:
            l_c, z_c = _chunk_ce_terms(h_c, w_in, b_in, t_c)
            return eff_grad_loss * l_c + (eff_grad_z * z_c if z_loss > 0.0 else 0.0)

        def _bwd_body(carry: tuple[Any, Any], inputs: tuple[Any, Any]) -> tuple[tuple[Any, Any], Any]:
            dw_acc, db_acc = carry
            h_c, t_c = inputs
            if has_bias:
                grad_fn = jax.grad(lambda hh, ww, bb: _chunk_loss_for_vjp(hh, ww, bb, t_c), argnums=(0, 1, 2))
                dh_c, dw_c, db_c = grad_fn(h_c, w, b)
                db_acc = db_acc + db_c.astype(jnp.float32)
            else:
                grad_fn = jax.grad(lambda hh, ww: _chunk_loss_for_vjp(hh, ww, b, t_c), argnums=(0, 1))
                dh_c, dw_c = grad_fn(h_c, w)
            return (dw_acc + dw_c.astype(jnp.float32), db_acc), dh_c

        init_dw = jnp.zeros_like(w, dtype=jnp.float32)
        init_db = jnp.zeros((vocab_size,), dtype=jnp.float32) if has_bias else jnp.zeros((1,), dtype=jnp.float32)
        (dw_tot, db_tot), dh_chunks = jax.lax.scan(_bwd_body, (init_dw, init_db), (h_chunks, tgt_chunks))
        dh_flat = dh_chunks.reshape(-1, hidden_size)[:num_tokens].astype(h.dtype)
        return dh_flat, dw_tot.astype(w.dtype), (db_tot.astype(b.dtype) if has_bias and b is not None else b)

    digest = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
    fwd_name = f"trainlm_ce_fwd_{digest}"
    bwd_name = f"trainlm_ce_bwd_{digest}"

    fwd_op = torch.library.custom_op(
        f"trainlm_pallas::{fwd_name}",
        custom_jax_kernel(ce_fwd, name=fwd_name),
        mutates_args=(),
        schema="(Tensor h, Tensor w, Tensor b, Tensor tgt) -> (Tensor, Tensor, Tensor)",
    )
    fwd_op.register_fake(
        lambda h, w, b, tgt: (
            torch.empty((), dtype=torch.float32, device=h.device),
            torch.empty((), dtype=torch.float32, device=h.device),
            torch.empty((), dtype=torch.float32, device=h.device),
        )
    )

    bwd_op = torch.library.custom_op(
        f"trainlm_pallas::{bwd_name}",
        custom_jax_kernel(ce_bwd, name=bwd_name),
        mutates_args=(),
        schema="(Tensor h, Tensor w, Tensor b, Tensor tgt, Tensor grad_loss, Tensor grad_z) -> (Tensor, Tensor, Tensor)",
    )
    bwd_op.register_fake(
        lambda h, w, b, tgt, grad_loss, grad_z: (
            torch.empty_like(h),
            torch.empty_like(w),
            torch.empty_like(b),
        )
    )

    def _setup_ctx(ctx: Any, inputs: Any, output: Any) -> None:
        del output
        h, w, b, tgt = inputs
        ctx.save_for_backward(h, w, b, tgt)

    def _backward(
        ctx: Any, grad_loss: torch.Tensor, grad_z: torch.Tensor, grad_denom: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        del grad_denom
        h, w, b, tgt = ctx.saved_tensors
        dh, dw, db = bwd_op(h, w, b, tgt, grad_loss, grad_z)
        return dh, dw, (db if has_bias else None), None

    fwd_op.register_autograd(_backward, setup_context=_setup_ctx)
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
    fwd_op = getattr(torch.ops.trainlm_pallas, fwd_name)
    dummy_bias = bias if bias is not None else shifted_hidden.new_zeros((1,))
    targets = (
        shifted_labels
        if shifted_labels.dtype == torch.int32
        else shifted_labels.to(dtype=torch.int32)
    )
    mean_loss, mean_z, _ = fwd_op(shifted_hidden, output_weight, dummy_bias, targets)
    return mean_loss, (mean_z if z_loss else None)
