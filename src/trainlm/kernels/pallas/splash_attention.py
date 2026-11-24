"""Native TPU Pallas Splash Attention bridged via torch_tpu's custom_jax_kernel.

Follows Google/Meta TorchTitan's verified TorchTPU Pallas integration:
wraps ``jax.experimental.pallas.ops.tpu.splash_attention`` forward and backward
kernels with ``torch_tpu._internal.pallas.custom_jax_kernel`` (or public
``torch.tpu.pallas.jax_op`` when exposed) + ``torch.library.custom_op``,
``register_fake``, and ``register_autograd`` so ``torch.compile(backend="tpu")``
traces both forward and backward passes with zero graph breaks.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

_SPLASH_OP_CACHE: dict[tuple[Any, ...], tuple[str, str]] = {}
_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None), "assume_constant_result", lambda fn: fn
)


def probe_pallas_bridge() -> Callable[..., Any] | None:
    """Return TorchTPU's ``custom_jax_kernel`` bridge if available."""
    try:
        import torch_tpu  # type: ignore[import-not-found]
    except Exception:
        torch_tpu = None  # type: ignore[assignment]

    if torch_tpu is not None:
        for mod in (
            getattr(getattr(torch_tpu, "_internal", None), "pallas", None),
            getattr(torch_tpu, "pallas", None),
        ):
            if mod is not None:
                fn = getattr(mod, "custom_jax_kernel", None)
                if callable(fn):
                    return fn

    tpu_mod = getattr(torch, "tpu", None)
    pallas_pub = getattr(tpu_mod, "pallas", None) if tpu_mod is not None else None
    if pallas_pub is not None:
        fn = getattr(pallas_pub, "custom_jax_kernel", None)
        if callable(fn):
            return fn
    return None


def _has_jax_splash() -> bool:
    try:
        from jax.experimental.pallas.ops.tpu.splash_attention import (  # type: ignore[import-not-found]
            splash_attention_kernel,
            splash_attention_mask,
        )
    except Exception:
        return False
    return splash_attention_kernel is not None and splash_attention_mask is not None


def is_pallas_splash_available() -> bool:
    """Return True when both TorchTPU's Pallas bridge and JAX Splash Attention exist."""
    return probe_pallas_bridge() is not None and _has_jax_splash()


def _resolve_block_size(seq_len: int, max_block: int = 512, min_block: int = 128) -> int:
    block = min(max_block, seq_len)
    while block >= min_block:
        if seq_len % block == 0:
            return block
        block //= 2
    return 0


@_assume_constant_result
def _build_splash_ops(
    *,
    seq_len: int,
    kv_seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    is_causal: bool,
    block_size: int,
    sliding_window: int | None,
) -> tuple[str, str]:
    cache_key = (seq_len, kv_seq_len, n_heads, n_kv_heads, is_causal, block_size, sliding_window)
    cached = _SPLASH_OP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    custom_jax_kernel = probe_pallas_bridge()
    if custom_jax_kernel is None:
        raise RuntimeError("TorchTPU Pallas custom_jax_kernel bridge is unavailable.")

    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]
    from jax.experimental.pallas.ops.tpu.splash_attention import (  # type: ignore[import-not-found]
        splash_attention_kernel,
        splash_attention_mask,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import (  # type: ignore[import-not-found]
        QKVLayout,
    )

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_size,
        block_kv=block_size,
        block_kv_compute=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        use_fused_bwd_kernel=True,
        q_layout=QKVLayout.HEAD_DIM_MINOR,
        k_layout=QKVLayout.HEAD_DIM_MINOR,
        v_layout=QKVLayout.HEAD_DIM_MINOR,
    )
    mask_shape = (seq_len, kv_seq_len)
    if sliding_window is not None:
        single_mask = splash_attention_mask.LocalMask(
            shape=mask_shape, window_size=(sliding_window, 0), offset=0
        )
    elif is_causal:
        single_mask = splash_attention_mask.CausalMask(shape=mask_shape)
    else:
        single_mask = splash_attention_mask.FullMask(shape=mask_shape)

    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(single_mask,) * n_heads)
    is_mqa = n_kv_heads == 1
    if is_mqa:
        splash_kernel = splash_attention_kernel.make_splash_mqa(
            mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
        )
    else:
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
        )

    fwd_mask_info = splash_kernel.fwd_mask_info
    dq_mask_info = splash_kernel.dq_mask_info
    dkv_mask_info = splash_kernel.dkv_mask_info
    mask_value = splash_kernel.kwargs.get("mask_value", splash_attention_kernel.DEFAULT_MASK_VALUE)
    mask_function = splash_kernel.kwargs.get("mask_function", None)
    soft_cap = splash_kernel.kwargs.get("attn_logits_soft_cap", None)
    res_ckpt = splash_kernel.kwargs.get("residual_checkpoint_name", None)
    interpret = splash_kernel.kwargs.get("interpret", False)

    @jax.jit
    def splash_fwd(q: Any, k: Any, v: Any) -> tuple[Any, Any]:
        scale = jnp.array(1.0 / math.sqrt(q.shape[-1]), dtype=q.dtype)
        q_scaled = q * scale

        def _single_fwd(q_b: Any, k_b: Any, v_b: Any) -> tuple[Any, Any]:
            out_b, (lse_b,) = splash_attention_kernel._splash_attention_forward(
                fwd_mask_info,
                q_b,
                k_b,
                v_b,
                segment_ids=None,
                sinks=None,
                mask_value=mask_value,
                is_mqa=is_mqa,
                block_sizes=block_sizes,
                residual_checkpoint_name=res_ckpt,
                save_residuals=True,
                mask_function=mask_function,
                attn_logits_soft_cap=soft_cap,
                interpret=interpret,
            )
            return out_b, lse_b

        return jax.vmap(_single_fwd)(q_scaled, k, v)

    @jax.jit
    def splash_bwd(q: Any, k: Any, v: Any, out: Any, lse: Any, grad_out: Any) -> tuple[Any, Any, Any]:
        scale = jnp.array(1.0 / math.sqrt(q.shape[-1]), dtype=q.dtype)
        q_scaled = q * scale

        def _single_bwd(
            q_b: Any, k_b: Any, v_b: Any, out_b: Any, lse_b: Any, g_b: Any
        ) -> tuple[Any, Any, Any]:
            res = (q_b, k_b, v_b, None, None, out_b, lse_b, dq_mask_info, dkv_mask_info)
            all_grads = splash_attention_kernel._splash_attention_bwd(
                False,
                mask_value,
                is_mqa,
                block_sizes,
                res_ckpt,
                mask_function,
                soft_cap,
                interpret,
                res,
                g_b,
            )
            return all_grads[3], all_grads[4], all_grads[5]

        def _scan_body(carry: Any, args: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            q_b, k_b, v_b, out_b, lse_b, g_b = args
            return carry, _single_bwd(q_b, k_b, v_b, out_b, lse_b, g_b)

        _, (dq_scaled, dk, dv) = jax.lax.scan(_scan_body, None, (q_scaled, k, v, out, lse, grad_out))
        return dq_scaled * scale, dk, dv

    digest = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
    fwd_name = f"trainlm_splash_fwd_{digest}"
    bwd_name = f"trainlm_splash_bwd_{digest}"

    fwd_wrapped = custom_jax_kernel(splash_fwd, name=fwd_name)
    bwd_wrapped = custom_jax_kernel(splash_bwd, name=bwd_name)

    fwd_op = torch.library.custom_op(
        f"trainlm_pallas::{fwd_name}",
        fwd_wrapped,
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor)",
    )
    fwd_op.register_fake(
        lambda q, k, v: (
            torch.empty_like(q),
            torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32, device=q.device),
        )
    )

    bwd_op = torch.library.custom_op(
        f"trainlm_pallas::{bwd_name}",
        bwd_wrapped,
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor grad_out) -> (Tensor, Tensor, Tensor)",
    )
    bwd_op.register_fake(
        lambda q, k, v, out, lse, grad_out: (
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
        )
    )

    def _setup_ctx(ctx: Any, inputs: Any, output: Any) -> None:
        q, k, v = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)

    def _backward(ctx: Any, grad_out: torch.Tensor, grad_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del grad_lse
        q, k, v, out, lse = ctx.saved_tensors
        return bwd_op(q, k, v, out, lse, grad_out)

    fwd_op.register_autograd(_backward, setup_context=_setup_ctx)
    _SPLASH_OP_CACHE[cache_key] = (fwd_name, bwd_name)
    return fwd_name, bwd_name


def pallas_splash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
    query_heads: int | None = None,
    key_value_heads: int | None = None,
    sliding_window: int | None = None,
    block_q: int = 512,
    block_k: int = 512,
) -> torch.Tensor:
    """Compute TPU Splash Attention in HF ``(B, H, S, D)`` -> ``(B, S, H, D)`` layout.

    Falls back to ``F.scaled_dot_product_attention`` on CPU, for sequences not
    divisible by a valid ``>= 128`` Pallas block, or when JAX Splash is absent.
    """
    del query_heads, key_value_heads
    n_heads = query.shape[1]
    n_kv_heads = key.shape[1]
    seq_len = query.shape[2]
    kv_seq_len = key.shape[2]
    head_dim = query.shape[3]
    block = _resolve_block_size(seq_len, max_block=max(128, min(block_q, block_k)))

    if (
        query.device.type != "tpu"
        or seq_len < 128
        or block == 0
        or kv_seq_len != seq_len
        or not is_pallas_splash_available()
    ):
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=causal,
            scale=scale,
            enable_gqa=(n_heads != n_kv_heads),
        )
        return out.transpose(1, 2).contiguous()

    fwd_name, _ = _build_splash_ops(
        seq_len=seq_len,
        kv_seq_len=kv_seq_len,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        is_causal=causal,
        block_size=block,
        sliding_window=sliding_window,
    )
    fwd_op = getattr(torch.ops.trainlm_pallas, fwd_name)

    q_in = query
    if scale is not None:
        default_scale = 1.0 / math.sqrt(head_dim)
        if abs(scale - default_scale) > 1e-6:
            q_in = q_in * (scale / default_scale)
    k_in = key.squeeze(1) if n_kv_heads == 1 else key
    v_in = value.squeeze(1) if n_kv_heads == 1 else value
    out, _ = fwd_op(q_in, k_in, v_in)
    if out.dtype != query.dtype:
        out = out.to(dtype=query.dtype)
    return out.transpose(1, 2).contiguous()
