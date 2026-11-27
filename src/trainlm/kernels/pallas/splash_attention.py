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
_REGISTERED_PALLAS_OPS: dict[str, Any] = {}
_DISABLED_PALLAS_BRIDGES: set[str] = set()
_VERIFIED_PALLAS_BRIDGES: set[tuple[str, bool]] = set()
_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None), "assume_constant_result", lambda fn: fn
)


def _iter_pallas_bridge_candidates() -> list[tuple[str, Callable[..., Any]]]:
    """Discover candidate JAX-to-PyTorch bridge callables in preference order."""
    candidates: list[tuple[str, Callable[..., Any]]] = []
    seen_ids: set[int] = set()

    def _add(kind: str, fn: Any) -> None:
        if callable(fn) and id(fn) not in seen_ids:
            seen_ids.add(id(fn))
            candidates.append((kind, fn))

    try:
        import torch_tpu  # type: ignore[import-not-found]
    except Exception:
        torch_tpu = None  # type: ignore[assignment]

    tpu_mod = getattr(torch, "tpu", None)
    pallas_pub = getattr(tpu_mod, "pallas", None) if tpu_mod is not None else None
    if pallas_pub is not None:
        _add("jax_op", getattr(pallas_pub, "jax_op", None))

    if torch_tpu is not None:
        for mod in (
            getattr(torch_tpu, "pallas", None),
            getattr(getattr(torch_tpu, "_internal", None), "pallas", None),
        ):
            if mod is not None:
                _add("jax_op", getattr(mod, "jax_op", None))

        for mod in (
            getattr(getattr(torch_tpu, "_internal", None), "pallas", None),
            getattr(torch_tpu, "pallas", None),
        ):
            if mod is not None:
                _add("custom_jax_kernel", getattr(mod, "custom_jax_kernel", None))

    if pallas_pub is not None:
        _add("custom_jax_kernel", getattr(pallas_pub, "custom_jax_kernel", None))

    return candidates


def probe_pallas_bridge_spec() -> tuple[str, Callable[..., Any]] | None:
    """Return ``(bridge_kind, fn)`` for TorchTPU's active JAX-to-PyTorch bridge.

    Prefers TorchTPU >=0.2.0's public positional ``torch.tpu.pallas.jax_op``
    and falls back to ``custom_jax_kernel(fn, name=...)``, skipping any bridge
    kind that failed live TPU verification in ``_DISABLED_PALLAS_BRIDGES``.
    """
    for kind, fn in _iter_pallas_bridge_candidates():
        if kind not in _DISABLED_PALLAS_BRIDGES:
            return (kind, fn)
    return None


def probe_pallas_bridge() -> Callable[..., Any] | None:
    """Return TorchTPU's Pallas/JAX bridge callable if available."""
    spec = probe_pallas_bridge_spec()
    return None if spec is None else spec[1]


def register_pallas_jax_op(
    qualname: str,
    jax_fn: Callable[..., Any],
    *,
    schema: str,
    fake_impl: Callable[..., Any],
    jax_annotations: dict[str, Any] | None = None,
    bridge_spec: tuple[str, Callable[..., Any]] | None = None,
) -> Any:
    """Register ``jax_fn`` as a PyTorch custom op via the active TorchTPU bridge."""
    spec = bridge_spec if bridge_spec is not None else probe_pallas_bridge_spec()
    if spec is None:
        raise RuntimeError("TorchTPU Pallas bridge is unavailable.")
    kind, bridge_fn = spec
    _, _, op_name = qualname.partition("::")
    short_name = op_name or qualname

    if jax_annotations is not None:
        jax_fn.__annotations__ = dict(jax_annotations)

    if kind == "custom_jax_kernel":
        try:
            import jax  # type: ignore[import-not-found]

            jit_fn = getattr(jax, "jit", None)
            target_fn = jit_fn(jax_fn) if callable(jit_fn) else jax_fn
        except Exception:
            target_fn = jax_fn
        wrapped = bridge_fn(target_fn, name=short_name)
        op_def = torch.library.custom_op(
            qualname,
            wrapped,
            mutates_args=(),
            schema=schema,
        )
        op_def.register_fake(fake_impl)
    else:
        # TorchTPU 0.2.0+: positional torch.tpu.pallas.jax_op(qualname, jax_fn)
        # infers the PyTorch custom op schema from runtime jax.Array annotations
        # on a plain Python function (must not be pre-wrapped in @jax.jit).
        op_def = bridge_fn(qualname, jax_fn)

    _REGISTERED_PALLAS_OPS[short_name] = op_def
    return op_def


def _register_pallas_autograd(
    qualname: str,
    op_def: Any,
    backward: Callable[..., Any],
    *,
    setup_context: Callable[..., Any],
) -> None:
    reg_method = getattr(op_def, "register_autograd", None)
    if callable(reg_method):
        reg_method(backward, setup_context=setup_context)
        return
    lib_reg = getattr(torch.library, "register_autograd", None)
    if callable(lib_reg):
        lib_reg(qualname, backward, setup_context=setup_context)
        return
    raise RuntimeError(f"Unable to register autograd for Pallas op {qualname}.")


def _resolve_pallas_op(op_name: str) -> Any:
    ns = getattr(torch.ops, "trainlm_pallas", None)
    if ns is not None:
        candidate = getattr(ns, op_name, None)
        if callable(candidate):
            return candidate
    cached = _REGISTERED_PALLAS_OPS.get(op_name)
    if cached is not None:
        return cached
    raise RuntimeError(f"Registered Pallas op trainlm_pallas::{op_name} was not found.")


def _tpu_runtime_initialized() -> bool:
    tpu_mod = getattr(torch, "tpu", None)
    if tpu_mod is None:
        return False
    checker = getattr(tpu_mod, "is_initialized", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def verify_pallas_bridge_runtime(*, compiled: bool = False) -> bool:
    """Verify the active Pallas bridge lowers and executes cleanly on initialized TPU.

    On CPU/CUDA or before TPU initialization this is a fast non-blocking True once
    a bridge symbol is present. When TPU is initialized, runs a tiny cached 2-arg
    ``jax.lax.scan`` op (and optional ``torch.compile``) so incompatible
    ``torch_tpu``/JAX MLIR bridges fall back cleanly instead of crashing training.
    """
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if callable(is_compiling) and is_compiling():
        return probe_pallas_bridge_spec() is not None
    if not _tpu_runtime_initialized():
        return probe_pallas_bridge_spec() is not None

    try:
        import gc
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
    except Exception:
        return False

    jax_array_cls = getattr(jax, "Array", Any)

    for kind, fn in _iter_pallas_bridge_candidates():
        if kind in _DISABLED_PALLAS_BRIDGES:
            continue
        if (kind, compiled) in _VERIFIED_PALLAS_BRIDGES:
            return True
        probe_name = f"trainlm_probe_{kind}"
        qualname = f"trainlm_pallas::{probe_name}"
        x = None
        idx = None
        out = None
        try:
            op = _REGISTERED_PALLAS_OPS.get(probe_name)
            if op is None:
                def _probe_scan(x: Any, idx: Any) -> Any:
                    def _body(acc: Any, row: Any) -> tuple[Any, None]:
                        return acc + jnp.sum(row.astype(jnp.float32)), None

                    tot, _ = jax.lax.scan(_body, jnp.array(0.0, dtype=jnp.float32), x)
                    mask_sum = jnp.sum((idx >= 0).astype(jnp.float32))
                    return jnp.reshape(tot + mask_sum, (1,))

                op = register_pallas_jax_op(
                    qualname,
                    _probe_scan,
                    schema="(Tensor x, Tensor idx) -> Tensor",
                    fake_impl=lambda x, idx: torch.empty(
                        (1,), dtype=torch.float32, device=x.device
                    ),
                    jax_annotations={
                        "x": jax_array_cls,
                        "idx": jax_array_cls,
                        "return": jax_array_cls,
                    },
                    bridge_spec=(kind, fn),
                )
            caller = _resolve_pallas_op(probe_name)
            x = torch.ones((2, 4), dtype=torch.bfloat16, device="tpu")
            idx = torch.tensor([0, 1], dtype=torch.int32, device="tpu")
            out = caller(x, idx)
            if compiled:
                compiled_fn = torch.compile(
                    lambda a, b: caller(a, b), backend="tpu", dynamic=False
                )
                out = compiled_fn(x, idx)
            accel = getattr(torch, "accelerator", None)
            sync = getattr(accel, "synchronize", None)
            if callable(sync):
                sync()
            _ = float(out.detach().cpu().reshape(()).item())
            _VERIFIED_PALLAS_BRIDGES.add((kind, compiled))
            return True
        except Exception:
            del x, idx, out
            gc.collect()
            _DISABLED_PALLAS_BRIDGES.add(kind)
    return False


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
    """Return True when a working TorchTPU Pallas bridge and JAX Splash exist."""
    if not _has_jax_splash():
        return False
    return verify_pallas_bridge_runtime(compiled=False)


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

    if probe_pallas_bridge() is None:
        raise RuntimeError("TorchTPU Pallas bridge is unavailable.")

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

    jax_array_cls = getattr(jax, "Array", Any)

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
            return out_b, lse_b.astype(jnp.float32)

        return jax.vmap(_single_fwd)(q_scaled, k, v)

    def splash_bwd(
        q: Any, k: Any, v: Any, out: Any, lse: Any, grad_out: Any
    ) -> tuple[Any, Any, Any]:
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

        dq_scaled, dk, dv = jax.vmap(_single_bwd)(q_scaled, k, v, out, lse, grad_out)
        return dq_scaled * scale, dk, dv

    digest = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
    fwd_name = f"trainlm_splash_fwd_{digest}"
    bwd_name = f"trainlm_splash_bwd_{digest}"
    fwd_qualname = f"trainlm_pallas::{fwd_name}"
    bwd_qualname = f"trainlm_pallas::{bwd_name}"

    fwd_op = register_pallas_jax_op(
        fwd_qualname,
        splash_fwd,
        schema="(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor)",
        fake_impl=lambda q, k, v: (
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(
                q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32, device=q.device
            ),
        ),
        jax_annotations={
            "q": jax_array_cls,
            "k": jax_array_cls,
            "v": jax_array_cls,
            "return": tuple[jax_array_cls, jax_array_cls],
        },
    )
    bwd_op = register_pallas_jax_op(
        bwd_qualname,
        splash_bwd,
        schema="(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor grad_out) -> (Tensor, Tensor, Tensor)",
        fake_impl=lambda q, k, v, out, lse, grad_out: (
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(k.shape, dtype=k.dtype, device=k.device),
            torch.empty(v.shape, dtype=v.dtype, device=v.device),
        ),
        jax_annotations={
            "q": jax_array_cls,
            "k": jax_array_cls,
            "v": jax_array_cls,
            "out": jax_array_cls,
            "lse": jax_array_cls,
            "grad_out": jax_array_cls,
            "return": tuple[jax_array_cls, jax_array_cls, jax_array_cls],
        },
    )

    def _setup_ctx(ctx: Any, inputs: Any, output: Any) -> None:
        q, k, v = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)

    def _backward(
        ctx: Any, grad_out: torch.Tensor | None, grad_lse: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del grad_lse
        q, k, v, out, lse = ctx.saved_tensors
        safe_grad_out = (
            torch.zeros(out.shape, dtype=out.dtype, device=out.device)
            if grad_out is None
            else grad_out.contiguous()
        )
        bwd_caller = _resolve_pallas_op(bwd_name) if bwd_op is None else bwd_op
        return bwd_caller(q, k, v, out, lse, safe_grad_out)

    _register_pallas_autograd(fwd_qualname, fwd_op, _backward, setup_context=_setup_ctx)
    _SPLASH_OP_CACHE[cache_key] = (fwd_name, bwd_name)
    return fwd_name, bwd_name


_UNSUPPORTED_SPLASH_SPECS: set[tuple[int, int, int, int, int, bool, int, int | None]] = set()


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

    Supports native GQA (``n_heads % n_kv_heads == 0``) without KV head
    replication. Falls back to ``F.scaled_dot_product_attention`` on CPU, for
    sequences or head dimensions unsupported by Pallas Splash (such as
    ``head_dim % 128 != 0``), or when JAX Splash is absent.
    """
    del query_heads, key_value_heads
    n_heads = query.shape[1]
    n_kv_heads = key.shape[1]
    seq_len = query.shape[2]
    kv_seq_len = key.shape[2]
    head_dim = query.shape[3]
    block = _resolve_block_size(seq_len, max_block=max(128, min(block_q, block_k)))
    spec_key = (
        seq_len,
        kv_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal,
        block,
        sliding_window,
    )

    def _sdpa_fallback() -> torch.Tensor:
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=causal,
            scale=scale,
            enable_gqa=(n_heads != n_kv_heads),
        )
        return out.transpose(1, 2).contiguous()

    if (
        query.device.type != "tpu"
        or seq_len < 128
        or block == 0
        or kv_seq_len != seq_len
        or head_dim < 128
        or head_dim % 128 != 0
        or (n_kv_heads != 1 and n_heads % n_kv_heads != 0)
        or spec_key in _UNSUPPORTED_SPLASH_SPECS
        or not is_pallas_splash_available()
    ):
        return _sdpa_fallback()

    try:
        fwd_name, _ = _build_splash_ops(
            seq_len=seq_len,
            kv_seq_len=kv_seq_len,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            is_causal=causal,
            block_size=block,
            sliding_window=sliding_window,
        )
    except (RuntimeError, ValueError, TypeError, NotImplementedError):
        _UNSUPPORTED_SPLASH_SPECS.add(spec_key)
        return _sdpa_fallback()
    fwd_op = _resolve_pallas_op(fwd_name)

    q_in = query
    if scale is not None:
        default_scale = 1.0 / math.sqrt(head_dim)
        if abs(scale - default_scale) > 1e-6:
            q_in = q_in * (scale / default_scale)
    if n_kv_heads == 1:
        k_in = key.squeeze(1)
        v_in = value.squeeze(1)
    else:
        k_in = key
        v_in = value
    q_in = q_in.contiguous()
    k_in = k_in.contiguous()
    v_in = v_in.contiguous()

    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if callable(is_compiling) and is_compiling():
        out, _ = fwd_op(q_in, k_in, v_in)
    else:
        try:
            out, _ = fwd_op(q_in, k_in, v_in)
        except (RuntimeError, ValueError, TypeError, NotImplementedError):
            _UNSUPPORTED_SPLASH_SPECS.add(spec_key)
            return _sdpa_fallback()
    if out.dtype != query.dtype:
        out = out.to(dtype=query.dtype)
    return out.transpose(1, 2).contiguous()
