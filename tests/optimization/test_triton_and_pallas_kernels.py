from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from trainlm.kernels import (
    is_pallas_splash_available,
    is_triton_available,
    pallas_splash_attention,
    probe_pallas_bridge,
    triton_rms_norm,
    triton_swiglu,
)
from trainlm.kernels.triton import apply_triton_rms_norm, restore_triton_rms_norm
from trainlm.optimization import probe_torchtpu_kernels, probe_torchtpu_splash_kernel


class _DummyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int = 16, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.variance_epsilon)).to(hidden_states.dtype)


def test_triton_kernels_fallback_cleanly_on_cpu() -> None:
    x = torch.randn(2, 4, 16, requires_grad=True)
    w = torch.ones(16, requires_grad=True)
    out = triton_rms_norm(x, w, eps=1e-6)
    expected = _DummyRMSNorm(16)(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    g = torch.randn(2, 4, 16, requires_grad=True)
    u = torch.randn(2, 4, 16, requires_grad=True)
    sw = triton_swiglu(g, u)
    torch.testing.assert_close(sw, torch.nn.functional.silu(g) * u)

    assert isinstance(is_triton_available(), bool)


def test_pallas_splash_fallback_and_bridge_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from trainlm.kernels.pallas.splash_attention import probe_pallas_bridge_spec

    def _fake_custom_jax(fn, name=""):
        return fn

    def _fake_jax_op(qualname, fn, /):
        return fn

    fake_torch_tpu = SimpleNamespace(
        __version__="0.2.0",
        _internal=SimpleNamespace(pallas=SimpleNamespace(custom_jax_kernel=_fake_custom_jax)),
    )
    monkeypatch.setitem(sys.modules, "torch_tpu", fake_torch_tpu)

    assert probe_pallas_bridge() is _fake_custom_jax
    assert probe_pallas_bridge_spec() == ("custom_jax_kernel", _fake_custom_jax)
    inv = probe_torchtpu_kernels()
    assert inv.has_pallas_bridge is True
    assert isinstance(is_pallas_splash_available(), bool)

    monkeypatch.setitem(
        sys.modules,
        "torch_tpu",
        SimpleNamespace(__version__="0.2.0", libtpu=None, pathlib=None),
    )
    monkeypatch.setattr(
        torch,
        "tpu",
        SimpleNamespace(pallas=SimpleNamespace(jax_op=_fake_jax_op)),
        raising=False,
    )
    assert probe_pallas_bridge() is _fake_jax_op
    assert probe_pallas_bridge_spec() == ("jax_op", _fake_jax_op)

    monkeypatch.setattr(torch, "tpu", None, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch_tpu",
        SimpleNamespace(__version__="0.2.0"),
    )
    assert probe_pallas_bridge() is None
    assert probe_pallas_bridge_spec() is None

    q = torch.randn(1, 4, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    out = pallas_splash_attention(q, k, v, causal=True)
    assert out.shape == (1, 16, 4, 8)
    assert probe_torchtpu_splash_kernel() is None or callable(probe_torchtpu_splash_kernel())


def test_apply_triton_rms_norm_no_triton_returns_clean_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trainlm.kernels.triton.rmsnorm as triton_rmsnorm_mod

    monkeypatch.setattr(triton_rmsnorm_mod, "_HAS_TRITON", False)
    model = nn.Sequential(_DummyRMSNorm(16))
    res = apply_triton_rms_norm(model)
    assert res.applied is False
    assert res.patched_modules == 0
    assert restore_triton_rms_norm(model) == 0


def test_pallas_cross_entropy_casts_int64_targets_to_int32(monkeypatch) -> None:
    import types
    import torch
    from trainlm.kernels.pallas import cross_entropy as ce_mod

    captured: dict[str, object] = {}

    def fake_fwd_op(hidden, weight, targets):
        del hidden, weight
        captured["targets_dtype"] = targets.dtype
        captured["targets_values"] = targets.clone()
        return (
            torch.tensor([1.5], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.zeros((targets.numel(),), dtype=torch.float32),
        )

    fake_ns = types.SimpleNamespace(fake_ce_fwd=fake_fwd_op)
    monkeypatch.setattr(ce_mod, "_build_pallas_ce_ops", lambda **kwargs: ("fake_ce_fwd", "fake_ce_bwd"))
    monkeypatch.setattr(torch.ops, "trainlm_pallas", fake_ns, raising=False)

    hidden = torch.randn(8, 16, dtype=torch.bfloat16)
    weight = torch.randn(32, 16, dtype=torch.bfloat16)
    labels = torch.tensor([0, 1, 2, -100, 4, 5, 6, 7], dtype=torch.int64)

    mean_loss, mean_z = ce_mod.pallas_chunked_linear_cross_entropy(hidden, weight, labels)
    assert mean_loss.shape == ()
    assert mean_loss.item() == 1.5
    assert mean_z is None
    assert captured["targets_dtype"] == torch.int32
    assert torch.equal(captured["targets_values"], labels.to(torch.int32))


def test_pallas_splash_gqa_head_dim_and_unsupported_fallback(monkeypatch) -> None:
    import types
    import torch
    from trainlm.kernels.pallas import splash_attention as sa_mod

    class _FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return types.SimpleNamespace(type="tpu")

    monkeypatch.setattr(sa_mod, "is_pallas_splash_available", lambda: True)
    sa_mod._UNSUPPORTED_SPLASH_SPECS.clear()

    build_calls: list[dict[str, object]] = []
    op_inputs: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        if kwargs["n_heads"] == 8:
            raise ValueError("unsupported spec")
        return ("fake_splash_fwd", "fake_splash_bwd")

    def fake_splash_fwd(q, k, v):
        op_inputs.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
        return torch.zeros(q.shape, dtype=q.dtype), torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)

    monkeypatch.setattr(sa_mod, "_build_splash_ops", fake_build)
    monkeypatch.setattr(
        torch.ops,
        "trainlm_pallas",
        types.SimpleNamespace(fake_splash_fwd=fake_splash_fwd),
        raising=False,
    )

    # 1. head_dim=64 (<128 / not multiple of 128) must fall back to SDPA without calling _build_splash_ops.
    q64 = torch.randn(1, 9, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    k64 = torch.randn(1, 3, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    v64 = torch.randn(1, 3, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    out64 = sa_mod.pallas_splash_attention(q64, k64, v64)
    assert out64.shape == (1, 128, 9, 64)
    assert build_calls == []

    # 2. GQA with head_dim=128 passes native un-repeated KV heads [1, 2, 128, 128] to Splash.
    q128 = torch.randn(1, 4, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    k128 = torch.randn(1, 2, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    v128 = torch.randn(1, 2, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    out128 = sa_mod.pallas_splash_attention(q128, k128, v128)
    assert out128.shape == (1, 128, 4, 128)
    assert len(build_calls) == 1
    assert build_calls[0]["n_kv_heads"] == 2
    assert op_inputs == [((1, 4, 128, 128), (1, 2, 128, 128), (1, 2, 128, 128))]

    # 3. _build_splash_ops errors are cached in _UNSUPPORTED_SPLASH_SPECS and fall back to SDPA.
    q_bad = torch.randn(1, 8, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    k_bad = torch.randn(1, 4, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    v_bad = torch.randn(1, 4, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    out_bad_1 = sa_mod.pallas_splash_attention(q_bad, k_bad, v_bad)
    out_bad_2 = sa_mod.pallas_splash_attention(q_bad, k_bad, v_bad)
    assert out_bad_1.shape == (1, 128, 8, 128)
    assert out_bad_2.shape == (1, 128, 8, 128)
    assert len([c for c in build_calls if c["n_heads"] == 8]) == 1


def test_pallas_cross_entropy_autograd_handles_none_grad_z(monkeypatch) -> None:
    import sys
    import types
    import torch
    from trainlm.kernels.pallas import cross_entropy as ce_mod
    from trainlm.kernels.pallas import splash_attention as sa_mod

    ce_mod._CE_OP_CACHE.clear()
    seen_bwd_grads: list[tuple[torch.dtype, torch.dtype, tuple[int, ...], tuple[int, ...]]] = []

    def fake_custom_jax_kernel(fn, *, name: str):
        del fn
        if "_fwd_" in name:
            def _fwd(h, w, tgt):
                del w, tgt
                return (
                    h.sum().reshape(1).to(dtype=torch.float32),
                    torch.zeros((1,), dtype=torch.float32, device=h.device),
                    torch.zeros((h.shape[0],), dtype=torch.float32, device=h.device),
                )
            return _fwd

        def _bwd(h, w, tgt, lse, grad_loss, grad_z):
            del tgt, lse
            seen_bwd_grads.append(
                (grad_loss.dtype, grad_z.dtype, tuple(grad_loss.shape), tuple(grad_z.shape))
            )
            return torch.ones_like(h), torch.zeros_like(w)

        return _bwd

    fake_jax = types.ModuleType("jax")
    fake_jax.Array = object
    fake_jax.jit = lambda fn: fn
    fake_jnp = types.ModuleType("jax.numpy")
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(
        sa_mod,
        "probe_pallas_bridge_spec",
        lambda: ("custom_jax_kernel", fake_custom_jax_kernel),
    )

    hidden = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    loss, z_loss = ce_mod.pallas_chunked_linear_cross_entropy(hidden, weight, labels)
    assert loss.shape == ()
    assert z_loss is None
    loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None
    assert seen_bwd_grads == [(torch.float32, torch.float32, (1,), (1,))]


def test_pallas_jax_op_020_registration_annotations_and_autograd(monkeypatch) -> None:
    import inspect
    import sys
    import types
    import torch
    from trainlm.kernels.pallas import cross_entropy as ce_mod
    from trainlm.kernels.pallas import splash_attention as sa_mod

    ce_mod._CE_OP_CACHE.clear()
    registered_qualnames: list[str] = []
    seen_annotations: list[dict[str, object]] = []

    class _FakeJaxArray:
        pass

    def _forbid_jit(fn):
        raise AssertionError("jax_op requires plain Python functions without @jax.jit")

    def fake_jax_op(qualname: str, fn, /):
        assert inspect.isfunction(fn)
        seen_annotations.append(dict(fn.__annotations__))
        registered_qualnames.append(qualname)
        if "_fwd_" in qualname:
            def _fwd(h, w, tgt):
                del w, tgt
                return (
                    h.sum().reshape(1).to(dtype=torch.float32),
                    torch.zeros((1,), dtype=torch.float32, device=h.device),
                    torch.zeros((h.shape[0],), dtype=torch.float32, device=h.device),
                )

            op = torch.library.custom_op(
                f"{qualname}_test020",
                _fwd,
                mutates_args=(),
                schema="(Tensor h, Tensor w, Tensor tgt) -> (Tensor, Tensor, Tensor)",
            )
            op.register_fake(
                lambda h, w, tgt: (
                    torch.empty((1,), dtype=torch.float32, device=h.device),
                    torch.empty((1,), dtype=torch.float32, device=h.device),
                    torch.empty((h.shape[0],), dtype=torch.float32, device=h.device),
                )
            )
            return op

        def _bwd(h, w, tgt, lse, grad_loss, grad_z):
            del tgt, lse, grad_loss, grad_z
            return torch.ones_like(h), torch.zeros_like(w)

        op = torch.library.custom_op(
            f"{qualname}_test020",
            _bwd,
            mutates_args=(),
            schema="(Tensor h, Tensor w, Tensor tgt, Tensor lse, Tensor grad_loss, Tensor grad_z) -> (Tensor, Tensor)",
        )
        op.register_fake(
            lambda h, w, tgt, lse, grad_loss, grad_z: (
                torch.empty(h.shape, dtype=h.dtype, device=h.device),
                torch.empty(w.shape, dtype=w.dtype, device=w.device),
            )
        )
        return op

    fake_jax = types.ModuleType("jax")
    fake_jax.Array = _FakeJaxArray
    fake_jax.jit = _forbid_jit
    fake_jnp = types.ModuleType("jax.numpy")
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(
        sa_mod,
        "probe_pallas_bridge_spec",
        lambda: ("jax_op", fake_jax_op),
    )

    hidden = torch.randn(4, 12, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(20, 12, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([0, 1, 2, -100], dtype=torch.int64)

    loss, z_loss = ce_mod.pallas_chunked_linear_cross_entropy(hidden, weight, labels)
    assert loss.shape == ()
    assert z_loss is None
    assert len(registered_qualnames) == 2
    assert all(q.startswith("trainlm_pallas::trainlm_ce_") for q in registered_qualnames)
    assert seen_annotations[0] == {
        "h": _FakeJaxArray,
        "w": _FakeJaxArray,
        "tgt": _FakeJaxArray,
        "return": tuple[_FakeJaxArray, _FakeJaxArray, _FakeJaxArray],
    }
    assert seen_annotations[1] == {
        "h": _FakeJaxArray,
        "w": _FakeJaxArray,
        "tgt": _FakeJaxArray,
        "lse": _FakeJaxArray,
        "grad_loss": _FakeJaxArray,
        "grad_z": _FakeJaxArray,
        "return": tuple[_FakeJaxArray, _FakeJaxArray],
    }
    loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("z_loss_coef", [0.0, 0.05])
def test_pallas_cross_entropy_analytical_backward_matches_reference(
    monkeypatch: pytest.MonkeyPatch, has_bias: bool, z_loss_coef: float
) -> None:
    import sys
    import types
    import torch
    import torch.nn.functional as F
    from trainlm.kernels.pallas import cross_entropy as ce_mod
    from trainlm.kernels.pallas import splash_attention as sa_mod
    from trainlm.tasks.chunked_loss import chunked_linear_causal_cross_entropy

    class _JnpArray(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types_, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            out = super().__torch_function__(func, types_, args, kwargs)
            if isinstance(out, torch.Tensor) and not isinstance(out, cls):
                return out.as_subclass(cls)
            if isinstance(out, tuple):
                return tuple(
                    x.as_subclass(cls)
                    if isinstance(x, torch.Tensor) and not isinstance(x, cls)
                    else x
                    for x in out
                )
            return out

        def astype(self, dtype):
            return self.to(dtype=dtype)

        @property
        def at(self):
            outer = self

            class _AtIndexer:
                def __getitem__(self, idx):
                    class _AtOps:
                        def add(self, val):
                            cloned = outer.as_subclass(torch.Tensor).clone()
                            rows = _unwrap(idx[0]).to(torch.long)
                            cols = _unwrap(idx[1]).to(torch.long)
                            cloned[rows, cols] = cloned[rows, cols] + val
                            return cloned.as_subclass(_JnpArray)

                    return _AtOps()

            return _AtIndexer()

    def _to_jnp(x):
        return x.as_subclass(_JnpArray) if isinstance(x, torch.Tensor) else x

    def _unwrap(x):
        return x.as_subclass(torch.Tensor) if isinstance(x, _JnpArray) else x

    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.float32 = torch.float32
    fake_jnp.int32 = torch.int32
    fake_jnp.array = lambda v, dtype=None: torch.tensor(v, dtype=dtype).as_subclass(_JnpArray)
    fake_jnp.zeros = lambda shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype).as_subclass(_JnpArray)
    fake_jnp.arange = lambda n, dtype=torch.int32: torch.arange(n, dtype=dtype).as_subclass(_JnpArray)
    fake_jnp.reshape = lambda x, shape: _to_jnp(x).reshape(shape)
    fake_jnp.abs = lambda x: torch.abs(_to_jnp(x))
    fake_jnp.exp = lambda x: torch.exp(_to_jnp(x))
    fake_jnp.log = lambda x: torch.log(_to_jnp(x))
    fake_jnp.square = lambda x: torch.square(_to_jnp(x))
    fake_jnp.maximum = lambda a, b: torch.maximum(
        _to_jnp(a), torch.as_tensor(b, dtype=a.dtype, device=a.device).as_subclass(_JnpArray)
    )
    fake_jnp.max = lambda x, axis=-1, keepdims=False: torch.amax(_to_jnp(x), dim=axis, keepdim=keepdims)
    fake_jnp.sum = lambda x, axis=None: (
        torch.sum(_to_jnp(x)) if axis is None else torch.sum(_to_jnp(x), dim=axis)
    )
    fake_jnp.take_along_axis = lambda arr, idx, axis=-1: torch.take_along_dim(
        _to_jnp(arr), _unwrap(idx).to(torch.long), dim=axis
    ).as_subclass(_JnpArray)

    def _where(cond, a, b):
        ref = a if isinstance(a, torch.Tensor) else b
        dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float32
        dev = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")
        ta = a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=dtype, device=dev)
        tb = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=dtype, device=dev)
        return torch.where(_unwrap(cond), _unwrap(ta), _unwrap(tb)).as_subclass(_JnpArray)

    def _matmul(a, b, preferred_element_type=None):
        res = torch.matmul(_unwrap(a).float(), _unwrap(b).float())
        out_dtype = preferred_element_type if preferred_element_type is not None else a.dtype
        return res.to(dtype=out_dtype).as_subclass(_JnpArray)

    def _pad(x, pad_width, constant_values=0):
        tx = _unwrap(x)
        if tx.ndim == 2:
            return F.pad(tx, (pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]), value=constant_values).as_subclass(_JnpArray)
        return F.pad(tx, (pad_width[0], pad_width[1]), value=constant_values).as_subclass(_JnpArray)

    def _scan(fn, init, xs):
        carry = init
        ys = []
        steps = xs[0].shape[0]
        for idx in range(steps):
            step_in = tuple(t[idx] for t in xs)
            carry, y = fn(carry, step_in)
            ys.append(y)
        stacked = None if ys[0] is None else torch.stack([_unwrap(y) for y in ys], dim=0).as_subclass(_JnpArray)
        return carry, stacked

    fake_jnp.where = _where
    fake_jnp.matmul = _matmul
    fake_jnp.pad = _pad

    fake_jax = types.ModuleType("jax")
    fake_jax.Array = _JnpArray
    fake_jax.jit = lambda fn: fn
    fake_jax.lax = types.SimpleNamespace(scan=_scan)

    def _exec_custom_jax_kernel(fn, *, name: str):
        del name
        def _wrapped(*args):
            res = fn(*(_to_jnp(a) for a in args))
            if isinstance(res, tuple):
                return tuple(_unwrap(r) for r in res)
            return _unwrap(res)
        return _wrapped

    ce_mod._CE_OP_CACHE.clear()
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(
        sa_mod,
        "probe_pallas_bridge_spec",
        lambda: ("custom_jax_kernel", _exec_custom_jax_kernel),
    )

    torch.manual_seed(42)
    # Shape (2, 6, 8) -> 10 shifted tokens with chunk_size=4 forces pad_tokens=2.
    h_ref = torch.randn(2, 6, 8, dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(19, 8, dtype=torch.float32, requires_grad=True)
    b_ref = torch.randn(19, dtype=torch.float32, requires_grad=True) if has_bias else None
    labels = torch.tensor([[1, 2, -100, 4, 5, 6], [0, 3, 7, -100, 11, 12]], dtype=torch.long)

    ref_loss, ref_z = chunked_linear_causal_cross_entropy(
        h_ref, w_ref, labels, bias=b_ref, chunk_size=4, z_loss=z_loss_coef
    )
    ref_loss.backward()

    h_pal = h_ref.detach().clone().requires_grad_(True)
    w_pal = w_ref.detach().clone().requires_grad_(True)
    b_pal = b_ref.detach().clone().requires_grad_(True) if b_ref is not None else None

    shifted_h = h_pal[:, :-1, :].reshape(-1, 8)
    shifted_l = labels[:, 1:].reshape(-1)
    pal_loss, pal_z = ce_mod.pallas_chunked_linear_cross_entropy(
        shifted_h, w_pal, shifted_l, bias=b_pal, chunk_size=4, z_loss=z_loss_coef
    )
    pal_loss.backward()

    torch.testing.assert_close(pal_loss, ref_loss, rtol=1e-5, atol=1e-5)
    if z_loss_coef > 0.0:
        assert pal_z is not None and ref_z is not None
        torch.testing.assert_close(pal_z, ref_z, rtol=1e-5, atol=1e-5)
    else:
        assert pal_z is None
    torch.testing.assert_close(h_pal.grad, h_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w_pal.grad, w_ref.grad, rtol=1e-5, atol=1e-5)
    if has_bias:
        assert b_pal is not None and b_ref is not None
        torch.testing.assert_close(b_pal.grad, b_ref.grad, rtol=1e-5, atol=1e-5)


def test_pallas_runtime_health_fallback_and_splash_contiguous_output(monkeypatch) -> None:
    import sys
    import types
    import torch
    from trainlm.kernels.pallas import cross_entropy as ce_mod
    from trainlm.kernels.pallas import splash_attention as sa_mod

    sa_mod._DISABLED_PALLAS_BRIDGES.clear()
    sa_mod._VERIFIED_PALLAS_BRIDGES.clear()

    # 1. Failing TPU bridge is disabled cleanly by verify_pallas_bridge_runtime().
    fake_jax = types.ModuleType("jax")
    fake_jax.Array = object
    fake_jax.jit = lambda fn: fn
    fake_jnp = types.ModuleType("jax.numpy")
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(sa_mod, "_tpu_runtime_initialized", lambda: True)

    def _failing_jax_op(qualname: str, fn, /):
        del fn
        def _raise(*args, **kwargs):
            del args, kwargs
            raise RuntimeError(f"call_custom_kernel() failed for {qualname}")
        return _raise

    monkeypatch.setattr(
        torch,
        "tpu",
        types.SimpleNamespace(pallas=types.SimpleNamespace(jax_op=_failing_jax_op)),
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_tpu",
        types.SimpleNamespace(__version__="0.2.0"),
    )

    assert sa_mod.verify_pallas_bridge_runtime(compiled=False) is False
    assert "jax_op" in sa_mod._DISABLED_PALLAS_BRIDGES
    assert sa_mod.probe_pallas_bridge_spec() is None
    assert ce_mod.is_pallas_ce_available() is False

    # Reset probe state so subsequent tests keep a clean module global state.
    sa_mod._DISABLED_PALLAS_BRIDGES.clear()
    sa_mod._VERIFIED_PALLAS_BRIDGES.clear()

    # 2. Transposed (non-contiguous) Q/K/V through Splash fake_impl produces contiguous
    #    [B, S, H, D] output AND passes contiguous grad_out to Splash backward.
    class _FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return types.SimpleNamespace(type="tpu")

    monkeypatch.setattr(sa_mod, "_tpu_runtime_initialized", lambda: False)
    monkeypatch.setattr(sa_mod, "is_pallas_splash_available", lambda: True)
    sa_mod._SPLASH_OP_CACHE.clear()
    sa_mod._UNSUPPORTED_SPLASH_SPECS.clear()
    seen_grad_out_contiguous: list[bool] = []

    def fake_jax_op_splash(qualname: str, fn, /):
        del fn
        if "_fwd_" in qualname:
            def _fwd(q, k, v):
                del k, v
                assert q.is_contiguous()
                return (
                    torch.zeros(q.shape, dtype=q.dtype),
                    torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32),
                )

            op = torch.library.custom_op(
                qualname,
                _fwd,
                mutates_args=(),
                schema="(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor)",
            )
            op.register_fake(
                lambda q, k, v: (
                    torch.empty(q.shape, dtype=q.dtype),
                    torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32),
                )
            )
            return op

        def _bwd(q, k, v, out, lse, grad_out):
            del out, lse
            seen_grad_out_contiguous.append(bool(grad_out.is_contiguous()))
            return torch.zeros(q.shape, dtype=q.dtype), torch.zeros(k.shape, dtype=k.dtype), torch.zeros(v.shape, dtype=v.dtype)

        op = torch.library.custom_op(
            qualname,
            _bwd,
            mutates_args=(),
            schema="(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor grad_out) -> (Tensor, Tensor, Tensor)",
        )
        op.register_fake(
            lambda q, k, v, out, lse, grad_out: (
                torch.empty(q.shape, dtype=q.dtype),
                torch.empty(k.shape, dtype=k.dtype),
                torch.empty(v.shape, dtype=v.dtype),
            )
        )
        return op

    fwd_op = fake_jax_op_splash("trainlm_pallas::trainlm_splash_fwd_contigtest", None)
    bwd_op = fake_jax_op_splash("trainlm_pallas::trainlm_splash_bwd_contigtest", None)
    fwd_op.register_autograd(
        lambda ctx, grad_out, grad_lse: bwd_op(
            *ctx.saved_tensors, grad_out.contiguous()
        ),
        setup_context=lambda ctx, inputs, output: ctx.save_for_backward(*inputs, *output),
    )
    monkeypatch.setattr(
        sa_mod,
        "_iter_pallas_bridge_candidates",
        lambda: [("jax_op", fake_jax_op_splash)],
    )
    monkeypatch.setattr(
        sa_mod,
        "_build_splash_ops",
        lambda **kwargs: (
            "trainlm_splash_fwd_contigtest",
            "trainlm_splash_bwd_contigtest",
        ),
    )

    q_base = torch.randn(2, 128, 4, 128, requires_grad=True)
    k_base = torch.randn(2, 128, 2, 128, requires_grad=True)
    v_base = torch.randn(2, 128, 2, 128, requires_grad=True)
    q_t = q_base.transpose(1, 2).as_subclass(_FakeTPUTensor)
    k_t = k_base.transpose(1, 2).as_subclass(_FakeTPUTensor)
    v_t = v_base.transpose(1, 2).as_subclass(_FakeTPUTensor)
    assert not q_t.is_contiguous()

    out = sa_mod.pallas_splash_attention(q_t, k_t, v_t)
    assert out.shape == (2, 128, 4, 128)
    assert out.is_contiguous()
    reshaped = out.view(2, 128, -1)
    assert reshaped.shape == (2, 128, 512)
    reshaped.sum().backward()
    assert seen_grad_out_contiguous == [True]


def test_packed_gated_mlp_checks_swiglu_activation_spec() -> None:
    from trainlm.optimization.mlp import GatedMLPProjectionSpec, PackedGatedMLPProjection

    spec = GatedMLPProjectionSpec(
        prefix="mlp",
        activation="swiglu",
        input_size=8,
        intermediate_size=16,
        gate_weight_key="mlp.gate_proj.weight",
        up_weight_key="mlp.up_proj.weight",
        packed_weight_key="mlp.packed_gate_up.weight",
    )
    g = nn.Linear(8, 16, bias=False)
    u = nn.Linear(8, 16, bias=False)
    packed = PackedGatedMLPProjection.from_separate(spec, g, u, torch.nn.functional.silu)
    x = torch.randn(2, 4, 8)
    torch.testing.assert_close(packed(x), torch.nn.functional.silu(g(x)) * u(x))


def test_splash_bwd_uses_vmap_and_compiled_tpu_prefers_xla_chunked_loss(monkeypatch) -> None:
    import inspect
    import types

    from trainlm.kernels import pallas as pallas_pkg
    from trainlm.kernels.pallas import cross_entropy as ce_mod
    from trainlm.kernels.pallas import splash_attention as sa_mod
    from trainlm.tasks import chunked_loss as cl_mod

    # 1. splash_bwd source should vectorize over batch with jax.vmap, not jax.lax.scan.
    src = inspect.getsource(sa_mod._build_splash_ops)
    bwd_section = src[src.index("def splash_bwd") : src.index("fwd_qualname")]
    assert "jax.vmap(_single_bwd)" in bwd_section
    assert "jax.lax.scan" not in bwd_section

    # 2. Compiled TPU prefers XLA chunked_linear unless TRAINLM_ENABLE_PALLAS_CE=1.
    class _FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return types.SimpleNamespace(type="tpu")

    pallas_calls: list[bool] = []
    fake_ce = lambda *args, **kwargs: (pallas_calls.append(True) or (torch.tensor(0.0), None))
    monkeypatch.setattr(ce_mod, "is_pallas_ce_available", lambda: True)
    monkeypatch.setattr(pallas_pkg, "is_pallas_ce_available", lambda: True)
    monkeypatch.setattr(ce_mod, "pallas_chunked_linear_cross_entropy", fake_ce)
    monkeypatch.setattr(pallas_pkg, "pallas_chunked_linear_cross_entropy", fake_ce)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.delenv("TRAINLM_ENABLE_PALLAS_CE", raising=False)

    h = torch.randn(1, 4, 8).as_subclass(_FakeTPUTensor)
    w = torch.randn(16, 8).as_subclass(_FakeTPUTensor)
    labels = torch.tensor([[1, 2, 3, -100]], dtype=torch.long).as_subclass(_FakeTPUTensor)

    loss, _ = cl_mod.chunked_linear_causal_cross_entropy(h, w, labels, chunk_size=2)
    assert float(loss) > 0.0
    assert pallas_calls == []

    monkeypatch.setenv("TRAINLM_ENABLE_PALLAS_CE", "1")
    cl_mod.chunked_linear_causal_cross_entropy(h, w, labels, chunk_size=2)
    assert pallas_calls == [True]

    # 3. pad_head_dim=True pads head_dim=64 -> 128, rescales Q, and slices back to 64.
    monkeypatch.setattr(sa_mod, "is_pallas_splash_available", lambda: True)
    seen_shapes: list[tuple[tuple[int, ...], float]] = []

    def fake_fwd_pad(q, k, v):
        del k, v
        seen_shapes.append((tuple(q.shape), float(q.abs().mean().item())))
        return q.clone(), torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)

    monkeypatch.setattr(sa_mod, "_build_splash_ops", lambda **kwargs: ("fake_fwd_pad", "fake_bwd_pad"))
    monkeypatch.setattr(sa_mod, "_resolve_pallas_op", lambda name: fake_fwd_pad)

    q64 = torch.ones(1, 4, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    k64 = torch.ones(1, 2, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    v64 = torch.ones(1, 2, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    out64 = sa_mod.pallas_splash_attention(q64, k64, v64, pad_head_dim=True)
    assert out64.shape == (1, 128, 4, 64)
    assert seen_shapes[0][0] == (1, 4, 128, 128)
    # Only first 64 cols are non-zero sqrt(2), remaining 64 zero-padded -> mean = sqrt(2) / 2.
    assert abs(seen_shapes[0][1] - (2.0 ** 0.5) / 2.0) < 1e-4

