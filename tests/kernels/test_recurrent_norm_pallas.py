"""Numerical parity and torch.compile tests for fused_recurrent_inject_rmsnorm."""

from __future__ import annotations

import pytest
import torch

from trainlm.kernels.pallas.recurrent_norm import (
    _jax_recurrent_norm_bwd_ref,
    _jax_recurrent_norm_fwd_ref,
    _pallas_recurrent_norm_bwd_2d,
    _pallas_recurrent_norm_fwd_2d,
    _pytorch_recurrent_inject_rmsnorm,
    fused_recurrent_inject_rmsnorm,
)


def _naive_reference(
    h_t: torch.Tensor,
    h_0: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    z = h_t.float() + torch.sigmoid(gate.float()) * h_0.float()
    var = z.pow(2).mean(dim=-1, keepdim=True)
    y = z * torch.rsqrt(var + eps) * weight.float()
    return y.to(dtype=h_t.dtype)


@pytest.mark.parametrize("hidden_size", [128, 256, 896, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_recurrent_inject_rmsnorm_forward_backward_parity(
    hidden_size: int, dtype: torch.dtype
):
    torch.manual_seed(0)
    bs, seq = 2, 16
    h_t = torch.randn(bs, seq, hidden_size, dtype=dtype, requires_grad=True)
    h_0 = torch.randn(bs, seq, hidden_size, dtype=dtype, requires_grad=True)
    gate = torch.randn(hidden_size, dtype=dtype, requires_grad=True)
    weight = (torch.ones(hidden_size) + 0.1 * torch.randn(hidden_size)).to(dtype=dtype)
    weight.requires_grad_(True)

    ref_h_t = h_t.detach().clone().requires_grad_(True)
    ref_h_0 = h_0.detach().clone().requires_grad_(True)
    ref_gate = gate.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)

    y = fused_recurrent_inject_rmsnorm(h_t, h_0, gate, weight, eps=1e-6, use_pallas=False)
    ref_y = _naive_reference(ref_h_t, ref_h_0, ref_gate, ref_weight, eps=1e-6)

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(y.float(), ref_y.float(), atol=atol, rtol=atol)

    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    ref_y.backward(grad_out)

    torch.testing.assert_close(h_t.grad.float(), ref_h_t.grad.float(), atol=atol, rtol=atol)
    torch.testing.assert_close(h_0.grad.float(), ref_h_0.grad.float(), atol=atol, rtol=atol)
    torch.testing.assert_close(gate.grad.float(), ref_gate.grad.float(), atol=atol, rtol=atol)
    torch.testing.assert_close(
        weight.grad.float(), ref_weight.grad.float(), atol=atol, rtol=atol
    )


def test_recurrent_inject_rmsnorm_torch_compile_eager_compatibility():
    torch.manual_seed(42)
    h_t = torch.randn(2, 32, 256, dtype=torch.bfloat16, requires_grad=True)
    h_0 = torch.randn(2, 32, 256, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn(256, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(256, dtype=torch.bfloat16, requires_grad=True)

    def step(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        out = fused_recurrent_inject_rmsnorm(a, b, g, w, eps=1e-6, use_pallas=True)
        return out.float().square().mean()

    compiled = torch.compile(step, backend="eager", fullgraph=True)
    loss = compiled(h_t, h_0, gate, weight)
    loss.backward()

    ref = _pytorch_recurrent_inject_rmsnorm(
        h_t.detach(), h_0.detach(), gate.detach(), weight.detach(), eps=1e-6
    )
    torch.testing.assert_close(loss.detach(), ref.float().square().mean(), atol=1e-3, rtol=1e-3)
    assert h_t.grad is not None and h_0.grad is not None
    assert gate.grad is not None and weight.grad is not None


def test_jax_pallas_forward_and_backward_math_parity_when_jax_installed():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(7)
    rows, hidden = 16, 128
    h_t_np = rng.normal(size=(rows, hidden)).astype(np.float32)
    h_0_np = rng.normal(size=(rows, hidden)).astype(np.float32)
    gate_np = rng.normal(size=(hidden,)).astype(np.float32)
    weight_np = (1.0 + 0.1 * rng.normal(size=(hidden,))).astype(np.float32)
    dy_np = rng.normal(size=(rows, hidden)).astype(np.float32)

    h_t_j = jnp.asarray(h_t_np)
    h_0_j = jnp.asarray(h_0_np)
    g_j = jnp.asarray(gate_np)
    w_j = jnp.asarray(weight_np)
    dy_j = jnp.asarray(dy_np)

    y_ref = _jax_recurrent_norm_fwd_ref(h_t_j, h_0_j, g_j, w_j, eps=1e-6)
    grads_ref = _jax_recurrent_norm_bwd_ref(h_t_j, h_0_j, g_j, w_j, dy_j, eps=1e-6)

    pt_h_t = torch.tensor(h_t_np, requires_grad=True)
    pt_h_0 = torch.tensor(h_0_np, requires_grad=True)
    pt_g = torch.tensor(gate_np, requires_grad=True)
    pt_w = torch.tensor(weight_np, requires_grad=True)
    pt_y = _pytorch_recurrent_inject_rmsnorm(pt_h_t, pt_h_0, pt_g, pt_w, eps=1e-6)
    pt_y.backward(torch.tensor(dy_np))

    np.testing.assert_allclose(np.asarray(y_ref), pt_y.detach().numpy(), atol=1e-5, rtol=1e-5)
    for g_jax, g_pt in zip(
        grads_ref, (pt_h_t.grad, pt_h_0.grad, pt_g.grad, pt_w.grad), strict=True
    ):
        np.testing.assert_allclose(np.asarray(g_jax), g_pt.detach().numpy(), atol=1e-4, rtol=1e-4)

    # Exercise Pallas interpret-mode path when supported by the installed JAX version.
    use_interpret = jax.default_backend() == "cpu"
    try:
        y_pal = _pallas_recurrent_norm_fwd_2d(
            h_t_j, h_0_j, g_j, w_j, eps=1e-6, interpret=use_interpret
        )
        bwd_pal = _pallas_recurrent_norm_bwd_2d(
            h_t_j, h_0_j, g_j, w_j, dy_j, eps=1e-6, interpret=use_interpret
        )
    except Exception as exc:
        if use_interpret:
            pytest.skip(f"Pallas interpret lowering unavailable on this JAX CPU build: {exc}")
        raise
    else:
        np.testing.assert_allclose(np.asarray(y_pal), np.asarray(y_ref), atol=1e-5, rtol=1e-5)
        for g_pal, g_ref in zip(bwd_pal, grads_ref, strict=True):
            np.testing.assert_allclose(np.asarray(g_pal), np.asarray(g_ref), atol=1e-4, rtol=1e-4)


def test_pallas_recurrent_norm_avoids_unsupported_scatter_primitive():
    import inspect
    from trainlm.kernels.pallas import recurrent_norm

    for fn in (
        recurrent_norm._pallas_recurrent_norm_fwd_2d,
        recurrent_norm._pallas_recurrent_norm_bwd_2d,
    ):
        src = inspect.getsource(fn)
        assert ".at[" not in src

