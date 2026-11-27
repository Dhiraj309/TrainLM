"""Mocked CPU unit tests for multi-chip TorchTPU distributed runtime (v5e-8 / v6e-8)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from trainlm.config import LaughLMLoopConfig
from trainlm.model import LaughLMLoopForCausalLM
from trainlm.runtime import distributed_overlap
import trainlm.runtime.torchtpu as native_runtime


def _report(*, count: int = 8, status: str = "ready") -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "device_type": "tpu",
        "device_count_visible": count,
        "device_name": "TPU v6e",
        "topology": "2x4",
        "hbm_bytes_per_device": 32 * 1024**3,
        "hbm_bytes_per_chip": 32 * 1024**3,
        "devices_per_chip": 1,
        "allocation": "passed" if status == "ready" else "failed",
        "error_type": None,
        "torch_tpu_version": "0.2.0",
        "libtpu_version": "0.0.50",
    }


def test_torchtpu_multichip_launch_init_device_and_mesh_diagnostics(monkeypatch):
    events: list[tuple[object, ...]] = []

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")

    tpu_dist = SimpleNamespace(
        set_tpu_launch_env=lambda: events.append(("set_tpu_launch_env",)),
    )
    tpu_mod = SimpleNamespace(
        distributed=tpu_dist,
        set_device=lambda idx: events.append(("set_device", idx)),
        topology_aware_mesh=lambda: "Mesh(2x4, 'data')",
    )

    state = {"initialized": False}
    dist_mod = SimpleNamespace(
        ReduceOp=SimpleNamespace(SUM="SUM"),
        is_available=lambda: True,
        is_initialized=lambda: state["initialized"],
        init_process_group=lambda **kw: (
            state.__setitem__("initialized", True),
            events.append(("init_process_group", kw["backend"], kw["world_size"], kw["rank"])),
        ),
        get_world_size=lambda: 8,
        get_rank=lambda: 3,
        all_reduce=lambda tensor, op=None: (
            events.append(("all_reduce", op)),
            tensor.mul_(2.0),
        ),
        barrier=lambda: events.append(("barrier",)),
    )

    fake_torch = SimpleNamespace(
        tpu=tpu_mod,
        distributed=dist_mod,
        accelerator=SimpleNamespace(synchronize=lambda: events.append(("synchronize",))),
        device=lambda kind, idx=None: (kind, idx) if idx is not None else kind,
        manual_seed=lambda s: events.append(("seed", s)),
    )

    monkeypatch.setattr(native_runtime, "discover_tpu", lambda *a, **k: _report(count=8))
    monkeypatch.setattr(native_runtime, "torch", fake_torch)

    runtime = native_runtime.TorchTPURuntime(precision="bf16", compile_enabled=True)

    assert ("set_tpu_launch_env",) in events
    assert ("set_device", 3) in events
    assert events.index(("set_tpu_launch_env",)) < events.index(
        ("init_process_group", "tpu", 8, 3)
    )
    assert runtime.device == ("tpu", 3)
    assert runtime.world_size == 8
    assert runtime.rank == 3
    assert runtime.local_rank == 3
    assert runtime.is_distributed is True
    assert runtime.is_primary_process is False
    assert runtime.mean_reduces_gradients_before_clipping is True
    assert runtime.compile_stateless_optimizer_step is True

    diag = runtime.diagnostics()
    assert diag.world_size == 8
    assert diag.rank == 3
    assert diag.values["dist_backend"] == "tpu"
    assert diag.values["topology_mesh"] == "Mesh(2x4, 'data')"

    reduced = runtime.reduce_sum(torch.tensor(3.0))
    assert float(reduced.item()) == pytest.approx(6.0)
    runtime.barrier("ckpt")
    assert ("synchronize",) in events and ("barrier",) in events


def test_torchtpu_explicit_grad_all_reduce_and_resume_validation(monkeypatch):
    monkeypatch.setattr(
        native_runtime,
        "torch",
        SimpleNamespace(
            tpu=SimpleNamespace(set_rng_state=lambda _s: None),
            distributed=SimpleNamespace(
                ReduceOp=SimpleNamespace(SUM="SUM"),
                is_initialized=lambda: True,
                all_reduce=lambda t, op=None: t.mul_(8.0),
            ),
        ),
    )

    runtime = object.__new__(native_runtime.TorchTPURuntime)
    runtime._world_size = 8
    runtime._rank = 0
    runtime._local_rank = 0
    runtime._precision = "bf16"
    runtime._compile_enabled = True
    runtime._distributed_strategy = "ddp"

    # With chunked_linear training_view, DDP forward hooks are skipped -> explicit reduce runs.
    assert runtime.needs_explicit_grad_all_reduce(
        model=torch.nn.Linear(4, 4), has_training_view=True
    )
    grad = torch.ones(2, 2)
    runtime.all_reduce_gradients([grad])
    assert torch.allclose(grad, torch.ones(2, 2))  # *8 / 8 == 1

    runtime._distributed_strategy = "fsdp2"
    assert not runtime.needs_explicit_grad_all_reduce(
        model=torch.nn.Linear(4, 4), has_training_view=True
    )

    runtime.validate_state_dict(
        {
            "backend": "torchtpu",
            "world_size": 8,
            "precision": "bf16",
            "compile_enabled": True,
            "device_rng_state": b"rng",
        }
    )


def test_find_decoder_blocks_discovers_laughlm_loop_modules():
    cfg = LaughLMLoopConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_prelude_layers=2,
        num_core_layers=3,
        num_coda_layers=1,
        use_pallas_recurrent_norm=False,
        max_position_embeddings=32,
    )
    model = LaughLMLoopForCausalLM(cfg)
    blocks = distributed_overlap._find_decoder_blocks(model)
    assert len(blocks) == 6
