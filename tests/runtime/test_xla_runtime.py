from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim import SGD

from trainlm.runtime import ExecutionBackend, LogicalMesh, XlaRuntime


class FakeXlaModel:

    def __init__(self):
        self.optimizer_steps = 0
        self.mark_steps = 0
        self.barriers = []

    def xla_device(self):
        return torch.device("cpu")

    def xrt_world_size(self):
        return 1

    def get_ordinal(self):
        return 0

    def mark_step(self):
        self.mark_steps += 1

    def optimizer_step(self, optimizer, barrier=False):
        del barrier
        optimizer.step()
        self.optimizer_steps += 1

    def rendezvous(self, name):
        self.barriers.append(name)


class FakeSpmd:

    class Mesh:
        def __init__(self, device_ids, mesh_shape, axis_names):
            self.device_ids = device_ids
            self.mesh_shape = mesh_shape
            self.axis_names = axis_names

    class PartitionSpec:
        def __init__(self, *axes):
            self.axes = axes

    def __init__(self):
        self.shardings = []

    def mark_sharding(self, tensor, mesh, spec):
        self.shardings.append((tensor, mesh, spec))
        return tensor


class FakeXlaRuntimeModule:

    def __init__(self):
        self.cache_calls = []

    def initialize_cache(self, path, readonly=False):
        self.cache_calls.append((path, readonly))


def runtime(*, spmd=False):
    xm = FakeXlaModel()
    fake_spmd = FakeSpmd() if spmd else None
    backend = XlaRuntime(
        device="cpu",
        precision="fp32",
        xm_module=xm,
        torch_xla_module=SimpleNamespace(__version__="2.9.0"),
        spmd_module=fake_spmd,
    )
    return backend, xm, fake_spmd


def test_xla_runtime_is_lazy_and_satisfies_backend_protocol():
    backend, _, _ = runtime()

    assert isinstance(backend, ExecutionBackend)
    assert backend.name == "pytorch-xla"
    assert backend.world_size == 1
    assert backend.rank == 0
    assert backend.diagnostics().values["torch_xla_version"] == "2.9.0"


def test_xla_runtime_uses_xla_step_and_barrier_hooks():
    backend, xm, _ = runtime()
    model = nn.Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 2)).sum()

    backend.backward(loss)
    backend.optimizer_step(optimizer)
    backend.synchronize()
    backend.barrier("test")

    assert xm.optimizer_steps == 1
    assert xm.mark_steps == 1
    assert xm.barriers == ["test"]


def test_xla_runtime_validates_mesh_ownership_and_state():
    backend, _, _ = runtime(spmd=True)

    mesh = backend.create_mesh(LogicalMesh({"data": 1}))
    assert mesh.logical.axis_sizes == {"data": 1}
    with pytest.raises(ValueError, match="world size"):
        backend.create_mesh(LogicalMesh({"data": 2}))

    backend.load_state_dict({"backend": "pytorch-xla"})
    with pytest.raises(ValueError, match="runtime state"):
        backend.load_state_dict({"backend": "other"})


def test_xla_runtime_marks_replicated_parameters_and_data_batches():
    backend, _, spmd = runtime(spmd=True)
    mesh = backend.create_mesh(LogicalMesh({"data": 1}))
    model = nn.Linear(2, 1)
    batch = {"input_ids": torch.ones(2, 2)}

    backend.shard_model(model, mesh)
    backend.prepare_batch(batch)

    assert len(spmd.shardings) == 3
    parameter_spec = spmd.shardings[0][2]
    batch_spec = spmd.shardings[-1][2]
    assert parameter_spec.axes == ()
    assert batch_spec.axes == ("data", None)


def test_xla_runtime_initializes_persistent_cache_before_device_use(tmp_path):
    xm = FakeXlaModel()
    cache_runtime = FakeXlaRuntimeModule()
    backend = XlaRuntime(
        device="cpu",
        xm_module=xm,
        torch_xla_module=SimpleNamespace(__version__="2.9.0"),
        torch_xla_runtime_module=cache_runtime,
        cache_dir=tmp_path / "xla-cache",
        cache_readonly=True,
    )

    backend.initialize()

    assert cache_runtime.cache_calls == [(str(tmp_path / "xla-cache"), True)]
    assert backend.diagnostics().values["compilation_cache_initialized"] is True


def test_xla_runtime_rejects_dynamic_batch_shapes_and_accumulation():
    backend, _, _ = runtime()
    backend.configure_static_shapes(accumulation_steps=2)
    backend.prepare_batch({"input_ids": torch.ones(2, 4)})

    with pytest.raises(RuntimeError, match="recompilation guard"):
        backend.prepare_batch({"input_ids": torch.ones(1, 4)})
    with pytest.raises(RuntimeError, match="accumulation structure"):
        backend.configure_static_shapes(accumulation_steps=1)
