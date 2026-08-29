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


def runtime():
    xm = FakeXlaModel()
    backend = XlaRuntime(
        device="cpu",
        precision="fp32",
        xm_module=xm,
        torch_xla_module=SimpleNamespace(__version__="2.9.0"),
    )
    return backend, xm


def test_xla_runtime_is_lazy_and_satisfies_backend_protocol():
    backend, _ = runtime()

    assert isinstance(backend, ExecutionBackend)
    assert backend.name == "pytorch-xla"
    assert backend.world_size == 1
    assert backend.rank == 0
    assert backend.diagnostics().values["torch_xla_version"] == "2.9.0"


def test_xla_runtime_uses_xla_step_and_barrier_hooks():
    backend, xm = runtime()
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
    backend, _ = runtime()

    assert backend.create_mesh(LogicalMesh({"data": 1}))
    with pytest.raises(ValueError, match="world size"):
        backend.create_mesh(LogicalMesh({"data": 2}))

    backend.load_state_dict({"backend": "pytorch-xla"})
    with pytest.raises(ValueError, match="runtime state"):
        backend.load_state_dict({"backend": "other"})
