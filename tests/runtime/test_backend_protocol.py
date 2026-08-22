from pathlib import Path

import pytest

from trainlm.runtime import (
    BackendDiagnostics,
    ExecutionBackend,
    LogicalMesh,
    Runtime,
    TorchRuntime,
)


def test_default_runtime_satisfies_execution_backend_protocol():
    runtime = Runtime()

    assert isinstance(runtime, ExecutionBackend)
    assert isinstance(runtime, TorchRuntime)
    assert runtime.name == "pytorch"
    assert runtime.world_size == 1
    assert runtime.rank == 0
    assert runtime.local_rank == 0
    assert runtime.is_distributed is False
    assert runtime.is_primary_process is True


def test_logical_mesh_validates_size_and_runtime_ownership():
    runtime = Runtime()
    mesh = LogicalMesh({"data": 1, "tensor": 1})

    assert mesh.size == 1
    assert runtime.create_mesh(mesh) is mesh

    with pytest.raises(ValueError, match="world size"):
        runtime.create_mesh(LogicalMesh({"data": 2}))


def test_backend_diagnostics_use_portable_schema():
    diagnostics = Runtime().diagnostics()

    assert isinstance(diagnostics, BackendDiagnostics)
    assert diagnostics.backend == "pytorch"
    assert diagnostics.device_type == "cpu"
    assert diagnostics.precision == "fp32"
    assert diagnostics.values == {"compile_enabled": False}


def test_trainer_facing_packages_do_not_import_torch_xla():
    repository_root = Path(__file__).parents[2]
    trainer_facing = (
        repository_root / "src" / "trainlm" / "training",
        repository_root / "src" / "trainlm" / "runtime" / "base.py",
    )

    for path in trainer_facing:
        files = path.rglob("*.py") if path.is_dir() else (path,)
        for source_file in files:
            source = source_file.read_text(encoding="utf-8")
            assert "torch_xla" not in source

