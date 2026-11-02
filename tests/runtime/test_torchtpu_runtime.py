from types import SimpleNamespace

import pytest

import trainlm.runtime.torchtpu as native_runtime
from trainlm import TrainLMTrainingArguments


def _report(*, status="ready", count=1, error_type=None):
    return {
        "status": status,
        "device_count_visible": count,
        "error_type": error_type,
        "torch_tpu_version": "0.2.0",
        "libtpu_version": "0.0.50",
    }


def test_native_tpu_is_an_explicit_training_argument():
    args = TrainLMTrainingArguments(accelerator="native_tpu", max_steps=1)
    assert args.accelerator == "native_tpu"


def test_native_runtime_requires_a_ready_single_device(monkeypatch):
    monkeypatch.setattr(native_runtime, "discover_tpu", lambda *a, **k: _report(count=1))
    monkeypatch.setattr(
        native_runtime,
        "torch",
        SimpleNamespace(device=lambda name: name),
    )

    runtime = native_runtime.TorchTPURuntime()

    assert runtime.device == "tpu"
    assert runtime.name == "torchtpu"
    assert runtime.world_size == 1


@pytest.mark.parametrize(
    ("report", "message"),
    [
        (_report(status="allocation_failed", error_type="RuntimeError"), "initialization failed"),
        (_report(count=8), "exactly one visible TPU device"),
    ],
)
def test_native_runtime_rejects_failed_or_multi_device_discovery(
    monkeypatch, report, message
):
    monkeypatch.setattr(native_runtime, "discover_tpu", lambda *a, **k: report)

    with pytest.raises(RuntimeError, match=message):
        native_runtime.TorchTPURuntime()


def test_native_runtime_uses_generic_accelerator_synchronization(monkeypatch):
    calls = []
    monkeypatch.setattr(
        native_runtime,
        "torch",
        SimpleNamespace(accelerator=SimpleNamespace(synchronize=lambda: calls.append(1))),
    )
    runtime = object.__new__(native_runtime.TorchTPURuntime)

    runtime.synchronize()

    assert calls == [1]


def test_native_runtime_requires_synchronization_api(monkeypatch):
    monkeypatch.setattr(native_runtime, "torch", SimpleNamespace())
    runtime = object.__new__(native_runtime.TorchTPURuntime)

    with pytest.raises(RuntimeError, match="synchronization API"):
        runtime.synchronize()


def test_native_runtime_checkpoints_and_restores_device_rng(monkeypatch):
    state = object()
    restored = []
    accelerator = SimpleNamespace(
        get_rng_state=lambda: state,
        set_rng_state=lambda value: restored.append(value),
    )
    monkeypatch.setattr(
        native_runtime, "torch", SimpleNamespace(accelerator=accelerator)
    )
    runtime = object.__new__(native_runtime.TorchTPURuntime)
    runtime._precision = "fp32"
    runtime._seed = 42

    payload = runtime.state_dict()
    runtime.load_state_dict(payload)

    assert payload["backend"] == "torchtpu"
    assert payload["world_size"] == 1
    assert payload["seed"] == 42
    assert payload["device_rng_state"] is state
    assert restored == [state]


def test_native_runtime_seeds_active_torch_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        native_runtime, "torch", SimpleNamespace(manual_seed=lambda seed: calls.append(seed))
    )
    runtime = object.__new__(native_runtime.TorchTPURuntime)
    runtime._seed = 123

    runtime.initialize()

    assert calls == [123]


def test_native_runtime_rejects_incompatible_or_incomplete_rng_state(monkeypatch):
    monkeypatch.setattr(
        native_runtime,
        "torch",
        SimpleNamespace(accelerator=SimpleNamespace(set_rng_state=lambda _: None)),
    )
    runtime = object.__new__(native_runtime.TorchTPURuntime)

    with pytest.raises(ValueError, match="created by the native"):
        runtime.load_state_dict({"backend": "cpu"})
    with pytest.raises(ValueError, match="missing device RNG"):
        runtime.load_state_dict({"backend": "torchtpu", "world_size": 1})


def test_native_runtime_refuses_checkpoint_without_rng_capture_api(monkeypatch):
    monkeypatch.setattr(native_runtime, "torch", SimpleNamespace())
    runtime = object.__new__(native_runtime.TorchTPURuntime)

    with pytest.raises(RuntimeError, match="accelerator RNG capture"):
        runtime.state_dict()
