from types import SimpleNamespace

import pytest

import trainlm.runtime.torchtpu as native_runtime
import trainlm.runtime.runtime as base_runtime
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
    assert args.torch_compile is False


def test_torch_compile_is_opt_in_for_native_tpu_only():
    args = TrainLMTrainingArguments(
        accelerator="native_tpu", torch_compile=True, max_steps=1
    )
    assert args.torch_compile is True

    with pytest.raises(ValueError, match="only with accelerator='native_tpu'"):
        TrainLMTrainingArguments(accelerator="cpu", torch_compile=True, max_steps=1)

    with pytest.raises(ValueError, match="torch_compile must be boolean"):
        TrainLMTrainingArguments(
            accelerator="native_tpu", torch_compile="yes", max_steps=1
        )


def test_native_runtime_compiles_only_when_explicitly_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_runtime,
        "torch",
        SimpleNamespace(compile=lambda model: calls.append(model) or model),
    )
    model = object()
    disabled = object.__new__(native_runtime.TorchTPURuntime)
    disabled._compile_enabled = False
    enabled = object.__new__(native_runtime.TorchTPURuntime)
    enabled._compile_enabled = True

    assert disabled.compile_model(model) is model
    assert enabled.compile_model(model) is model
    assert calls == [model]


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
    runtime._compile_enabled = False

    payload = runtime.state_dict()
    runtime.validate_state_dict(payload)

    assert restored == []

    runtime.load_state_dict(payload)

    assert payload["backend"] == "torchtpu"
    assert payload["world_size"] == 1
    assert payload["seed"] == 42
    assert payload["compile_enabled"] is False
    assert payload["device_rng_state"] is state
    assert restored == [state]


def test_native_runtime_rejects_resume_with_changed_compile_mode():
    runtime = object.__new__(native_runtime.TorchTPURuntime)
    runtime._precision = "fp32"
    runtime._compile_enabled = True

    with pytest.raises(ValueError, match="compile mode does not match"):
        runtime.load_state_dict(
            {
                "backend": "torchtpu",
                "world_size": 1,
                "precision": "fp32",
                "compile_enabled": False,
                "device_rng_state": object(),
            }
        )


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
    runtime._precision = "fp32"
    runtime._compile_enabled = False

    with pytest.raises(ValueError, match="created by the native"):
        runtime.load_state_dict({"backend": "cpu"})
    with pytest.raises(ValueError, match="missing device RNG"):
        runtime.load_state_dict(
            {"backend": "torchtpu", "world_size": 1, "precision": "fp32"}
        )


def test_native_runtime_refuses_checkpoint_without_rng_capture_api(monkeypatch):
    monkeypatch.setattr(native_runtime, "torch", SimpleNamespace())
    runtime = object.__new__(native_runtime.TorchTPURuntime)

    with pytest.raises(RuntimeError, match="accelerator RNG capture"):
        runtime.state_dict()


def test_native_runtime_compiles_training_step_with_tpu_backend_when_enabled(monkeypatch):
    compile_calls = []
    monkeypatch.setattr(
        base_runtime,
        "torch",
        SimpleNamespace(
            compile=lambda fn, **kwargs: compile_calls.append((fn, kwargs)) or (lambda *a, **k: fn(*a, **k))
        ),
    )
    step = lambda batch, counts: (batch, counts)
    disabled = object.__new__(native_runtime.TorchTPURuntime)
    disabled._compile_enabled = False
    enabled = object.__new__(native_runtime.TorchTPURuntime)
    enabled._compile_enabled = True

    assert disabled.compile_training_step(step) is step
    compiled = enabled.compile_training_step(step)
    assert compiled is not step
    assert compile_calls == [(step, {"backend": "tpu", "dynamic": False})]
    assert compiled("b", 4) == ("b", 4)


def test_native_runtime_toggles_defer_and_fuse_only_when_not_compiled(monkeypatch):
    modes = ["STRICT"]

    def set_mode(mode):
        modes.append(mode)

    tpu_ns = SimpleNamespace(
        EagerMode=SimpleNamespace(DEFER_AND_FUSE="DEFER_AND_FUSE"),
        get_eager_mode=lambda: modes[-1],
        set_eager_mode=set_mode,
    )
    monkeypatch.setattr(native_runtime, "torch", SimpleNamespace(tpu=tpu_ns))

    compiled = object.__new__(native_runtime.TorchTPURuntime)
    compiled._compile_enabled = True
    compiled.on_train_begin()
    compiled.on_train_end()
    assert modes == ["STRICT"]

    eager = object.__new__(native_runtime.TorchTPURuntime)
    eager._compile_enabled = False
    eager.on_train_begin()
    assert modes == ["STRICT", "DEFER_AND_FUSE"]
    eager.on_train_end()
    assert modes == ["STRICT", "DEFER_AND_FUSE", "STRICT"]


def test_native_runtime_prepares_bf16_model_and_restores_mxu_precision(monkeypatch):
    import torch

    calls = []

    class _Model:
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    runtime = object.__new__(native_runtime.TorchTPURuntime)
    runtime._device = "tpu"
    runtime._precision = "bf16"
    runtime._compile_enabled = False
    runtime._allow_excess_precision = True

    runtime.prepare_model(_Model())
    assert calls == [((), {"device": "tpu", "dtype": torch.bfloat16})]

    matmul_precisions = ["high"]
    tpu_ns = SimpleNamespace(allow_excess_precision=False)
    monkeypatch.setattr(
        native_runtime,
        "torch",
        SimpleNamespace(
            get_float32_matmul_precision=lambda: matmul_precisions[-1],
            set_float32_matmul_precision=lambda p: matmul_precisions.append(p),
            tpu=tpu_ns,
        ),
    )

    runtime.on_train_begin()
    assert matmul_precisions[-1] == "medium"
    assert tpu_ns.allow_excess_precision is True
    runtime.on_train_end()
    assert matmul_precisions[-1] == "high"
    assert tpu_ns.allow_excess_precision is False
