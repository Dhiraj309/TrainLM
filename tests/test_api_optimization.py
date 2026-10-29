"""Public optimization lifecycle tests; fixture providers make no hardware claims."""

import builtins
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainlm import TrainLMTrainer, TrainLMTrainingArguments, OptimizationOptions, OptimizationError
from trainlm.optimization import (
    CapabilityRequirement, ModelTransformation, OperationRequest, ProviderDefinition,
    TransformHandler, TransformRegistry, StateLayout, TensorPack, ShapeConstraint,
)
from trainlm.runtime import TorchRuntime
from trainlm.training import TrainerCallback
from tests.optimization.test_layouts import FixtureLM, Triplet, PackedTriplet, local_layout, unpack
from tests.data.test_partition import _reader
from tests.test_tpu_lifecycle import cpu_worker, worker_args
from tests.test_tpu_entrypoint import load_script


class APIModel(FixtureLM):
    def __init__(self, dropout=0.0):
        super().__init__(dropout)
        self.embedding = nn.Embedding(128, 4)
        self.head = nn.Linear(4, 128, bias=False)
        self.tie_weights()
        self.config = SimpleNamespace(use_cache=False, tie_word_embeddings=True,
                                      to_dict=lambda: {"vocab_size": 128})

    def tie_weights(self):
        self.head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None, attention_mask=None):
        return SimpleNamespace(logits=super().forward(input_ids))

    def save_pretrained(self, destination, *, state_dict=None, safe_serialization=True):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict() if state_dict is None else state_dict, destination / "weights.pt")


def mapped_profile():
    transform = ModelTransformation("fixture.pack-qkv", "projections", "fixture.qkv", ("block",),
                                    "fixture.unpack-qkv", "Equivalent public API fixture.", True)
    provider = ProviderDefinition("fixture.qkv", "forward_backward", CapabilityRequirement("projections", ("separate_qkv",)),
        ("cpu", "xla"), ("fp32",), ("causal",), supports_backward=True, transformations=(transform,))
    handler = TransformHandler("fixture.pack-qkv", "fixture.unpack-qkv", "fixture.qkv", "projections", (Triplet,),
        PackedTriplet, unpack, layout=local_layout)
    return OptimizationOptions(requests=(OperationRequest("projections", "forward_backward"),), providers=(provider,),
                               handlers=TransformRegistry((handler,)), mask_kind="causal")


PROFILE = f"{__name__}:mapped_profile"


def dataset():
    return [{"input_ids": torch.tensor([1, 2, 3, 4])}, {"input_ids": torch.tensor([4, 3, 2, 1])}]


class StopAtOne(TrainerCallback):
    def on_step_end(self, state, control):
        if state.step == 1:
            control.request_stop()


def args(path, **changes):
    return replace(TrainLMTrainingArguments(output_dir=path, accelerator="cpu", max_steps=3,
                                           sequence_length=4, save_steps=1, eval_steps=1,
                                           logging_steps=1, lr_scheduler_type="constant"), **changes)


def test_public_mapped_train_evaluate_checkpoint_resume_and_export(tmp_path):
    torch.manual_seed(3)
    initial = APIModel(dropout=0.2)
    full = TrainLMTrainer(model=deepcopy(initial), train_dataset=dataset(), eval_dataset=dataset(),
                         args=args(tmp_path / "full"), optimization=mapped_profile())
    assert type(full.model.block) is PackedTriplet
    assert any(parameter is full.model.block.qkv_weight for group in full.optimizer.param_groups for parameter in group["params"])
    torch.manual_seed(99)
    full.train()
    expected = full._optimization_session.applied.canonical_state_dict()
    stopped = TrainLMTrainer(model=deepcopy(initial), train_dataset=dataset(), eval_dataset=dataset(),
                            args=args(tmp_path / "stopped"), optimization=PROFILE, callbacks=[StopAtOne()])
    torch.manual_seed(99)
    stopped.train()
    checkpoint = tmp_path / "stopped" / "checkpoint-1"
    saved = torch.load(checkpoint / "trainer_state.pt", weights_only=True)
    assert saved["schema_version"] == 2
    assert "block.q_proj.weight" in saved["optimization"]["model"]
    assert not any("qkv_weight" in name for name in saved["optimization"]["model"])
    resumed = TrainLMTrainer(model=APIModel(dropout=0.2), train_dataset=dataset(), eval_dataset=dataset(),
                            args=args(tmp_path / "resumed"), optimization=mapped_profile())
    state = resumed.train(resume_from_checkpoint=checkpoint)
    assert state.step == 3 and state.tokens_seen == full.engine.state.tokens_seen
    torch.testing.assert_close(resumed._optimization_session.applied.canonical_state_dict(), expected, rtol=0, atol=0)
    export = resumed.save_model(tmp_path / "export")
    torch.testing.assert_close(torch.load(export / "weights.pt", weights_only=True), expected, rtol=0, atol=0)
    state_file = resumed.save_state(tmp_path / "manual")
    assert torch.load(state_file, weights_only=True)["schema_version"] == 2
    report = resumed.explain()
    assert report["optimization"]["status"] == "applied"
    assert report["capabilities"]["projections"]["kind"] == "separate_qkv"
    assert report["optimization"]["applied_transform_ids"] == ["fixture.pack-qkv"]
    assert not report["optimization"]["certification"]["hardware_validated"]
    assert json.loads(resumed.explain(format="json")) == json.loads(json.dumps(report))
    assert "inverse=fixture.unpack-qkv" in resumed.explain(format="text")
    assert "Applied transforms: fixture.pack-qkv" in resumed.explain(format="text")
    assert "Hardware validated: False" in resumed.explain(format="text")


def test_reference_disabled_and_auto_fallback_paths_do_not_change_model(tmp_path):
    for policy, options in (("auto", None), ("disabled", mapped_profile()),
                            ("auto", replace(mapped_profile(), mask_kind="padding"))):
        model = APIModel()
        original = model.block
        trainer = TrainLMTrainer(model=model, train_dataset=dataset(), args=args(tmp_path / policy, eval_steps=None,
                                                                                     optimization_policy=policy),
                                 optimization=options)
        assert model.block is original
        assert trainer.explain()["optimization"]["applied_transform_ids"] == []
        assert trainer.explain()["optimization"]["execution_plan"]["decisions"][0]["status"] in {"skipped", "fallback"}


def test_strict_failures_happen_before_runtime_or_optimizer_construction(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime/optimizer constructed before strict validation")
    monkeypatch.setattr(TrainLMTrainer, "_make_runtime", forbidden)
    monkeypatch.setattr(TrainLMTrainer, "_make_optimizer", forbidden)
    with pytest.raises(OptimizationError, match="blocked") as error:
        TrainLMTrainer(model=APIModel(), train_dataset=dataset(),
            args=args(tmp_path, eval_steps=None, optimization_policy="required"),
            optimization=replace(mapped_profile(), mask_kind="padding"))
    assert error.value.report["execution_plan"]["status"] == "blocked"
    pending = APIModel()
    pending(input_ids=dataset()[0]["input_ids"].unsqueeze(0)).logits.sum().backward()
    with pytest.raises(OptimizationError, match="cleared gradients before runtime"):
        TrainLMTrainer(model=pending, train_dataset=dataset(),
            args=args(tmp_path, eval_steps=None, optimization_policy="required"), optimization=mapped_profile())
    with pytest.raises(OptimizationError, match="blocked"):
        TrainLMTrainer(model="org/model", train_dataset=tmp_path,
            args=TrainLMTrainingArguments(accelerator="tpu", optimization_policy="required"))


def test_backend_prepares_once_before_optimizer_and_failed_setup_reverts(tmp_path, monkeypatch):
    events = []
    class Runtime(TorchRuntime):
        def prepare_model(self, model):
            events.append("prepare")
            return super().prepare_model(model)
    real = TrainLMTrainer._make_optimizer
    def optimizer(self):
        events.append("optimizer")
        assert type(self.model.block) is PackedTriplet
        return real(self)
    monkeypatch.setattr(TrainLMTrainer, "_make_optimizer", optimizer)
    TrainLMTrainer(model=APIModel(), train_dataset=dataset(), runtime=Runtime(device="cpu"),
                   args=args(tmp_path, eval_steps=None), optimization=mapped_profile())
    assert events == ["prepare", "optimizer"]
    model = APIModel()
    original = model.block
    monkeypatch.setattr(TrainLMTrainer, "_make_optimizer", lambda self: (_ for _ in ()).throw(RuntimeError("factory failed")))
    with pytest.raises(RuntimeError, match="factory failed"):
        TrainLMTrainer(model=model, train_dataset=dataset(), args=args(tmp_path, eval_steps=None), optimization=mapped_profile())
    assert model.block is original and model.embedding.weight is model.head.weight


def test_mapped_checkpoints_require_matching_profile(tmp_path):
    trainer = TrainLMTrainer(model=APIModel(), train_dataset=dataset(), args=args(tmp_path / "mapped", eval_steps=None),
                             optimization=mapped_profile())
    trainer.train()
    reference = TrainLMTrainer(model=APIModel(), train_dataset=dataset(), args=args(tmp_path / "ref", eval_steps=None))
    before = deepcopy(reference.model.state_dict())
    with pytest.raises(ValueError, match="matching optimization profile"):
        reference.train(resume_from_checkpoint=tmp_path / "mapped" / "checkpoint-1")
    torch.testing.assert_close(reference.model.state_dict(), before, rtol=0, atol=0)


def test_private_worker_strict_preflight_never_imports_xla(monkeypatch, tmp_path):
    import sys
    worker = load_script("trainlm_tpu_worker")
    original_import = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.startswith("torch_xla"):
            raise AssertionError("XLA imported before required policy failed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    monkeypatch.setattr(worker, "configure_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["worker", "--probe-only", "--optimization-policy", "required",
                                      "--output-dir", str(tmp_path)])
    with pytest.raises(OptimizationError, match="blocked"):
        worker.main()


def test_tpu_worker_profile_preflight_train_resume_and_canonical_export(tmp_path, monkeypatch, cpu_worker):
    import trainlm.model
    from trainlm._tpu_optimization import host_preflight
    def load_model(source):
        model = APIModel()
        return SimpleNamespace(model=model, config=model.config,
            metadata=SimpleNamespace(resolved_revision=None, to_dict=lambda: {}))
    monkeypatch.setattr(trainlm.model, "load_huggingface_causal_lm", load_model)
    monkeypatch.setattr(cpu_worker.module, "load_huggingface_causal_lm", load_model)
    request = worker_args(cpu_worker, tmp_path / "train", monkeypatch,
        "--optimization-factory", PROFILE, "--optimization-policy", "required",
        "--final-checkpoint-dir", str(tmp_path / "final"))
    preflight = host_preflight(request)
    request.optimization_plan_id = preflight["execution_plan"]["plan_id"]
    with _reader(tmp_path / "data", (96,)) as reader:
        cpu_worker.module.train_fn(0, request, reader.shards)
        summary = json.loads((tmp_path / "train" / "summary.json").read_text())
        assert summary["optimization"]["status"] == "applied"
        payload = torch.load(tmp_path / "final" / "rank-0.pt", weights_only=True)
        assert payload["schema_version"] == 2 and payload["model"] is None
        assert "block.q_proj.weight" in payload["optimization"]["model"]
        request.operation, request.resume_from_checkpoint = "export", str(tmp_path / "final")
        request.output_dir, request.export_dir = str(tmp_path / "operation"), str(tmp_path / "export")
        request.final_checkpoint_dir = None
        cpu_worker.module.train_fn(0, request, reader.shards)
        exported = torch.load(tmp_path / "export" / "weights.pt", weights_only=True)
        torch.testing.assert_close(exported, payload["optimization"]["model"], rtol=0, atol=0)
    assert preflight["applied_transform_ids"] == []


@pytest.mark.parametrize("policy", ["auto", "required"])
def test_evaluation_geometry_is_checked_before_applying_a_train_only_shape(tmp_path, policy):
    options = mapped_profile()
    options = replace(options, providers=(replace(options.providers[0], shapes=(ShapeConstraint("batch_size", maximum=1),)),))
    model = APIModel()
    original = model.block
    arguments = args(tmp_path, per_device_eval_batch_size=2, optimization_policy=policy)
    if policy == "required":
        with pytest.raises(OptimizationError, match="Evaluation:"):
            TrainLMTrainer(model=model, train_dataset=dataset(), eval_dataset=dataset(), args=arguments, optimization=options)
    else:
        trainer = TrainLMTrainer(model=model, train_dataset=dataset(), eval_dataset=dataset(), args=arguments, optimization=options)
        report = trainer.explain()["optimization"]
        assert report["applied_transform_ids"] == []
        assert "Evaluation:" in report["excluded_providers"][0]["reason"]
    assert model.block is original


def test_public_hf_provider_mapped_training_exports_plain_transformers(tmp_path):
    from transformers import AutoModelForCausalLM
    from trainlm.config import ModelSourceConfig
    class Renamed(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.packed_weight = nn.Parameter(module.weight.detach().clone())
            self.train(module.training)
        def forward(self, inputs):
            return torch.nn.functional.linear(inputs, self.packed_weight)
    def inverse(module):
        result = nn.Linear(module.packed_weight.shape[1], module.packed_weight.shape[0], bias=False)
        result.weight = nn.Parameter(module.packed_weight.detach().clone())
        return result.train(module.training)
    transform = ModelTransformation("fixture.rename", "projections", "fixture.rename",
        ("model.layers.0.self_attn.q_proj",), "fixture.unrename", "Public HF fixture.", True)
    provider = ProviderDefinition("fixture.rename", "forward_backward", CapabilityRequirement("projections", ("separate_qkv",)),
        ("cpu",), ("fp32",), ("causal",), supports_backward=True, transformations=(transform,))
    handler = TransformHandler("fixture.rename", "fixture.unrename", "fixture.rename", "projections", (nn.Linear,),
        Renamed, inverse, layout=lambda module: StateLayout.from_module(module, (TensorPack(("weight",), "packed_weight"),)))
    options = OptimizationOptions(requests=(OperationRequest("projections", "forward_backward"),),
        providers=(provider,), handlers=TransformRegistry((handler,)), mask_kind="causal")
    source = ModelSourceConfig(provider="huggingface", initialization="config", model_type="llama",
        config_overrides={"vocab_size": 16, "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                          "num_attention_heads": 2, "num_key_value_heads": 2, "tie_word_embeddings": True, "use_cache": False})
    trainer = TrainLMTrainer(model=source, train_dataset=dataset(), optimization=options,
                            args=args(tmp_path / "run", max_steps=1, eval_steps=None))
    trainer.train()
    trainer.model.eval()
    expected = trainer.model(input_ids=dataset()[0]["input_ids"].unsqueeze(0)).logits.detach()
    path = trainer.save_model(tmp_path / "hf")
    restored = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).eval()
    torch.testing.assert_close(restored(input_ids=dataset()[0]["input_ids"].unsqueeze(0)).logits, expected, rtol=1e-6, atol=1e-6)
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert not any("packed_weight" in name for name in restored.state_dict())
    assert trainer.explain()["optimization"]["status"] == "applied"


def test_worker_transform_application_preserves_device_seed_on_failure():
    from trainlm._tpu_optimization import apply_worker_profile
    seed = [7]
    class Session:
        plan = SimpleNamespace(transformations=("fixture",))
        def apply(self, model):
            seed[0] = 99
            raise RuntimeError("injected transform failure")
    with pytest.raises(RuntimeError, match="injected transform failure"):
        apply_worker_profile(Session(), None, get_rng=lambda: seed[0], set_rng=lambda value: seed.__setitem__(0, value))
    assert seed[0] == 7


def test_default_training_tolerates_unimplemented_optional_inspection_hooks(tmp_path):
    class ExternalModel(APIModel):
        def get_input_embeddings(self):
            raise NotImplementedError("Optional hook is not part of forward execution")
    trainer = TrainLMTrainer(model=ExternalModel(), train_dataset=dataset(),
                            args=args(tmp_path, max_steps=1, eval_steps=None))
    assert trainer.train().step == 1
    report = trainer.explain()
    assert report["capabilities"]["embedding"]["status"] == "unknown"
    assert "not implemented" in report["capabilities"]["warnings"][0]
