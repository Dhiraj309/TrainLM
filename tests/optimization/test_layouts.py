"""CPU-only pack/split, optimizer continuation, and plain-HF export evidence."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from trainlm.config import OptimizerConfig
from trainlm.optimization import (
    CapabilityRequirement, ModelTransformation, OperationRequest, PlanningContext,
    ProviderDefinition, StateLayout, StateTensor, TensorPack, TransformHandler, TransformRegistry,
    TransformationError, apply_execution_plan, convert_optimizer_state, create_optimizer,
    inspect_dense_causal_lm, named_optimizer_state, plan_optimizations,
)
from trainlm.optimization.layouts import module_tensors
from trainlm.optimization.state_io import copy_tensors


class Triplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(4, 4) for _ in range(3))
        self.register_buffer("scale", torch.tensor(1.0), persistent=False)

    def forward(self, inputs):
        return (self.q_proj(inputs).sigmoid() * self.k_proj(inputs).tanh() + self.v_proj(inputs)) * self.scale


def local_layout(module):
    return StateLayout.from_module(module, (
        TensorPack(("q_proj.weight", "k_proj.weight", "v_proj.weight"), "qkv_weight"),
        TensorPack(("q_proj.bias", "k_proj.bias", "v_proj.bias"), "qkv_bias"),
    ))


class PackedTriplet(nn.Module):
    def __init__(self, source):
        super().__init__()
        state = local_layout(source).convert(module_tensors(source), include_nonpersistent=True)
        self.qkv_weight = nn.Parameter(state["qkv_weight"])
        self.qkv_bias = nn.Parameter(state["qkv_bias"])
        self.register_buffer("scale", state["scale"], persistent=False)
        self.train(source.training)

    def forward(self, inputs):
        q, k, v = F.linear(inputs, self.qkv_weight, self.qkv_bias).chunk(3, dim=-1)
        return (q.sigmoid() * k.tanh() + v) * self.scale


def unpack(module):
    result = Triplet().to(dtype=module.qkv_weight.dtype, device=module.qkv_weight.device)
    canonical = local_layout(result).convert(module_tensors(module), to_canonical=True, include_nonpersistent=True)
    copy_tensors(result, canonical)
    result.train(module.training)
    return result


class FixtureLM(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.block = Triplet()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(4, 16, bias=False)
        self.head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids):
        return self.head(self.dropout(self.block(self.embedding(input_ids))))


DATA = torch.tensor([[1, 2, 3, 4]])


def apply_fixture(model, *, layout_factory=local_layout, forward=PackedTriplet):
    transform = ModelTransformation("fixture.pack-qkv", "projections", "fixture.qkv", ("block",),
                                    "fixture.unpack-qkv", "Equivalent CPU fixture packing.", True)
    definition = ProviderDefinition(
        "fixture.qkv", "forward_backward", CapabilityRequirement("projections", ("separate_qkv",)),
        ("cpu",), ("fp32",), ("causal",), supports_backward=True, transformations=(transform,),
    )
    context = PlanningContext("cpu", "fp32", "causal")
    requests = (OperationRequest("projections", "forward_backward"),)
    plan = plan_optimizations(inspect_dense_causal_lm(model), context=context, requests=requests, providers=(definition,))
    handler = TransformHandler("fixture.pack-qkv", "fixture.unpack-qkv", "fixture.qkv", "projections",
                               (Triplet,), forward, unpack, layout=layout_factory)
    return apply_execution_plan(model, plan, context=context, requests=requests, providers=(definition,),
                                registry=TransformRegistry((handler,)))


def step(model, optimizer, scheduler=None):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(DATA).reshape(-1, 16), DATA.reshape(-1))
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.detach().clone()


def make_optimizer(model, compact=False):
    return (create_optimizer(model.parameters(), OptimizerConfig(
        learning_rate=0.01, fused=False, mu_dtype="bfloat16", nu_dtype="float32",
    )) if compact else torch.optim.AdamW(model.parameters(), lr=0.01, fused=False, foreach=False))


def test_pack_split_alias_and_metadata_roundtrip_is_lossless_and_nonmutating():
    model = FixtureLM()
    packs = local_layout(model.block).prefixed("block").packs
    layout = StateLayout.from_module(model, packs)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    transformed = layout.convert(model.state_dict())
    assert transformed["block.qkv_weight"].shape == (12, 4)
    assert transformed["block.qkv_bias"].shape == (12,)
    assert transformed["embedding.weight"] is transformed["head.weight"]
    canonical = layout.convert(transformed, to_canonical=True)
    torch.testing.assert_close(canonical, before, rtol=0, atol=0)
    canonical["block.q_proj.weight"].zero_()
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert StateLayout.from_dict(json.loads(layout.to_json())) == layout
    assert StateLayout.from_dict(layout.to_dict()).fingerprint == layout.fingerprint


def test_pack_dimension_and_unequal_split_sizes_are_explicit():
    layout = StateLayout((StateTensor("left", (2, 3), "torch.float32"), StateTensor("right", (2, 5), "torch.float32")),
                         (TensorPack(("left", "right"), "packed", 1),))
    source = {"left": torch.randn(2, 3), "right": torch.randn(2, 5)}
    packed = layout.convert(source)
    assert packed["packed"].shape == (2, 8)
    torch.testing.assert_close(layout.convert(packed, to_canonical=True), source, rtol=0, atol=0)


def test_trained_layout_revert_converts_weights_gradients_and_existing_optimizer():
    torch.manual_seed(7)
    model = FixtureLM()
    reference = deepcopy(model)
    original_embedding, original_block, original_scale = model.embedding.weight, model.block, model.block.scale
    transaction = apply_fixture(model)
    optimizer, baseline_optimizer = make_optimizer(model), make_optimizer(reference)
    torch.testing.assert_close(model(DATA), reference(DATA), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(step(model, optimizer), step(reference, baseline_optimizer), rtol=1e-6, atol=1e-6)
    canonical = transaction.canonical_state_dict()
    torch.testing.assert_close(canonical, reference.state_dict(), rtol=1e-5, atol=1e-6)
    assert model.embedding.weight is model.head.weight is original_embedding
    assert model.block.scale is original_scale
    model.eval()
    exported_copy = transaction.canonical_model()
    assert type(exported_copy.block) is Triplet
    assert exported_copy.embedding.weight is exported_copy.head.weight
    torch.testing.assert_close(exported_copy(DATA), model(DATA), rtol=1e-6, atol=1e-6)
    transaction.revert(optimizer)
    assert model.block is original_block and not model.block.training
    assert model.embedding.weight is model.head.weight is original_embedding
    assert model.block.q_proj.weight in optimizer.state
    assert all(parameter is not getattr(original_block, "qkv_weight", None)
               for group in optimizer.param_groups for parameter in group["params"])
    torch.testing.assert_close(model.state_dict(), canonical, rtol=0, atol=0)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(step(model, optimizer), step(reference, baseline_optimizer), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("compact", [False, True])
def test_transformed_checkpoint_resume_matches_next_update_and_moment_precision(tmp_path, compact):
    torch.manual_seed(11)
    model = FixtureLM(dropout=0.2)
    transaction = apply_fixture(model)
    optimizer = make_optimizer(model, compact)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    step(model, optimizer, scheduler)
    model.block.scale.add_(0.125)  # Runtime-only buffers also affect the next update.
    payload = transaction.training_state_dict(optimizer, scheduler)
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    expected_loss = step(model, optimizer, scheduler)
    expected = transaction.canonical_state_dict()
    expected_optimizer = transaction.training_state_dict(optimizer, scheduler)["optimizer"]
    resumed = FixtureLM(dropout=0.2)
    resumed_transaction = apply_fixture(resumed)
    resumed_optimizer = make_optimizer(resumed, compact)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 1, gamma=0.9)
    resumed_transaction.load_training_state_dict(torch.load(path, weights_only=True), resumed_optimizer, resumed_scheduler)
    actual_loss = step(resumed, resumed_optimizer, resumed_scheduler)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(resumed_transaction.canonical_state_dict(), expected, rtol=0, atol=0)
    actual_optimizer = resumed_transaction.training_state_dict(resumed_optimizer, resumed_scheduler)["optimizer"]
    torch.testing.assert_close(actual_optimizer["state"], expected_optimizer["state"], rtol=0, atol=0)
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for slots in resumed_optimizer.state.values():
        assert slots["exp_avg"].dtype == (torch.bfloat16 if compact else torch.float32)
        assert slots["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize("damage,reason", [("dtype", "shape/dtype"), ("alias", "Aliased state values"),
    ("partial", "inconsistent initialized"), ("step", "scalar steps must match"),
    ("moment", "moment precision"), ("plan", "identity"), ("slot", "Unsupported optimizer slot")])
def test_bad_checkpoint_is_rejected_without_mutating_live_weights_or_optimizer(damage, reason):
    model = FixtureLM()
    transaction = apply_fixture(model)
    optimizer = make_optimizer(model)
    step(model, optimizer)
    payload = transaction.training_state_dict(optimizer)
    if damage == "dtype":
        payload["model"]["block.q_proj.weight"] = payload["model"]["block.q_proj.weight"].half()
    elif damage == "alias":
        payload["model"]["head.weight"] = payload["model"]["head.weight"] + 1
    elif damage == "partial":
        payload["optimizer"]["state"].pop("block.q_proj.weight")
    elif damage == "step":
        payload["optimizer"]["state"]["block.q_proj.weight"]["step"].add_(1)
    elif damage == "moment":
        item = payload["optimizer"]["state"]["block.q_proj.weight"]
        item["exp_avg"] = item["exp_avg"].half()
    elif damage == "plan":
        payload["plan_id"] = "wrong"
    elif damage == "slot":
        payload["optimizer"]["state"]["block.q_proj.weight"]["mystery"] = torch.tensor(1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    old_state, old_groups = optimizer.state, optimizer.param_groups
    with pytest.raises(ValueError, match=reason):
        transaction.load_training_state_dict(payload, optimizer)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert optimizer.state is old_state and optimizer.param_groups is old_groups


def test_different_optimizer_groups_and_tied_pack_sources_are_rejected_before_forward():
    model = FixtureLM()
    layout = StateLayout.from_module(model, local_layout(model.block).prefixed("block").packs)
    optimizer = make_optimizer(model)
    step(model, optimizer)
    named = named_optimizer_state(model, optimizer)
    group = named["param_groups"][0]
    group["params"].remove("block.q_proj.weight")
    named["param_groups"].append({**deepcopy(group), "params": ["block.q_proj.weight"]})
    with pytest.raises(ValueError, match="share one complete parameter group"):
        convert_optimizer_state(layout, named)
    optimizer.zero_grad(set_to_none=True)
    model.block.q_proj.weight = model.embedding.weight
    calls = []
    with pytest.raises(TransformationError, match="Packing tied parameters"):
        apply_fixture(model, forward=lambda module: calls.append(module))
    assert calls == [] and type(model.block) is Triplet


def test_layout_application_rejects_pending_gradients_before_forward():
    model = FixtureLM()
    model(DATA).sum().backward()
    gradient = model.block.q_proj.weight.grad.clone()
    calls = []
    with pytest.raises(TransformationError, match="cleared gradients"):
        apply_fixture(model, forward=lambda module: calls.append(module))
    assert calls == [] and type(model.block) is Triplet
    torch.testing.assert_close(model.block.q_proj.weight.grad, gradient, rtol=0, atol=0)


def test_restore_rolls_back_weights_optimizer_and_rng_after_a_late_failure():
    model = FixtureLM()
    transaction = apply_fixture(model)
    optimizer = make_optimizer(model)
    step(model, optimizer)
    payload = transaction.training_state_dict(optimizer)
    payload["model"]["block.q_proj.weight"].add_(0.5)
    payload["nonpersistent_buffers"]["block.scale"].add_(0.5)
    payload["cpu_rng"] = torch.tensor([0.0])  # Invalid byte-state fails after tensor writes.
    before = {name: value.clone() for name, value in model.state_dict().items()}
    old_state, old_groups, old_rng = optimizer.state, optimizer.param_groups, torch.get_rng_state().clone()
    old_scale = model.block.scale.clone()
    with pytest.raises((TypeError, RuntimeError)):
        transaction.load_training_state_dict(payload, optimizer)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert optimizer.state is old_state and optimizer.param_groups is old_groups
    assert torch.equal(torch.get_rng_state(), old_rng)
    torch.testing.assert_close(model.block.scale, old_scale, rtol=0, atol=0)


def test_resume_rejects_changed_optimizer_membership_and_implementation():
    model = FixtureLM()
    transaction = apply_fixture(model)
    optimizer = make_optimizer(model)
    step(model, optimizer)
    payload = transaction.training_state_dict(optimizer)
    wrong_group = torch.optim.AdamW([model.embedding.weight], lr=0.01, fused=False)
    with pytest.raises(ValueError, match="group parameter membership"):
        transaction.load_training_state_dict(payload, wrong_group)
    with pytest.raises(ValueError, match="implementation or state precision"):
        transaction.load_training_state_dict(payload, torch.optim.SGD(model.parameters(), lr=0.01))


@pytest.mark.parametrize("damage", ["overlap", "target", "dimension", "dtype", "alias-cycle"])
def test_invalid_declarative_mappings_fail_before_conversion(damage):
    specs = (StateTensor("a", (2, 3), "torch.float32"), StateTensor("b", (2, 3), "torch.float32"))
    packs = (TensorPack(("a", "b"), "packed"),)
    aliases = ()
    if damage == "overlap":
        packs += (TensorPack(("a",), "other"),)
    elif damage == "target":
        packs = (TensorPack(("a",), "b"),)
    elif damage == "dimension":
        packs = (TensorPack(("a", "b"), "packed", 2),)
    elif damage == "dtype":
        specs = (specs[0], replace(specs[1], dtype="torch.float16"))
    elif damage == "alias-cycle":
        packs, aliases = (), (("a", "b"), ("b", "a"))
    with pytest.raises(ValueError):
        StateLayout(specs, packs, aliases)


def test_plain_transformers_export_after_mapped_training_reloads_weights_aliases_and_logits(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig
    model = AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
        tie_word_embeddings=True,
    ))
    class RenamedLinear(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.packed_weight = nn.Parameter(source.weight.detach().clone())
            self.train(source.training)
        def forward(self, inputs):
            return F.linear(inputs, self.packed_weight)
    def inverse(module):
        result = nn.Linear(module.packed_weight.shape[1], module.packed_weight.shape[0], bias=False)
        result.weight = nn.Parameter(module.packed_weight.detach().clone())
        return result.train(module.training)
    mapping = lambda module: StateLayout.from_module(module, (TensorPack(("weight",), "packed_weight"),))
    transform = ModelTransformation("fixture.rename", "projections", "fixture.rename", ("model.layers.0.self_attn.q_proj",),
                                    "fixture.unrename", "Plain-HF round trip fixture.", True)
    provider = ProviderDefinition("fixture.rename", "forward_backward", CapabilityRequirement("projections", ("separate_qkv",)),
                                  ("cpu",), ("fp32",), ("causal",), supports_backward=True, transformations=(transform,))
    arguments = dict(context=PlanningContext("cpu", "fp32", "causal"),
                     requests=(OperationRequest("projections", "forward_backward"),), providers=(provider,))
    plan = plan_optimizations(inspect_dense_causal_lm(model), **arguments)
    handler = TransformHandler("fixture.rename", "fixture.unrename", "fixture.rename", "projections", (nn.Linear,),
                               RenamedLinear, inverse, layout=mapping)
    transaction = apply_execution_plan(model, plan, registry=TransformRegistry((handler,)), **arguments)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=False)
    model(input_ids=DATA, labels=DATA).loss.backward()
    optimizer.step()
    model.eval()
    expected = model(input_ids=DATA).logits.detach()
    transaction.export_huggingface(tmp_path / "export")
    restored = AutoModelForCausalLM.from_pretrained(tmp_path / "export", local_files_only=True).eval()
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert "model.layers.0.self_attn.q_proj.weight" in restored.state_dict()
    assert not any("packed_weight" in name for name in restored.state_dict())
    torch.testing.assert_close(restored.state_dict(), transaction.canonical_state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored(input_ids=DATA).logits, expected, rtol=1e-6, atol=1e-6)
