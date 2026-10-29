"""CPU transaction evidence using equivalent fixture modules, not real kernels."""

from copy import deepcopy
from dataclasses import replace
import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from trainlm.optimization import (
    CapabilityRequirement, ModelTransformation, OperationRequest, PlanningContext,
    ProviderDefinition, TransformHandler, TransformRegistry, TransformationError,
    apply_execution_plan, inspect_dense_causal_lm, plan_optimizations,
)


class FixtureLinear(nn.Linear):
    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)


class FixtureLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.heads = nn.ModuleList([nn.Linear(4, 16, bias=False), nn.Linear(4, 16, bias=False)])
        self.heads[0].weight = self.embedding.weight
        self.heads[1].weight = self.embedding.weight
        self.register_buffer("counter", torch.tensor(3), persistent=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.heads[0]

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        return self.heads[0](hidden) + self.heads[1](hidden)


def convert(module, kind):
    result = deepcopy(module)
    result.__class__ = kind  # Changes an isolated fixture instance, not a class/global method.
    return result


def handler(**changes):
    return replace(TransformHandler(
        transform_id="fixture.linear", inverse_transform_id="fixture.reference",
        provider="fixture.provider", component="lm_head", source_types=(nn.Linear,),
        forward=lambda module: convert(module, FixtureLinear),
        inverse=lambda module: convert(module, nn.Linear),
    ), **changes)


def inputs(model, *, targets=("heads.*",), policy="auto", **transform_options):
    transform = ModelTransformation(
        transform_id="fixture.linear", component="lm_head", provider="fixture.provider",
        target_paths=targets, inverse_transform_id="fixture.reference",
        reason="Equivalent CPU fixture module replacement.", **transform_options,
    )
    provider = ProviderDefinition(
        provider_id="fixture.provider", operation="forward_backward",
        capability=CapabilityRequirement("lm_head", ("linear",)),
        backends=("cpu",), precisions=("fp32",), masks=("none",), supports_backward=True,
        transformations=(transform,),
    )
    arguments = dict(context=PlanningContext("cpu", "fp32", "none"),
                     requests=(OperationRequest("lm_head", "forward_backward"),), providers=(provider,))
    plan = plan_optimizations(inspect_dense_causal_lm(model), policy=policy, **arguments)
    return plan, arguments


def state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_apply_before_optimizer_preserves_outputs_gradients_aliases_and_trained_undo():
    model = FixtureLM().train()
    reference = deepcopy(model)
    originals = tuple(model.heads)
    parameters = dict(model.named_parameters())
    counter = model.counter
    plan, arguments = inputs(model)
    applied = apply_execution_plan(model, plan, registry=TransformRegistry((handler(),)), **arguments)
    assert applied.applied_transform_ids == ("fixture.linear",)
    assert all(type(module) is FixtureLinear for module in model.heads)
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())
    assert model.embedding.weight is model.heads[0].weight is model.heads[1].weight
    assert model.counter is counter
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    baseline_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    data = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(data), reference(data), rtol=0, atol=0)
    model(data).square().mean().backward()
    reference(data).square().mean().backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    optimizer.step()
    baseline_optimizer.step()
    model.eval()
    assert applied.revert() is model
    assert applied.is_reverted and applied.revert() is model
    assert tuple(model.heads) == originals
    assert not model.heads[0].training
    assert model.embedding.weight is model.heads[0].weight is model.heads[1].weight
    torch.testing.assert_close(state(model), state(reference), rtol=0, atol=0)
    torch.testing.assert_close(model(data), reference(data), rtol=0, atol=0)
    assert optimizer.param_groups[0]["params"][0] is model.embedding.weight


@pytest.mark.parametrize("failure", ["forward", "inverse", "validator"])
def test_partial_handler_failure_leaves_original_model_usable_and_rng_unchanged(failure):
    model = FixtureLM().eval()
    plan, arguments = inputs(model)
    original_modules = tuple(model.heads)
    before = state(model)
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    calls = []
    def forward(module):
        torch.rand(3)
        random.random()
        calls.append("forward")
        if failure == "forward" and len(calls) > 1:
            module.weight.data.zero_()
            raise RuntimeError("injected forward failure")
        return convert(module, FixtureLinear)
    def inverse(module):
        if failure == "inverse":
            module.weight.data.zero_()
            raise RuntimeError("injected inverse failure")
        return convert(module, nn.Linear)
    def validate(source, candidate):
        source.weight.data.zero_()  # Even validator source arguments are isolated.
        if failure == "validator":
            candidate.weight.data.zero_()
            raise RuntimeError("injected validator failure")
    registry = TransformRegistry((handler(forward=forward, inverse=inverse, validate=validate),))
    with pytest.raises(TransformationError, match=f"injected {failure} failure"):
        apply_execution_plan(model, plan, registry=registry, **arguments)
    assert tuple(model.heads) == original_modules
    assert model.embedding.weight is model.heads[0].weight is model.heads[1].weight
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    torch.testing.assert_close(state(model), before, rtol=0, atol=0)
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    assert model.embedding.weight.grad is not None


def test_commit_failure_restores_already_installed_replacements(monkeypatch):
    import trainlm.optimization.application as application
    model = FixtureLM()
    plan, arguments = inputs(model)
    original_modules = tuple(model.heads)
    before = state(model)
    install = application._install
    def fail_second(model, path, replacement):
        if path == "heads.1" and type(replacement) is FixtureLinear:
            raise RuntimeError("injected commit failure")
        install(model, path, replacement)
    monkeypatch.setattr(application, "_install", fail_second)
    with pytest.raises(TransformationError, match="injected commit failure"):
        apply_execution_plan(model, plan, registry=TransformRegistry((handler(),)), **arguments)
    assert tuple(model.heads) == original_modules
    torch.testing.assert_close(state(model), before, rtol=0, atol=0)


@pytest.mark.parametrize("damage,reason", [("values", "values changed"), ("layout", "tensor names"),
                                          ("hooks", "hooks"), ("mode", "training mode")])
def test_unadvertised_state_or_hook_mutations_are_rejected(damage, reason):
    model = FixtureLM().eval()
    plan, arguments = inputs(model)
    before = state(model)
    def forward(module):
        replacement = convert(module, FixtureLinear)
        if damage == "values":
            replacement.weight.data.add_(1)
        elif damage == "layout":
            replacement.extra = nn.Parameter(torch.zeros(1))
        elif damage == "hooks":
            replacement.register_forward_hook(lambda *args: None)
        elif damage == "mode":
            replacement.train()
        return replacement
    with pytest.raises(TransformationError, match=reason):
        apply_execution_plan(model, plan, registry=TransformRegistry((handler(forward=forward),)), **arguments)
    assert all(type(module) is nn.Linear for module in model.heads)
    torch.testing.assert_close(state(model), before, rtol=0, atol=0)


def test_preflight_checks_every_handler_target_owner_and_layout_before_callbacks():
    model = FixtureLM()
    calls = []
    registry = TransformRegistry((handler(forward=lambda module: calls.append(module)),))
    for targets, options, reason in [(("heads.0", "missing"), {}, "matched no modules"),
                                      (("heads.*",), {"parameter_layout_change": True}, "M8-F5"),
                                      (("embedding",), {}, "Unsupported target class")]:
        plan, arguments = inputs(model, targets=targets, **options)
        with pytest.raises(TransformationError, match=reason):
            apply_execution_plan(model, plan, registry=registry, **arguments)
    plan, arguments = inputs(model)
    with pytest.raises(TransformationError, match="Missing transform handler"):
        apply_execution_plan(model, plan, **arguments)
    with pytest.raises(TransformationError, match="ownership/inverse mismatch"):
        apply_execution_plan(model, plan, registry=TransformRegistry((handler(inverse_transform_id="wrong"),)), **arguments)
    assert calls == []


def test_stale_context_metadata_plan_and_existing_optimizer_are_rejected():
    model = FixtureLM()
    plan, arguments = inputs(model)
    registry = TransformRegistry((handler(),))
    with pytest.raises(TransformationError, match="before optimizer"):
        apply_execution_plan(model, plan, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                             registry=registry, **arguments)
    for changed in ({**arguments, "context": replace(arguments["context"], mask_kind="causal")},
                    {**arguments, "providers": (replace(arguments["providers"][0], priority=5),)}):
        with pytest.raises(TransformationError, match="stale or changed"):
            apply_execution_plan(model, plan, registry=registry, **changed)
    model.heads[0].weight = nn.Parameter(model.heads[0].weight.detach().clone())
    with pytest.raises(TransformationError, match="stale or changed"):
        apply_execution_plan(model, plan, registry=registry, **arguments)


def test_module_aliases_must_be_selected_together_and_remain_shared():
    model = FixtureLM()
    model.heads[1] = model.heads[0]
    plan, arguments = inputs(model, targets=("heads.0",))
    with pytest.raises(TransformationError, match="All aliases"):
        apply_execution_plan(model, plan, registry=TransformRegistry((handler(),)), **arguments)
    plan, arguments = inputs(model)
    original = model.heads[0]
    transaction = apply_execution_plan(model, plan, registry=TransformRegistry((handler(),)), **arguments)
    assert model.heads[0] is model.heads[1]
    transaction.revert()
    assert model.heads[0] is model.heads[1] is original


def test_undo_refuses_stale_tensor_ownership_or_failed_inverse_without_changing_live_modules():
    model = FixtureLM()
    plan, arguments = inputs(model)
    fail_inverse = [False]
    def inverse(module):
        if fail_inverse[0]:
            module.weight.data.zero_()
            raise RuntimeError("injected later inverse failure")
        return convert(module, nn.Linear)
    transaction = apply_execution_plan(model, plan, registry=TransformRegistry((handler(inverse=inverse),)), **arguments)
    modules = tuple(model.heads)
    before = state(model)
    fail_inverse[0] = True
    with pytest.raises(TransformationError, match="live replacements remain installed"):
        transaction.revert()
    assert tuple(model.heads) == modules and not transaction.is_reverted
    torch.testing.assert_close(state(model), before, rtol=0, atol=0)
    fail_inverse[0] = False
    model.heads[0].weight = nn.Parameter(model.heads[0].weight.detach().clone())
    with pytest.raises(TransformationError, match="Tensor ownership changed"):
        transaction.revert()
    assert tuple(model.heads) == modules


def test_disabled_and_reference_plans_are_noops_but_blocked_plans_never_apply():
    model = FixtureLM()
    before = tuple(model.heads)
    for policy in ("disabled", "auto"):
        arguments = dict(context=PlanningContext("cpu", "fp32", "none"),
                         requests=(OperationRequest("lm_head", "forward_backward"),), providers=())
        plan = plan_optimizations(inspect_dense_causal_lm(model), policy=policy, **arguments)
        transaction = apply_execution_plan(model, plan, **arguments)
        assert transaction.applied_transform_ids == () and tuple(model.heads) == before
        transaction.revert()
    blocked = plan_optimizations(inspect_dense_causal_lm(model), policy="required", **arguments)
    with pytest.raises(TransformationError, match="blocked execution plan"):
        apply_execution_plan(model, blocked, **arguments)


def test_existing_hooks_and_nonpersistent_target_buffers_survive_apply_and_undo():
    model = FixtureLM().eval()
    model.heads[0].register_buffer("scale", torch.ones(1), persistent=False)
    model.heads[0].register_parameter("optional", None)
    original_buffer = model.heads[0].scale
    calls = []
    def hook(module, args, kwargs, result):
        calls.append(type(module))
    handle = model.heads[0].register_forward_hook(hook, with_kwargs=True)
    plan, arguments = inputs(model)
    transaction = apply_execution_plan(model, plan, registry=TransformRegistry((handler(),)), **arguments)
    assert model.heads[0].scale is original_buffer
    assert "scale" in model.heads[0]._non_persistent_buffers_set
    assert model.heads[0].optional is None
    model(torch.tensor([[1, 2, 3]]))
    assert calls == [FixtureLinear]
    handle.remove()
    model(torch.tensor([[1, 2, 3]]))
    assert calls == [FixtureLinear]  # Existing handles still remove installed hooks.
    transaction.revert()
    assert model.heads[0].scale is original_buffer
    model(torch.tensor([[1, 2, 3]]))
    assert calls == [FixtureLinear]


def test_successful_handler_staging_does_not_advance_training_rng():
    model = FixtureLM()
    plan, arguments = inputs(model)
    def forward(module):
        torch.rand(4)
        random.random()
        return convert(module, FixtureLinear)
    before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
    transaction = apply_execution_plan(model, plan, registry=TransformRegistry((handler(forward=forward),)), **arguments)
    assert torch.equal(before_torch, torch.get_rng_state()) and before_python == random.getstate()
    transaction.revert()
    assert torch.equal(before_torch, torch.get_rng_state()) and before_python == random.getstate()


def test_overlapping_targets_are_rejected_before_any_handler_runs():
    model = FixtureLM()
    calls = []
    plan, arguments = inputs(model, targets=("heads", "heads.0"))
    broad_handler = handler(source_types=(nn.ModuleList, nn.Linear), forward=lambda module: calls.append(module))
    with pytest.raises(TransformationError, match="targets overlap"):
        apply_execution_plan(model, plan, registry=TransformRegistry((broad_handler,)), **arguments)
    assert calls == []


def test_external_shared_child_alias_cannot_be_broken_by_parent_replacement():
    model = FixtureLM()
    model.shared_head = model.heads[0]
    plan, arguments = inputs(model, targets=("heads",))
    original = model.heads
    # Copying the parent detaches its child module from the external alias,
    # even though rebinding parameters preserves every tensor alias.
    identity = handler(source_types=(nn.ModuleList,), forward=deepcopy, inverse=deepcopy)
    with pytest.raises(TransformationError, match="Shared module aliases changed"):
        apply_execution_plan(model, plan, registry=TransformRegistry((identity,)), **arguments)
    assert model.heads is original and model.heads[0] is model.shared_head
