import pytest
import torch
from torch import nn

from trainlm.optimization import (
    ExecutionPlan,
    ModelTransformation,
    ModelTransformRegistry,
    TransformApplicationError,
    TransformHandler,
)

from .test_capabilities import capabilities


def _transform(name, target, *, parameter_layout_change=True):
    return ModelTransformation(
        transform_id=name,
        component="projections",
        provider="fixture",
        target_paths=(target,),
        inverse_transform_id=f"undo-{name}",
        reason="fixture transform",
        parameter_layout_change=parameter_layout_change,
    )


def _plan(*transforms, status="ready"):
    return ExecutionPlan(
        1, "fixture-plan", status, "auto", capabilities().fingerprint,
        "pytorch", "fp32", transformations=transforms,
        errors=(("blocked",) if status == "blocked" else ()),
    )


def _handler(name, *, fail=False):
    def capture(model, specification):
        return getattr(model, specification.target_paths[0])

    def apply(model, specification):
        setattr(model, specification.target_paths[0], nn.Identity())
        if fail:
            raise RuntimeError("injected failure")

    def rollback(model, specification, snapshot):
        setattr(model, specification.target_paths[0], snapshot)

    return TransformHandler(name, f"undo-{name}", capture, apply, rollback)


def test_successful_transaction_can_commit_model_changes():
    model = nn.Sequential()
    model.first = nn.Linear(2, 2)
    original = model.first
    registry = ModelTransformRegistry()
    registry.register(_handler("replace"))

    transaction = registry.apply(model, _plan(_transform("replace", "first")))

    assert isinstance(model.first, nn.Identity)
    assert transaction.applied_transform_ids == ("replace",)
    assert transaction.commit() is model
    assert model.first is not original


def test_injected_failure_rolls_back_failing_and_prior_transforms():
    model = nn.Sequential()
    model.first = nn.Linear(2, 2)
    model.second = nn.Linear(2, 2)
    originals = (model.first, model.second)
    registry = ModelTransformRegistry()
    registry.register(_handler("first"))
    registry.register(_handler("second", fail=True))

    with pytest.raises(TransformApplicationError, match="rolled back"):
        registry.apply(
            model,
            _plan(_transform("first", "first"), _transform("second", "second")),
        )

    assert (model.first, model.second) == originals
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_context_manager_rolls_back_on_downstream_failure():
    model = nn.Sequential()
    model.first = nn.Linear(2, 2)
    original = model.first
    registry = ModelTransformRegistry()
    registry.register(_handler("replace"))

    with pytest.raises(RuntimeError, match="optimizer construction"):
        with registry.apply(model, _plan(_transform("replace", "first"))):
            raise RuntimeError("optimizer construction failed")

    assert model.first is original


def test_undeclared_parameter_alias_change_is_rolled_back():
    model = nn.Sequential()
    model.first = nn.Linear(2, 2)
    original = model.first
    registry = ModelTransformRegistry()
    registry.register(_handler("replace"))

    with pytest.raises(TransformApplicationError, match="parameter aliases"):
        registry.apply(
            model,
            _plan(_transform("replace", "first", parameter_layout_change=False)),
        )

    assert model.first is original


def test_missing_handler_and_blocked_plan_never_mutate_model():
    model = nn.Linear(2, 2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    registry = ModelTransformRegistry()

    with pytest.raises(TransformApplicationError, match="No registered handler"):
        registry.apply(model, _plan(_transform("missing", "weight")))
    with pytest.raises(TransformApplicationError, match="blocked execution plan"):
        registry.apply(model, _plan(status="blocked"))

    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
