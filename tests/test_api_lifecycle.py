"""HF-like facade lifecycle, cadence, metrics, and local resume."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from trainlm import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.training import TokenWSD, TrainerCallback


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.projection = torch.nn.Linear(4, 8)

    def forward(self, input_ids, labels=None, **kwargs):
        del labels, kwargs
        logits = self.projection(self.embedding(input_ids))
        return SimpleNamespace(logits=logits)


class MetricsRecorder(TrainerCallback):
    def __init__(self):
        self.metrics = []

    def on_metrics(self, state, control, metrics):
        del state, control
        self.metrics.append(dict(metrics))


def _dataset():
    return [
        {
            "input_ids": torch.tensor([0, 1, 2, 3]),
            "labels": torch.tensor([0, 1, 2, 3]),
        }
        for _ in range(4)
    ]


def test_facade_runs_eval_save_and_forwards_metrics(tmp_path):
    recorder = MetricsRecorder()
    trainer = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
        callbacks=[recorder],
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=1,
            sequence_length=4,
            accelerator="cpu",
            logging_steps=1,
            eval_steps=1,
            save_steps=1,
        ),
    )

    state = trainer.train()

    assert state.step == 1
    assert any("loss" in metrics for metrics in recorder.metrics)
    assert any("eval_loss" in metrics for metrics in recorder.metrics)
    assert (tmp_path / "checkpoint-1" / "trainer_state.pt").is_file()


def test_facade_resumes_local_training_state(tmp_path):
    first = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=1,
            sequence_length=4,
            accelerator="cpu",
            save_steps=1,
        ),
    )
    first.train()
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint_state = checkpoint / "trainer_state.pt"
    payload = torch.load(checkpoint_state, map_location="cpu", weights_only=False)
    payload.pop("optimizer_type")
    payload.pop("scheduler_type")
    torch.save(payload, checkpoint_state)

    resumed = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=2,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    state = resumed.train(resume_from_checkpoint=checkpoint)

    assert state.step == 2
    assert state.micro_step == 2


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("optimizer_type", "optimizer type does not match"),
        ("scheduler_type", "scheduler type does not match"),
        ("optimizer_groups", "parameter-group count does not match"),
        ("optimizer_slot_shape", "optimizer slot 'exp_avg' shape"),
        (
            "optimizer_step_shape",
            "optimizer state 'step' must be a non-negative scalar",
        ),
        ("scheduler_lr_count", "scheduler field 'base_lrs' must contain one value"),
        (
            "scheduler_epoch_type",
            "scheduler field 'last_epoch' has an incompatible value",
        ),
        ("scheduler_missing_field", "scheduler state is missing required fields"),
        (
            "trainer_epoch_negative",
            "trainer field 'epoch' must be a non-negative integer",
        ),
        (
            "trainer_tokens_bool",
            "trainer field 'tokens_seen' must be a non-negative integer",
        ),
        (
            "trainer_samples_type",
            "trainer field 'samples_seen' must be a non-negative integer",
        ),
        (
            "trainer_batch_negative",
            "trainer field 'global_batch_size' must be a non-negative integer",
        ),
        (
            "trainer_learning_rate_nan",
            "trainer field 'learning_rate' must be a finite non-negative scalar",
        ),
        ("trainer_loss_infinite", "trainer field 'loss' must be a finite scalar"),
        (
            "trainer_grad_norm_negative",
            "trainer field 'grad_norm' must be a finite non-negative scalar",
        ),
        ("trainer_phase", "trainer field 'phase' is not persisted"),
        ("trainer_is_training", "trainer field 'is_training' is not persisted"),
        ("trainer_should_stop", "trainer field 'should_stop' is not persisted"),
        ("trainer_failure", "trainer field 'failure' is not persisted"),
    ],
)
def test_resume_preflights_checkpoint_state_before_mutation(
    tmp_path, corruption, message
):
    first = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path / "source",
            max_steps=1,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    first.train()
    checkpoint = first.save_state(tmp_path / "checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if corruption == "optimizer_type":
        payload["optimizer_type"] = "torch.optim.SGD"
    elif corruption == "scheduler_type":
        payload["scheduler_type"] = "torch.optim.lr_scheduler.StepLR"
    elif corruption == "optimizer_slot_shape":
        parameter_id = next(iter(payload["optimizer"]["state"]))
        payload["optimizer"]["state"][parameter_id]["exp_avg"] = torch.zeros(1)
    elif corruption == "optimizer_step_shape":
        parameter_id = next(iter(payload["optimizer"]["state"]))
        payload["optimizer"]["state"][parameter_id]["step"] = torch.ones(2)
    elif corruption == "scheduler_lr_count":
        payload["scheduler"]["base_lrs"] = []
    elif corruption == "scheduler_epoch_type":
        payload["scheduler"]["last_epoch"] = "zero"
    elif corruption == "scheduler_missing_field":
        payload["scheduler"].pop("_last_lr")
    elif corruption == "trainer_epoch_negative":
        payload["trainer"]["epoch"] = -1
    elif corruption == "trainer_tokens_bool":
        payload["trainer"]["tokens_seen"] = True
    elif corruption == "trainer_samples_type":
        payload["trainer"]["samples_seen"] = "three"
    elif corruption == "trainer_batch_negative":
        payload["trainer"]["global_batch_size"] = -1
    elif corruption == "trainer_learning_rate_nan":
        payload["trainer"]["learning_rate"] = float("nan")
    elif corruption == "trainer_loss_infinite":
        payload["trainer"]["loss"] = float("inf")
    elif corruption == "trainer_grad_norm_negative":
        payload["trainer"]["grad_norm"] = -0.5
    elif corruption == "trainer_phase":
        payload["trainer"]["phase"] = "failed"
    elif corruption == "trainer_is_training":
        payload["trainer"]["is_training"] = True
    elif corruption == "trainer_should_stop":
        payload["trainer"]["should_stop"] = True
    elif corruption == "trainer_failure":
        payload["trainer"]["failure"] = "stale failure"
    else:
        payload["optimizer"]["param_groups"] = []
    torch.save(payload, checkpoint)

    resumed = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path / "resume",
            max_steps=2,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    before_model = {
        key: value.detach().clone()
        for key, value in resumed.model.state_dict().items()
    }
    before_scheduler = deepcopy(resumed.scheduler.state_dict())
    before_trainer_state = deepcopy(resumed.engine.state)

    with pytest.raises(ValueError, match=message):
        resumed._load_training_checkpoint(resumed.engine, checkpoint)

    assert all(
        torch.equal(before_model[key], value)
        for key, value in resumed.model.state_dict().items()
    )
    assert resumed.engine.optimizer.state == {}
    assert resumed.scheduler.state_dict() == before_scheduler
    assert resumed.engine.state == before_trainer_state


def test_token_scheduler_preflight_rejects_inconsistent_cursor_without_mutation():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = TokenWSD(
        optimizer,
        horizon_tokens=100,
        warmup_fraction=0.1,
        stable_fraction=0.5,
    )
    state = scheduler.state_dict()

    TrainLMTrainer._preflight_scheduler_state(
        scheduler, state, optimizer_group_count=1
    )
    corrupt = {**state, "stable_end_tokens": 80}

    with pytest.raises(ValueError, match="token scheduler state is inconsistent"):
        TrainLMTrainer._preflight_scheduler_state(
            scheduler, corrupt, optimizer_group_count=1
        )

    assert scheduler.last_tokens == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_resume_preflights_model_shapes_before_mutating_trainer(tmp_path):
    first = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path / "source",
            max_steps=1,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    first.train()
    checkpoint = first.save_state(tmp_path / "checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["projection.bias"] = torch.zeros(1)
    torch.save(payload, checkpoint)

    resumed = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path / "resume",
            max_steps=2,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    before = {
        key: value.detach().clone()
        for key, value in resumed.model.state_dict().items()
    }

    with pytest.raises(ValueError, match="model tensor shape mismatch"):
        resumed.train(resume_from_checkpoint=checkpoint)

    assert all(
        torch.equal(before[key], value)
        for key, value in resumed.model.state_dict().items()
    )
    assert resumed.engine.optimizer.state == {}
    assert resumed.engine.state.step == 0


def test_resume_preflights_runtime_state_before_mutating_trainer(tmp_path):
    first = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path / "source",
            max_steps=1,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    checkpoint = first.save_state(tmp_path / "checkpoint")

    resumed = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path / "resume",
            max_steps=2,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    before = {
        key: value.detach().clone()
        for key, value in resumed.model.state_dict().items()
    }
    validated = []

    def reject_runtime_state(state):
        validated.append(state)
        raise ValueError("runtime state is incompatible")

    resumed.runtime.validate_state_dict = reject_runtime_state

    with pytest.raises(ValueError, match="runtime state is incompatible"):
        resumed._load_training_checkpoint(resumed.engine, checkpoint)

    assert validated == [{}]
    assert all(
        torch.equal(before[key], value)
        for key, value in resumed.model.state_dict().items()
    )
    assert resumed.engine.optimizer.state == {}
    assert resumed.engine.state.step == 0


def test_manual_state_export_is_resumable(tmp_path):
    first = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=1,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    first.train()
    state_path = first.save_state(tmp_path / "manual-state")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    assert {"model", "optimizer", "scheduler", "runtime", "trainer"} <= payload.keys()

    resumed = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=2,
            sequence_length=4,
            accelerator="cpu",
        ),
    )
    state = resumed.train(resume_from_checkpoint=state_path)

    assert state.step == 2
    assert state.micro_step == 2


def test_cpu_resumed_weights_match_uninterrupted_four_step_run(tmp_path):
    initial_model = TinyModel()
    rows = _dataset()

    def make_trainer(model, max_steps, output_dir):
        return TrainLMTrainer(
            model=model,
            train_dataset=rows,
            args=TrainLMTrainingArguments(
                output_dir=output_dir,
                max_steps=max_steps,
                sequence_length=4,
                accelerator="cpu",
                lr_scheduler_type="constant",
                seed=42,
            ),
        )

    first = make_trainer(deepcopy(initial_model), 3, tmp_path / "resumed")
    assert first.train().step == 3
    checkpoint = first.save_state(tmp_path / "resumed" / "checkpoint")
    weights_after_three = {
        name: value.detach().clone() for name, value in first.model.state_dict().items()
    }

    resumed = make_trainer(deepcopy(initial_model), 4, tmp_path / "resumed")
    assert resumed.train(resume_from_checkpoint=checkpoint).step == 4

    uninterrupted = make_trainer(deepcopy(initial_model), 4, tmp_path / "reference")
    assert uninterrupted.train().step == 4

    resumed_weights = resumed.model.state_dict()
    reference_weights = uninterrupted.model.state_dict()
    assert resumed_weights.keys() == reference_weights.keys()
    assert any(
        not torch.equal(weights_after_three[name], value)
        for name, value in resumed_weights.items()
    )
    for name, value in resumed_weights.items():
        torch.testing.assert_close(value, reference_weights[name], rtol=0, atol=0)


class _IterableRows(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter(_dataset())


def _native_resume_validator(dataset, sampler, *, workers=0, generator=True):
    trainer = object.__new__(TrainLMTrainer)
    trainer.runtime = SimpleNamespace(name="torchtpu")
    trainer._train_loader_generator = object() if generator else None
    loader = SimpleNamespace(dataset=dataset, sampler=sampler, num_workers=workers)
    engine = SimpleNamespace(train_dataloader=loader)
    return trainer, engine


def test_native_resume_allows_owned_random_or_sequential_map_loaders():
    rows = _dataset()
    random_trainer, random_engine = _native_resume_validator(
        rows, RandomSampler(rows), generator=True
    )
    random_trainer._validate_native_resume_data(random_engine)

    sequential_trainer, sequential_engine = _native_resume_validator(
        rows, SequentialSampler(rows), generator=False
    )
    sequential_trainer._validate_native_resume_data(sequential_engine)


def test_native_resume_rejects_iterable_sources_workers_and_unknown_samplers():
    iterable = _IterableRows()
    trainer, engine = _native_resume_validator(
        iterable, SequentialSampler(_dataset()), generator=False
    )
    with pytest.raises(ValueError, match="map-style dataset"):
        trainer._validate_native_resume_data(engine)

    rows = _dataset()
    trainer, engine = _native_resume_validator(
        rows, SequentialSampler(rows), workers=1, generator=False
    )
    with pytest.raises(ValueError, match="dataloader_num_workers=0"):
        trainer._validate_native_resume_data(engine)

    trainer, engine = _native_resume_validator(rows, object())
    with pytest.raises(ValueError, match="custom sampler state"):
        trainer._validate_native_resume_data(engine)


def test_native_random_resume_requires_trainlm_owned_generator():
    rows = _dataset()
    trainer, engine = _native_resume_validator(
        rows, RandomSampler(rows), generator=False
    )

    with pytest.raises(ValueError, match="TrainLM-owned deterministic"):
        trainer._validate_native_resume_data(engine)
