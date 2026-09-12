"""Token-based WSD schedule tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.optim import SGD

from trainlm.config import SchedulerConfig
from trainlm.training import SchedulerFactory, TokenWSD, create_scheduler
from trainlm.training import Trainer


def _scheduler(**kwargs):
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=0.1)
    return optimizer, TokenWSD(optimizer, **kwargs)


def test_wsd_warmup_stable_decay_uses_cumulative_tokens():
    optimizer, scheduler = _scheduler(
        horizon_tokens=100,
        warmup_fraction=0.1,
        stable_fraction=0.5,
        min_lr_ratio=0.2,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    scheduler.step_tokens(5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    scheduler.step_tokens(10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    scheduler.step_tokens(60)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    scheduler.step_tokens(80)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.06)
    scheduler.step_tokens(100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    scheduler.step_tokens(200)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)


def test_wsd_requires_monotonic_token_progress_and_restores_state():
    optimizer, scheduler = _scheduler(horizon_tokens=100)
    scheduler.step_tokens(25)
    state = scheduler.state_dict()
    with pytest.raises(ValueError, match="monotonic"):
        scheduler.step_tokens(24)

    restored_optimizer, restored = _scheduler(horizon_tokens=100)
    restored.load_state_dict(state)
    assert restored.last_tokens == 25
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )


def test_trainer_advances_wsd_by_global_dp_tokens():
    optimizer, scheduler = _scheduler(
        horizon_tokens=1000, warmup_fraction=0.8, stable_fraction=0.2,
    )
    trainer = SimpleNamespace(
        scheduler=scheduler, state=SimpleNamespace(tokens_seen=50),
        runtime=SimpleNamespace(world_size=8),
    )
    Trainer._advance_scheduler(trainer, total_tokens=50)
    assert scheduler.last_tokens == 400
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_scheduler_factory_builds_wsd_and_rejects_missing_horizon():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=0.1)
    scheduler = SchedulerFactory.create(
        optimizer,
        SchedulerConfig(
            name="wsd",
            horizon_tokens=1_000,
            warmup_fraction=0.01,
            stable_fraction=0.95,
            min_lr_ratio=0.05,
        ),
    )
    assert isinstance(scheduler, TokenWSD)
    assert scheduler.horizon_tokens == 1_000

    with pytest.raises(ValueError, match="horizon_tokens"):
        SchedulerConfig(name="wsd")

    assert isinstance(
        create_scheduler(optimizer, SchedulerConfig(name="constant")),
        torch.optim.lr_scheduler.LRScheduler,
    )


@pytest.mark.parametrize(
    ("tokens", "expected_multiplier"),
    (
        (0, 0.0),
        (1, 0.1),
        (9, 0.9),
        (10, 1.0),
        (59, 1.0),
        (60, 1.0),
        (61, 0.98),
        (80, 0.6),
        (99, 0.22),
        (100, 0.2),
        (101, 0.2),
    ),
)
def test_wsd_boundaries_and_resume_are_exact(tokens, expected_multiplier):
    optimizer, scheduler = _scheduler(
        horizon_tokens=100,
        warmup_fraction=0.1,
        stable_fraction=0.5,
        min_lr_ratio=0.2,
    )
    scheduler.step_tokens(tokens)
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()

    resumed_optimizer, resumed = _scheduler(
        horizon_tokens=100,
        warmup_fraction=0.1,
        stable_fraction=0.5,
        min_lr_ratio=0.2,
    )
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed.load_state_dict(saved_scheduler)

    expected_lr = 0.1 * expected_multiplier
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert resumed.last_tokens == tokens
    assert resumed.state_dict() == saved_scheduler


def test_wsd_resume_produces_identical_future_schedule():
    optimizer, uninterrupted = _scheduler(
        horizon_tokens=1_000,
        warmup_fraction=0.05,
        stable_fraction=0.8,
        min_lr_ratio=0.1,
    )
    for tokens in (17, 50, 411):
        uninterrupted.step_tokens(tokens)

    resumed_optimizer, resumed = _scheduler(
        horizon_tokens=1_000,
        warmup_fraction=0.05,
        stable_fraction=0.8,
        min_lr_ratio=0.1,
    )
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed.load_state_dict(uninterrupted.state_dict())

    for tokens in (849, 850, 925, 1_000, 1_250):
        uninterrupted.step_tokens(tokens)
        resumed.step_tokens(tokens)
        assert resumed.get_last_lr() == pytest.approx(uninterrupted.get_last_lr())
        assert resumed.state_dict() == uninterrupted.state_dict()
