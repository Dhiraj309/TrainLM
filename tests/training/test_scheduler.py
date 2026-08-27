"""Token-based WSD schedule tests."""

from __future__ import annotations

import pytest
import torch
from torch.optim import SGD

from trainlm.config import SchedulerConfig
from trainlm.training import SchedulerFactory, TokenWSD, create_scheduler


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
