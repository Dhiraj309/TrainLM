"""AdamW factory and optimizer-state dtype policy tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from trainlm.config import OptimizerConfig
from trainlm.optimization import (
    AdamWStateDtype,
    OptimizerFactory,
    create_optimizer,
)


def test_optimizer_factory_constructs_decoupled_adamw_policy():
    model = nn.Linear(4, 2)
    config = OptimizerConfig(
        learning_rate=1e-3,
        weight_decay=0.1,
        fused=False,
        parameter_dtype="float32",
        mu_dtype="bfloat16",
        nu_dtype="float32",
    )

    optimizer = OptimizerFactory.create(model.parameters(), config)

    assert isinstance(optimizer, AdamWStateDtype)
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.1
    assert optimizer.state_policy.mu_dtype is torch.bfloat16
    assert optimizer.state_policy.nu_dtype is torch.float32


def test_optimizer_factory_casts_moment_state_after_update():
    parameter = nn.Parameter(torch.ones(2, dtype=torch.float32))
    optimizer = create_optimizer(
        [parameter],
        OptimizerConfig(
            learning_rate=1e-2,
            fused=False,
            mu_dtype="bfloat16",
            nu_dtype="float32",
        ),
    )

    (parameter.square().sum()).backward()
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype is torch.bfloat16
    assert state["exp_avg_sq"].dtype is torch.float32


def test_optimizer_config_rejects_unsupported_policy():
    with pytest.raises(ValueError, match="weight_decay"):
        OptimizerConfig(weight_decay=-1.0)
    with pytest.raises(ValueError, match="decoupled"):
        OptimizerConfig(decay_mode="l2")
    with pytest.raises(ValueError, match="mu_dtype"):
        OptimizerConfig(mu_dtype="int8")


def test_optimizer_factory_rejects_parameter_dtype_mismatch():
    parameter = nn.Parameter(torch.ones(2, dtype=torch.float32))

    with pytest.raises(ValueError, match="Parameter dtype"):
        create_optimizer(
            [parameter],
            OptimizerConfig(
                fused=False,
                parameter_dtype="bfloat16",
            ),
        )
