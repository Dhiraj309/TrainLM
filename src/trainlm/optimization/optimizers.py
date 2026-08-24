"""Backend-neutral optimizer construction and state-dtype policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from trainlm.config import OptimizerConfig

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class OptimizerStatePolicy:
    """Independent AdamW first- and second-moment dtype policy."""

    mu_dtype: torch.dtype
    nu_dtype: torch.dtype


class AdamWStateDtype(torch.optim.AdamW):
    """AdamW that keeps moment tensors in an explicit state dtype policy."""

    def __init__(self, *args: Any, state_policy: OptimizerStatePolicy, **kwargs: Any):
        self.state_policy = state_policy
        super().__init__(*args, **kwargs)

    def _cast_moment_state(self) -> None:
        for state in self.state.values():
            exp_avg = state.get("exp_avg")
            if isinstance(exp_avg, torch.Tensor):
                state["exp_avg"] = exp_avg.to(dtype=self.state_policy.mu_dtype)
            exp_avg_sq = state.get("exp_avg_sq")
            if isinstance(exp_avg_sq, torch.Tensor):
                state["exp_avg_sq"] = exp_avg_sq.to(
                    dtype=self.state_policy.nu_dtype
                )

    def step(self, closure=None):
        self._cast_moment_state()
        result = super().step(closure)
        self._cast_moment_state()
        return result


class OptimizerFactory:
    """Construct validated optimizers before backend preparation."""

    @staticmethod
    def create(
        parameters: Iterable[nn.Parameter],
        config: OptimizerConfig,
    ) -> AdamWStateDtype:
        if not isinstance(config, OptimizerConfig):
            raise TypeError("config must be an OptimizerConfig.")
        parameter_list = list(parameters)
        if not parameter_list:
            raise ValueError("OptimizerFactory requires at least one parameter.")
        if any(not isinstance(parameter, nn.Parameter) for parameter in parameter_list):
            raise TypeError("Optimizer parameters must be torch.nn.Parameter values.")
        if config.parameter_dtype is not None:
            expected_dtype = _TORCH_DTYPES[config.parameter_dtype]
            mismatched = [
                parameter.dtype
                for parameter in parameter_list
                if parameter.dtype != expected_dtype
            ]
            if mismatched:
                raise ValueError(
                    "Parameter dtype does not match optimizer parameter_dtype."
                )
        policy = OptimizerStatePolicy(
            mu_dtype=_TORCH_DTYPES[config.mu_dtype],
            nu_dtype=_TORCH_DTYPES[config.nu_dtype],
        )
        return AdamWStateDtype(
            parameter_list,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            fused=config.fused,
            state_policy=policy,
        )


def create_optimizer(
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> AdamWStateDtype:
    """Convenience wrapper for the public optimizer factory."""

    return OptimizerFactory.create(parameters, config)


__all__ = [
    "AdamWStateDtype",
    "OptimizerFactory",
    "OptimizerStatePolicy",
    "create_optimizer",
]
