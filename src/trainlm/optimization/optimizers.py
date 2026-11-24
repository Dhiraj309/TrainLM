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


def _compiler_disable(fn):
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    return disable(fn) if callable(disable) else fn


class AdamWStateDtype(torch.optim.AdamW):
    """AdamW that keeps moment tensors in an explicit state dtype policy."""

    def __init__(self, *args: Any, state_policy: OptimizerStatePolicy, **kwargs: Any):
        self.state_policy = state_policy
        super().__init__(*args, **kwargs)

    def _cast_moment_state(self) -> None:
        for state in self.state.values():
            exp_avg = state.get("exp_avg")
            if (
                isinstance(exp_avg, torch.Tensor)
                and exp_avg.dtype != self.state_policy.mu_dtype
            ):
                state["exp_avg"] = exp_avg.to(dtype=self.state_policy.mu_dtype)
            exp_avg_sq = state.get("exp_avg_sq")
            if (
                isinstance(exp_avg_sq, torch.Tensor)
                and exp_avg_sq.dtype != self.state_policy.nu_dtype
            ):
                state["exp_avg_sq"] = exp_avg_sq.to(
                    dtype=self.state_policy.nu_dtype
                )

    def _prepare_step_dtypes(self) -> None:
        """Make in-place AdamW kernels dtype-compatible with current grads.

        PyTorch/XLA's single-tensor AdamW path requires lerp_ and addcmul_
        operands to have matching dtypes. We keep the requested compact state
        policy between steps, but temporarily use each parameter's gradient
        dtype during the in-place optimizer update.
        """
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                state = self.state.get(parameter)
                if gradient is None or not state:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    value = state.get(key)
                    if (
                        isinstance(value, torch.Tensor)
                        and value.dtype != gradient.dtype
                    ):
                        state[key] = value.to(dtype=gradient.dtype)

    @_compiler_disable
    def step(self, closure=None):
        if (
            self.defaults.get("fused", False)
            and self.state_policy.mu_dtype == torch.float32
            and self.state_policy.nu_dtype == torch.float32
        ):
            return super().step(closure)
        self._cast_moment_state()
        self._prepare_step_dtypes()
        result = super().step(closure)
        self._cast_moment_state()
        return result


class OptimizerFactory:
    """Construct validated optimizers before backend preparation."""

    @staticmethod
    def create(
        parameters: Iterable[nn.Parameter],
        config: OptimizerConfig,
    ) -> torch.optim.Optimizer:
        if not isinstance(config, OptimizerConfig):
            raise TypeError("config must be an OptimizerConfig.")
        parameter_list = list(parameters)
        if not parameter_list:
            raise ValueError("OptimizerFactory requires at least one parameter.")
        if any(not isinstance(parameter, nn.Parameter) for parameter in parameter_list):
            raise TypeError("Optimizer parameters must be torch.nn.Parameter values.")
        trainable = [parameter for parameter in parameter_list if parameter.requires_grad]
        if trainable:
            parameter_list = trainable
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
        if config.name in {"adamw_8bit", "paged_adamw_8bit"}:
            return _create_bnb_adamw8bit(parameter_list, config)
        if config.name == "torchao_adamw_8bit":
            return _create_torchao_adamw8bit(parameter_list, config)
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


def _create_bnb_adamw8bit(
    parameters: list[nn.Parameter],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"optim={config.name!r} requires the 'bitsandbytes' package "
            "(pip install bitsandbytes)."
        ) from exc
    optim_cls = (
        bnb.optim.PagedAdamW8bit
        if config.name == "paged_adamw_8bit"
        else bnb.optim.AdamW8bit
    )
    return optim_cls(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def _create_torchao_adamw8bit(
    parameters: list[nn.Parameter],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    optim_cls = None
    try:
        from torchao.optim import AdamW8bit  # type: ignore[import-not-found]

        optim_cls = AdamW8bit
    except ModuleNotFoundError:
        try:
            from torchao.prototype.low_bit_optim import AdamW8bit  # type: ignore[import-not-found]

            optim_cls = AdamW8bit
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "optim='torchao_adamw_8bit' requires the 'torchao' package "
                "(pip install torchao)."
            ) from exc
    return optim_cls(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def create_optimizer(
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Convenience wrapper for the public optimizer factory."""

    return OptimizerFactory.create(parameters, config)


__all__ = [
    "AdamWStateDtype",
    "OptimizerFactory",
    "OptimizerStatePolicy",
    "create_optimizer",
]
