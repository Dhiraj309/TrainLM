"""Token-based learning-rate schedules and scheduler construction."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, LambdaLR

from trainlm.config import SchedulerConfig


class TokenWSD(LRScheduler):
    """Warmup-stable-decay schedule indexed by cumulative consumed tokens."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        horizon_tokens: int,
        warmup_fraction: float = 0.0,
        stable_fraction: float = 1.0,
        min_lr_ratio: float = 0.0,
        last_tokens: int = 0,
    ) -> None:
        if horizon_tokens < 1:
            raise ValueError("horizon_tokens must be positive.")
        if not 0 <= warmup_fraction <= 1:
            raise ValueError("warmup_fraction must be between 0 and 1.")
        if not 0 <= stable_fraction <= 1:
            raise ValueError("stable_fraction must be between 0 and 1.")
        if warmup_fraction + stable_fraction > 1:
            raise ValueError("warmup_fraction + stable_fraction must be <= 1.")
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between 0 and 1.")
        if (
            isinstance(last_tokens, bool)
            or not isinstance(last_tokens, int)
            or last_tokens < 0
        ):
            raise ValueError("last_tokens must be non-negative.")
        self.horizon_tokens = horizon_tokens
        self.warmup_tokens = int(horizon_tokens * warmup_fraction)
        self.stable_tokens = int(horizon_tokens * stable_fraction)
        self.stable_end_tokens = self.warmup_tokens + self.stable_tokens
        self.min_lr_ratio = float(min_lr_ratio)
        self.last_tokens = last_tokens
        self._initializing = True
        super().__init__(optimizer, last_epoch=-1)
        self._initializing = False
        self.last_tokens = last_tokens
        self._last_lr = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr

    def _multiplier(self, tokens: int) -> float:
        if self.warmup_tokens > 0 and tokens < self.warmup_tokens:
            return tokens / self.warmup_tokens
        if tokens < self.stable_end_tokens:
            return 1.0
        decay_tokens = self.horizon_tokens - self.stable_end_tokens
        if decay_tokens <= 0 or tokens >= self.horizon_tokens:
            return self.min_lr_ratio
        progress = (tokens - self.stable_end_tokens) / decay_tokens
        return 1.0 - progress * (1.0 - self.min_lr_ratio)

    def get_lr(self) -> list[float]:
        multiplier = self._multiplier(self.last_tokens)
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def step_tokens(self, total_tokens: int) -> None:
        if (
            isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens < self.last_tokens
        ):
            raise ValueError("total_tokens must be monotonic non-negative integer.")
        self.last_tokens = total_tokens
        self.last_epoch += 1
        self._step_count += 1
        self._last_lr = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr

    def step(self, tokens: int | None = None) -> None:
        """Advance by explicit cumulative tokens or one legacy step."""

        if self._initializing:
            self.last_epoch = 0
            self._step_count = 1
            self._last_lr = self.get_lr()
            for group, lr in zip(self.optimizer.param_groups, self._last_lr):
                group["lr"] = lr
            return
        self.step_tokens(self.last_tokens + 1 if tokens is None else tokens)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._initializing = False
        self._last_lr = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr


class SchedulerFactory:
    """Create a scheduler from the versioned TrainLM scheduler config."""

    @staticmethod
    def create(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler:
        if not isinstance(config, SchedulerConfig):
            raise TypeError("config must be a SchedulerConfig.")
        if config.name == "wsd":
            return TokenWSD(
                optimizer,
                horizon_tokens=config.horizon_tokens,
                warmup_fraction=config.warmup_fraction,
                stable_fraction=config.stable_fraction,
                min_lr_ratio=config.min_lr_ratio,
            )
        if config.name == "constant":
            return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        if config.name == "linear":
            if config.horizon_steps is None:
                raise ValueError("Linear scheduler requires horizon_steps.")
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=config.min_lr_ratio,
                total_iters=config.horizon_steps,
            )
        if config.horizon_steps is None:
            raise ValueError("Cosine scheduler requires horizon_steps.")
        return CosineAnnealingLR(
            optimizer,
            T_max=config.horizon_steps,
            eta_min=config.min_lr_ratio,
        )


def create_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler:
    """Convenience wrapper for the public scheduler factory."""

    return SchedulerFactory.create(optimizer, config)


__all__ = ["SchedulerFactory", "TokenWSD", "create_scheduler"]
