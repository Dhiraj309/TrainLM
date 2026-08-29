"""Static input-shape contracts for compiled accelerator backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

import torch


def _describe(value: Any) -> tuple[Any, ...]:
    """Return a host-only, value-independent description of a batch pytree."""

    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.dtype),
            tuple(int(size) for size in value.shape),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__qualname__,
            tuple(
                (repr(key), _describe(item))
                for key, item in value.items()
            ),
        )
    if isinstance(value, tuple):
        return (
            "tuple",
            type(value).__qualname__,
            tuple(_describe(item) for item in value),
        )
    if isinstance(value, list):
        return ("list", tuple(_describe(item) for item in value))
    if value is None:
        return ("none",)
    if isinstance(value, (bool, int, float, str)):
        # Scalar values are intentionally excluded: changing a token count or
        # a metadata value must not create a new graph. Their type and location
        # remain part of the static pytree contract.
        return ("scalar", type(value).__qualname__)
    return ("object", type(value).__qualname__)


@dataclass(slots=True)
class StaticShapeGuard:
    """Reject graph-changing batch or accumulation structures after warmup."""

    _batch_description: tuple[Any, ...] | None = None
    _accumulation_steps: int | None = None

    def configure_accumulation_steps(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("accumulation steps must be a positive integer.")
        if (
            self._accumulation_steps is not None
            and self._accumulation_steps != steps
        ):
            raise RuntimeError(
                "XLA recompilation guard: gradient accumulation structure "
                f"changed from {self._accumulation_steps} to {steps}."
            )
        self._accumulation_steps = steps

    def observe_accumulation_steps(self, steps: int) -> None:
        """Check the actual microstep count used by a compiled update."""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("accumulation steps must be a positive integer.")
        if self._accumulation_steps is None:
            self._accumulation_steps = steps
        elif self._accumulation_steps != steps:
            raise RuntimeError(
                "XLA recompilation guard: actual microstep count "
                f"{steps} differs from configured {self._accumulation_steps}."
            )

    def observe_batch(self, batch: Any) -> None:
        description = _describe(batch)
        if self._batch_description is None:
            self._batch_description = description
            return
        if description != self._batch_description:
            raise RuntimeError(
                "XLA recompilation guard: batch structure or static shapes "
                "changed after warmup; use fixed batch/sequence/mask shapes "
                "and pad or drop incomplete batches."
            )

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (self._batch_description, self._accumulation_steps)
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_signature": repr(self._batch_description),
            "accumulation_steps": self._accumulation_steps,
            "fingerprint": self.fingerprint,
        }
