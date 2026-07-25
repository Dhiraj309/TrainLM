from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn


class Runtime:
    """Execution backend for model training.

    Runtime encapsulates execution-specific concerns such as automatic mixed
    precision, distributed execution, compilation, and synchronization.

    The default implementation provides a pass-through backend suitable for
    single-device eager execution.
    """

    def prepare_model(self, model: nn.Module) -> nn.Module:
        return model

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def autocast(self):
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def synchronize(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict
