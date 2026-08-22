"""Forward-signature-aware model batch dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from types import MappingProxyType
from typing import Any

from torch import nn


class ForwardSignatureError(ValueError):
    """A model forward contract cannot be safely dispatched by keyword."""


@dataclass(frozen=True, slots=True)
class ForwardSignature:
    """Stable keyword contract extracted from one model's bound forward."""

    model_class: str
    keyword_parameters: tuple[str, ...]
    required_parameters: tuple[str, ...]
    accepts_var_kwargs: bool


@dataclass(frozen=True, slots=True)
class BatchDispatch:
    """One filtered model call and the fields intentionally left behind."""

    inputs: Mapping[str, Any]
    forwarded_fields: tuple[str, ...]
    dropped_fields: tuple[str, ...]
    signature: ForwardSignature


class ForwardBatchDispatcher:
    """Analyze a model once and filter subsequent batches without family rules."""

    def __init__(
        self,
        signature: ForwardSignature,
        *,
        passthrough_fields: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(signature, ForwardSignature):
            raise TypeError("signature must be a ForwardSignature.")
        if any(not isinstance(name, str) or not name for name in passthrough_fields):
            raise ValueError("Passthrough field names must be non-empty strings.")
        if len(passthrough_fields) != len(set(passthrough_fields)):
            raise ValueError("Passthrough field names must be unique.")
        if passthrough_fields and not signature.accepts_var_kwargs:
            raise ForwardSignatureError(
                "Passthrough fields require a forward method with **kwargs."
            )
        self.signature = signature
        self.passthrough_fields = passthrough_fields
        self._accepted = frozenset(
            (*signature.keyword_parameters, *passthrough_fields)
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        passthrough_fields: tuple[str, ...] = (),
    ) -> "ForwardBatchDispatcher":
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")
        try:
            signature = inspect.signature(model.forward, follow_wrapped=True)
        except (TypeError, ValueError) as exc:
            raise ForwardSignatureError(
                f"Cannot inspect {type(model).__name__}.forward."
            ) from exc

        keywords = []
        required = []
        accepts_var_kwargs = False
        required_positional_only = []
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                accepts_var_kwargs = True
            elif parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                keywords.append(parameter.name)
                if parameter.default is inspect.Parameter.empty:
                    required.append(parameter.name)
            elif (
                parameter.kind is inspect.Parameter.POSITIONAL_ONLY
                and parameter.default is inspect.Parameter.empty
            ):
                required_positional_only.append(parameter.name)

        if required_positional_only:
            names = ", ".join(required_positional_only)
            raise ForwardSignatureError(
                "TrainLM dispatches model batches by keyword, but "
                f"{type(model).__name__}.forward requires positional-only: {names}."
            )

        return cls(
            ForwardSignature(
                model_class=(
                    f"{type(model).__module__}.{type(model).__qualname__}"
                ),
                keyword_parameters=tuple(keywords),
                required_parameters=tuple(required),
                accepts_var_kwargs=accepts_var_kwargs,
            ),
            passthrough_fields=passthrough_fields,
        )

    def dispatch(self, batch: Mapping[str, Any]) -> BatchDispatch:
        if not isinstance(batch, Mapping):
            raise TypeError("Model inputs must be a mapping.")
        if any(not isinstance(name, str) for name in batch):
            raise TypeError("Model input names must be strings.")

        forwarded = {
            name: value for name, value in batch.items()
            if name in self._accepted
        }
        dropped = tuple(name for name in batch if name not in self._accepted)

        missing = tuple(
            name for name in self.signature.required_parameters
            if name not in forwarded
        )
        if missing:
            names = ", ".join(missing)
            raise ForwardSignatureError(
                f"Model batch is missing required forward inputs: {names}."
            )

        return BatchDispatch(
            inputs=MappingProxyType(forwarded),
            forwarded_fields=tuple(forwarded),
            dropped_fields=dropped,
            signature=self.signature,
        )


def dispatch_model_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
) -> BatchDispatch:
    """Inspect and dispatch a single model batch."""

    return ForwardBatchDispatcher.from_model(model).dispatch(batch)
