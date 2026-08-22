"""Model acquisition policy independent of model architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True, slots=True)
class ModelSourceConfig:
    """Describe how a model is supplied without duplicating its HF config.

    ``config_overrides`` are passed to the selected model provider. They are
    deliberately not interpreted by the training configuration loader:
    Hugging Face ``PretrainedConfig`` remains authoritative for architecture.
    """

    provider: Literal["external", "huggingface", "trainlm"] = "external"
    initialization: Literal["config", "pretrained"] = "config"
    name_or_path: str | None = None
    model_type: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in {"external", "huggingface", "trainlm"}:
            raise ValueError(f"Unsupported model provider: {self.provider}")
        if self.initialization not in {"config", "pretrained"}:
            raise ValueError(
                f"Unsupported model initialization: {self.initialization}"
            )
        if not isinstance(self.config_overrides, Mapping):
            raise ValueError("'model.config_overrides' must be a mapping.")

        if self.provider == "external":
            if (
                self.name_or_path is not None
                or self.model_type is not None
                or self.revision is not None
                or self.trust_remote_code
                or self.config_overrides
                or self.initialization != "config"
            ):
                raise ValueError(
                    "The 'external' model provider accepts no loading or "
                    "architecture options; pass an already constructed model."
                )
            return

        if self.name_or_path is not None and self.model_type is not None:
            raise ValueError(
                "Set only one of 'model.name_or_path' and 'model.model_type'."
            )
        if self.revision is not None and self.name_or_path is None:
            raise ValueError("'model.revision' requires 'model.name_or_path'.")

        if self.initialization == "pretrained" and self.name_or_path is None:
            raise ValueError(
                "'model.initialization: pretrained' requires "
                "'model.name_or_path'."
            )

        if self.provider == "huggingface":
            if self.name_or_path is None and self.model_type is None:
                raise ValueError(
                    "The Hugging Face provider requires 'model.name_or_path' "
                    "or 'model.model_type'."
                )
        else:
            if self.model_type is not None:
                raise ValueError(
                    "'model.model_type' is owned by the Hugging Face provider; "
                    "omit it when 'model.provider: trainlm'."
                )
            if self.trust_remote_code:
                raise ValueError(
                    "'model.trust_remote_code' is only valid for the "
                    "Hugging Face provider."
                )
