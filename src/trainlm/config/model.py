"""Model acquisition policy independent of model architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


_ACQUISITION_FIELDS = {
    "provider",
    "initialization",
    "name_or_path",
    "model_type",
    "revision",
    "trust_remote_code",
    "dtype",
    "cache_dir",
    "local_files_only",
    "subfolder",
    "use_safetensors",
}


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
    dtype: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    subfolder: str | None = None
    use_safetensors: bool | None = None
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
        misplaced_fields = _ACQUISITION_FIELDS & set(self.config_overrides)
        if misplaced_fields:
            names = ", ".join(sorted(misplaced_fields))
            raise ValueError(
                "Model acquisition fields must not appear in "
                f"'model.config_overrides': {names}."
            )
        for name in (
            "name_or_path",
            "model_type",
            "revision",
            "dtype",
            "cache_dir",
            "subfolder",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"'model.{name}' must be a non-empty string.")
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("'model.trust_remote_code' must be boolean.")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("'model.local_files_only' must be boolean.")
        if self.use_safetensors is not None and not isinstance(
            self.use_safetensors, bool
        ):
            raise ValueError("'model.use_safetensors' must be boolean when set.")
        if self.dtype == "auto" and self.initialization != "pretrained":
            raise ValueError(
                "'model.dtype: auto' is valid only for pretrained loading."
            )

        if self.provider == "external":
            if (
                self.name_or_path is not None
                or self.model_type is not None
                or self.revision is not None
                or self.trust_remote_code
                or self.dtype is not None
                or self.cache_dir is not None
                or self.local_files_only
                or self.subfolder is not None
                or self.use_safetensors is not None
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
        if self.subfolder is not None and self.name_or_path is None:
            raise ValueError("'model.subfolder' requires 'model.name_or_path'.")
        if self.local_files_only and self.name_or_path is None:
            raise ValueError(
                "'model.local_files_only' requires 'model.name_or_path'."
            )
        if self.use_safetensors is not None and self.initialization != "pretrained":
            raise ValueError(
                "'model.use_safetensors' is valid only for pretrained loading."
            )

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
