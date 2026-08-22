"""Generic Hugging Face causal-language-model acquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from trainlm.config import ModelSourceConfig


@dataclass(frozen=True, slots=True)
class HuggingFaceModelMetadata:
    """Immutable provenance captured when a Hugging Face model is acquired."""

    initialization: str
    requested_source: str
    requested_revision: str | None
    resolved_revision: str | None
    local_source: bool
    model_type: str
    model_class: str
    config_class: str
    architectures: tuple[str, ...]
    requested_dtype: str | None
    resolved_dtype: str | None
    tied_parameter_groups: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedCausalLM:
    """An unchanged HF model together with its acquisition provenance."""

    model: PreTrainedModel
    config: PretrainedConfig
    metadata: HuggingFaceModelMetadata


class HuggingFaceCausalLMProvider:
    """Construct any AutoModelForCausalLM-supported architecture."""

    name = "huggingface"

    def load(self, source: ModelSourceConfig) -> LoadedCausalLM:
        if not isinstance(source, ModelSourceConfig):
            raise TypeError("source must be a ModelSourceConfig.")
        if source.provider != self.name:
            raise ValueError(
                "HuggingFaceCausalLMProvider requires 'model.provider: huggingface'."
            )

        config = self._load_config(source)
        model = self._load_model(source, config)

        # ``from_pretrained`` returns an eval-mode model while ``from_config``
        # returns a train-mode model. The training provider makes both paths
        # consistent without changing the implementation or parameter layout.
        model.train()

        return LoadedCausalLM(
            model=model,
            config=config,
            metadata=self._metadata(source, config, model),
        )

    def _load_config(self, source: ModelSourceConfig) -> PretrainedConfig:
        config_kwargs = dict(source.config_overrides)
        if source.dtype not in {None, "auto"}:
            config_kwargs["dtype"] = source.dtype

        if source.name_or_path is None:
            return AutoConfig.for_model(source.model_type, **config_kwargs)

        return AutoConfig.from_pretrained(
            source.name_or_path,
            **self._hub_kwargs(source),
            **config_kwargs,
        )

    def _load_model(
        self,
        source: ModelSourceConfig,
        config: PretrainedConfig,
    ) -> PreTrainedModel:
        model_kwargs: dict[str, Any] = {}
        if source.dtype is not None:
            # Keep dtype explicit at the model boundary. Transformers may
            # otherwise prefer a checkpoint/config dtype after AutoConfig.
            model_kwargs["dtype"] = source.dtype

        if source.initialization == "config":
            model_kwargs["trust_remote_code"] = source.trust_remote_code
            return AutoModelForCausalLM.from_config(config, **model_kwargs)

        if source.use_safetensors is not None:
            model_kwargs["use_safetensors"] = source.use_safetensors
        return AutoModelForCausalLM.from_pretrained(
            source.name_or_path,
            config=config,
            **self._hub_kwargs(source),
            **model_kwargs,
        )

    @staticmethod
    def _hub_kwargs(source: ModelSourceConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": source.trust_remote_code,
            "local_files_only": source.local_files_only,
        }
        for name in ("revision", "cache_dir", "subfolder"):
            value = getattr(source, name)
            if value is not None:
                kwargs[name] = value
        return kwargs

    @staticmethod
    def _metadata(
        source: ModelSourceConfig,
        config: PretrainedConfig,
        model: PreTrainedModel,
    ) -> HuggingFaceModelMetadata:
        source_name = source.name_or_path or f"model_type:{source.model_type}"
        architectures = getattr(config, "architectures", None) or ()
        resolved_revision = getattr(config, "_commit_hash", None)
        model_type = getattr(config, "model_type", None) or source.model_type

        return HuggingFaceModelMetadata(
            initialization=source.initialization,
            requested_source=source_name,
            requested_revision=source.revision,
            resolved_revision=resolved_revision,
            local_source=(
                source.name_or_path is not None
                and Path(source.name_or_path).expanduser().exists()
            ),
            model_type=str(model_type),
            model_class=type(model).__name__,
            config_class=type(config).__name__,
            architectures=tuple(str(item) for item in architectures),
            requested_dtype=source.dtype,
            resolved_dtype=_dtype_name(getattr(model, "dtype", None)),
            tied_parameter_groups=_tied_parameter_groups(model),
        )


def _dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def _tied_parameter_groups(model: PreTrainedModel) -> tuple[tuple[str, ...], ...]:
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    groups = (tuple(names) for names in aliases.values() if len(names) > 1)
    return tuple(sorted(groups))


def load_huggingface_causal_lm(source: ModelSourceConfig) -> LoadedCausalLM:
    """Load a generic HF causal LM through the default provider."""

    return HuggingFaceCausalLMProvider().load(source)
