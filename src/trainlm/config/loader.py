"""
YAML configuration loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .train import TrainConfig
from .checkpoint import CheckpointConfig
from .dataset import DatasetConfig, DatasetSource
from .evaluation import EvaluationConfig
from .loss import LossConfig
from .logging import LoggingConfig
from .model import ModelSourceConfig
from .monitoring import MonitoringConfig
from .optimization import OptimizationConfig
from .optimizer import OptimizerConfig
from .parallelism import ParallelismConfig
from .runtime import RuntimeConfig
from .scheduler import SchedulerConfig
from .trainer import TrainerConfig


def _normalize_path(path: str | Path) -> Path:
    """Convert a filesystem path to a Path object."""

    return path if isinstance(path, Path) else Path(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge dictionaries."""

    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = value

    return result


def _build_dataset(data: dict[str, Any]) -> DatasetConfig:
    sources = [
        DatasetSource(**source)
        for source in data.get("sources", [])
    ]

    kwargs = dict(data)
    kwargs["sources"] = sources

    return DatasetConfig(**kwargs)


_ROOT_SECTIONS = {
    "model", "dataset", "loss", "runtime", "parallelism",
    "optimizations", "optimizer", "scheduler", "trainer", "checkpoint",
    "logging", "monitoring", "evaluation",
}


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one validated top-level configuration mapping."""

    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return dict(value)


def _normalize_legacy_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate only legacy fields whose ownership is unambiguous."""

    normalized = dict(config)
    runtime = _section(normalized, "runtime")
    optimizations = _section(normalized, "optimizations")

    if "compile" in runtime:
        if "compile" in optimizations:
            raise ValueError(
                "Set compilation only in 'optimizations.compile'; the legacy "
                "'runtime.compile' value is ambiguous when both are present."
            )
        optimizations["compile"] = runtime.pop("compile")
        normalized["runtime"] = runtime
        normalized["optimizations"] = optimizations

    model = _section(normalized, "model")
    model_source_fields = {
        "provider", "initialization", "name_or_path", "model_type",
        "revision", "trust_remote_code", "config_overrides",
    }
    legacy_architecture_fields = set(model) - model_source_fields
    if legacy_architecture_fields:
        fields = ", ".join(sorted(legacy_architecture_fields))
        raise ValueError(
            "Model architecture fields are no longer owned by TrainConfig "
            f"({fields}). Select 'model.provider' explicitly and move them "
            "under 'model.config_overrides'."
        )

    return normalized


def load_config(
    base_config: str | Path,
    override_config: str | Path | None = None,
) -> TrainConfig:
    """
    Load a TrainConfig from YAML.
    """

    base = _load_yaml(
        _normalize_path(base_config)
    )

    if override_config is not None:
        override = _load_yaml(
            _normalize_path(override_config)
        )
        merged = _deep_merge(base, override)
    else:
        merged = base

    unknown_sections = set(merged) - _ROOT_SECTIONS
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"Unknown top-level configuration sections: {names}")

    merged = _normalize_legacy_fields(merged)

    config = TrainConfig(
        model=ModelSourceConfig(**_section(merged, "model")),
        dataset=_build_dataset(
            _section(merged, "dataset")
        ),
        loss=LossConfig(**_section(merged, "loss")),
        runtime=RuntimeConfig(**_section(merged, "runtime")),
        parallelism=ParallelismConfig(**_section(merged, "parallelism")),
        optimizations=OptimizationConfig(**_section(merged, "optimizations")),
        optimizer=OptimizerConfig(**_section(merged, "optimizer")),
        scheduler=SchedulerConfig(**_section(merged, "scheduler")),
        trainer=TrainerConfig(**_section(merged, "trainer")),
        checkpoint=CheckpointConfig(**_section(merged, "checkpoint")),
        logging=LoggingConfig(**_section(merged, "logging")),
        monitoring=MonitoringConfig(**_section(merged, "monitoring")),
        evaluation=EvaluationConfig(**_section(merged, "evaluation")),
    )

    config.validate()

    return config
