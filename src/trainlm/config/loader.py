"""
YAML configuration loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .train import TrainConfig
from .configuration_trainlm import TrainLMConfig
from .checkpoint import CheckpointConfig
from .dataset import DatasetConfig, DatasetSource
from .evaluation import EvaluationConfig
from .logging import LoggingConfig
from .optimizer import OptimizerConfig
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

    return {} if data is None else data


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

    return TrainConfig(
        model=TrainLMConfig(
            **merged.get("model", {})
        ),
        dataset=_build_dataset(
            merged.get("dataset", {})
        ),
        runtime=RuntimeConfig(
            **merged.get("runtime", {})
        ),
        optimizer=OptimizerConfig(
            **merged.get("optimizer", {})
        ),
        scheduler=SchedulerConfig(
            **merged.get("scheduler", {})
        ),
        trainer=TrainerConfig(
            **merged.get("trainer", {})
        ),
        checkpoint=CheckpointConfig(
            **merged.get("checkpoint", {})
        ),
        logging=LoggingConfig(
            **merged.get("logging", {})
        ),
        evaluation=EvaluationConfig(
            **merged.get("evaluation", {})
        ),
    )
