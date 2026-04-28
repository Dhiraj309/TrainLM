import yaml
from pathlib import Path
from typing import Any, Dict, Union, Optional

from trainlm.config.schema import TrainConfig


# ------------------------------------------------------------
# YAML loading
# ------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML file into dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return data or {}


# ------------------------------------------------------------
# Deep merge
# ------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionaries.
    Override always takes precedence.
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# ------------------------------------------------------------
# Path normalization
# ------------------------------------------------------------

def _normalize_path(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        return Path(path)
    return path


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def load_config(
    base_config: Union[str, Path],
    override_config: Optional[Union[str, Path]] = None,
) -> TrainConfig:
    """
    Load and validate configuration.

    Supports:
    - base config
    - optional override config
    """

    base_config = _normalize_path(base_config)
    base_dict = _load_yaml(base_config)

    if override_config is not None:
        override_config = _normalize_path(override_config)
        override_dict = _load_yaml(override_config)
        merged = _deep_merge(base_dict, override_dict)
    else:
        merged = base_dict

    # Validate via Pydantic
    config = TrainConfig(**merged)

    return config
