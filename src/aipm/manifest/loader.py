from pathlib import Path
from typing import Any, Dict

import yaml

from .models import Capability, ToolSpec, Permissions
from .validator import validate_manifest
from .errors import (
    ManifestLoadError,
    ManifestValidationError,
    ManifestStructureError,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ManifestLoadError(f"Failed to read manifest: {path} ({e})") from e


def _build_permissions(data: Dict[str, Any]) -> Permissions:
    if data is None:
        return Permissions()

    return Permissions(
        network=data.get("network", False),
        filesystem=data.get("filesystem", False),
        subprocess=data.get("subprocess", False),
    )


def _build_tools(data: Any) -> list[ToolSpec]:
    if not isinstance(data, list):
        raise ManifestStructureError("tools must be a list")

    tools: list[ToolSpec] = []

    for item in data:
        try:
            tool = ToolSpec(
                name=item["name"],
                description=item["description"],
                input_schema=item["input_schema"],
            )
        except KeyError as e:
            raise ManifestStructureError(
                f"Invalid tool definition, missing key: {e}"
            ) from e

        tools.append(tool)

    return tools


def load_manifest(path: str | Path) -> Capability:
    """
    Load and validate a capability manifest.

    Returns:
        Capability

    Raises:
        ManifestLoadError
        ManifestValidationError
        ManifestStructureError
    """
    path = Path(path)

    if not path.exists():
        raise ManifestLoadError(f"Manifest file not found: {path}")

    raw = _read_yaml(path)

    if not isinstance(raw, dict):
        raise ManifestStructureError("Manifest root must be a mapping/object")

    # Step 1: schema validation
    validate_manifest(raw)

    # Step 2: structural conversion
    try:
        capability = Capability(
            name=raw["name"],
            version=raw["version"],
            entrypoint=raw["entrypoint"],
            description=raw.get("description"),
            tools=_build_tools(raw["tools"]),
            permissions=_build_permissions(raw.get("permissions")),
            dependencies=raw.get("dependencies", []),
        )
    except KeyError as e:
        raise ManifestStructureError(
            f"Missing required field after validation: {e}"
        ) from e
    except Exception as e:
        raise ManifestStructureError(f"Failed to build capability: {e}") from e

    return capability
