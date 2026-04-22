import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft7Validator

from .errors import ManifestValidationError


SCHEMA_PATH = Path(__file__).resolve().parents[3] / "spec" / "capability.schema.json"


def _load_schema() -> Dict[str, Any]:
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ManifestValidationError(f"Failed to load schema: {e}") from e


# Load once at import time
_SCHEMA = _load_schema()
_VALIDATOR = Draft7Validator(_SCHEMA)


def validate_manifest(data: Dict[str, Any]) -> None:
    """
    Validate a manifest dict against the capability schema.

    Raises:
        ManifestValidationError: if validation fails
    """
    errors = sorted(_VALIDATOR.iter_errors(data), key=lambda e: e.path)

    if not errors:
        return

    messages = []
    for err in errors:
        path = ".".join(str(p) for p in err.path)
        if path:
            messages.append(f"{path}: {err.message}")
        else:
            messages.append(err.message)

    raise ManifestValidationError(
        "Manifest validation failed:\n" + "\n".join(messages)
    )
