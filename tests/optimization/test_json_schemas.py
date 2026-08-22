import json
from pathlib import Path

import jsonschema

from .test_capabilities import capabilities
from .test_execution_plan import plan


REPOSITORY_ROOT = Path(__file__).parents[2]
SCHEMA_ROOT = REPOSITORY_ROOT / "schemas" / "optimization"


def _schema(name):
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


def test_capabilities_match_versioned_json_schema():
    jsonschema.validate(
        json.loads(capabilities().to_json()),
        _schema("model_capabilities_v1.schema.json"),
    )


def test_execution_plan_matches_versioned_json_schema():
    jsonschema.validate(
        json.loads(plan().to_json()),
        _schema("execution_plan_v1.schema.json"),
    )

