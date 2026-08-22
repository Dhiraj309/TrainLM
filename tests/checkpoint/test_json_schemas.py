import json
from pathlib import Path

import jsonschema

from .test_export_contract import export_manifest
from .test_resume_contract import resume_manifest


REPOSITORY_ROOT = Path(__file__).parents[2]
SCHEMA_ROOT = REPOSITORY_ROOT / "schemas" / "checkpoint"


def schema(name):
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


def test_resume_manifest_matches_versioned_json_schema():
    jsonschema.validate(
        json.loads(resume_manifest().to_json()),
        schema("resume_manifest_v1.schema.json"),
    )


def test_hf_export_manifest_matches_versioned_json_schema():
    jsonschema.validate(
        json.loads(export_manifest().to_json()),
        schema("hf_export_manifest_v1.schema.json"),
    )

