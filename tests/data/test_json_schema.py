"""Language-neutral schema validation for packed shard manifests."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from .test_packed_binary_manifest import legacy_shard


REPOSITORY_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "schemas"
    / "data"
    / "packed_binary_shard_v1.schema.json"
)


def test_packed_binary_manifest_matches_versioned_json_schema(tmp_path):
    _, manifest = legacy_shard(tmp_path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.validate(json.loads(manifest.to_json()), schema)
