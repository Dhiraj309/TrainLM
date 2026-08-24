"""Language-neutral schema validation for packed shard manifests."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from trainlm.data import plan_packed_batch_partition

from .test_partition import _reader
from .test_packed_binary_manifest import legacy_shard


REPOSITORY_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "schemas"
    / "data"
    / "packed_binary_shard_v1.schema.json"
)


def _schema(name):
    path = REPOSITORY_ROOT / "schemas" / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_packed_binary_manifest_matches_versioned_json_schema(tmp_path):
    _, manifest = legacy_shard(tmp_path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.validate(json.loads(manifest.to_json()), schema)


def test_batch_partition_matches_versioned_json_schema(tmp_path):
    reader = _reader(tmp_path, (16, 16))
    plan = plan_packed_batch_partition(
        reader,
        split="train",
        seed=11,
        epoch=2,
        world_size=2,
        rank=0,
    )

    jsonschema.validate(
        json.loads(plan.to_json()),
        _schema("batch_partition_v1.schema.json"),
    )
    reader.close()
