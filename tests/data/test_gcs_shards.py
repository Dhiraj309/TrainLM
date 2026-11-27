"""Mocked CPU tests for zero-egress same-region gs:// shard staging and loading."""

from __future__ import annotations

from pathlib import Path
import shutil
import struct

import pytest
import torch

from trainlm.data import PackedBinDataset, parse_gcs_uri, stage_gcs_prefix


class _FakeGCSFilesystem:
    def __init__(self, remote_root: Path, bucket: str) -> None:
        self._root = remote_root
        self._bucket = bucket
        self.get_calls: list[tuple[str, str]] = []

    def find(self, prefix: str) -> list[str]:
        rel_prefix = prefix.split("/", 1)[1] if "/" in prefix else ""
        base = self._root / rel_prefix
        results = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self._root).as_posix()
                results.append(f"{self._bucket}/{rel}")
        return results

    def get(self, remote_path: str, local_path: str) -> None:
        self.get_calls.append((remote_path, local_path))
        _, rel = remote_path.split("/", 1)
        shutil.copy2(self._root / rel, local_path)


def test_parse_gcs_uri_validation():
    assert parse_gcs_uri("gs://my-bucket/stage1/shards/") == ("my-bucket", "stage1/shards")
    with pytest.raises(ValueError):
        parse_gcs_uri("https://storage.googleapis.com/bucket")
    with pytest.raises(ValueError):
        parse_gcs_uri("gs://")


def test_packed_bin_dataset_from_gcs_raw_bins_and_cache_reuse(tmp_path: Path):
    remote = tmp_path / "remote_bucket" / "stage1"
    remote.mkdir(parents=True)
    tokens_1 = [1, 2, 3, 4, 5, 6, 7, 8]
    tokens_2 = [9, 10, 11, 12, 13, 14, 15, 16]
    (remote / "shard_00001.bin").write_bytes(struct.pack("<" + "H" * 8, *tokens_1))
    (remote / "shard_00002.bin").write_bytes(struct.pack("<" + "H" * 8, *tokens_2))

    fake_fs = _FakeGCSFilesystem(tmp_path / "remote_bucket", "trainlm-us-central1")
    cache_dir = tmp_path / "cache"

    ds = PackedBinDataset.from_directory(
        "gs://trainlm-us-central1/stage1",
        sequence_length=4,
        cache_dir=cache_dir,
        max_shards=1,
        header_bytes=0,
        vocab_size=64,
        filesystem=fake_fs,
    )
    assert len(ds.shards) == 1
    assert len(fake_fs.get_calls) == 1
    examples = list(ds)
    assert len(examples) == 2
    assert torch.equal(examples[0]["input_ids"], torch.tensor([1, 2, 3, 4], dtype=torch.long))

    # Re-staging should reuse already cached files with zero additional downloads.
    stage_gcs_prefix(
        "gs://trainlm-us-central1/stage1",
        cache_dir=cache_dir,
        max_shards=1,
        filesystem=fake_fs,
    )
    assert len(fake_fs.get_calls) == 1
