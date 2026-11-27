"""Mocked unit tests for zero-egress same-region gs:// staging, manifests, checksums, and mmap."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import shutil
import struct
from types import SimpleNamespace

import pytest
import torch

from trainlm.data import (
    ContiguousPackedBatchReader,
    PackedBinaryShardManifest,
    PackedBinDataset,
    stage_gcs_prefix,
)


def _write_explicit_v1_shard(directory: Path, shard_id: str, tokens: list[int]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    bin_name = f"{shard_id}.bin"
    bin_path = directory / bin_name
    raw = struct.pack("<" + "H" * len(tokens), *tokens)
    bin_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = PackedBinaryShardManifest(
        schema_version=1,
        shard_id=shard_id,
        data_path=bin_name,
        compatibility_profile="explicit_v1",
        header_bytes=0,
        token_dtype="uint16",
        byte_order="little",
        token_count=len(tokens),
        token_id_min=min(tokens),
        token_id_max=max(tokens),
        vocab_size=256,
        file_size_bytes=len(raw),
        sha256=digest,
    )
    manifest_path = directory / f"{shard_id}.manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest_path


class _MockGCSFs:
    def __init__(self, remote_root: Path, bucket: str) -> None:
        self._root = remote_root
        self._bucket = bucket
        self.get_calls: list[tuple[str, str]] = []

    def find(self, prefix: str) -> list[str]:
        rel_prefix = prefix.split("/", 1)[1] if "/" in prefix else ""
        base = self._root / rel_prefix
        return [
            f"{self._bucket}/{p.relative_to(self._root).as_posix()}"
            for p in sorted(base.rglob("*"))
            if p.is_file()
        ]

    def get(self, remote_path: str, local_path: str) -> None:
        self.get_calls.append((remote_path, local_path))
        _, rel = remote_path.split("/", 1)
        shutil.copy2(self._root / rel, local_path)


def test_gcs_gcloud_cli_staging_and_same_region_reminder(tmp_path: Path, capsys, caplog):
    remote = tmp_path / "gcs_remote" / "stage1"
    remote.mkdir(parents=True)
    (remote / "shard_00002.bin").write_bytes(struct.pack("<HHHH", 10, 11, 12, 13))
    raw_1 = struct.pack("<HHHH", 1, 2, 3, 4)
    (remote / "shard_00001.bin").write_bytes(raw_1)
    (remote / "shard_00001.bin.sha256").write_text(
        f"{hashlib.sha256(raw_1).hexdigest()}  shard_00001.bin\n",
        encoding="utf-8",
    )

    commands: list[list[str]] = []

    def fake_runner(cmd: list[str], **_kwargs):
        commands.append(cmd)
        if cmd[1:3] == ["storage", "ls"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "gs://trainlm-us-central1/stage1/shard_00002.bin\n"
                    "gs://trainlm-us-central1/stage1/shard_00001.bin\n"
                    "gs://trainlm-us-central1/stage1/shard_00001.bin.sha256\n"
                ),
            )
        if cmd[1:3] == ["storage", "cp"]:
            src, dst = cmd[3], Path(cmd[4])
            rel = src[len("gs://trainlm-us-central1/") :]
            shutil.copy2(tmp_path / "gcs_remote" / rel, dst)
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"Unexpected command: {cmd}")

    caplog.set_level(logging.INFO)
    ds = PackedBinDataset.from_gcs(
        "gs://trainlm-us-central1/stage1",
        sequence_length=2,
        cache_dir=tmp_path / "local_cache",
        max_shards=1,
        header_bytes=0,
        vocab_size=64,
        which=lambda name: "/usr/bin/gcloud" if name == "gcloud" else None,
        runner=fake_runner,
    )

    captured = capsys.readouterr().out + "\n" + caplog.text
    assert "Single-Region" in captured
    assert "SAME region" in captured
    assert "$0.00" in captured
    assert any(cmd[1:3] == ["storage", "cp"] for cmd in commands)
    assert len(ds.shards) == 1 and ds.shards[0].shard_id == "shard_00001"

    reader = ContiguousPackedBatchReader(ds.shards, batch_size=2, sequence_length=2)
    batch = reader.read_batch(0)
    assert tuple(batch["input_ids"].shape) == (2, 2)


def test_gcs_gsutil_fallback_and_sha256_corruption_detection(tmp_path: Path):
    remote = tmp_path / "remote" / "train"
    remote.mkdir(parents=True)
    raw = struct.pack("<HHHH", 1, 2, 3, 4)
    (remote / "s0.bin").write_bytes(raw)
    (remote / "s0.bin.sha256").write_text("0" * 64 + "  s0.bin\n", encoding="utf-8")

    def fake_runner(cmd: list[str], **_kwargs):
        assert cmd[0] == "/usr/bin/gsutil"
        dst = Path(cmd[-1])
        for src_file in remote.iterdir():
            shutil.copy2(src_file, dst / src_file.name)
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage_gcs_prefix(
            "gs://bucket-us-east1/train",
            cache_dir=tmp_path / "cache",
            which=lambda name: "/usr/bin/gsutil" if name == "gsutil" else None,
            runner=fake_runner,
        )


def test_gcs_explicit_v1_manifest_deterministic_ordering(tmp_path: Path):
    remote = tmp_path / "remote" / "v1"
    _write_explicit_v1_shard(remote, "shard_00002", [5, 6, 7, 8])
    _write_explicit_v1_shard(remote, "shard_00001", [1, 2, 3, 4])

    fake_fs = _MockGCSFs(tmp_path / "remote", "trainlm-v6e-us-central1")
    ds = PackedBinDataset.from_gcs(
        "gs://trainlm-v6e-us-central1/v1",
        sequence_length=4,
        cache_dir=tmp_path / "cache",
        max_shards=1,
        filesystem=fake_fs,
    )
    assert [s.shard_id for s in ds.shards] == ["shard_00001"]
    batch = next(iter(ds))
    assert torch.equal(batch["input_ids"], torch.tensor([1, 2, 3, 4], dtype=torch.long))


def test_gcs_distributed_staging_writes_marker_on_rank0_only(tmp_path: Path, monkeypatch):
    remote = tmp_path / "remote" / "dist"
    _write_explicit_v1_shard(remote, "shard_00001", [1, 2, 3, 4])
    fake_fs = _MockGCSFs(tmp_path / "remote", "trainlm-us-central1")
    cache_dir = tmp_path / "dist_cache"

    barriers = []
    state = {"rank": 0}
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 8,
        get_rank=lambda: state["rank"],
        barrier=lambda: barriers.append(state["rank"]),
    )
    monkeypatch.setattr(torch, "distributed", fake_dist, raising=False)

    # Rank 0 downloads/verifies and writes .staged.ok before barrier.
    monkeypatch.setenv("LOCAL_RANK", "0")
    local_root = stage_gcs_prefix(
        "gs://trainlm-us-central1/dist",
        cache_dir=cache_dir,
        filesystem=fake_fs,
    )
    assert (local_root / ".staged.ok").is_file()
    assert len(fake_fs.get_calls) == 2
    assert barriers == [0]

    # Worker rank 3 skips download completely and checks .staged.ok after barrier.
    state["rank"] = 3
    monkeypatch.setenv("LOCAL_RANK", "3")
    worker_root = stage_gcs_prefix(
        "gs://trainlm-us-central1/dist",
        cache_dir=cache_dir,
        filesystem=fake_fs,
    )
    assert worker_root == local_root
    assert len(fake_fs.get_calls) == 2  # zero additional downloads
    assert barriers == [0, 3]

