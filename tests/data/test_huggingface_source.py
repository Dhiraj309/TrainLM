"""Deterministic Hugging Face packed-shard source resolution."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import struct

import pytest

from trainlm.data import (
    HuggingFacePackedShardSource,
    HuggingFaceShardSourceConfig,
    HuggingFaceShardSpec,
    PackedBinaryShardManifest,
)


REVISION = "a" * 40


class FakeHubDownload:
    def __init__(self, files: dict[str, Path]) -> None:
        self.files = files
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return str(self.files[kwargs["filename"]])


def _add_legacy_shard(
    root: Path,
    *,
    shard_id: str,
    tokens: tuple[int, ...],
) -> tuple[HuggingFaceShardSpec, dict[str, Path]]:
    data_name = f"fineweb-edu/{shard_id}.bin"
    manifest_name = f"fineweb-edu/{shard_id}.manifest.json"
    payload = bytes(1024) + struct.pack(f"<{len(tokens)}H", *tokens)
    data_file = root / data_name
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_bytes(payload)
    manifest = PackedBinaryShardManifest(
        schema_version=1,
        shard_id=shard_id,
        data_path=data_name,
        compatibility_profile="legacy_1024_uint16",
        header_bytes=1024,
        token_dtype="uint16",
        byte_order="little",
        token_count=len(tokens),
        token_id_min=min(tokens),
        token_id_max=max(tokens),
        vocab_size=32,
        file_size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest_file = root / manifest_name
    manifest_file.write_text(manifest.to_json(), encoding="utf-8")
    return (
        HuggingFaceShardSpec(shard_id, manifest_name),
        {manifest_name: manifest_file, data_name: data_file},
    )


def test_source_resolves_in_declared_order_with_pinned_hub_arguments(tmp_path):
    first, first_files = _add_legacy_shard(
        tmp_path,
        shard_id="train-00001",
        tokens=(1, 2, 3),
    )
    second, second_files = _add_legacy_shard(
        tmp_path,
        shard_id="train-00000",
        tokens=(0, 4, 31),
    )
    downloader = FakeHubDownload({**first_files, **second_files})
    config = HuggingFaceShardSourceConfig(
        repo_id="LaughTaleAI/LaughLM-Tokenized-Fine",
        revision=REVISION,
        shards=(first, second),
        cache_dir=tmp_path / "hub-cache",
        local_files_only=True,
    )

    resolved = HuggingFacePackedShardSource(
        config,
        download_file=downloader,
    ).resolve()

    assert tuple(item.shard_id for item in resolved) == (
        "train-00001",
        "train-00000",
    )
    assert tuple(item.validation.token_count for item in resolved) == (3, 3)
    assert all(item.revision == REVISION for item in resolved)
    assert [call["filename"] for call in downloader.calls] == [
        first.manifest_path,
        "fineweb-edu/train-00001.bin",
        second.manifest_path,
        "fineweb-edu/train-00000.bin",
    ]
    for call in downloader.calls:
        assert call["repo_id"] == config.repo_id
        assert call["repo_type"] == "dataset"
        assert call["revision"] == REVISION
        assert call["cache_dir"] == config.cache_dir
        assert call["local_files_only"] is True
        assert call["library_name"] == "trainlm"
        assert "token" not in call

    repeated_downloader = FakeHubDownload({**first_files, **second_files})
    repeated = HuggingFacePackedShardSource(
        config,
        download_file=repeated_downloader,
    ).resolve()
    assert tuple(item.data_path for item in repeated) == tuple(
        item.data_path for item in resolved
    )
    assert repeated_downloader.calls == downloader.calls


def test_offline_resolution_reuses_only_cached_files(tmp_path):
    spec, files = _add_legacy_shard(
        tmp_path,
        shard_id="validation-00000",
        tokens=(2, 3, 4),
    )

    def cached_only(**kwargs):
        assert kwargs["local_files_only"] is True
        return str(files[kwargs["filename"]])

    source = HuggingFacePackedShardSource(
        HuggingFaceShardSourceConfig(
            repo_id="owner/dataset",
            revision=REVISION,
            shards=(spec,),
            local_files_only=True,
        ),
        download_file=cached_only,
    )

    assert source.resolve()[0].shard_id == "validation-00000"


def test_source_rejects_identity_mismatch_and_corrupt_payload(tmp_path):
    spec, files = _add_legacy_shard(
        tmp_path,
        shard_id="train-00000",
        tokens=(0, 1, 2),
    )
    downloader = FakeHubDownload(files)
    config = HuggingFaceShardSourceConfig(
        repo_id="owner/dataset",
        revision=REVISION,
        shards=(spec,),
    )

    manifest_file = files[spec.manifest_path]
    manifest = PackedBinaryShardManifest.from_json(
        manifest_file.read_text(encoding="utf-8")
    )
    manifest_file.write_text(
        replace(manifest, shard_id="different").to_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        HuggingFacePackedShardSource(
            config,
            download_file=downloader,
        ).resolve()

    manifest_file.write_text(manifest.to_json(), encoding="utf-8")
    data_file = files[manifest.data_path]
    corrupted = bytearray(data_file.read_bytes())
    corrupted[-1] ^= 1
    data_file.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum"):
        HuggingFacePackedShardSource(
            config,
            download_file=downloader,
        ).resolve()


def test_source_config_requires_safe_unique_commit_pinned_shards():
    spec = HuggingFaceShardSpec("train-00000", "data/train-00000.json")

    with pytest.raises(ValueError, match="commit SHA"):
        HuggingFaceShardSourceConfig("owner/data", "main", (spec,))
    with pytest.raises(ValueError, match="owner/name"):
        HuggingFaceShardSourceConfig("invalid", REVISION, (spec,))
    with pytest.raises(ValueError, match="unique"):
        HuggingFaceShardSourceConfig("owner/data", REVISION, (spec, spec))
    with pytest.raises(ValueError, match="relative POSIX"):
        HuggingFaceShardSpec("train-00001", "../outside.json")
