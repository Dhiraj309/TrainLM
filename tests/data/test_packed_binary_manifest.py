"""Packed binary shard manifest and integrity validation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import struct

import pytest

from trainlm.data import (
    DocumentIndex,
    PackedBinaryShardManifest,
    validate_packed_binary_shard,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def legacy_shard(tmp_path, tokens=(0, 4, 31)):
    payload = struct.pack(f"<{len(tokens)}H", *tokens)
    data = bytes(1024) + payload
    path = tmp_path / "train-00000.bin"
    path.write_bytes(data)
    manifest = PackedBinaryShardManifest(
        schema_version=1,
        shard_id="train-00000",
        data_path="fineweb-edu/train-00000.bin",
        compatibility_profile="legacy_1024_uint16",
        header_bytes=1024,
        token_dtype="uint16",
        byte_order="little",
        token_count=len(tokens),
        token_id_min=min(tokens),
        token_id_max=max(tokens),
        vocab_size=32,
        file_size_bytes=len(data),
        sha256=_sha256(data),
    )
    return path, manifest


def test_legacy_manifest_round_trips_and_validates_complete_shard(tmp_path):
    path, manifest = legacy_shard(tmp_path)

    restored = PackedBinaryShardManifest.from_json(manifest.to_json())
    validated = validate_packed_binary_shard(restored, path, chunk_tokens=2)

    assert restored == manifest
    assert validated.token_count == 3
    assert validated.token_id_min == 0
    assert validated.token_id_max == 31
    assert validated.sha256 == manifest.sha256
    assert validated.document_count is None


def test_manifest_rejects_corrupt_metadata_before_file_access(tmp_path):
    _, manifest = legacy_shard(tmp_path)

    with pytest.raises(ValueError, match="legacy_1024_uint16"):
        replace(manifest, header_bytes=0, file_size_bytes=6)
    with pytest.raises(ValueError, match="vocab_size"):
        replace(manifest, token_id_max=32)
    with pytest.raises(ValueError, match="file_size_bytes"):
        replace(manifest, file_size_bytes=manifest.file_size_bytes + 1)
    with pytest.raises(ValueError, match="relative POSIX"):
        replace(manifest, data_path="../outside.bin")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(manifest, sha256="not-a-digest")


def test_validation_rejects_size_checksum_and_token_corruption(tmp_path):
    path, manifest = legacy_shard(tmp_path)

    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="size"):
        validate_packed_binary_shard(manifest, path)

    path, manifest = legacy_shard(tmp_path)
    corrupted = bytearray(path.read_bytes())
    corrupted[-2:] = struct.pack("<H", 30)
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum"):
        validate_packed_binary_shard(manifest, path)

    checksum = _sha256(bytes(corrupted))
    with pytest.raises(ValueError, match="bounds"):
        validate_packed_binary_shard(replace(manifest, sha256=checksum), path)

    corrupted[-2:] = struct.pack("<H", 99)
    path.write_bytes(corrupted)
    checksum = _sha256(bytes(corrupted))
    with pytest.raises(ValueError, match="outside"):
        validate_packed_binary_shard(replace(manifest, sha256=checksum), path)


def test_document_offsets_are_content_addressed_and_cover_payload(tmp_path):
    tokens = (1, 2, 3, 4)
    data = struct.pack("<4H", *tokens)
    data_path = tmp_path / "documents.bin"
    data_path.write_bytes(data)
    offsets = struct.pack("<3Q", 0, 2, 4)
    index_path = tmp_path / "documents.idx"
    index_path.write_bytes(offsets)
    documents = DocumentIndex(
        storage="uint64_offsets",
        document_count=2,
        path="fineweb-edu/documents.idx",
        sha256=_sha256(offsets),
        size_bytes=len(offsets),
    )
    manifest = PackedBinaryShardManifest(
        schema_version=1,
        shard_id="documents",
        data_path="fineweb-edu/documents.bin",
        compatibility_profile="explicit_v1",
        header_bytes=0,
        token_dtype="uint16",
        byte_order="little",
        token_count=4,
        token_id_min=1,
        token_id_max=4,
        vocab_size=32,
        file_size_bytes=len(data),
        sha256=_sha256(data),
        documents=documents,
    )

    validated = validate_packed_binary_shard(
        manifest,
        data_path,
        document_index_file=index_path,
    )
    assert validated.document_count == 2

    bad_offsets = struct.pack("<3Q", 0, 4, 4)
    index_path.write_bytes(bad_offsets)
    bad_documents = replace(documents, sha256=_sha256(bad_offsets))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_packed_binary_shard(
            replace(manifest, documents=bad_documents),
            data_path,
            document_index_file=index_path,
        )


def test_document_index_declaration_is_all_or_nothing():
    with pytest.raises(ValueError, match="cannot declare"):
        DocumentIndex(document_count=2)
    with pytest.raises(ValueError, match="size"):
        DocumentIndex(
            storage="uint64_offsets",
            document_count=2,
            path="documents.idx",
            sha256="a" * 64,
            size_bytes=16,
        )
