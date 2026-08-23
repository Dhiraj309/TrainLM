"""Contiguous fixed-shape reads from validated packed shards."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import struct

import pytest
import torch

from trainlm.data import (
    ContiguousPackedBatchReader,
    PackedBinaryShardManifest,
    ValidatedPackedBinaryShard,
    validate_packed_binary_shard,
)


@dataclass(frozen=True)
class LocalShard:
    shard_id: str
    data_file: Path
    manifest: PackedBinaryShardManifest
    validation: ValidatedPackedBinaryShard


def _local_shard(
    root: Path,
    *,
    shard_id: str,
    tokens: tuple[int, ...],
    byte_order: str = "little",
    header_bytes: int = 1024,
) -> LocalShard:
    prefix = "<" if byte_order == "little" else ">"
    payload = bytes(header_bytes) + struct.pack(
        f"{prefix}{len(tokens)}H",
        *tokens,
    )
    path = root / f"{shard_id}.bin"
    path.write_bytes(payload)
    profile = (
        "legacy_1024_uint16"
        if byte_order == "little" and header_bytes == 1024
        else "explicit_v1"
    )
    manifest = PackedBinaryShardManifest(
        schema_version=1,
        shard_id=shard_id,
        data_path=f"data/{shard_id}.bin",
        compatibility_profile=profile,
        header_bytes=header_bytes,
        token_dtype="uint16",
        byte_order=byte_order,
        token_count=len(tokens),
        token_id_min=min(tokens),
        token_id_max=max(tokens),
        vocab_size=256,
        file_size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return LocalShard(
        shard_id=shard_id,
        data_file=path,
        manifest=manifest,
        validation=validate_packed_binary_shard(manifest, path),
    )


def test_reader_batches_complete_contiguous_regions_without_crossing_shards(
    tmp_path,
):
    first_tokens = tuple(range(18))
    second_tokens = tuple(range(100, 116))
    shards = (
        _local_shard(tmp_path, shard_id="first", tokens=first_tokens),
        _local_shard(tmp_path, shard_id="second", tokens=second_tokens),
    )

    with ContiguousPackedBatchReader(
        shards,
        batch_size=2,
        sequence_length=4,
    ) as reader:
        assert len(reader) == 4
        assert reader.layout.batches_per_shard == (2, 2)
        assert reader.layout.dropped_tokens_per_shard == (2, 0)
        assert reader.layout.total_tokens == 32

        expected_batches = (
            first_tokens[0:8],
            first_tokens[8:16],
            second_tokens[0:8],
            second_tokens[8:16],
        )
        for index, expected in enumerate(expected_batches):
            batch = reader.read_batch(index)
            reference = torch.tensor(expected, dtype=torch.int64).reshape(2, 4)
            assert torch.equal(batch["input_ids"], reference)
            assert batch["input_ids"].shape == (2, 4)
            assert batch["input_ids"].dtype is torch.int64
            assert batch["labels"] is batch["input_ids"]
            assert batch["attention_mask"] is batch["loss_mask"]
            assert batch["attention_mask"].dtype is torch.bool
            assert batch["attention_mask"].all()

        assert reader.locate(1).shard_id == "first"
        assert reader.locate(1).token_offset == 8
        assert reader.locate(2).shard_id == "second"
        assert reader.locate(2).token_offset == 0

    with pytest.raises(RuntimeError, match="closed"):
        reader.read_batch(0)
    reader.close()


def test_reader_matches_reference_for_non_native_endian_payload(tmp_path):
    tokens = tuple(range(8))
    shard = _local_shard(
        tmp_path,
        shard_id="big-endian",
        tokens=tokens,
        byte_order="big",
        header_bytes=3,
    )

    with ContiguousPackedBatchReader(
        (shard,),
        batch_size=2,
        sequence_length=4,
    ) as reader:
        batch = reader.read_batch(0)

    expected = torch.tensor(tokens, dtype=torch.int64).reshape(2, 4)
    assert torch.equal(batch["input_ids"], expected)


def test_reader_rejects_stale_validation_and_post_validation_size_change(tmp_path):
    shard = _local_shard(
        tmp_path,
        shard_id="stale",
        tokens=tuple(range(8)),
    )
    stale = replace(
        shard,
        validation=replace(shard.validation, sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="does not match"):
        ContiguousPackedBatchReader(
            (stale,),
            batch_size=2,
            sequence_length=4,
        )

    reader = ContiguousPackedBatchReader(
        (shard,),
        batch_size=2,
        sequence_length=4,
    )
    shard.data_file.write_bytes(shard.data_file.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="size changed"):
        reader.read_batch(0)
    reader.close()


def test_reader_rejects_invalid_geometry_and_batch_indices(tmp_path):
    shard = _local_shard(
        tmp_path,
        shard_id="small",
        tokens=(1, 2, 3),
    )
    with pytest.raises(ValueError, match="complete fixed-shape batch"):
        ContiguousPackedBatchReader(
            (shard,),
            batch_size=2,
            sequence_length=2,
        )
    with pytest.raises(ValueError, match="sequence_length"):
        ContiguousPackedBatchReader(
            (shard,),
            batch_size=1,
            sequence_length=1,
        )

    full = _local_shard(
        tmp_path,
        shard_id="full",
        tokens=tuple(range(8)),
    )
    reader = ContiguousPackedBatchReader(
        (full,),
        batch_size=2,
        sequence_length=4,
    )
    with pytest.raises(IndexError, match="out of range"):
        reader.read_batch(-1)
    with pytest.raises(IndexError, match="out of range"):
        reader.read_batch(1)
    reader.close()
