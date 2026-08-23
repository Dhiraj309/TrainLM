"""Contiguous fixed-shape batch reads from validated packed token shards."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
import mmap
from pathlib import Path
import struct
import sys
from typing import Any, Protocol

import torch

from .manifest import PackedBinaryShardManifest, ValidatedPackedBinaryShard

_TORCH_DTYPES = {
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "int32": torch.int32,
    "int64": torch.int64,
}
_STRUCT_CODES = {
    "uint16": "H",
    "uint32": "I",
    "int32": "i",
    "int64": "q",
}


class PackedShardFile(Protocol):
    """Source-neutral local shard contract consumed by the reader."""

    shard_id: str
    data_file: Path
    manifest: PackedBinaryShardManifest
    validation: ValidatedPackedBinaryShard


@dataclass(frozen=True, slots=True)
class PackedBatchLocation:
    """Stable location metadata for one complete contiguous batch."""

    batch_index: int
    shard_index: int
    shard_id: str
    local_batch_index: int
    token_offset: int
    token_count: int


@dataclass(frozen=True, slots=True)
class PackedReaderLayout:
    """Fixed geometry and remainder accounting for an opened reader."""

    batch_size: int
    sequence_length: int
    tokens_per_batch: int
    batches_per_shard: tuple[int, ...]
    dropped_tokens_per_shard: tuple[int, ...]
    total_batches: int
    total_tokens: int


class _MappedTokenShard:
    def __init__(self, shard: PackedShardFile) -> None:
        self.shard = shard
        self.path = Path(shard.data_file)
        if not self.path.is_file():
            raise FileNotFoundError(f"Packed token shard does not exist: {self.path}")
        if self.path.stat().st_size != shard.manifest.file_size_bytes:
            raise ValueError("Packed token shard size changed after validation.")

        self._file = self.path.open("rb")
        try:
            self._mapping = mmap.mmap(
                self._file.fileno(),
                length=0,
                access=mmap.ACCESS_COPY,
            )
            self._native_tokens: torch.Tensor | None = None
            if shard.manifest.byte_order == sys.byteorder:
                self._native_tokens = torch.frombuffer(
                    self._mapping,
                    dtype=_TORCH_DTYPES[shard.manifest.token_dtype],
                    count=shard.manifest.token_count,
                    offset=shard.manifest.header_bytes,
                )
        except BaseException:
            mapping = getattr(self, "_mapping", None)
            if mapping is not None:
                mapping.close()
            self._file.close()
            raise

    def read(self, token_offset: int, token_count: int) -> torch.Tensor:
        manifest = self.shard.manifest
        if token_offset < 0 or token_count < 1:
            raise ValueError("Token offset and count must describe a non-empty span.")
        if token_offset + token_count > manifest.token_count:
            raise IndexError("Packed token read exceeds the shard payload.")
        if self._native_tokens is not None:
            return self._native_tokens.narrow(
                0,
                token_offset,
                token_count,
            ).to(dtype=torch.int64, copy=True)

        item_size = manifest.item_size
        byte_start = manifest.header_bytes + token_offset * item_size
        byte_end = byte_start + token_count * item_size
        payload = self._mapping[byte_start:byte_end]
        prefix = "<" if manifest.byte_order == "little" else ">"
        format_code = _STRUCT_CODES[manifest.token_dtype]
        values = (
            item[0] for item in struct.iter_unpack(f"{prefix}{format_code}", payload)
        )
        return torch.tensor(tuple(values), dtype=torch.int64)

    def close(self) -> None:
        native_tokens = self._native_tokens
        self._native_tokens = None
        del native_tokens
        self._mapping.close()
        self._file.close()


class ContiguousPackedBatchReader:
    """Read whole batches without per-sample memmap indexing."""

    def __init__(
        self,
        shards: Sequence[PackedShardFile],
        *,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if isinstance(shards, (str, bytes)) or not isinstance(shards, Sequence):
            raise TypeError("shards must be a sequence of validated shard files.")
        self.shards = tuple(shards)
        if not self.shards:
            raise ValueError("At least one validated shard is required.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(sequence_length, bool) or not isinstance(
            sequence_length, int
        ):
            raise ValueError("sequence_length must be an integer >= 2.")
        if sequence_length < 2:
            raise ValueError("sequence_length must be an integer >= 2.")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        tokens_per_batch = batch_size * sequence_length
        batches_per_shard = []
        dropped_tokens = []
        for shard in self.shards:
            self._validate_shard_contract(shard)
            batches, remainder = divmod(
                shard.manifest.token_count,
                tokens_per_batch,
            )
            batches_per_shard.append(batches)
            dropped_tokens.append(remainder)
        total_batches = sum(batches_per_shard)
        if total_batches == 0:
            raise ValueError("No shard contains one complete fixed-shape batch.")

        self.layout = PackedReaderLayout(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokens_per_batch=tokens_per_batch,
            batches_per_shard=tuple(batches_per_shard),
            dropped_tokens_per_shard=tuple(dropped_tokens),
            total_batches=total_batches,
            total_tokens=total_batches * tokens_per_batch,
        )
        cumulative = []
        running = 0
        for batches in batches_per_shard:
            running += batches
            cumulative.append(running)
        self._cumulative_batches = tuple(cumulative)
        self._mapped: list[_MappedTokenShard | None] = [None] * len(self.shards)
        self._closed = False

    def __len__(self) -> int:
        return self.layout.total_batches

    def locate(self, batch_index: int) -> PackedBatchLocation:
        if isinstance(batch_index, bool) or not isinstance(batch_index, int):
            raise TypeError("batch_index must be an integer.")
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(f"Batch index out of range: {batch_index}")
        shard_index = bisect_right(self._cumulative_batches, batch_index)
        previous = 0 if shard_index == 0 else self._cumulative_batches[
            shard_index - 1
        ]
        local_batch = batch_index - previous
        token_offset = local_batch * self.layout.tokens_per_batch
        return PackedBatchLocation(
            batch_index=batch_index,
            shard_index=shard_index,
            shard_id=self.shards[shard_index].shard_id,
            local_batch_index=local_batch,
            token_offset=token_offset,
            token_count=self.layout.tokens_per_batch,
        )

    def read_batch(self, batch_index: int) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Packed batch reader is closed.")
        location = self.locate(batch_index)
        mapped = self._mapped[location.shard_index]
        if mapped is None:
            mapped = _MappedTokenShard(self.shards[location.shard_index])
            self._mapped[location.shard_index] = mapped
        input_ids = mapped.read(
            location.token_offset,
            location.token_count,
        ).reshape(self.batch_size, self.sequence_length)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": mask,
            "loss_mask": mask,
            "shard_id": location.shard_id,
            "batch_index": location.batch_index,
            "token_offset": location.token_offset,
        }

    def __iter__(self):
        for batch_index in range(len(self)):
            yield self.read_batch(batch_index)

    def close(self) -> None:
        if self._closed:
            return
        for mapped in self._mapped:
            if mapped is not None:
                mapped.close()
        self._mapped = [None] * len(self.shards)
        self._closed = True

    def __enter__(self) -> "ContiguousPackedBatchReader":
        if self._closed:
            raise RuntimeError("Packed batch reader is closed.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _validate_shard_contract(shard: PackedShardFile) -> None:
        for name in ("shard_id", "data_file", "manifest", "validation"):
            if not hasattr(shard, name):
                raise TypeError(f"Packed shard is missing required field: {name}")
        if not isinstance(shard.shard_id, str) or not shard.shard_id.strip():
            raise ValueError("Packed shard ID cannot be empty.")
        if not isinstance(shard.manifest, PackedBinaryShardManifest):
            raise TypeError("Packed shard manifest has the wrong type.")
        if not isinstance(shard.validation, ValidatedPackedBinaryShard):
            raise TypeError("Packed shard validation has the wrong type.")
        expected = shard.manifest
        observed = shard.validation
        if shard.shard_id != expected.shard_id:
            raise ValueError("Packed shard ID does not match its manifest.")
        if (
            observed.token_count != expected.token_count
            or observed.token_id_min != expected.token_id_min
            or observed.token_id_max != expected.token_id_max
            or observed.sha256 != expected.sha256
        ):
            raise ValueError("Packed shard validation does not match its manifest.")
