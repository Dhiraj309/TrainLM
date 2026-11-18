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
    target_ids_file: Path | None
    loss_mask_file: Path | None


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
            self._target_file = None
            self._target_mapping = None
            if shard.manifest.target_ids is not None:
                target_path = getattr(shard, "target_ids_file", None)
                if target_path is None or not Path(target_path).is_file():
                    raise FileNotFoundError("Target-ID sidecar is required by manifest.")
                target_path = Path(target_path)
                if target_path.stat().st_size != shard.manifest.target_ids.size_bytes:
                    raise ValueError("Target-ID sidecar size changed after validation.")
                self._target_file = target_path.open("rb")
                self._target_mapping = mmap.mmap(
                    self._target_file.fileno(), length=0, access=mmap.ACCESS_COPY
                )
            self._loss_mask_file = None
            self._loss_mask_mapping = None
            if shard.manifest.loss_mask is not None:
                mask_path = getattr(shard, "loss_mask_file", None)
                if mask_path is None or not Path(mask_path).is_file():
                    raise FileNotFoundError("Loss-mask sidecar is required by manifest.")
                mask_path = Path(mask_path)
                if mask_path.stat().st_size != shard.manifest.loss_mask.size_bytes:
                    raise ValueError("Loss-mask sidecar size changed after validation.")
                self._loss_mask_file = mask_path.open("rb")
                self._loss_mask_mapping = mmap.mmap(
                    self._loss_mask_file.fileno(), length=0, access=mmap.ACCESS_COPY
                )
        except BaseException:
            for name in ("_target_mapping", "_loss_mask_mapping"):
                mapping = getattr(self, name, None)
                if mapping is not None:
                    mapping.close()
            for name in ("_target_file", "_loss_mask_file"):
                handle = getattr(self, name, None)
                if handle is not None:
                    handle.close()
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

    def read_target_ids(self, token_offset: int, token_count: int) -> torch.Tensor | None:
        if self._target_mapping is None:
            return None
        if token_offset < 0 or token_count < 1:
            raise ValueError("Token offset and count must describe a non-empty span.")
        if token_offset + token_count > self.shard.manifest.token_count:
            raise IndexError("Target-ID read exceeds the sidecar payload.")
        if sys.byteorder == "little":
            values = torch.frombuffer(
                self._target_mapping,
                dtype=torch.int64,
                count=self.shard.manifest.token_count,
            ).narrow(0, token_offset, token_count)
            return values.to(dtype=torch.int64, copy=True)
        byte_start = token_offset * 8
        byte_stop = byte_start + token_count * 8
        values = struct.iter_unpack(
            "<q", self._target_mapping[byte_start:byte_stop]
        )
        return torch.tensor(tuple(value[0] for value in values), dtype=torch.int64)

    def read_loss_mask(self, token_offset: int, token_count: int) -> torch.Tensor | None:
        if self._loss_mask_mapping is None:
            return None
        if token_offset < 0 or token_count < 1:
            raise ValueError("Token offset and count must describe a non-empty span.")
        if token_offset + token_count > self.shard.manifest.token_count:
            raise IndexError("Loss-mask read exceeds the sidecar payload.")
        values = torch.frombuffer(
            self._loss_mask_mapping,
            dtype=torch.uint8,
            count=self.shard.manifest.token_count,
        ).narrow(0, token_offset, token_count)
        return values.to(dtype=torch.bool, copy=True)

    def close(self) -> None:
        native_tokens = self._native_tokens
        self._native_tokens = None
        del native_tokens
        self._mapping.close()
        self._file.close()
        for name in ("_target_mapping", "_loss_mask_mapping"):
            mapping = getattr(self, name, None)
            if mapping is not None:
                mapping.close()
                setattr(self, name, None)
        for name in ("_target_file", "_loss_mask_file"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()
                setattr(self, name, None)


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
        labels = mapped.read_target_ids(
            location.token_offset, location.token_count
        )
        loss_mask = mapped.read_loss_mask(
            location.token_offset, location.token_count
        )
        return {
            "input_ids": input_ids,
            "labels": input_ids if labels is None else labels.reshape(
                self.batch_size, self.sequence_length
            ),
            "attention_mask": mask,
            "loss_mask": mask if loss_mask is None else loss_mask.reshape(
                self.batch_size, self.sequence_length
            ),
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
