"""Exactly resumable cursors for deterministic packed-batch schedules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping

from .partition import DataSplit, PartitionedPackedBatchReader


def _require_nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _require_sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hexadecimal.")
    return value


@dataclass(frozen=True, slots=True)
class PackedDataCursorState:
    """Portable state describing the next rank-local packed batch."""

    schema_version: int
    split: DataSplit
    seed: int
    epoch: int
    world_size: int
    rank: int
    dataset_fingerprint: str
    schedule_fingerprint: str
    source_revision: str | None
    next_partition_index: int
    next_global_batch_index: int | None
    next_shard_id: str | None
    next_local_batch_index: int | None
    next_token_offset: int | None
    batches_consumed: int
    tokens_consumed: int
    rng_state: Any = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("PackedDataCursorState supports schema_version=1 only.")
        if self.split not in {"train", "validation"}:
            raise ValueError(f"Unsupported data split: {self.split}")
        _require_nonnegative_integer("seed", self.seed)
        _require_nonnegative_integer("epoch", self.epoch)
        if self.split == "validation" and (self.seed != 0 or self.epoch != 0):
            raise ValueError("Validation cursors require seed=0 and epoch=0.")
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 1
        ):
            raise ValueError("world_size must be a positive integer.")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank < self.world_size
        ):
            raise ValueError("rank must be within world_size.")
        _require_sha256("dataset_fingerprint", self.dataset_fingerprint)
        _require_sha256("schedule_fingerprint", self.schedule_fingerprint)
        if self.source_revision is not None and (
            not isinstance(self.source_revision, str)
            or not self.source_revision.strip()
        ):
            raise ValueError("source_revision must be non-empty when provided.")
        _require_nonnegative_integer(
            "next_partition_index", self.next_partition_index
        )
        _require_nonnegative_integer("batches_consumed", self.batches_consumed)
        _require_nonnegative_integer("tokens_consumed", self.tokens_consumed)
        if self.batches_consumed != self.next_partition_index:
            raise ValueError("batches_consumed must equal next_partition_index.")
        location_values = (
            self.next_global_batch_index,
            self.next_shard_id,
            self.next_local_batch_index,
            self.next_token_offset,
        )
        if self.next_global_batch_index is not None:
            _require_nonnegative_integer(
                "next_global_batch_index", self.next_global_batch_index
            )
            if (
                not isinstance(self.next_shard_id, str)
                or not self.next_shard_id.strip()
            ):
                raise ValueError("next_shard_id is required for a pending batch.")
            _require_nonnegative_integer(
                "next_local_batch_index", self.next_local_batch_index
            )
            _require_nonnegative_integer("next_token_offset", self.next_token_offset)
        elif any(value is not None for value in location_values[1:]):
            raise ValueError("Pending batch location fields must be set together.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PackedDataCursorState":
        if not isinstance(data, Mapping):
            raise ValueError("Packed data cursor JSON root must be an object.")
        return cls(**dict(data))

    @classmethod
    def from_json(cls, value: str) -> "PackedDataCursorState":
        data = json.loads(value)
        return cls.from_dict(data)


class PackedDataCursor:
    """Consume a rank-local plan while exposing an exact next-batch snapshot."""

    def __init__(
        self,
        reader: PartitionedPackedBatchReader,
        *,
        source_revision: str | None = None,
        rng_state: Any = None,
        state: PackedDataCursorState | None = None,
    ) -> None:
        if not isinstance(reader, PartitionedPackedBatchReader):
            raise TypeError("reader must be a PartitionedPackedBatchReader.")
        if source_revision is not None and (
            not isinstance(source_revision, str) or not source_revision.strip()
        ):
            raise ValueError("source_revision must be non-empty when provided.")
        self.reader = reader
        self._source_revision = source_revision
        self._rng_state = deepcopy(rng_state)
        self._position = 0
        if state is not None:
            self._restore_state(state)

    @classmethod
    def from_state(
        cls,
        reader: PartitionedPackedBatchReader,
        state: PackedDataCursorState,
        *,
        source_revision: str | None = None,
    ) -> "PackedDataCursor":
        return cls(reader, source_revision=source_revision, state=state)

    def __iter__(self) -> "PackedDataCursor":
        return self

    def __len__(self) -> int:
        return len(self.reader) - self._position

    def __next__(self) -> dict[str, Any]:
        if self._position >= len(self.reader):
            raise StopIteration
        batch = self.reader.read_batch(self._position)
        self._position += 1
        return batch

    @property
    def position(self) -> int:
        """Rank-local index of the next batch to be returned."""

        return self._position

    @property
    def state(self) -> PackedDataCursorState:
        plan = self.reader.plan
        location = None
        global_index = None
        if self._position < len(self.reader):
            global_index = plan.batch_indices[self._position]
            location = self.reader.reader.locate(global_index)
        return PackedDataCursorState(
            schema_version=1,
            split=plan.split,
            seed=plan.seed,
            epoch=plan.epoch,
            world_size=plan.world_size,
            rank=plan.rank,
            dataset_fingerprint=plan.dataset_fingerprint,
            schedule_fingerprint=plan.schedule_fingerprint,
            source_revision=self._source_revision,
            next_partition_index=self._position,
            next_global_batch_index=global_index,
            next_shard_id=None if location is None else location.shard_id,
            next_local_batch_index=(
                None if location is None else location.local_batch_index
            ),
            next_token_offset=None if location is None else location.token_offset,
            batches_consumed=self._position,
            tokens_consumed=(
                self._position * self.reader.reader.layout.tokens_per_batch
            ),
            rng_state=deepcopy(self._rng_state),
        )

    def set_rng_state(self, rng_state: Any) -> None:
        """Replace the data RNG snapshot included in the next state."""

        self._rng_state = deepcopy(rng_state)

    def _restore_state(self, state: PackedDataCursorState) -> None:
        if not isinstance(state, PackedDataCursorState):
            raise TypeError("state must be a PackedDataCursorState.")
        plan = self.reader.plan
        for name in (
            "split",
            "seed",
            "epoch",
            "world_size",
            "rank",
            "dataset_fingerprint",
            "schedule_fingerprint",
        ):
            if getattr(state, name) != getattr(plan, name, None):
                raise ValueError(f"Cursor state does not match partition plan: {name}.")
        if (
            self._source_revision is not None
            and state.source_revision != self._source_revision
        ):
            raise ValueError("Cursor source revision does not match saved state.")
        if state.next_partition_index > len(self.reader):
            raise ValueError("Cursor position exceeds rank partition length.")
        self._position = state.next_partition_index
        self._source_revision = state.source_revision
        self._rng_state = deepcopy(state.rng_state)
        expected = self.state
        if expected != state:
            raise ValueError("Cursor state does not match the deterministic reader.")


__all__ = ["PackedDataCursor", "PackedDataCursorState"]
