"""Deterministic shard ordering and host partitioning for packed batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal, Mapping

from .reader import ContiguousPackedBatchReader

DataSplit = Literal["train", "validation"]
RemainderPolicy = Literal["drop", "error"]


def _require_nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def packed_dataset_fingerprint(reader: ContiguousPackedBatchReader) -> str:
    """Fingerprint ordered shard content and fixed reader geometry."""

    if not isinstance(reader, ContiguousPackedBatchReader):
        raise TypeError("reader must be a ContiguousPackedBatchReader.")
    identity = {
        "schema_version": 1,
        "batch_size": reader.batch_size,
        "sequence_length": reader.sequence_length,
        "shards": [
            {
                "shard_id": shard.shard_id,
                "sha256": shard.manifest.sha256,
                "token_count": shard.manifest.token_count,
                "batches": reader.layout.batches_per_shard[index],
                "dropped_tokens": (
                    reader.layout.dropped_tokens_per_shard[index]
                ),
            }
            for index, shard in enumerate(reader.shards)
        ],
    }
    return _sha256_json(identity)


@dataclass(frozen=True, slots=True)
class BatchPartitionPlan:
    """Serializable rank-local view of one deterministic global schedule."""

    schema_version: int
    split: DataSplit
    seed: int
    epoch: int
    world_size: int
    rank: int
    cross_shard_remainder: RemainderPolicy
    host_remainder: RemainderPolicy
    dataset_fingerprint: str
    schedule_fingerprint: str
    shard_order: tuple[str, ...]
    dropped_token_count: int
    global_batch_count: int
    retained_batch_count: int
    dropped_batch_indices: tuple[int, ...]
    batch_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("BatchPartitionPlan supports schema_version=1 only.")
        if self.split not in {"train", "validation"}:
            raise ValueError(f"Unsupported data split: {self.split}")
        _require_nonnegative_integer("seed", self.seed)
        _require_nonnegative_integer("epoch", self.epoch)
        if self.split == "validation" and (self.seed != 0 or self.epoch != 0):
            raise ValueError("Validation partition plans require seed=0 and epoch=0.")
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
        for name in ("cross_shard_remainder", "host_remainder"):
            if getattr(self, name) not in {"drop", "error"}:
                raise ValueError(f"Unsupported remainder policy: {getattr(self, name)}")
        for name in ("dataset_fingerprint", "schedule_fingerprint"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hexadecimal.")
        for name in (
            "shard_order",
            "dropped_batch_indices",
            "batch_indices",
        ):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(
                value, (list, tuple)
            ):
                raise ValueError(f"{name} must be a list or tuple.")
            object.__setattr__(self, name, tuple(value))
        if not self.shard_order or any(
            not isinstance(item, str) or not item.strip()
            for item in self.shard_order
        ):
            raise ValueError("shard_order must contain non-empty shard IDs.")
        if len(self.shard_order) != len(set(self.shard_order)):
            raise ValueError("shard_order must contain unique shard IDs.")
        _require_nonnegative_integer(
            "dropped_token_count",
            self.dropped_token_count,
        )
        _require_nonnegative_integer("global_batch_count", self.global_batch_count)
        _require_nonnegative_integer(
            "retained_batch_count",
            self.retained_batch_count,
        )
        if self.retained_batch_count > self.global_batch_count:
            raise ValueError("Retained batches cannot exceed global batches.")
        for name in ("dropped_batch_indices", "batch_indices"):
            indices = getattr(self, name)
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.global_batch_count
                for index in indices
            ):
                raise ValueError(f"{name} contains an invalid global batch index.")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{name} must not contain duplicate indices.")
        if set(self.dropped_batch_indices) & set(self.batch_indices):
            raise ValueError("Dropped and rank-owned batch indices must be disjoint.")
        if self.retained_batch_count % self.world_size:
            raise ValueError("Retained batch count must divide evenly across hosts.")
        expected_rank_batches = self.retained_batch_count // self.world_size
        if len(self.batch_indices) != expected_rank_batches:
            raise ValueError("Rank batch count does not match retained schedule.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatchPartitionPlan":
        values = dict(data)
        for name in (
            "shard_order",
            "dropped_batch_indices",
            "batch_indices",
        ):
            values[name] = tuple(values.get(name, ()))
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "BatchPartitionPlan":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Batch partition JSON root must be an object.")
        return cls.from_dict(data)


def _ordered_shard_indices(
    reader: ContiguousPackedBatchReader,
    *,
    split: DataSplit,
    seed: int,
    epoch: int,
) -> tuple[int, ...]:
    indices = tuple(range(len(reader.shards)))
    if split == "validation":
        return indices

    def key(index: int) -> tuple[bytes, int]:
        shard_id = reader.shards[index].shard_id
        value = (
            f"trainlm-shard-order-v1\0{seed}\0{epoch}\0{shard_id}\0{index}"
        ).encode("utf-8")
        return hashlib.sha256(value).digest(), index

    return tuple(sorted(indices, key=key))


def _global_batch_order(
    reader: ContiguousPackedBatchReader,
    shard_indices: tuple[int, ...],
) -> tuple[int, ...]:
    starts = []
    running = 0
    for count in reader.layout.batches_per_shard:
        starts.append(running)
        running += count
    order = []
    for shard_index in shard_indices:
        start = starts[shard_index]
        count = reader.layout.batches_per_shard[shard_index]
        order.extend(range(start, start + count))
    return tuple(order)


def plan_packed_batch_partition(
    reader: ContiguousPackedBatchReader,
    *,
    split: DataSplit,
    seed: int,
    epoch: int,
    world_size: int,
    rank: int,
    cross_shard_remainder: RemainderPolicy = "drop",
    host_remainder: RemainderPolicy = "drop",
) -> BatchPartitionPlan:
    """Create one rank's deterministic, non-overlapping batch schedule."""

    if not isinstance(reader, ContiguousPackedBatchReader):
        raise TypeError("reader must be a ContiguousPackedBatchReader.")
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported data split: {split}")
    _require_nonnegative_integer("seed", seed)
    _require_nonnegative_integer("epoch", epoch)
    if split == "validation" and (seed != 0 or epoch != 0):
        raise ValueError("Validation partition plans require seed=0 and epoch=0.")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise ValueError("world_size must be a positive integer.")
    if world_size < 1:
        raise ValueError("world_size must be a positive integer.")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < world_size
    ):
        raise ValueError("rank must be within world_size.")
    if cross_shard_remainder not in {"drop", "error"}:
        raise ValueError("Unsupported cross_shard_remainder policy.")
    if host_remainder not in {"drop", "error"}:
        raise ValueError("Unsupported host_remainder policy.")

    shard_ids = tuple(shard.shard_id for shard in reader.shards)
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("Partitioning requires unique shard IDs.")
    if cross_shard_remainder == "error" and any(
        reader.layout.dropped_tokens_per_shard
    ):
        raise ValueError(
            "Shard token remainder would require a cross-shard batch; "
            "use cross_shard_remainder='drop' to exclude it explicitly."
        )

    shard_indices = _ordered_shard_indices(
        reader,
        split=split,
        seed=seed,
        epoch=epoch,
    )
    global_order = _global_batch_order(reader, shard_indices)
    host_remainder_count = len(global_order) % world_size
    if host_remainder == "error" and host_remainder_count:
        raise ValueError("Global batch count does not divide evenly across hosts.")
    retained_count = len(global_order) - host_remainder_count
    if retained_count == 0:
        raise ValueError("Partitioning leaves no complete batch per host.")
    retained = global_order[:retained_count]
    dropped = global_order[retained_count:]
    batches_per_rank = retained_count // world_size
    rank_start = rank * batches_per_rank
    rank_batches = retained[rank_start:rank_start + batches_per_rank]
    dataset_fingerprint = packed_dataset_fingerprint(reader)
    schedule_fingerprint = _sha256_json(
        {
            "schema_version": 1,
            "split": split,
            "seed": seed,
            "epoch": epoch,
            "world_size": world_size,
            "dataset_fingerprint": dataset_fingerprint,
            "shard_indices": shard_indices,
            "global_order": global_order,
            "retained_count": retained_count,
            "dropped": dropped,
            "cross_shard_remainder": cross_shard_remainder,
            "host_remainder": host_remainder,
        }
    )
    return BatchPartitionPlan(
        schema_version=1,
        split=split,
        seed=seed,
        epoch=epoch,
        world_size=world_size,
        rank=rank,
        cross_shard_remainder=cross_shard_remainder,
        host_remainder=host_remainder,
        dataset_fingerprint=dataset_fingerprint,
        schedule_fingerprint=schedule_fingerprint,
        shard_order=tuple(reader.shards[index].shard_id for index in shard_indices),
        dropped_token_count=sum(reader.layout.dropped_tokens_per_shard),
        global_batch_count=len(global_order),
        retained_batch_count=retained_count,
        dropped_batch_indices=dropped,
        batch_indices=rank_batches,
    )


class PartitionedPackedBatchReader:
    """Read only the deterministic global batches assigned to one rank."""

    def __init__(
        self,
        reader: ContiguousPackedBatchReader,
        plan: BatchPartitionPlan,
    ) -> None:
        if not isinstance(reader, ContiguousPackedBatchReader):
            raise TypeError("reader must be a ContiguousPackedBatchReader.")
        if not isinstance(plan, BatchPartitionPlan):
            raise TypeError("plan must be a BatchPartitionPlan.")
        if packed_dataset_fingerprint(reader) != plan.dataset_fingerprint:
            raise ValueError("Partition plan does not match the packed dataset.")
        if len(reader) != plan.global_batch_count:
            raise ValueError("Partition plan global batch count is stale.")
        expected = plan_packed_batch_partition(
            reader,
            split=plan.split,
            seed=plan.seed,
            epoch=plan.epoch,
            world_size=plan.world_size,
            rank=plan.rank,
            cross_shard_remainder=plan.cross_shard_remainder,
            host_remainder=plan.host_remainder,
        )
        if expected != plan:
            raise ValueError("Partition plan does not match deterministic schedule.")
        self.reader = reader
        self.plan = plan

    def __len__(self) -> int:
        return len(self.plan.batch_indices)

    def read_batch(self, partition_index: int) -> dict[str, Any]:
        if isinstance(partition_index, bool) or not isinstance(
            partition_index, int
        ):
            raise TypeError("partition_index must be an integer.")
        if partition_index < 0 or partition_index >= len(self):
            raise IndexError(f"Partition index out of range: {partition_index}")
        global_index = self.plan.batch_indices[partition_index]
        batch = self.reader.read_batch(global_index)
        return {
            **batch,
            "global_batch_index": global_index,
            "partition_index": partition_index,
            "replica_rank": self.plan.rank,
        }

    def __iter__(self):
        for partition_index in range(len(self)):
            yield self.read_batch(partition_index)
