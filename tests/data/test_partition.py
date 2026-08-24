"""Deterministic shard shuffle and exact host partitioning."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from trainlm.data import (
    BatchPartitionPlan,
    ContiguousPackedBatchReader,
    PartitionedPackedBatchReader,
    plan_packed_batch_partition,
)

from .test_contiguous_reader import _local_shard


def _reader(tmp_path, token_counts):
    tmp_path.mkdir(parents=True, exist_ok=True)
    shards = tuple(
        _local_shard(
            tmp_path,
            shard_id=f"shard-{index}",
            tokens=tuple(range(index * 32, index * 32 + token_count)),
        )
        for index, token_count in enumerate(token_counts)
    )
    return ContiguousPackedBatchReader(
        shards,
        batch_size=2,
        sequence_length=4,
    )


def test_all_hosts_cover_retained_training_batches_exactly_once(tmp_path):
    reader = _reader(tmp_path, (24, 24, 24, 24))
    plans = tuple(
        plan_packed_batch_partition(
            reader,
            split="train",
            seed=17,
            epoch=3,
            world_size=4,
            rank=rank,
        )
        for rank in range(4)
    )

    assert all(len(plan.batch_indices) == 3 for plan in plans)
    assert len({plan.dataset_fingerprint for plan in plans}) == 1
    assert len({plan.schedule_fingerprint for plan in plans}) == 1
    assert len({plan.shard_order for plan in plans}) == 1
    owned = [set(plan.batch_indices) for plan in plans]
    assert set.union(*owned) == set(range(len(reader)))
    assert sum(len(indices) for indices in owned) == len(set.union(*owned))
    assert all(not plans[index].dropped_batch_indices for index in range(4))

    restored = BatchPartitionPlan.from_json(plans[2].to_json())
    assert restored == plans[2]
    assert restored.to_json() == plans[2].to_json()

    repeated = plan_packed_batch_partition(
        reader,
        split="train",
        seed=17,
        epoch=3,
        world_size=4,
        rank=2,
    )
    next_epoch = plan_packed_batch_partition(
        reader,
        split="train",
        seed=17,
        epoch=4,
        world_size=4,
        rank=2,
    )
    assert repeated == plans[2]
    assert next_epoch.schedule_fingerprint != plans[2].schedule_fingerprint
    reader.close()


def test_partitioned_reader_follows_rank_plan_and_reports_metadata(tmp_path):
    reader = _reader(tmp_path, (16, 16))
    plan = plan_packed_batch_partition(
        reader,
        split="train",
        seed=7,
        epoch=0,
        world_size=2,
        rank=1,
    )
    partitioned = PartitionedPackedBatchReader(reader, plan)

    for partition_index, batch in enumerate(partitioned):
        global_index = plan.batch_indices[partition_index]
        reference = reader.read_batch(global_index)
        assert torch.equal(batch["input_ids"], reference["input_ids"])
        assert batch["global_batch_index"] == global_index
        assert batch["partition_index"] == partition_index
        assert batch["replica_rank"] == 1
    reader.close()


def test_validation_order_is_declared_and_requires_zero_seed_epoch(tmp_path):
    reader = _reader(tmp_path, (16, 16, 16))
    plan = plan_packed_batch_partition(
        reader,
        split="validation",
        seed=0,
        epoch=0,
        world_size=1,
        rank=0,
    )

    assert plan.shard_order == ("shard-0", "shard-1", "shard-2")
    assert plan.batch_indices == tuple(range(len(reader)))
    with pytest.raises(ValueError, match="seed=0 and epoch=0"):
        plan_packed_batch_partition(
            reader,
            split="validation",
            seed=1,
            epoch=0,
            world_size=1,
            rank=0,
        )
    reader.close()


def test_remainder_policies_are_explicit_and_balanced(tmp_path):
    reader = _reader(tmp_path, (18, 24, 16, 16))
    plans = tuple(
        plan_packed_batch_partition(
            reader,
            split="train",
            seed=5,
            epoch=0,
            world_size=4,
            rank=rank,
            cross_shard_remainder="drop",
            host_remainder="drop",
        )
        for rank in range(4)
    )

    assert all(plan.dropped_token_count == 2 for plan in plans)
    assert all(len(plan.dropped_batch_indices) == 1 for plan in plans)
    assert all(len(plan.batch_indices) == 2 for plan in plans)
    retained = set.union(*(set(plan.batch_indices) for plan in plans))
    dropped = set(plans[0].dropped_batch_indices)
    assert retained.isdisjoint(dropped)
    assert retained | dropped == set(range(len(reader)))

    with pytest.raises(ValueError, match="cross-shard batch"):
        plan_packed_batch_partition(
            reader,
            split="train",
            seed=5,
            epoch=0,
            world_size=4,
            rank=0,
            cross_shard_remainder="error",
        )
    with pytest.raises(ValueError, match="divide evenly"):
        plan_packed_batch_partition(
            reader,
            split="train",
            seed=5,
            epoch=0,
            world_size=4,
            rank=0,
            host_remainder="error",
        )
    reader.close()


def test_partition_rejects_stale_dataset_plan(tmp_path):
    first_reader = _reader(tmp_path / "first", (16, 16))
    plan = plan_packed_batch_partition(
        first_reader,
        split="train",
        seed=1,
        epoch=0,
        world_size=1,
        rank=0,
    )
    second_reader = _reader(tmp_path / "second", (24, 16))

    altered = replace(plan, batch_indices=tuple(reversed(plan.batch_indices)))
    with pytest.raises(ValueError, match="deterministic schedule"):
        PartitionedPackedBatchReader(first_reader, altered)
    with pytest.raises(ValueError, match="does not match"):
        PartitionedPackedBatchReader(second_reader, plan)
    first_reader.close()
    second_reader.close()
