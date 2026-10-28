"""Exact packed-data cursor snapshots and deterministic resume."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trainlm.data import (
    PackedDataCursor,
    PackedDataCursorState,
    PartitionedPackedBatchReader,
    plan_packed_batch_partition,
)

from .test_partition import _reader


def _partitioned(tmp_path):
    reader = _reader(tmp_path, (16, 16, 16, 16))
    plan = plan_packed_batch_partition(
        reader,
        split="train",
        seed=19,
        epoch=2,
        world_size=2,
        rank=1,
    )
    return reader, PartitionedPackedBatchReader(reader, plan)


def test_cursor_state_round_trips_and_resumes_next_batch(tmp_path):
    reader, partitioned = _partitioned(tmp_path)
    cursor = PackedDataCursor(
        partitioned,
        source_revision="refs/heads/main",
        rng_state={"shuffle_counter": 7},
    )
    first = next(cursor)
    state = cursor.state

    assert first["partition_index"] == 0
    assert state.next_partition_index == 1
    assert state.next_global_batch_index == partitioned.plan.batch_indices[1]
    assert state.next_shard_id
    assert state.next_token_offset is not None
    assert state.batches_consumed == 1
    assert state.tokens_consumed == 8

    restored_state = PackedDataCursorState.from_json(state.to_json())
    resumed = PackedDataCursor.from_state(partitioned, restored_state)
    uninterrupted = next(cursor)
    assert next(resumed)["input_ids"].tolist() == uninterrupted["input_ids"].tolist()
    assert resumed.state == cursor.state
    reader.close()


def test_cursor_end_state_has_no_pending_location(tmp_path):
    reader, partitioned = _partitioned(tmp_path)
    cursor = PackedDataCursor(partitioned)
    list(cursor)
    state = cursor.state

    assert state.next_partition_index == len(partitioned)
    assert state.next_global_batch_index is None
    assert state.next_shard_id is None
    assert state.next_local_batch_index is None
    assert state.next_token_offset is None
    with pytest.raises(StopIteration):
        next(cursor)
    reader.close()


def test_cursor_rejects_stale_or_inconsistent_state(tmp_path):
    reader, partitioned = _partitioned(tmp_path)
    cursor = PackedDataCursor(partitioned)
    state = cursor.state

    with pytest.raises(ValueError, match="does not match partition plan"):
        PackedDataCursor.from_state(
            partitioned,
            replace(state, schedule_fingerprint="0" * 64),
        )
    with pytest.raises(ValueError, match="batches_consumed"):
        replace(state, batches_consumed=1)
    with pytest.raises(ValueError, match="source revision"):
        PackedDataCursor.from_state(
            partitioned,
            replace(state, source_revision="refs/heads/dev"),
            source_revision="refs/heads/main",
        )
    reader.close()


def test_reference_28_2_split_partitions_and_resumes_exactly(tmp_path):
    train_reader = _reader(tmp_path / "train", (16,) * 28)
    validation_reader = _reader(tmp_path / "validation", (16,) * 2)

    try:
        train_plans = tuple(
            plan_packed_batch_partition(
                train_reader,
                split="train",
                seed=2026,
                epoch=3,
                world_size=8,
                rank=rank,
            )
            for rank in range(8)
        )
        owned = [set(plan.batch_indices) for plan in train_plans]
        assert set.union(*owned) == set(range(len(train_reader)))
        assert sum(map(len, owned)) == len(set.union(*owned))

        for plan in train_plans:
            partitioned = PartitionedPackedBatchReader(train_reader, plan)
            expected = [
                batch["input_ids"].tolist()
                for batch in PackedDataCursor(partitioned)
            ]
            for interruption in range(len(partitioned) + 1):
                cursor = PackedDataCursor(
                    partitioned,
                    source_revision="a" * 40,
                    rng_state={"epoch": 3, "draw": interruption},
                )
                prefix = [next(cursor)["input_ids"].tolist() for _ in range(interruption)]
                state = PackedDataCursorState.from_json(cursor.state.to_json())
                resumed = PackedDataCursor.from_state(partitioned, state)
                suffix = [batch["input_ids"].tolist() for batch in resumed]
                assert prefix + suffix == expected
                assert resumed.state.tokens_consumed == (
                    len(partitioned) * train_reader.layout.tokens_per_batch
                )

        validation_plans = tuple(
            plan_packed_batch_partition(
                validation_reader,
                split="validation",
                seed=0,
                epoch=0,
                world_size=2,
                rank=rank,
            )
            for rank in range(2)
        )
        validation_owned = [set(plan.batch_indices) for plan in validation_plans]
        assert set.union(*validation_owned) == set(range(len(validation_reader)))
        assert sum(map(len, validation_owned)) == len(set.union(*validation_owned))
    finally:
        train_reader.close()
        validation_reader.close()
