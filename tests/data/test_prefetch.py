"""Bounded asynchronous prefetch ordering, transfer, and backpressure."""

from __future__ import annotations

import threading

import pytest

from trainlm.data import (
    AsyncBatchPrefetcher,
    BackendBatchTransfer,
    PrefetchWorkerError,
)


class RecordingSource:
    def __init__(self, count: int, *, fail_at: int | None = None) -> None:
        self.count = count
        self.fail_at = fail_at
        self.calls: list[int] = []

    def __len__(self) -> int:
        return self.count

    def read_batch(self, index: int):
        self.calls.append(index)
        if index == self.fail_at:
            raise ValueError(f"bad batch {index}")
        return {"value": index}


class RecordingTransfer:
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def transfer(self, batch):
        value = batch["value"]
        self.calls.append(value)
        return {"value": value * 10}


class BackpressureSource(RecordingSource):
    def __init__(self, count: int, capacity: int) -> None:
        super().__init__(count)
        self.capacity_reached = threading.Event()
        self.next_read = threading.Event()
        self.capacity = capacity

    def read_batch(self, index: int):
        batch = super().read_batch(index)
        if len(self.calls) == self.capacity:
            self.capacity_reached.set()
        if len(self.calls) == self.capacity + 1:
            self.next_read.set()
        return batch


def test_prefetch_preserves_order_applies_transfer_and_reports_timing():
    source = RecordingSource(5)
    transfer = RecordingTransfer()
    prefetcher = AsyncBatchPrefetcher(
        source,
        capacity=2,
        transfer=transfer,
    )

    batches = list(prefetcher)
    metrics = prefetcher.metrics

    assert [batch["value"] for batch in batches] == [0, 10, 20, 30, 40]
    assert source.calls == [0, 1, 2, 3, 4]
    assert transfer.calls == [0, 1, 2, 3, 4]
    assert prefetcher.state == "exhausted"
    assert metrics.state == "exhausted"
    assert metrics.transfer == "recording"
    assert metrics.capacity == 2
    assert metrics.queue_depth == 0
    assert 1 <= metrics.max_queue_depth <= 2
    assert metrics.produced_batches == 5
    assert metrics.consumed_batches == 5
    assert metrics.read_seconds >= 0.0
    assert metrics.transfer_seconds >= 0.0
    assert metrics.backpressure_seconds >= 0.0
    assert metrics.producer_seconds >= 0.0
    assert metrics.consumer_wait_seconds >= 0.0


def test_capacity_blocks_read_ahead_until_consumer_releases_slot():
    source = BackpressureSource(count=6, capacity=2)
    prefetcher = AsyncBatchPrefetcher(source, capacity=2)
    prefetcher.start()

    assert source.capacity_reached.wait(timeout=2.0)
    assert source.calls == [0, 1]
    assert next(prefetcher) == {"value": 0}
    assert source.next_read.wait(timeout=2.0)
    assert source.calls[:3] == [0, 1, 2]
    prefetcher.close()
    assert prefetcher.state == "closed"


def test_worker_failure_is_raised_in_order_with_original_cause():
    source = RecordingSource(5, fail_at=2)
    prefetcher = AsyncBatchPrefetcher(source, capacity=3)

    assert next(prefetcher) == {"value": 0}
    assert next(prefetcher) == {"value": 1}
    with pytest.raises(PrefetchWorkerError, match="batch 2") as captured:
        next(prefetcher)
    assert isinstance(captured.value.__cause__, ValueError)
    assert "bad batch 2" in str(captured.value.__cause__)
    assert prefetcher.state == "failed"


def test_index_slice_and_early_close_are_deterministic():
    source = RecordingSource(10)
    prefetcher = AsyncBatchPrefetcher(
        source,
        capacity=2,
        start_index=3,
        stop_index=7,
    )

    assert [batch["value"] for batch in prefetcher] == [3, 4, 5, 6]
    assert source.calls == [3, 4, 5, 6]

    blocked_source = BackpressureSource(count=10, capacity=2)
    blocked = AsyncBatchPrefetcher(blocked_source, capacity=2)
    blocked.start()
    assert blocked_source.capacity_reached.wait(timeout=2.0)
    blocked.close()
    assert blocked.state == "closed"
    assert blocked.metrics.produced_batches == 2
    blocked.close()


def test_backend_transfer_delegates_without_backend_imports():
    class FakeBackend:
        name = "fake-tpu"

        def prepare_batch(self, batch):
            return {**batch, "prepared": True}

    transfer = BackendBatchTransfer(FakeBackend())
    prefetcher = AsyncBatchPrefetcher(
        RecordingSource(1),
        transfer=transfer,
    )

    assert list(prefetcher) == [{"value": 0, "prepared": True}]
    assert prefetcher.metrics.transfer == "backend:fake-tpu"


def test_prefetch_rejects_invalid_geometry_and_adapters():
    source = RecordingSource(1)
    with pytest.raises(ValueError, match="capacity"):
        AsyncBatchPrefetcher(source, capacity=0)
    with pytest.raises(ValueError, match="start_index"):
        AsyncBatchPrefetcher(source, start_index=2)
    with pytest.raises(ValueError, match="stop_index"):
        AsyncBatchPrefetcher(source, start_index=1, stop_index=0)
    with pytest.raises(TypeError, match="prepare_batch"):
        BackendBatchTransfer(object())
