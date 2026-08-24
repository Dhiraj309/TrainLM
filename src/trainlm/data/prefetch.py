"""Bounded asynchronous batch prefetch with replaceable transfer policy."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
import threading
import time
from typing import Any, Literal, Protocol

PrefetchState = Literal["new", "running", "exhausted", "failed", "closed"]


class BatchSource(Protocol):
    """Random-access batch source used by the prefetch worker."""

    def __len__(self) -> int: ...

    def read_batch(self, index: int) -> Any: ...


class BatchTransfer(Protocol):
    """Backend-supplied preparation performed after a host batch read."""

    @property
    def name(self) -> str: ...

    def transfer(self, batch: Any) -> Any: ...


class IdentityBatchTransfer:
    """Leave batches on the host for backend-neutral consumption."""

    name = "identity"

    def transfer(self, batch: Any) -> Any:
        return batch


class BackendBatchTransfer:
    """Adapt an execution backend's batch preparation to prefetch policy."""

    def __init__(self, backend: Any) -> None:
        prepare_batch = getattr(backend, "prepare_batch", None)
        name = getattr(backend, "name", None)
        if not callable(prepare_batch):
            raise TypeError("backend must provide a callable prepare_batch method.")
        if not isinstance(name, str) or not name.strip():
            raise TypeError("backend must provide a non-empty name.")
        self.backend = backend
        self.name = f"backend:{name}"

    def transfer(self, batch: Any) -> Any:
        return self.backend.prepare_batch(batch)


@dataclass(frozen=True, slots=True)
class PrefetchMetrics:
    """Thread-safe point-in-time counters for one prefetch lifecycle."""

    state: PrefetchState
    transfer: str
    capacity: int
    queue_depth: int
    max_queue_depth: int
    produced_batches: int
    consumed_batches: int
    read_seconds: float
    transfer_seconds: float
    backpressure_seconds: float
    producer_seconds: float
    consumer_wait_seconds: float


@dataclass(frozen=True, slots=True)
class _ReadyBatch:
    index: int
    batch: Any


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    index: int
    error: Exception


class _EndOfSource:
    pass


_END = _EndOfSource()


class PrefetchWorkerError(RuntimeError):
    """The asynchronous reader or transfer policy failed."""


class AsyncBatchPrefetcher:
    """Single-producer ordered queue with strict capacity backpressure."""

    def __init__(
        self,
        source: BatchSource,
        *,
        capacity: int = 16,
        transfer: BatchTransfer | None = None,
        start_index: int = 0,
        stop_index: int | None = None,
    ) -> None:
        if not hasattr(source, "read_batch") or not callable(source.read_batch):
            raise TypeError("source must provide a callable read_batch method.")
        try:
            source_length = len(source)
        except (TypeError, ValueError) as exc:
            raise TypeError("source must provide a finite batch count.") from exc
        if (
            isinstance(source_length, bool)
            or not isinstance(source_length, int)
            or source_length < 0
        ):
            raise ValueError("source length must be a non-negative integer.")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or not 0 <= start_index <= source_length
        ):
            raise ValueError("start_index must be within the source.")
        final_index = source_length if stop_index is None else stop_index
        if (
            isinstance(final_index, bool)
            or not isinstance(final_index, int)
            or not start_index <= final_index <= source_length
        ):
            raise ValueError(
                "stop_index must be between start_index and source length."
            )
        selected_transfer = IdentityBatchTransfer() if transfer is None else transfer
        if not callable(getattr(selected_transfer, "transfer", None)):
            raise TypeError("transfer must provide a callable transfer method.")
        transfer_name = getattr(selected_transfer, "name", None)
        if not isinstance(transfer_name, str) or not transfer_name.strip():
            raise TypeError("transfer must provide a non-empty name.")

        self.source = source
        self.capacity = capacity
        self.transfer = selected_transfer
        self.start_index = start_index
        self.stop_index = final_index
        self._queue: Queue[_ReadyBatch | _WorkerFailure | _EndOfSource] = Queue(
            maxsize=capacity
        )
        self._slots = threading.Semaphore(capacity)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: PrefetchState = "new"
        self._produced_batches = 0
        self._consumed_batches = 0
        self._max_queue_depth = 0
        self._read_seconds = 0.0
        self._transfer_seconds = 0.0
        self._backpressure_seconds = 0.0
        self._producer_started_at: float | None = None
        self._producer_finished_at: float | None = None
        self._consumer_wait_seconds = 0.0

    def __iter__(self) -> "AsyncBatchPrefetcher":
        return self

    def __next__(self) -> Any:
        if self._state == "new":
            self.start()
        if self._state in {"exhausted", "closed"}:
            raise StopIteration
        if self._state == "failed":
            raise RuntimeError("Failed prefetcher cannot be consumed again.")

        wait_started = time.perf_counter()
        item = self._queue.get()
        wait_seconds = time.perf_counter() - wait_started
        self._slots.release()
        with self._lock:
            self._consumer_wait_seconds += wait_seconds

        if isinstance(item, _ReadyBatch):
            with self._lock:
                self._consumed_batches += 1
            return item.batch
        if isinstance(item, _WorkerFailure):
            self._state = "failed"
            self._stop.set()
            self._join_worker()
            raise PrefetchWorkerError(
                f"Prefetch worker failed while preparing batch {item.index}."
            ) from item.error

        self._state = "exhausted"
        self._join_worker()
        raise StopIteration

    @property
    def state(self) -> PrefetchState:
        return self._state

    @property
    def metrics(self) -> PrefetchMetrics:
        with self._lock:
            if self._producer_started_at is None:
                producer_seconds = 0.0
            else:
                finished = self._producer_finished_at or time.perf_counter()
                producer_seconds = finished - self._producer_started_at
            return PrefetchMetrics(
                state=self._state,
                transfer=self.transfer.name,
                capacity=self.capacity,
                queue_depth=self._queue.qsize(),
                max_queue_depth=self._max_queue_depth,
                produced_batches=self._produced_batches,
                consumed_batches=self._consumed_batches,
                read_seconds=self._read_seconds,
                transfer_seconds=self._transfer_seconds,
                backpressure_seconds=self._backpressure_seconds,
                producer_seconds=producer_seconds,
                consumer_wait_seconds=self._consumer_wait_seconds,
            )

    def start(self) -> None:
        if self._state != "new":
            raise RuntimeError(f"Cannot start prefetcher in state '{self._state}'.")
        self._state = "running"
        self._thread = threading.Thread(
            target=self._produce,
            name="trainlm-batch-prefetch",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._state == "closed":
            return
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
            else:
                self._slots.release()
        self._join_worker()
        self._state = "closed"

    def __enter__(self) -> "AsyncBatchPrefetcher":
        if self._state == "closed":
            raise RuntimeError("Closed prefetcher cannot be reused.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _produce(self) -> None:
        with self._lock:
            self._producer_started_at = time.perf_counter()
        try:
            for index in range(self.start_index, self.stop_index):
                if not self._acquire_slot():
                    return
                if self._stop.is_set():
                    self._slots.release()
                    return
                try:
                    read_started = time.perf_counter()
                    batch = self.source.read_batch(index)
                    read_seconds = time.perf_counter() - read_started
                    transfer_started = time.perf_counter()
                    batch = self.transfer.transfer(batch)
                    transfer_seconds = time.perf_counter() - transfer_started
                except Exception as exc:
                    self._queue.put_nowait(_WorkerFailure(index, exc))
                    self._record_queue_depth()
                    return
                if self._stop.is_set():
                    self._slots.release()
                    return
                self._queue.put_nowait(_ReadyBatch(index, batch))
                with self._lock:
                    self._produced_batches += 1
                    self._read_seconds += read_seconds
                    self._transfer_seconds += transfer_seconds
                self._record_queue_depth()

            if self._acquire_slot():
                if self._stop.is_set():
                    self._slots.release()
                else:
                    self._queue.put_nowait(_END)
                    self._record_queue_depth()
        finally:
            with self._lock:
                self._producer_finished_at = time.perf_counter()

    def _acquire_slot(self) -> bool:
        started = time.perf_counter()
        while not self._stop.is_set():
            if self._slots.acquire(timeout=0.1):
                with self._lock:
                    self._backpressure_seconds += time.perf_counter() - started
                return True
        with self._lock:
            self._backpressure_seconds += time.perf_counter() - started
        return False

    def _record_queue_depth(self) -> None:
        depth = self._queue.qsize()
        with self._lock:
            self._max_queue_depth = max(self._max_queue_depth, depth)

    def _join_worker(self) -> None:
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("Prefetch worker did not stop within 5 seconds.")
