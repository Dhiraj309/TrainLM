"""Bounded background batch prefetcher for overlapping input loading with step compute."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import queue
import threading
from typing import Any


_SENTINEL = object()


class AsyncBatchPrefetcher(Iterator[Any]):
    """Prefetch items from ``source`` on a background thread with bounded queue depth."""

    def __init__(
        self,
        source: Iterator[Any],
        *,
        transform: Callable[[Any], Any] | None = None,
        prefetch_depth: int = 2,
    ) -> None:
        if isinstance(prefetch_depth, bool) or not isinstance(prefetch_depth, int) or prefetch_depth < 1:
            raise ValueError("prefetch_depth must be a positive integer.")
        self._source = source
        self._transform = transform
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_depth)
        self._stop_event = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="trainlm-batch-prefetcher",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    item = next(self._source)
                except StopIteration:
                    self._put_or_stop((_SENTINEL, None))
                    return
                if self._transform is not None:
                    item = self._transform(item)
                if not self._put_or_stop((None, item)):
                    return
        except BaseException as exc:  # pragma: no cover - defensive forwarding
            self._put_or_stop((exc, None))

    def _put_or_stop(self, payload: Any) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(payload, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def __iter__(self) -> "AsyncBatchPrefetcher":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        payload, value = self._queue.get()
        if payload is _SENTINEL:
            self.close()
            raise StopIteration
        if isinstance(payload, BaseException):
            self.close()
            raise payload
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
