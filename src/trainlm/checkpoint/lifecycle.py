"""Thread-safe lifecycle state for asynchronous checkpoint publication."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

CheckpointPhase = Literal[
    "idle", "in_flight", "committed", "failed", "shutdown"
]


@dataclass(frozen=True, slots=True)
class CheckpointLifecycleSnapshot:
    """Host-visible checkpoint state; committed means durable and published."""

    phase: CheckpointPhase
    transaction_id: str | None
    durable: bool
    retained_checkpoints: tuple[str, ...]
    error: str | None


class AsyncCheckpointLifecycle:
    """Coordinate async save completion without claiming durability early."""

    def __init__(self, *, max_retained: int = 1) -> None:
        if isinstance(max_retained, bool) or not isinstance(max_retained, int):
            raise ValueError("max_retained must be a positive integer.")
        if max_retained < 1:
            raise ValueError("max_retained must be a positive integer.")
        self._lock = Lock()
        self._max_retained = max_retained
        self._phase: CheckpointPhase = "idle"
        self._transaction_id: str | None = None
        self._durable = False
        self._retained: list[str] = []
        self._error: str | None = None

    def _require(self, phase: CheckpointPhase) -> None:
        if self._phase != phase:
            raise RuntimeError(
                f"Checkpoint lifecycle is {self._phase}; expected {phase}."
            )

    def _snapshot_unlocked(self) -> CheckpointLifecycleSnapshot:
        return CheckpointLifecycleSnapshot(
            phase=self._phase,
            transaction_id=self._transaction_id,
            durable=self._durable,
            retained_checkpoints=tuple(self._retained),
            error=self._error,
        )

    def begin(self, transaction_id: str) -> CheckpointLifecycleSnapshot:
        """Start staging a transaction; it is not durable until ``complete``."""

        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise ValueError("transaction_id cannot be empty.")
        with self._lock:
            if self._phase == "shutdown":
                raise RuntimeError("Checkpoint lifecycle is shut down.")
            if self._phase == "in_flight":
                raise RuntimeError("A checkpoint transaction is already in flight.")
            self._phase = "in_flight"
            self._transaction_id = transaction_id
            self._durable = False
            self._error = None
            return self._snapshot_unlocked()

    def complete(self) -> CheckpointLifecycleSnapshot:
        """Publish a completed transaction and apply retention."""

        with self._lock:
            self._require("in_flight")
            assert self._transaction_id is not None
            self._phase = "committed"
            self._durable = True
            self._error = None
            self._retained.append(self._transaction_id)
            del self._retained[:-self._max_retained]
            return self._snapshot_unlocked()

    def fail(self, error: str) -> CheckpointLifecycleSnapshot:
        """Mark an in-flight transaction failed; it is never durable."""

        if not isinstance(error, str) or not error.strip():
            raise ValueError("error cannot be empty.")
        with self._lock:
            self._require("in_flight")
            self._phase = "failed"
            self._durable = False
            self._error = error
            return self._snapshot_unlocked()

    def shutdown(self) -> CheckpointLifecycleSnapshot:
        """Stop accepting work and invalidate any incomplete transaction."""

        with self._lock:
            if self._phase == "shutdown":
                return self._snapshot_unlocked()
            if self._phase == "in_flight":
                self._error = "shutdown before checkpoint completion"
                self._durable = False
            self._phase = "shutdown"
            return self._snapshot_unlocked()

    def snapshot(self) -> CheckpointLifecycleSnapshot:
        """Return an immutable, host-materialized lifecycle snapshot."""

        with self._lock:
            return self._snapshot_unlocked()
