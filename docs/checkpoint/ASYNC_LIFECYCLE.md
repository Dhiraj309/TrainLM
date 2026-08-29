# Asynchronous checkpoint lifecycle

`AsyncCheckpointLifecycle` is the backend-neutral host state machine for an
asynchronous checkpoint writer.  It separates an in-flight transaction from a
committed transaction so compute never treats a staged or partially written
checkpoint as durable.

Lifecycle rules:

- `begin(id)` enters `in_flight` and clears the previous error;
- `complete()` enters `committed`, marks the ID durable, and applies bounded
  retention;
- `fail(error)` enters `failed` without adding the ID to retained checkpoints;
- `shutdown()` rejects new work and invalidates any in-flight transaction; and
- `snapshot()` returns immutable host-visible state for callbacks and logs.

The state machine does not perform file I/O, barriers, or backend synchronization.
Writers must call `complete()` only after all shard writes, worker barriers,
manifest verification, and atomic publication have succeeded.  A failure or
shutdown before that point must remain non-durable and resumable readers must
ignore it.

M7-F2 therefore provides the lifecycle contract; backend-specific async writers
and overlap measurements can be added without changing the checkpoint schema.
