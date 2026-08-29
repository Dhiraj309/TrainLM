# Distributed resume-state planning

M7-F1 defines the state-ownership gate before checkpoint I/O is distributed.
`plan_distributed_resume(manifest)` validates a committed `ResumeManifest` and
returns a backend-neutral `DistributedResumePlan` for the participating ranks.

The plan requires the canonical state set:

- model and optimizer parameters/slots;
- scheduler, trainer, and runtime state;
- per-rank RNG state; and
- per-worker data state with an exact cursor for every replica.

For a multi-rank topology, model and optimizer artifacts must be sharded, RNG
must be per-rank, data must be per-worker, and every rank must have an exact
cursor.  A `direct_shard_io` flag records whether the backend may write shards
without gathering a global state; it is a capability decision, not permission
to change the canonical manifest.

Incomplete or failed transactions are rejected before any allocation or file
write.  Backend-specific writers, resharding, and asynchronous publication are
deliberately separate from this validation boundary and are implemented by the
following M7 features.
