# Resumable packed-data cursor

`PackedDataCursor` is the consumer-side resume boundary for a deterministic
`PartitionedPackedBatchReader`. It records the next rank-local batch, rather
than the last batch that a background prefetch worker happened to read.

## State recorded

`PackedDataCursorState` is schema-versioned and JSON serializable. It binds the
cursor to:

- split, seed, epoch, rank, and world size;
- dataset and deterministic schedule fingerprints;
- the optional Hugging Face source revision;
- next partition index and its global batch, shard, local-batch, and token
  offset;
- consumed batch and input-token counters; and
- an optional data-pipeline RNG snapshot.

The location fields are derived from the partition plan and reader. Restoring a
state with a changed dataset, permutation, topology, or next-batch location is
rejected before reading. A completed partition uses null pending-location
fields, which makes end-of-epoch state unambiguous.

## Resume boundary

Advance the cursor only after the consumer has accepted a batch. If a
prefetcher has read ahead, close it before checkpointing; unconsumed prefetched
batches are intentionally discarded and will be reread from the saved next
partition index. Because the schedule and shard bytes are fingerprinted, the
next batch sequence is identical after restart.

```python
cursor = PackedDataCursor(rank_reader, source_revision=hf_revision)
for batch in cursor:
    train_step(batch)
    if should_checkpoint():
        save_data_cursor(cursor.state.to_json())

# On restart, load the JSON state and validate it against the rebuilt plan.
cursor = PackedDataCursor.from_state(
    rank_reader,
    PackedDataCursorState.from_json(saved_json),
    source_revision=hf_revision,
)
```

`tokens_consumed` counts fixed input tokens (`batch_size * sequence_length`),
not the task's supervised-token reduction. Trainer progress owns the latter;
both counters must be stored in an exact checkpoint.

The cursor does not write checkpoint files or coordinate distributed barriers.
Those lifecycle responsibilities belong to M7. Its contract is intentionally
small so a future backend can restore the same data schedule without importing
PyTorch/XLA or TorchTPU APIs.
