# Deterministic shard ordering and host partitioning

`plan_packed_batch_partition` creates a versioned rank-local schedule from a
validated contiguous reader. It uses no process-global RNG, distributed API,
backend import, filesystem listing, or Python hash value.

## Dataset identity

The dataset fingerprint binds:

- declared shard order and IDs;
- every shard's content SHA-256 and token count;
- batch size and sequence length; and
- complete-batch and dropped-token counts per shard.

A partition plan cannot attach to a reader with a different fingerprint or
batch count. The reader also recomputes the complete expected plan, rejecting a
modified schedule even when its JSON remains structurally valid.

## Training order

Training computes a SHA-256 ordering key from the schema domain, seed, epoch,
shard ID, and original shard index. Shards are sorted by that key while batches
inside each shard remain sequential. This changes shard order across epochs
without converting mmap access into per-sample random I/O.

After constructing one global batch order, TrainLM retains a prefix divisible
by `world_size`, divides it into equal contiguous slices, and assigns one slice
to each rank:

```text
N = retained_batch_count / world_size
rank r <- retained_global_order[r*N:(r+1)*N]
```

Consequently, ranks have equal batch counts, no batch is owned twice, and the
union of rank schedules exactly covers the retained global schedule. Contiguous
ownership also preserves sequential reads and limits the number of shard maps
opened by each host.

## Remainder policy

M3-F4 never creates a batch spanning two shard files. A shard-tail remainder
can be:

- `drop`: excluded and reported as `dropped_token_count`; or
- `error`: rejected before iteration.

Likewise, a final global batch remainder that cannot divide evenly across hosts
can be dropped and reported as exact `dropped_batch_indices`, or rejected.
There is no implicit repetition or padding, because either would violate
exactly-once ownership and complicate exact resume.

## Validation behavior

Validation requires `seed=0` and `epoch=0`, preserves declared shard order, and
uses the same disjoint host partitioning. This makes repeated validation order
stable while retaining equal rank lengths for collective-safe execution.

## Usage

```python
from trainlm.data import (
    PartitionedPackedBatchReader,
    plan_packed_batch_partition,
)

plan = plan_packed_batch_partition(
    reader,
    split="train",
    seed=17,
    epoch=3,
    world_size=8,
    rank=process_index,
)
rank_reader = PartitionedPackedBatchReader(reader, plan)

for batch in rank_reader:
    result = task.training_step(model, batch, backend)
```

The plan serializes deterministically and includes dataset and global schedule
fingerprints. Its language-neutral contract is
`schemas/data/batch_partition_v1.schema.json`. M3-F6 will bind its exact resume
cursor to these identities.
