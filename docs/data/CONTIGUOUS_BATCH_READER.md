# Contiguous packed batch reader

`ContiguousPackedBatchReader` converts validated packed shards into fixed-shape
causal-language-model batches without per-sample memmap indexing. It consumes a
source-neutral structural contract: shard ID, local data path, manifest, and
matching validation result.

## Batch geometry

For batch size `B` and sequence length `S`, one read consumes exactly `B * S`
contiguous tokens from one shard and returns:

| Field | Shape | Dtype | Behavior |
|---|---:|---|---|
| `input_ids` | `[B, S]` | `torch.int64` | One contiguous converted token region |
| `labels` | `[B, S]` | `torch.int64` | Aliases `input_ids`; shift remains task-owned |
| `attention_mask` | `[B, S]` | `torch.bool` | All true for packed fixed-length input |
| `loss_mask` | `[B, S]` | `torch.bool` | Aliases the attention mask |

The batch also carries `shard_id`, global `batch_index`, and shard-local
`token_offset` metadata. The forward-aware dispatcher removes these metadata
fields before invoking a model.

TrainLM's causal task owns the next-token shift. The reader must not pre-shift
labels or read a hidden extra token, because doing either would create different
loss semantics between data backends.

## Native fast path

The production `legacy_1024_uint16` format maps the complete file copy-on-write,
creates one typed buffer view after the 1,024-byte header, narrows one whole
batch, and performs one copy/conversion to contiguous `int64`. It does not call
Python once per sample or token.

Non-native byte order remains correct through an explicit portable decode path,
but it is not the performance path intended for TPU training.

## Shard boundaries and remainders

M3-F3 never combines two shards. It exposes, for every shard:

- the number of complete batches; and
- the number of remainder tokens excluded from this reader geometry.

This makes data loss measurable. M3-F4 will define deterministic partitioning,
shuffle, and cross-shard policy; M3-F3 does not make those policy decisions
implicitly.

```python
from trainlm.data import ContiguousPackedBatchReader

with ContiguousPackedBatchReader(
    resolved_shards,
    batch_size=16,
    sequence_length=2048,
) as reader:
    for batch in reader:
        result = task.training_step(model, batch, backend)
```

Mappings open lazily on first access and are released by `close()` or the
context manager. File size is checked again when a mapping opens, catching a
payload replaced after its original integrity validation.
