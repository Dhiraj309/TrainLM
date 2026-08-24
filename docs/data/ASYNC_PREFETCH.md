# Asynchronous batch prefetch

TrainLM overlaps packed-shard reads with training through
`AsyncBatchPrefetcher`. The queue is deliberately independent of PyTorch,
PyTorch/XLA, and any future TorchTPU runtime so data iteration does not import
or select an accelerator backend.

## Contract

- One producer reads monotonically increasing batch indices, so consumer order
  is identical to source order.
- A semaphore is acquired before each read. The configured capacity therefore
  bounds batches being read, transferred, or waiting in the queue, not only
  completed queue entries.
- Capacity defaults to 16. This is a starting point for the TPU benchmark
  matrix, not a claim that 16 is optimal for every model or storage path.
- `start_index` and `stop_index` select an exact half-open source range. The
  [resumable cursor](RESUMABLE_CURSOR.md) supplies the next range when a
  consumer chooses to prefetch a resumed segment.
- Reader or transfer failures are delivered after every earlier ready batch,
  then raised as `PrefetchWorkerError` with the original exception as its
  cause.
- `close()` stops read-ahead and joins the producer. It is idempotent and must
  be used when consumption ends early; a context manager provides this cleanup
  boundary.

The source needs only `__len__()` and `read_batch(index)`. This allows the
contiguous reader, deterministic rank-partition reader, and later compatible
sources to share the same queue.

## Transfer ownership

The default `IdentityBatchTransfer` leaves each batch on the host. This is the
safe policy while the task or trainer owns `backend.prepare_batch()`.

`BackendBatchTransfer` is an opt-in adapter that invokes the selected execution
backend's `prepare_batch()` in the producer. When it is selected, the trainer
must treat the yielded batch as already prepared and must not prepare or move
it again. There is exactly one preparation boundary per batch. M4 trainer
integration will make that ownership explicit rather than performing a second
backend call.

This adapter makes asynchronous host-to-device preparation possible where a
backend supports it without placing XLA behavior in the data core. A backend
that requires device operations on the training thread should retain the
identity policy.

## Observability and tuning

`PrefetchMetrics` provides queue depth, maximum depth, produced/consumed batch
counts, and cumulative read, transfer, backpressure, producer, and consumer
wait time. Reading metrics does not synchronize an accelerator.

TPU tuning should compare a small bounded capacity matrix around 16 using real
shards. A useful queue hides storage latency without increasing host memory
without bound. Sustained consumer wait suggests an input or transfer
bottleneck; sustained backpressure with negligible consumer wait suggests the
queue is already deep enough. The final capacity is selected from end-to-end
throughput, host memory, and input-idle evidence in M11-F6.

The queue preserves semantics but does not by itself prove accelerator overlap.
That claim requires a target-TPU profile showing input and transfer activity
relative to compiled device execution.
