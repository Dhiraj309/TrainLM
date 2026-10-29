# TPU worker checkpoint and evaluation lifecycle

The public TPU path accepts `save_steps`, `eval_steps`,
`per_device_eval_batch_size`, and `train(resume_from_checkpoint=...)`.
Supply validation data as a `PackedBinDataset` or local manifest directory,
just as for training. Host validation checks shard integrity before launch.
All ranks evaluate equally sized deterministic validation partitions; incomplete
batches and rank remainders are dropped. Metrics reduce loss sums and supervised
token counts across ranks before computing loss and perplexity. Each cadence
boundary evaluates before saving, matching the backend-neutral trainer engine.

## Checkpoint ownership and publication

Each worker writes its full replicated model, optimizer slots keyed by canonical
parameter name, scheduler, runtime, trainer progress, CPU/XLA RNG state, and
packed-data cursor to `checkpoint-N/rank-R.pt`. The format is versioned as
`trainlm_replicated_worker`, independently of the M7 sharded resume manifest.
Rank payloads are written through temporary files. Rank zero publishes
`commit.json` only after every rank reports a successful write and matching
configuration/progress. Checksums and byte sizes detect corrupt/truncated rank
payloads. A shared output filesystem is required; this is synchronous local
publication, without asynchronous uploads, resharding, or retention deletion.
Existing checkpoint directories are never overwritten, including interrupted
transactions; use a new output directory when retrying such a save.

The data cursor uses **completed training microbatches**, not the position of a
prefetch producer. Resume validates the saved partition fingerprint and cursor,
then positions the data source before creating the XLA parallel loader. Loader
generators are isolated from the model RNG. This preserves the next training
batch even if a previous run prefetched additional data.

## Resume restrictions

Resume requires the same topology, precision, model source/config/revision,
dataset and partition, training/evaluation geometry, optimizer, scheduler,
evaluation cadence, and PyTorch/XLA versions. Every worker verifies its own
checksum, and rank errors are exchanged before continuation. Linear/cosine
schedules must retain their original `max_steps` horizon. Constant and token-WSD
schedules may extend `max_steps`, provided enough complete batches remain.
CPU/CUDA facade checkpoints and M7 sharded artifacts are not accepted by this
private worker format. Only complete committed checkpoints are resumable.

## Validation boundary

CPU tests cover bit-identical interrupted/uninterrupted training with dropout
and both ordinary and compact AdamW moments, data consumption despite prefetch,
save/evaluation cadence through the real worker body with CPU runtime doubles,
weighted cross-rank evaluation, two-rank publication/failure coordination, and
rejection of corrupt/incompatible state. These checks establish implementation
behavior without TPU certification. The owner must compare interrupted and
uninterrupted next updates, RNG/data continuity, evaluation metrics, and
checkpoint artifacts on Kaggle/Cloud TPU before marking the hardware gate done.

The complete CPU regression command was
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q`.
It executed 341 tests: 335 passed and 6 failed, matching the six pre-existing
onboarding failures (five missing Phi padding configuration errors and one
TrainLM save/reload weight mismatch). All new lifecycle tests passed.

Standalone TPU `evaluate()`, public model export/state-save methods, and
max-tokens stopping remain separate unfinished stories. The existing worker
`--export-hf` flag exports final model weights only; it is not a resume checkpoint.
