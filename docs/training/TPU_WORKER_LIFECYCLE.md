# TPU worker checkpoint and evaluation lifecycle

The public TPU path accepts `save_steps`, `eval_steps`,
`per_device_eval_batch_size`, and `train(resume_from_checkpoint=...)`.
Supply validation data as a `PackedBinDataset` or local manifest directory,
just as for training. Host validation checks shard integrity before launch.
All ranks evaluate equally sized deterministic validation partitions; incomplete
batches and rank remainders are dropped. Metrics reduce loss sums and supervised
token counts across ranks before computing loss and perplexity. Each cadence
boundary evaluates before saving, matching the backend-neutral trainer engine.

## Standalone public operations

`trainer.evaluate()` evaluates the latest trained state and returns evaluation
metrics, also forwarding them to public metric callbacks. It requires validation
data. `trainer.save_model(output_dir)` writes a plain Hugging Face model directory
with canonical CPU weights and the original attention configuration; private XLA
attention registry names are not exported. `trainer.save_state(output_dir)` saves
full replicated training state and returns a unique committed checkpoint directory
under the requested output directory. That path can be passed directly to
`trainer.train(resume_from_checkpoint=path)` in a compatible new trainer.

Because workers exit after each operation, public training retains a private
final checkpoint under `.trainlm_state/` even when `save_steps` is unset. It is
written at a completed-update boundary before runtime teardown. Standalone
operations transparently reconstruct workers from that snapshot; they perform
no optimizer updates. Before training, the same methods use initial pretrained
state. A successful `save_state()` becomes the latest retained checkpoint.
Private `.trainlm_operations/` directories isolate operation logs and summaries
from training artifacts. Retained checkpoints cost disk space for complete
per-rank replicas; there is no automatic retention deletion.

## Token-based stopping

Set `max_steps=None, max_tokens=N` for a token-only run. N counts supervised
tokens **globally**, including every replica, and is an absolute progress limit
when resuming. A packed optimizer update contributes
`(sequence_length - 1) * per_device_train_batch_size * gradient_accumulation_steps * world_size`
supervised tokens. The worker rounds the token budget up to an optimizer-update
boundary, preserving fixed accumulation shapes. Overshoot is less than one full
update. When both step and token limits are configured, the first limit wins.
This TPU policy differs from the portable engine's partial-accumulation stopping.

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
schedules must retain their original effective step horizon (after applying any
token limit). Constant and token-WSD
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
It executed 357 tests: 351 passed and 6 failed, matching the six pre-existing
onboarding failures (five missing Phi padding configuration errors and one
TrainLM save/reload weight mismatch). The focused facade/worker/lifecycle group
passed all 44 tests; all new lifecycle tests passed.

Additional CPU tests cover reopening final trained state for standalone
evaluation/export/state saving, saving initial state and resuming it, token-only
and combined stopping limits, plain HF export/reload with weight ties, and failed
export propagation. Real TPU lifecycle correctness remains a hardware gate.
Public CLI/config alignment remains unfinished. The private worker's
`--operation` modes support these operations; its `--export-hf` flag continues to
export final model weights only, rather than a resume checkpoint.
