# XLA compilation cache and static-shape guard

M5-F3 makes compiled PyTorch/XLA training predictable. `XlaRuntime` can
initialize the official persistent compilation cache before the first device
operation:

```python
runtime = XlaRuntime(
    precision="bf16",
    cache_dir="/tmp/trainlm-xla-cache",
)
```

The cache directory should be on durable, worker-local storage in production.
`cache_readonly=True` is useful for a pre-populated cache image. TrainLM does
not silently enable a cache path; the path is an explicit runtime choice.

The backend also records the first prepared batch's pytree, tensor dtypes, and
all dimensions. Later batches must keep the same mapping/tuple structure and
batch, sequence, and mask shapes. `Trainer.prepare()` registers the configured
gradient-accumulation count. A changed structure raises before it can trigger
an unbounded XLA recompilation. Configure the data loader to pad or drop the
last incomplete batch and keep masks structurally consistent. The trainer also
checks the actual microstep count of every update, so a token-limit boundary
cannot silently create a second compiled update shape.

`state_dict()` and `diagnostics()` expose a SHA-256 compilation fingerprint
covering the XLA version, precision, mesh, static batch contract, and
accumulation count. This is an evidence key for benchmark runs; it is not a
replacement for checkpoint contents.

The guard is intentionally backend-owned and model-agnostic. It does not edit
Hugging Face model implementations or require a family adapter.
