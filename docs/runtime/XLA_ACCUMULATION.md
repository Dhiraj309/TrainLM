# XLA accumulation strategy

TrainLM keeps gradient accumulation as an explicit strategy decision. The
current safe baseline is the existing generic `microstep` loop: it preserves
token-normalized loss and gradient semantics without assuming a compiler or
HBM profile.

The selector models four paths:

- `microstep`: backend-neutral loop and the default fallback;
- `unrolled`: one statically shaped compiled update containing all microsteps;
- `xla_loop`: a backend loop/scan implementation when the installed XLA
  version exposes and validates one;
- `native`: a future backend-owned accumulation primitive.

`select_accumulation_plan()` chooses an optimized path only when the benchmark
evidence says compilation, dispatch, and HBM headroom are all supported. An
explicit request falls back to `microstep` when `allow_fallbacks=True`, and
raises before training otherwise. The selected plan records the microbatch,
sequence length, accumulation count, evidence, and reason so benchmark output
cannot hide a fallback.

The v5e-8 decision must compare compiled microstep/update, unrolled GA32,
XLA loop/scan, and native candidates at MB2/S2048. This comparison remains a
TPU benchmark task; no strategy is marked as an automatic default by this
stage.
