# Trainer lifecycle and control state

TrainLM owns a small explicit state machine around backend execution. The
state is observable through `TrainerState.phase`; callbacks and checkpoint
managers never need to infer lifecycle from a step counter.

```text
created -> prepared -> training -> stopping -> finalized
   |          |          |
   |          |          +-> evaluating -> training
   |          +------------> saving ----> prepared/training
   +-----------------------> resuming -> prepared

Any active phase -> failed -> stopping -> finalized
```

## Operation boundaries

- `prepare()` initializes the selected backend exactly once and enters
  `prepared`.
- `train()` enters `training`, dispatches begin/step/end callbacks, and always
  calls backend end and finalize hooks, including after a failure.
- `evaluate()` temporarily enters `evaluating`, restores model mode, emits
  `on_evaluate`, and returns to the previous active phase.
- `request_stop()` sets a persistent stop request. The loop exits only at a
  completed-step boundary, then enters `stopping`.
- `save_checkpoint()` and `load_checkpoint()` require explicit injected hooks.
  They wrap the hook with backend checkpoint barriers and publish save/resume
  callback events. M7 owns durable file formats and atomic publication.
- `finalize()` is available for a prepared, non-training trainer. Calling it
  during `training` is rejected so a caller cannot bypass end hooks.

Failure text is retained in `TrainerState.failure` even after resources reach
`finalized`. A finalized trainer cannot be reused; this prevents accidentally
running a second loop against released backend resources.

The state machine contains no device or model-family logic. XLA, CUDA, and a
future TorchTPU backend implement the existing `ExecutionBackend` protocol and
are selected without changing lifecycle semantics.
