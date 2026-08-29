# Sync-safe callbacks

TrainLM keeps callback execution on the host side of the device/host
boundary. Existing lifecycle hooks (`on_step_end`, `on_evaluate`, and related
hooks) remain compatible and receive only trainer state and control flags.

Callbacks that need metrics implement the additive `on_metrics` hook:

```python
from trainlm.training import TrainerCallback


class Logger(TrainerCallback):
    def on_metrics(self, state, control, metrics):
        del state, control
        print(dict(metrics))
```

`metrics` is an immutable `MetricSnapshot` containing only Python `float`
values. Live tensors are rejected rather than implicitly calling `.item()` in
the callback path. Training metrics are emitted at
`config.logging.log_every_steps`; evaluation metrics are emitted once after
streaming evaluation completes. Both boundaries are intentionally sparse.

The trainer materializes metrics before dispatching callbacks. Callback code
must not access model parameters, optimizer tensors, device batches, or force
backend synchronization. Backend-specific runtimes remain responsible for
their own compiled reductions and synchronization policy.
