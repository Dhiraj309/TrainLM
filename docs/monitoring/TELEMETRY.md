# Sync-safe TPU telemetry

`TelemetryRecorder` collects already-synchronized, host-boundary samples and
returns an immutable `TelemetrySnapshot`.  Each `StepTelemetry` sample records
actual supervised tokens, wall/device seconds, compile time/count, peak HBM,
input-idle fraction, collective time, and CPU-fallback counters.

The recorder intentionally has no tensor `.item()` calls, device waits, logging,
or callbacks in the training hot path.  The backend/trainer must synchronize
before materializing a sample and should record samples sparsely (for example,
only during a benchmark window or every configured logging interval).  An
unsynchronized sample is rejected rather than producing an untrusted metric.

The snapshot derives global and device tokens/second from actual supervised
tokens and measured synchronized windows.  Compile and fallback counters are
reported separately so a fast result with hidden recompilation or CPU work
cannot be mistaken for a valid TPU measurement.  MFU is computed from the
existing benchmark schema once model FLOPs and device peak are supplied.

## TPU acceptance procedure

1. Warm up the permitted graph set outside the measured window.
2. Synchronize before and after each measured device step.
3. Record a bounded sample without host callbacks in the compiled callable.
4. Export the snapshot with runtime versions, geometry, compile count, HBM, and
   fallback counters.
5. Feed the synchronized window into `BenchmarkResult` for throughput/MFU
   validation.
