# Training-integrity gates

`check_training_integrity` provides sparse, configurable checks at a host
boundary after an optimizer update.  It returns an immutable
`IntegrityReport`; callers can stop the run or persist the violations before
the next checkpoint.

Default checks cover:

- finite loss, gradients, and parameters;
- optional maximum update norm;
- expected versus actual supervised-token delta; and
- optional exact data-cursor continuity.

Checks can be disabled individually for experiments, but a production profile
should keep all numerical checks enabled and require token continuity.  Cursor
continuity is enabled when the data pipeline can provide its exact state.

The implementation materializes values only at the configured integrity
interval.  It must not be called from a compiled device step or every
microstep, so normal training remains free of host synchronization and scalar
extraction overhead.

## TPU acceptance procedure

1. Run a finite-update smoke test with all checks enabled.
2. Inject NaN/Inf loss, gradient, and parameter values and verify the report
   fails with actionable violations.
3. Inject token and cursor discontinuities and verify the corresponding gates
   fail.
4. Run a 200-update target-TPU segment with sparse checks and record overhead,
   trusted metrics, and the resulting report alongside the checkpoint.

The integrity report is a safety gate, not a replacement for synchronized
throughput/MFU telemetry or checkpoint manifest validation.
