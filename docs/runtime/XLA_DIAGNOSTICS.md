# XLA diagnostics and profiling artifacts

Diagnostics are opt-in so ordinary training does not add host-side metrics
work to the hot path. Inject a metrics module in tests, or let `XlaRuntime`
load the installed PyTorch/XLA metrics module on TPU:

```python
runtime = XlaRuntime(
    precision="bf16",
    collect_diagnostics=True,
)

# Call at a sparse host boundary, never for every tensor operation.
snapshot = runtime.collect_diagnostics()
```

Snapshots include compile and execute sample counts, `aten::` fallback
counters, bounded short/full metrics reports, optional HLO text, and optional
XProf metadata. `record_hlo()` and `record_profile_metadata()` let a TPU
runner attach artifacts captured by its approved XLA/XProf tooling without
making those tools mandatory dependencies.

`clear_diagnostics()` resets the backend metrics between cold-compile,
warm-cache, and steady-state windows. Reports and counter maps are bounded to
prevent an accidental unbounded artifact or log payload. Runtime diagnostics
also expose whether collection is enabled; `state_dict()` stores the latest
snapshot for checkpoint/evidence plumbing.

The official [PyTorch/XLA metrics API](
https://docs.pytorch.org/xla/master/learn/api-guide.html) exposes
`metrics_report()`, metric sample data, counter names, and counter values.
Counters beginning with `aten::` identify potential CPU fallback/context-switch
operations. [OpenXLA XProf](https://openxla.org/xprof/pytorch_xla_profiling)
traces remain an explicit profiling concern and are not enabled automatically.
