# Dense-AR release certification tiers

TrainLM release certification is cumulative and commit-specific:

| Tier | Environment | Required evidence |
| --- | --- | --- |
| 0 | CPU CI | contracts, unit tests, public API and export smoke |
| 1 | CUDA CI | dense-AR correctness, update/resume, fallback diagnostics |
| 2 | Scheduled v5e-8 | multi-family correctness, graph stability, checkpoint/export |
| 3 | Release v5e-8 | current parity, performance, HBM, stability, and reproducibility artifacts |

Tier 0 runs for ordinary pushes and pull requests. Tiers 1 and 2 belong on
appropriately labelled hardware runners; Tier 2 is scheduled so dependency and
runtime drift is detected even without code changes. Tier 3 runs for a release
candidate only after its workload and comparison targets are review-locked.

A release is certified only when all four tiers passed for the exact same
40-character commit SHA, each result links a non-empty artifact, and no result
is older than the configured evidence window (14 days by default). Missing,
failed, stale, future-dated, or wrong-commit evidence blocks release. Tier 3 is
therefore mandatory rather than an optional performance report.

The repository’s existing CI test suite supplies the Tier 0 foundation. CUDA and TPU runner
workflows and real Tier 1-3 evidence are still release blockers; documentation
or synthetic evaluator fixtures are not hardware certification.
