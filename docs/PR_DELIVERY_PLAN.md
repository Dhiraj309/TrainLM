# TrainLM sequential pull-request delivery plan

> **Scope:** Candidate PRs after legacy [PR #18](https://github.com/Dhiraj309/TrainLM/pull/18)
> merged into `main` on 2026-10-28. These are planning identifiers, not actual
> GitHub PR numbers or commitments. T0.1's read-only inventory is recorded in
> `TORCHTPU_SALVAGE_INVENTORY.md`; review and remap this XLA-oriented queue
> before creating the first implementation branch.

## Native TorchTPU T0–T11 rebaseline: proposed PR remapping

The [new target plan](TORCHTPU_REBASELINE.md) proposes native TorchTPU-first
execution. **The 18 candidate cards below reflect an XLA-oriented roadmap,
not an approved TorchTPU delivery queue.** T0 must inventory reusable and
XLA-specific implementation and remap these cards with owner approval
before starting plan 01. Keep the established multiple sequential PRs,
multiple focused commits per PR, matching branch/title/body and strict
merge-to-main-before-next-branch rules. T0–T11 are architectural gates,
not a requirement to create one giant PR per milestone.

## Development outcome and user experience

TrainLM should feel like Hugging Face to its users: supply an HF model/model
source, processing class/tokenizer, train/eval datasets and familiar training
arguments; call `TrainLMTrainer.train()`; receive reproducible progress,
evaluation, portable checkpoints/resume and canonical Transformers export.
Users do not manage PJRT rank topology, worker scripts, preflight, stage-log
parsing, internal manifests or caches. A CLI/YAML workflow must use the same
contracts. Preserve HF semantics, correct generic fallbacks and explain
optimized-provider choices; optimization must not hide data loss or weaken
checkpoint safety.

## PR-by-PR serial execution

**Legacy PR #18 merges → audit merged `main` → create PR-plan 01 branch from
that `main` → make multiple small commits → review/test/merge →
verify new `main` → create PR-plan 02 branch → repeat.**

- A **PR-plan number** is a planned delivery slot, not a GitHub PR number.
  GitHub assigns its own PR number on creation. Do not create the next branch
  until the previous PR is merged and the remote `main` SHA verified.
- The branch is the deliverable: `<type>/<NN>-<outcome-slug>`. Its name,
  PR title and PR description must all describe the **same set of changes**
  and user/engineering outcome. Never use generic `update`, `misc`,
  `codex`, or an obsolete milestone branch.
- A PR may include **multiple focused, incremental commits** (tests, features,
  bugs, regression fixtures, documentation, review changes). No fixed count
  or one-commit rule. Preserve commits at merge through merge-commit or
  rebase-merge (owner choice); no default squash or force-push.
- Keep PR scope reviewable; separate unrelated areas. New independent
  requirements are added to the next *approved* PR only after the predecessor
  merges, except urgent owner-authorized hotfixes.
- On every implementation turn, update `docs/ROADMAP.md` and
  `docs/IMPLEMENTATION_CONTEXT.md` as required by `AGENTS.md`. Record
  verified branch/base/head SHAs, CPU tests, TPU artifacts when applicable,
  known failures, rollback and decision. Do not fabricate results.
- For code infrastructure PRs, disabled-by-default experimental providers
  may merge after CPU/contract checks if the PR explicitly does **not** claim
  hardware certification; only benchmarks and target-TPU evidence permit
  declaring optimization defaults or release performance gates.
- Block rather than silently skip a mandatory acceptance criterion. If a
  performance gate is unattainable, document the measured failure and obtain
  an explicit owner-approved *planning revision* before changing the next PR.
- Legacy PR #18 is merged. Its tree and owner-run TPU evidence still require
  the post-merge audit; a successful merge does not establish native TorchTPU
  compatibility or make the XLA-oriented candidate queue automatically valid.

## Planned PR index

| Plan | Branch | GitHub PR title | Tier | Traceability |
|---:|---|---|---|---|
| 01 | `fix/01-cpu-regressions-test-baseline` | fix(test): establish a clean, reproducible CPU regression baseline | P0 | M2-F5, M13-F3, M14-F3 |
| 02 | `feature/02-tpu-trainer-lifecycle-resume` | feat(tpu): complete public trainer lifecycle, checkpoints and resume | P0 | M8-F0, M7-F1-F5 |
| 03 | `perf/03-chunked-loss-no-full-logits` | perf(loss): integrate chunked causal loss without full logits | P0 | M9-F1-F5 |
| 04 | `perf/04-compiled-gradient-accumulation` | perf(xla): reduce host-unrolled accumulation and recompilation | P0 | M5-F4-F5, M11-F7 |
| 05 | `perf/05-pallas-mha-attention` | perf(attention): add guarded TPU-native Pallas MHA | P0 | M10-F1-F3 |
| 06 | `perf/06-compact-gqa-window-alibi` | perf(attention): support compact GQA/MQA, windows and ALiBi | P0 | M10-F4-F6 |
| 07 | `perf/07-reversible-qkv-mlp-fusion` | perf(model): activate reversible QKV and gated-MLP fusion | P0 | M8-F4-F5, M11-F1-F3 |
| 08 | `perf/08-optimizer-remat-memory-policy` | perf(train): tune XLA optimizer state and rematerialization | P0 | M11-F4-F5 |
| 09 | `perf/09-data-prefetch-cache-graph-stability` | perf(input): overlap packed data, prefetch and XLA compilation cache | P0 | M5-F3, M11-F6 |
| 10 | `perf/10-async-telemetry-xla-profile` | perf(telemetry): add low-overhead TPU profiling and async reporting | P0 | M7-F4, M11-F7, telemetry backlog |
| 11 | `benchmark/11-optimized-loss-attention-850k-gate` | perf(benchmark): certify the optimized loss and attention 850K gate | P0 | M10-F7, M11-F7 |
| 12 | `validation/12-numerical-resume-hf-export-parity` | test(parity): validate numerical equivalence, resume and HF export | P0 | M12-F1, M12-F4, M7-F1-F5 |
| 13 | `benchmark/13-laughlm-135m-three-run-certification` | perf(parity): run 135M LaughLM-class three-run certification | P0 | M12-F2-F3-F5 |
| 14 | `validation/14-dense-family-conformance-matrix` | test(compat): certify dense HF family support and fallbacks | P1 | M6-F1-F4, M13-F1-F5 |
| 15 | `feature/15-fsdp-sharding-1p3b-scale` | feat(scale): validate FSDP sharding and 1.3B training | P1 | M13-F6-F7 |
| 16 | `feature/16-hf-like-cli-config-user-journey` | feat(ux): polish HF-like trainer, CLI and secure TPU workflows | P1 | M8-F0, M14-F1-F2 |
| 17 | `ci/17-release-ci-preemption-evidence` | ci(release): enforce reproducibility, hardware gates and recovery | P1 | M14-F3-F4 |
| 18 | `release/18-dense-ar-v1-support-certification` | release(v1): publish dense-AR support, limitations and evidence | P1 | M14-F5 |

**Candidate-only caveat:** Legacy PR #18 is merged and its code is in current
`main`. T0.1 has started the source-level salvage audit; **audit each entry**
against the merged tree and real TPU evidence. Mark satisfied
deliverables with evidence, consolidate or resequence redundant candidates in
a separately reviewed plan update, and number the *first newly activated
real branch* `01`. Don't implement the same code again to satisfy a table.

## Exact PR title, description and commit trace contract

For **each planned PR** below, the **Branch** and **Proposed PR title** fields
are the canonical, copyable planned values. The **PR description — why/outcome**,
scope, planned commits, acceptance, tests and rollback together form the
description; do not publish only the one-line outcome. At activation, produce
the actual PR body from the corresponding card using this structure:

```markdown
## Summary / outcome
[The exact card title, intended outcome, and problems it solves.]

## Desired user experience and technical result
[User-facing behavior; for an internal perf/validation PR, the measurable benefit.]

## Included changes
[In-scope interfaces and capabilities from the card; name the historical
M/F references only for traceability.]

## Excluded from this PR
[Explicit non-goals and later PR dependencies; no unrelated scope creep.]

## Planned commits and implementation trail
- [ ] <small, coherent implementation or profiling commit>
- [ ] <focused test/regression commit>
- [ ] <fix or follow-up commit, if necessary>
- [ ] <measurement/benchmark commit, if applicable>
- [ ] <docs and decision record commit>
[Replace plans with actual short SHAs and results before review.]

## Tests and measured evidence
- CPU command, pinned environment, result, and known failures:
- TPU command, immutable model/data revisions, hardware, result:
- HLO/compile-count, HBM, throughput/MFU, warmup and measurement window:
- Artifact links and independent reproduction instructions:
- Explicitly unmeasured capabilities / forbidden certification claims:

## Risk, compatibility, fallback and rollback
- HF semantic and public API compatibility:
- Disabled/opt-in provider, strict-mode behavior, fallback and rollback:
- Checkpoint, resume, export and process-cleanup implications:

## Delivery gate
- Intended base: main at <verified post-predecessor-merge SHA>
- Branch: <the planned branch, created only now>
- Previous delivery PR: <actual merged PR URL and verification>
- Required review/CI/evidence: <links and statuses>
- Merge method: preserve logical commits (merge commit or rebase merge)
- Owner approval required; no automatic merge

## Follow-ups
[Only next *planned* PR; do not create its branch until this PR is merged.]
```

At PR creation, **title must match the card** and the description must be
specific to the branch's actual commits—not a generic template left blank.
Commit titles are examples, not a demand to manufacture work or to combine
unrelated code into a PR. Mark actual committed changes and evidence accurately
during review. GitHub numbers are assigned only on creation; the numbers
01–18 are delivery-plan slots.

## Detailed PR cards

### Plan 01 — fix(test): establish a clean, reproducible CPU regression baseline

- **Branch:** `fix/01-cpu-regressions-test-baseline`
- **Proposed PR title:** `fix(test): establish a clean, reproducible CPU regression baseline`
- **PR description — why/outcome:** Differentiate six previously observed Phi/config and weight round-trip failures from dependency or fixture drift; make CPU CI deterministic before optimizing.
- **Scope and boundaries:** Failing tests, pinned environment, tiny model round-trips, CI fixtures; no TPU kernel changes.
- **Milestone lineage:** M2-F5, M13-F3, M14-F3; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `test(cpu): reproduce Phi config and state round-trip failures`
  - `fix(compat): address verified model or fixture regression only`
  - `test(ci): enforce reproducible pinned CPU matrix and round-trip checks`
  - `docs(validation): record baseline and unresolved accelerator evidence`
- **Acceptance / merge gate:** Pinned CPU suite passes or documented non-regression exemption reviewed; fixes preserve HF semantics and do not mask failures.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 02 — feat(tpu): complete public trainer lifecycle, checkpoints and resume

- **Branch:** `feature/02-tpu-trainer-lifecycle-resume`
- **Proposed PR title:** `feat(tpu): complete public trainer lifecycle, checkpoints and resume`
- **PR description — why/outcome:** A user calls TrainLMTrainer.train() without worker scripts; eval/save/resume and interruption handling work through the private TPU coordinator.
- **Scope and boundaries:** Facade/coordinator/worker lifecycle, bounded evaluation, rank state/checkpoint resume and failure cleanup; no new attention kernels.
- **Milestone lineage:** M8-F0, M7-F1-F5; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(tpu): unify coordinator-owned train and validation lifecycle`
  - `test(tpu): cover bounded eval, checkpoint and interrupt contracts`
  - `fix(checkpoint): enforce exact topology, RNG and data-position resume`
  - `test(tpu): add owner-run v5e-8 lifecycle smoke and evidence harness`
  - `docs(training): publish public UX and remaining hardware gates`
- **Acceptance / merge gate:** CPU simulations pass; owner-run v5e lifecycle succeeds or explicitly leaves unvalidated capabilities gated; no stage-log parsing for end users.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 03 — perf(loss): integrate chunked causal loss without full logits

- **Branch:** `perf/03-chunked-loss-no-full-logits`
- **Proposed PR title:** `perf(loss): integrate chunked causal loss without full logits`
- **PR description — why/outcome:** Real TPU training avoids full B×S×V logits when supported and retains canonical HF output semantics.
- **Scope and boundaries:** Guarded hidden-state training view, FP32 chunked CE, z-loss, rematerialized backward and fallback.
- **Milestone lineage:** M9-F1-F5; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(loss): wire inspected hidden-state view to training engine`
  - `perf(loss): select bounded chunking and non-reentrant recompute`
  - `test(loss): prove tied/untied gradient, ignore-index and export parity`
  - `fix(loss): handle unsupported heads through explicit fallback`
  - `docs(perf): record HBM/HLO measurements and eligibility`
- **Acceptance / merge gate:** Loss/gradient parity; HLO proves no full logits for the selected path; defaults remain opt-in until TPU HBM/throughput evidence.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 04 — perf(xla): reduce host-unrolled accumulation and recompilation

- **Branch:** `perf/04-compiled-gradient-accumulation`
- **Proposed PR title:** `perf(xla): reduce host-unrolled accumulation and recompilation`
- **PR description — why/outcome:** Decrease host overhead from GA32 and unstable XLA graph boundaries without changing token-normalized gradients.
- **Scope and boundaries:** Supported compiled accumulation strategies, static-shape controls, compile-count and fallback evidence.
- **Milestone lineage:** M5-F4-F5, M11-F7; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `test(xla): benchmark host-unrolled GA and graph compile counts`
  - `feat(xla): add guarded compiled accumulation strategy`
  - `test(training): compare updates, masks and gradient normalization`
  - `fix(xla): stabilize shapes and step boundaries`
  - `docs(perf): capture HLO and matched hardware results`
- **Acceptance / merge gate:** Equivalent optimizer updates and stable graphs; use fallback if compiled path does not beat baseline on matched target TPU.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 05 — perf(attention): add guarded TPU-native Pallas MHA

- **Branch:** `perf/05-pallas-mha-attention`
- **Proposed PR title:** `perf(attention): add guarded TPU-native Pallas MHA`
- **PR description — why/outcome:** Make causal MHA faster with correct attention semantics and a safe portable fallback.
- **Scope and boundaries:** Version-guarded HF AttentionInterface integration, kernel forward/backward, causal masking and fallback.
- **Milestone lineage:** M10-F1-F3; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(attention): bind eligible HF attention to guarded Pallas MHA`
  - `test(attention): compare forward, backward, masks and dtypes`
  - `fix(attention): reject unsupported layouts without silent fallback`
  - `perf(attention): collect HLO/custom-call, HBM and wall-time evidence`
  - `docs(attention): explain selected and rejected providers`
- **Acceptance / merge gate:** Forward/backward and masking parity; no automatic activation without target TPU performance and compilation evidence.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 06 — perf(attention): support compact GQA/MQA, windows and ALiBi

- **Branch:** `perf/06-compact-gqa-window-alibi`
- **Proposed PR title:** `perf(attention): support compact GQA/MQA, windows and ALiBi`
- **PR description — why/outcome:** Maintain MHA/GQA/MQA and causal/window/ALiBi correctness without hidden KV repetition or dense-mask blowup.
- **Scope and boundaries:** Head mapping, guard rails for sliding windows and ALiBi, tuning-key evidence.
- **Milestone lineage:** M10-F4-F6; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(attention): preserve compact grouped-query KV layouts`
  - `test(attention): cover MHA, GQA, MQA and gradient geometry`
  - `feat(attention): support guarded window and explicit ALiBi slopes`
  - `fix(attention): reject unsupported masks and dropout safely`
  - `perf(attention): benchmark tuning keys and HBM`
- **Acceptance / merge gate:** Representative head geometries and masks pass; target HBM confirms no unintended KV/mask blowup.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 07 — perf(model): activate reversible QKV and gated-MLP fusion

- **Branch:** `perf/07-reversible-qkv-mlp-fusion`
- **Proposed PR title:** `perf(model): activate reversible QKV and gated-MLP fusion`
- **PR description — why/outcome:** Improve TPU kernels while preserving HF parameter semantics, optimizer updates and canonical export.
- **Scope and boundaries:** Selected reversible QKV/partial-QKV and SwiGLU/GeGLU transforms, alias safety and fusion audit.
- **Milestone lineage:** M8-F4-F5, M11-F1-F3; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(model): apply guarded QKV layout transforms before optimizer`
  - `test(model): verify tied aliases, rollback and canonical conversion`
  - `feat(model): apply guarded gated-MLP transforms`
  - `fix(model): preserve gradient/update semantics for partial layouts`
  - `perf(model): audit HLO fusion and transpose/copy costs`
- **Acceptance / merge gate:** Reversible parity and clean export; activate only transformations showing measured HLO/performance benefit.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 08 — perf(train): tune XLA optimizer state and rematerialization

- **Branch:** `perf/08-optimizer-remat-memory-policy`
- **Proposed PR title:** `perf(train): tune XLA optimizer state and rematerialization`
- **PR description — why/outcome:** Reduce HBM and update stalls with explicit precision and recompute policies, without checkpoint drift.
- **Scope and boundaries:** AdamW state/update policy, selective remat, gradient clipping/decay and resume integrity.
- **Milestone lineage:** M11-F4-F5; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `test(optimizer): lock AdamW and checkpoint numerical reference`
  - `feat(optimizer): integrate guarded dtype/state update policy`
  - `perf(memory): select remat policy from measured HBM budget`
  - `test(checkpoint): cover updates, restore and step boundaries`
  - `docs(perf): report throughput versus memory tradeoff`
- **Acceptance / merge gate:** Numerical update/resume parity; improvements confirmed on v5e; no unsafe precision default.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 09 — perf(input): overlap packed data, prefetch and XLA compilation cache

- **Branch:** `perf/09-data-prefetch-cache-graph-stability`
- **Proposed PR title:** `perf(input): overlap packed data, prefetch and XLA compilation cache`
- **PR description — why/outcome:** Reduce TPU idle time and repeated compilation through verified input pipeline overlap and stable cache ownership.
- **Scope and boundaries:** Memmap reader/buffer geometry, MpDeviceLoader interaction, worker cache keying, per-rank persistent XLA cache.
- **Milestone lineage:** M5-F3, M11-F6; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `test(input): instrument input idle time and shard correctness`
  - `perf(input): tune pinned host buffers and device prefetch`
  - `feat(xla): verify early per-worker compilation cache initialization`
  - `fix(input): eliminate shape changes and duplicate stage work`
  - `docs(perf): record cold/warm compile and real-shard results`
- **Acceptance / merge gate:** No missing/duplicate tokens; cold/warm compile counters and device idle improve or documented no-op; cache validity/version checks enforced.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 10 — perf(telemetry): add low-overhead TPU profiling and async reporting

- **Branch:** `perf/10-async-telemetry-xla-profile`
- **Proposed PR title:** `perf(telemetry): add low-overhead TPU profiling and async reporting`
- **PR description — why/outcome:** Users see useful ranked throughput/loss and clean diagnostics without per-step host sync or log spam.
- **Scope and boundaries:** Rank-zero aggregation, async writer, XLA traces, recompilation and CPU fallback attribution, measurement overhead limits.
- **Milestone lineage:** M7-F4, M11-F7, telemetry backlog; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(metrics): formalize versioned rank and phase metric records`
  - `perf(metrics): add bounded async aggregator and writer`
  - `test(metrics): cover rank dedupe, failure and log overhead`
  - `feat(xla): record compile, HLO and host/device idle diagnostics`
  - `docs(metrics): document overhead budgets and trace artifacts`
- **Acceptance / merge gate:** Structured diagnostics attributable to training stage; no hot-loop host transfers; measured telemetry overhead within explicit budget.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 11 — perf(benchmark): certify the optimized loss and attention 850K gate

- **Branch:** `benchmark/11-optimized-loss-attention-850k-gate`
- **Proposed PR title:** `perf(benchmark): certify the optimized loss and attention 850K gate`
- **PR description — why/outcome:** Compare optimized path against the locked 319K baseline and verify the 850K intermediate target without unsupported claims.
- **Scope and boundaries:** Matched fake/real shard benchmarks; HLO, HBM, provider identity, cache, compile and fallback reports.
- **Milestone lineage:** M10-F7, M11-F7; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `test(perf): lock 135M geometry, warmup and run fingerprint`
  - `perf(benchmark): collect measured candidate runs and HBM graphs`
  - `fix(perf): address measured bottleneck found in this PR scope`
  - `docs(perf): publish 850K pass/fail and reproducible artifacts`
- **Acceptance / merge gate:** At least 850K *measured* global supervised tokens/s, semantics stable, named providers, HBM and graph evidence; blocked if threshold unmet until owner-approved plan revised.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 12 — test(parity): validate numerical equivalence, resume and HF export

- **Branch:** `validation/12-numerical-resume-hf-export-parity`
- **Proposed PR title:** `test(parity): validate numerical equivalence, resume and HF export`
- **PR description — why/outcome:** Show the optimized model remains semantically equivalent to HF and survives exact interruptions and clean-process export.
- **Scope and boundaries:** Numerical alignment, precision tests, canonical checkpoints and independent Transformers loading.
- **Milestone lineage:** M12-F1, M12-F4, M7-F1-F5; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `test(parity): lock numerical and masked token comparisons`
  - `test(checkpoint): replay distributed resume with recorded RNG/data state`
  - `test(export): verify clean-process plain-Transformers loading`
  - `fix(parity): address evidence-backed discrepancy only`
  - `docs(parity): publish accepted tolerances and artifacts`
- **Acceptance / merge gate:** Numerical, resume and clean-process export checks pass on the required backend/hardware; no family-specific hidden behavior.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 13 — perf(parity): run 135M LaughLM-class three-run certification

- **Branch:** `benchmark/13-laughlm-135m-three-run-certification`
- **Proposed PR title:** `perf(parity): run 135M LaughLM-class three-run certification`
- **PR description — why/outcome:** Prove throughput/MFU release thresholds using matched steady-state v5e-8 runs and real-shard stability.
- **Scope and boundaries:** Three repeatable matched measurements, 200-update real-shard run, artifact bundling and evidence verdict.
- **Milestone lineage:** M12-F2-F3-F5; **priority:** P0
- **Planned logical commits** (examples, not an artificial limit):
  - `test(perf): lock matched baseline and candidate manifests`
  - `perf(parity): capture three steady-state v5e-8 runs`
  - `test(stability): archive 200-update real-shard reliability`
  - `docs(parity): publish tokens/s, MFU, HBM and limitations`
- **Acceptance / merge gate:** Hard 912.6K tokens/s and 47.8% MFU or explicitly blocked; preferred 963.3K/50.4% recorded separately; no forecast treated as proof.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 14 — test(compat): certify dense HF family support and fallbacks

- **Branch:** `validation/14-dense-family-conformance-matrix`
- **Proposed PR title:** `test(compat): certify dense HF family support and fallbacks`
- **PR description — why/outcome:** Plain HF model families work through generic training/export even without optimization adapters.
- **Scope and boundaries:** GPT-2, OPT, NeoX, BLOOM, Falcon/Phi, Llama/Mistral/Qwen/Gemma conformance; semantic/shape/alias fixtures.
- **Milestone lineage:** M6-F1-F4, M13-F1-F5; **priority:** P1
- **Planned logical commits** (examples, not an artificial limit):
  - `test(compat): add learned-position and LayerNorm models`
  - `test(compat): add packed/partial-QKV and dense Phi cases`
  - `test(compat): add RoPE/GQA/RMSNorm models`
  - `fix(compat): repair proven generic-path or fallback regression`
  - `docs(compat): publish per-capability support and certification`
- **Acceptance / merge gate:** Every in-scope family passes generic semantics and honest optimization explanations; missing TPU proof remains uncertified.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 15 — feat(scale): validate FSDP sharding and 1.3B training

- **Branch:** `feature/15-fsdp-sharding-1p3b-scale`
- **Proposed PR title:** `feat(scale): validate FSDP sharding and 1.3B training`
- **PR description — why/outcome:** Enable a larger dense model under topology-aware sharding with recoverable optimizer and checkpoint state.
- **Scope and boundaries:** DP×FSDP mesh, full-state ownership, checkpoint topology and measured v5e memory/throughput.
- **Milestone lineage:** M13-F6-F7; **priority:** P1
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(scale): define supported FSDP mesh and ownership policy`
  - `test(scale): cover shard mapping and topology guards`
  - `fix(checkpoint): preserve sharded save/reload semantics`
  - `perf(scale): measure locked 1.3B geometry on target v5e`
  - `docs(scale): record memory, throughput and limitations`
- **Acceptance / merge gate:** 1.3B model smoke, checkpoint integrity and measured scale evidence; unsupported topology fails before expensive allocation.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 16 — feat(ux): polish HF-like trainer, CLI and secure TPU workflows

- **Branch:** `feature/16-hf-like-cli-config-user-journey`
- **Proposed PR title:** `feat(ux): polish HF-like trainer, CLI and secure TPU workflows`
- **PR description — why/outcome:** New users train with model/dataset/tokenizer/standard arguments, readable progress and portable checkpoints—not TPU scripts.
- **Scope and boundaries:** TrainLMTrainer, CLI/YAML config parity, secure immutable HF model/shard selection, progress/errors, user documentation.
- **Milestone lineage:** M8-F0, M14-F1-F2; **priority:** P1
- **Planned logical commits** (examples, not an artificial limit):
  - `feat(ux): unify public API, CLI and validated config errors`
  - `test(ux): add end-user train/eval/resume smoke examples`
  - `fix(ux): remove leaked internal worker/PJRT arguments`
  - `docs(ux): write secure end-to-end CPU and TPU guides`
  - `test(ux): verify CLI and Python parity without TPU launch in CPU tests`
- **Acceptance / merge gate:** Copy-ready HF-like examples and deterministic CLI config; none require manual ranks, scripts, manifests, stage-log parsing.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 17 — ci(release): enforce reproducibility, hardware gates and recovery

- **Branch:** `ci/17-release-ci-preemption-evidence`
- **Proposed PR title:** `ci(release): enforce reproducibility, hardware gates and recovery`
- **PR description — why/outcome:** Prevent unsupported releases by separating CPU CI from owner-run TPU certification and checking preemption safety.
- **Scope and boundaries:** Version lock, targeted CPU CI, release evidence artifact checks, failure-injection recovery and PR template contract.
- **Milestone lineage:** M14-F3-F4; **priority:** P1
- **Planned logical commits** (examples, not an artificial limit):
  - `ci(test): enforce dependency and CPU conformance matrix`
  - `ci(evidence): validate published TPU artifact schemas`
  - `test(recovery): exercise interrupted save and restart`
  - `fix(ci): surface missing evidence without greenwashing`
  - `docs(release): document support matrix and exemptions`
- **Acceptance / merge gate:** Required checks are enforceable without a cloud TPU in standard CI; hardware certification only accepts real artifacts.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


### Plan 18 — release(v1): publish dense-AR support, limitations and evidence

- **Branch:** `release/18-dense-ar-v1-support-certification`
- **Proposed PR title:** `release(v1): publish dense-AR support, limitations and evidence`
- **PR description — why/outcome:** Ship an accurate, auditable V1 promise for dense decoder-only HF training and certified TPU optimizations.
- **Scope and boundaries:** Support manifest, version policy, changelog, certification matrix, docs and release checklist.
- **Milestone lineage:** M14-F5; **priority:** P1
- **Planned logical commits** (examples, not an artificial limit):
  - `docs(release): assemble supported-model and capability matrix`
  - `test(release): validate versioned support manifest`
  - `docs(release): link numerical, throughput and stability evidence`
  - `release(v1): finalize release notes and migration guidance`
- **Acceptance / merge gate:** All V1 gates satisfied and linked; no missing evidence or unsupported 'certified' claims.
- **Dependencies:** Previous *actual* delivery PR merged into `main`,
  accepted review/CI, and evidence from required earlier provider contracts.
- **Validation:** Pinned CPU/fixture tests are mandatory. For TPU-specific
  correctness, optimization or throughput claims, collect the exact v5e-8
  commands, source SHA, model/dataset revisions, environment, warmup,
  timing boundaries, HLO/compile counts, HBM and metric artifacts. If not
  measured, say so and keep feature disabled/un-certified.
- **Rollback / fallbacks:** Preserve generic HF behavior and a reversible
  feature/provider path. On regression, disable optimization and retain
  documented baseline and checkpoint interoperability.
- **PR template contract:** Include summary, user impact, scoped changes,
  per-commit intent, validation matrix, TPU evidence, risk/fallback, docs,
  dependencies, merge approval and follow-up.


## Performance and engineering optimization decision register

These are **hypotheses to measure, not benchmark results or automatic new
PRs**. Place each in the PR above that owns its impact; splitting is
allowed only through an owner-approved plan revision.

| Candidate lever | Location | Measurement before changing defaults |
|---|---|---|
| Zero full-logits projection during training | 03 | HLO `[B,S,V]` elimination, exact CE/z-loss gradients, HBM and steps/s |
| Fewer host-controlled GA steps and recompiles | 04 | Compile counts, graph shape cache, update/gradient parity, host idle |
| Native attention and compact KV | 05–06 | Kernel HLO, GQA/MQA shape/mask correctness, custom-call evidence, HBM |
| Reversible projection and MLP fusion | 07 | Update/export parity plus absence of harmful layout/copy transposes |
| Optimizer precision and selective remat | 08 | State/gradient error bounds, exact resume, HBM/tok/s tradeoff |
| Device prefetch and per-worker persistent XLA cache | 09 | Input idle, cache hit/miss/cold/warm compile, immutable revision keys |
| Async rank-0 metrics and low-sync XLA profiling | 10 | Host sync counts, callback overhead, trace attribution and error path |
| Dynamic shapes and unwanted CPU fallback | 04, 09–10 | Recompilation count and explicit fallback events on steady-state steps |
| Exact throughput quality/supervised token metric | 11–13 | Locked geometry, data revisions, warmup exclusion, 3 matched runs |
| Larger-model sharding and collectives | 15 | Owner-run topology, peak HBM, step wall time, checkpoint integrity |

Additional ideas **not automatically approved for V1**: sharded vocabulary
loss, tensor/sequence parallelism, overlapping collective and computation,
layer scan, and optimizer research (Muon/hybrid). These require isolated
benchmarks and an explicit roadmap decision after generic v5e-8 correctness
and parity: do not sacrifice core HF semantics to chase unverified speed.

### Additional engineering optimizations to evaluate (not assumed wins)

These opportunities come from current [implementation handoff](HANDOFF.md)
and profiling methodology. Each requires a controlled, *same-model,
same-data, same-global-token-budget* comparison on target hardware before
promoting an optimization. Until that proof exists, these are **candidate
follow-ups**, not new approved branches or PR numbers.

| Optimization hypothesis | Placement / priority | Experimental acceptance and guardrails |
|---|---|---|
| **Vocabulary-parallel LM head and distributed exact CE** | P1 after PR 15, only if 1.3B HBM is limiting | Implement numerically stable distributed max/sum normalization; test tied embeddings, padding/ignore index, z-loss, updates, HF export and collective memory/latency; compare exact CE to unsharded reference |
| **Tensor/sequence parallelism and collective overlap** | P1 after FSDP path is sound | Inspect model semantics and legal sharding first; show equivalent gradients and exact resume; profile all-gather/reduce-scatter utilization and ensure overlap reduces actual step wall time (not just isolated collective timing) |
| **Scanned decoder layers / compiled step boundaries** | Consider within PR 04 and PR 09 | Compare compiled loop versus unrolled graph size, graph-cache reuse, recompilation, peak HBM, host stalls and steady-state tokens/s; reject regressions in partial-gradient accumulation, dropout RNG or checkpointing |
| **Precise precision/memory policy experiments** | PR 08, opt-in only | Compare BF16/FP32 states and selective remat with update/resume numerical tolerances, NaN/overflow incidence, HBM and tokens-to-target-loss; never promise that lower precision is safe across all HF families |
| **Evaluation/checkpoint interruption overhead and async I/O** | PR 02, 10, 12, 17 | Measure eval graph compilation, save p50/p95 stalls, atomic durable commits, worker cancellation, resumable data cursor and failure injection; never trade away exactness for mean throughput |
| **Performance regression baselines with environment fingerprints** | PR 01, 11, 13, 17 | Bind git SHA, HF/tokenizer/data revision, XLA/torch/libtpu versions, TPU topology, cache hit/miss, measured supervised tokens, compilation counts, HBM and matched run windows; reject non-comparable improvements |
| **Optimizer algorithm research (Muon/hybrid), context/pipeline parallelism** | Deferred post-V1 research | Separate benchmarks and explicit product/architecture decisions; require matched quality or tokens-to-target-loss, compatibility/export and restart semantics before considering release |

**Suggested optimization triage:** First fix deterministic CPU conformance
and successful public TPU lifecycle; then attribute the measured ~319K
baseline's time to host, compile, dense logits, attention, data and
optimizer via repeatable profiles. Optimize the largest verified bottleneck
first. Promote a provider to default only after semantic, hardware, HBM,
graph and reproducibility gates; a prediction is not evidence.

## Recommended repository controls

- Protect `main` through repository settings: required PR reviews and
  required status checks, conversation resolution, no force-push.
- Add `.github/pull_request_template.md` as the required author checklist.
- Keep changelog/architecture-decision and measured benchmark evidence linked
  from the PR. Consider CI contract smoke and checkpoint round-trip gates.
- Preserve logical commit history, review traceability and a reversible
  path; avoid PRs containing 100+ unrelated commits.
- Do not use GitHub auto-merge or start another branch while the current
  sequential PR is still open.
