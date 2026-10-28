## What this PR delivers

<!-- Same coherent outcome as the descriptive branch and PR title. -->

## Why it matters / desired user experience

<!-- User journey or quantified performance/reliability objective. -->

## Scope and intentional non-goals

<!-- List changed public interfaces and explicitly excluded follow-up work. -->

## Commit-by-commit development trail

<!-- Summarize distinct implementation, tests, bug fixes, regression and docs commits.
     Several small meaningful commits are expected; no fixed number. -->

## Verification

- [ ] Relevant pinned CPU tests / regression suite (record commands and output)
- [ ] HF semantics, generic fallback and checkpoint/export regression considered
- [ ] TPU hardware commands/artifacts linked when this PR requires hardware evidence
- [ ] HLO, compile count, HBM and matched throughput when claiming optimization
- [ ] New tests and known failures documented; no invented results
- [ ] `docs/ROADMAP.md` and `docs/IMPLEMENTATION_CONTEXT.md` updated

## Risks, fallback and rollback

<!-- Which guarded path is selected? How is it disabled/reverted safely? -->

## PR sequencing / approvals

- [ ] Branch created from the latest merged `main` after previous PR merged
- [ ] Intended base: `main`; record base/head SHAs
- [ ] GitHub checks and review accepted
- [ ] Owner explicitly approves merge; no automatic merge
- [ ] Do not create the next PR branch until this one merges and `main` is verified

## Follow-ups and remaining certification gaps

<!-- If TPU certification is unproven, say so; leave guarded features off by default. -->
