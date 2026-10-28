# TrainLM native TorchTPU rebaseline — T0–T11

> **Proposed target architecture, not an implementation or certification claim.**
> This document reconciles the 2026-10-27 consolidated handoff with the
> repository's existing XLA-era M0–M14 work and 18 candidate PR cards.
> Existing code and benchmarks are salvage material; no rewrite is approved
> solely by publishing this plan.

## Project vision and desired developer experience

TrainLM is an open-source, Python-first language-model training framework
combining Hugging Face interoperability with a **native TorchTPU-first target**
for Kaggle TPU v5e-8. Users control configuration and can inspect hardware
backend, optimization provider, fallback, metrics and resume state. Ordinary
training must not require manually managing worker launch or rank topology.

The desired API below is illustrative; verify actual repository signatures:

```python
trainer = TrainLMTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    config=config,
)
trainer.train()
trainer.save_pretrained("./trained-model")
```

Preserve supported HF model semantics and generic correctness while making
performance-sensitive training operations explicit, capability-driven and
reversible.

## Repository facts and status boundaries

| Item | Verified/recorded context | Required next action |
|---|---|---|
| Repository | `Dhiraj309/TrainLM` | Verify checkout, branch, origin, permissions |
| Legacy integration | PR #18 merged into `main` on 2026-10-28; merge commit `23959d29434fd2ae3f78b20469e1d61c6b1e18c9` | T0.1 audits the merged tree before activating a new PR |
| Earlier salvage | PR #16 was merged into PR #18, which is now merged into `main` | Inventory its code as part of the merged tree; do not merge it again |
| Existing TPU backend | Primarily PyTorch/XLA with historical measurements | Classify reusable vs XLA-specific components |
| Native TorchTPU | **Target only**; no native v5e-8 certification established here | Prove device, autograd, distributed and compile support |
| GitHub access | Previous authenticated collaborator `Dignity045` branch push succeeded | Recheck actual identity each task; owner account not assumed |
| CPU tests | PR #18 snapshot: 700 passed / 21 failed under Python 3.12.14, torch 2.9.0+cpu, Transformers 5.0.0, Hub 1.3.5, pytest 9.1.1 | Re-run on merged `main` before the next code PR; failure groups are recorded in `HANDOFF.md` |
| CPU profiles | `core-minimum` Python 3.12/torch 2.9; `core-current` Python 3.13/torch 2.13 | Inspect actual environment and constraints |

The old report that GitHub pushes were unverified is superseded by the
verified branch-write exercise. Do not claim current shell authentication
without checking. Managed network proxy variables and credentials must not
be modified; request supported per-command network permission where needed.

The path-level read-only salvage classification is in
[the T0.1 inventory](TORCHTPU_SALVAGE_INVENTORY.md). PR #18 is merged;
this inventory does not establish native TorchTPU compatibility.

## Layered target architecture

1. **Public API:** `TrainLMTrainer`, typed configs, HF tokenizer/model/dataset
   compatibility, diagnostics, `train()`, `save_pretrained()`.
2. **Model:** HF causal LM semantics, transformer blocks, attention, MLP,
   normalization and embeddings; avoid model-family forks.
3. **Data:** HF sources and packed `.bin` tokens with explicit manifest,
   general `uint32` representation and validated legacy `uint16` support.
   Reject overflow and incompatible token ranges.
4. **Runtime:** single TPU device, eight-device DP, accumulation, checkpoint,
   resume, loss/metric synchronization and deterministic boundaries.
5. **Kernel dispatch:** explicit reference/HF, native TorchTPU and optional
   verified providers. Tokamax, Pallas/Mosaic or Helion require demonstrated
   backend support; no implicit optimized fallback claims.
6. **Optimization:** reference AdamW, gated Muon research, conservative
   per-block compilation and eager fallback, then broader functional graphs.
7. **Target:** native TorchTPU on actual Kaggle TPU v5e-8 with measured
   correctness, memory, throughput and reproducibility.

**Migration policy:** do not mass-delete XLA. Inventory each component as
*reusable* (preserve/adapt), *XLA-specific* (replace behind backend boundary)
or *uncertain* (investigate/test). Include models, training loop, launch,
attention, loss, optimizer, data, checkpoint, tests and documentation.

## Numerical, distributed and checkpoint contracts

- Define global vs per-device batch, accumulation boundaries, token masks,
  unequal valid-token counts and final partial batches. **Gradient
  normalization must use the global sum of valid tokens**. Averaging worker
  mean losses is wrong when worker valid-token counts differ; validate
  against a reference calculation.
- Training checkpoints must preserve model, optimizer, scheduler, RNG,
  progress counters, data position and topology-relevant state under a
  documented exact-resume contract. Compare uninterrupted and resumed runs
  with explicit numerical tolerance and nondeterminism notes.
- HF `save_pretrained` export is a separate portable model artifact,
  including model configuration, weights and tokenizer files when applicable.
- Validate reference correctness first, native TPU correctness second,
  explicit optimized dispatch third and matched hardware performance last.
  Never infer performance from fusion/compilation alone.
- Record actual Kaggle v5e-8 evidence for TPU-only gates; Codex CPU results
  are not TPU certification. FSDP/tensor parallelism are gated future scope.

## Proposed T0–T11 milestones and acceptance gates

| Gate | Outcome | Required acceptance evidence |
|---|---|---|
| T0 | Architecture rebaseline and salvage | PR #16/#18 inventory; reusable/XLA-specific/uncertain classification; approved boundaries and PR mapping |
| T1 | Native TorchTPU environment | Real v5e-8 import, device ops, autograd, optimizer step, multi-device feasibility and reproducible setup |
| T2 | Python-first configs/API | Verified public API, HF tokenizer/dataset/model contracts, CPU tests and actionable errors |
| T3 | Single-device training/data | TPU reference causal LM step, packed data manifest, uint32/legacy uint16 validation and loss parity |
| T4 | Eight-device data parallelism | Global valid-token-weighted gradients, unequal masks, accumulation and reference-parity proof |
| T5 | Checkpoint/resume and HF export | Clean-process resume equivalence and `save_pretrained` round trip |
| T6 | Compilation foundations | Per-block compile parity, eager fallback, graph stability and debug/recompile evidence |
| T7 | Optimized kernels | Guarded attention/fused CE, gradient/mask tests, explicit fallback and real TPU benchmarks |
| T8 | Functional optimizers | Native AdamW parity and resume; opt-in Muon only after numerical/quality/TPU evidence |
| T9 | Advanced compiled training | Optional compiled updates/accumulation without breaking DP8 or exact resume |
| T10 | Reproducible benchmarks | Locked workload, warmup/compile accounting, throughput, memory, MFU and repeated-run evidence |
| T11 | Future scaling | Gated FSDP/TP collective correctness, checkpoint portability and scaling tests |

Each gate is an architectural milestone, **not necessarily one PR**.
The earlier 18 PR cards explicitly target XLA in several places and
therefore require T0 mapping/replacement, not automatic execution.
Keep the approved workflow: **multiple sequential, outcome-scoped PRs;
multiple small, traceable commits per PR; each next branch created from
verified merged `main` only after the preceding PR merges**. Branch name,
PR title, description and commit scope must agree. Preserve commit history
by default; no direct main pushes, force-pushes, auto-merges or stacked PRs.

## T0.2 — proposed legacy-card mapping

The source review is recorded in [the salvage inventory](TORCHTPU_SALVAGE_INVENTORY.md).
The former 18-card XLA queue has a proposed T0–T11 disposition in
[`PR_DELIVERY_PLAN.md`](PR_DELIVERY_PLAN.md#t02-candidate-mapping-to-native-torchtpu-gates).
It is a review artifact, not permission to start implementation. Preserve old
IDs for traceability; create fresh sequential PR IDs only after owner review.
T1 must establish TorchTPU package provenance and real v5e-8 device,
autograd, optimizer, collective, and compile behavior before dependency or
support claims change.

## T0.1 — read-only Codex discovery task

T0.1 records source-level findings and proposes follow-up work. The inventory
is not an implementation plan approval; code, dependency, and architecture
changes still require owner review.

1. Verify repository, GitHub identity, permissions and push readiness without
   pushing; inspect PR #18 and the PR #16 code now present in merged `main`.
2. Inventory modules by path as reusable, XLA-specific or uncertain; propose
   native TorchTPU boundaries and dependency feasibility.
3. Read `AGENTS.md`, `ROADMAP.md`, `IMPLEMENTATION_CONTEXT.md` and
   `DEPENDENCIES.md`; identify conflicts and outdated instructions.
4. Record actual CPU Python/torch/transformers/hub versions, constraints and
   test baseline, including the six historical failure categories.
5. Propose the T0.1 diff, acceptance tests, next PR title/branch and exact
   logical commit messages; await owner approval before file changes,
   new branches, commits, pushes, PR creation or merge.

Owner-requested commit trailer for approved work, when appropriate:
`Co-authored-by: Dhiraj309 <patildhiraj309@outlook.com>`.
Do not add Codex/OpenAI/bot co-author trailers. Attribution and actual
Git author identity must remain truthful.
