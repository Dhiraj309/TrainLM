# TrainLM native TorchTPU rebaseline — T0–T11

> **Proposed target architecture, not an implementation or certification claim.**
> This document reconciles the 2026-10-27 consolidated handoff with the
> repository's existing XLA-era M0–M14 work and 18 candidate PR cards.
> Existing code and benchmarks are salvage material; no rewrite is approved
> solely by publishing this plan.

**Current gate:** T0 is closed; PR #20 merged the accepted T0.2 mapping. T1
starts on v5e-1 for single-device feasibility; use v5e-8 for tests that need
multiple devices. The owner reported a successful eager native TPU smoke;
hardware generation, explicit compilation and TrainLM integration remain
unverified. PR #28 merged the source-level T1 probes and explicit one-device
trainer foundation; hardware compile, lifecycle and collective evidence still
gate T1 closure. Follow the [utility inventory and delivery order](TORCHTPU_UTILITIES.md)
for the remaining T1.1–T1.4 evidence.

PR #45 rebase-merged agent-discoverability documentation at verified `main`
SHA `d983408221b92d631aea817bee7539e07de1bbf5`. It did not close a T0–T11
gate or add runtime/accelerator evidence. The active implementation path
remains experimental and explicit while owner-run T1 evidence is pending.

PR #47 aligned the current handoff and roadmap status at verified `main` SHA
`e109ad253754303e44eb6162aacb046c415b990b`; it did not change runtime support.
PR #48 rebase-merged the distributed token correctness slice at verified `main`
SHA `7ae837d085fbbaab9f6c2d3617ff129b89dd454e`; its full GitHub CI passed.
That slice implements global valid-token
gradient normalization for pre-clipping gradient-averaging backends and
global `max_tokens` stopping at matching microbatch boundaries. It also reduces
loss/progress metrics globally while preserving per-rank checkpoint counters.
Two CPU/Gloo regressions cover gradient parity and rank-consistent stopping
with global reports. The focused CPU/Gloo and related trainer/runtime/API
suite passed locally in an isolated Python 3.12.14, torch 2.9.0+cpu, pytest
9.2.0 environment (79 tests). Native multi-device TPU execution remains gated.

PR #49 rebase-merged T5 rank-shard integrity at verified `main` SHA
`01a71f990fc5578b77be6e2e0a84473486fbe488`; its Python 3.11 GitHub CI runs
#296 and #297 passed. Schema v3 records each rank shard's size and SHA-256,
and coordinated CPU/Gloo preflight catches corruption or rank-local write
failures before state restore. Native TPU checkpoint validation is pending.

PR #50 rebase-merged the T5 public trainer checkpoint preflight at verified
`main` SHA
`f57d6d71339753a6a3cee4082a43dfa2f5113017`; CI run #299 passed.

PR #51 rebase-merged at verified `main` SHA
`c63e6903767f3a2aab2ff733e6248ff0c4e4bd3b`; CI run #302 passed. It records
optimizer and scheduler class identity and checks optimizer parameter-group
compatibility before restore. Older schema-v1 checkpoints without the optional
class metadata remain accepted. PR #52 rebase-merged optimizer slot-shape/step
and scheduler-payload preflight at verified `main` SHA
`9cf8d4ba0de0b91cc52908d9d98f984b29873a33`; CI run #304 passed. PR #53
rebase-merged trainer progress metadata preflight at verified `main` SHA
`8ea810c2e5e54f492cb4e29e0844a74c9935f23e`; CI run #306 passed. PR #54
preflights packed-data cursor identity at verified `main` SHA
`6ab2b8c9fcde5895c71d5f8450808b7f5756c520`; CI run #308 passed. PR #55
rebase-merged at verified `main` SHA
`ea9dbadfb86f17cd3713b2abcee808677c38e845`; it rejects transient trainer
fields, with GitHub CI passed (run ID `41578130077`). The complete CPU suite
passed locally (871 tests and 41 subtests in Python 3.12.14 with torch 2.9.0+cpu
and pytest 9.2.0). PR #56 rebase-merged at verified `main` SHA
`4aa3614858065c6255c4ecc2d0949c24ee715f70`; GitHub pytest passed. PR #57
rebase-merged at verified `main` SHA
`e62052ae39d6216bce0dd5aec86fc42d9f10a837`; GitHub pytest passed. PR #58 is
open for custom iterable identity/cursor rejection; its focused
lifecycle/data-resume suite passes (37 tests). The last local full suite
reported 865 passed, 7 unrelated environment failures, and 41 subtests. Native
TPU checkpoint validation remains pending.

Owner direction allows CPU-verifiable implementation to proceed while scarce
TPU slots are pending. A provisional, explicit `accelerator="native_tpu"`
single-device path now reuses the portable engine; this does not close T1 or
change the dependency profile. `accelerator="auto"` remains guarded, and the
legacy `accelerator="tpu"` continues to select XLA. Checkpoint/resume,
compilation and distributed training remain later gates.

## Project vision and desired developer experience

TrainLM is an open-source, Python-first language-model training framework
combining Hugging Face interoperability with a **native TorchTPU-first target**
for Kaggle TPU v5e-8. Users control configuration and can inspect hardware
backend, optimization provider, fallback, metrics and resume state. Ordinary
training must not require manually managing worker launch or rank topology.
TrainLM must internally discover hardware identity, available devices, memory,
precision and runtime capabilities using supported TorchTPU/PyTorch utilities.
Notebook-provider labels are context, not hardware identification. The shared
report powers preflight, notebook guidance and `trainer.explain()`; missing
hardware and broken initialization have distinct outcomes. See the
[UX and discovery contract](TORCHTPU_UTILITIES.md#desired-user-experience-and-internal-discovery-contract).

The example below shows the target UX after native routing. Current `auto`
selects CPU/CUDA when no TPU is detected and fails during construction on a
detected TPU until the native route is validated. CPU/CUDA trainers can call
the non-allocating `preflight()` now; explicit `tpu` still uses XLA. Native TPU
training auto-selection and the expanded worker hardware report are planned:

```python
from trainlm import TrainLMTrainer, TrainLMTrainingArguments

trainer = TrainLMTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=TrainLMTrainingArguments(
        output_dir="runs/example",
        accelerator="auto",
        max_steps=100,
    ),
)
print(trainer.explain(include_hardware=True))
trainer.train()
trainer.save_model("./trained-model")
```

Preserve supported HF model semantics and generic correctness while making
performance-sensitive training operations explicit, capability-driven and
reversible.

## Repository facts and status boundaries

| Item | Verified/recorded context | Required next action |
|---|---|---|
| Repository | `Dhiraj309/TrainLM` | Verify checkout, branch, origin, permissions |
| Current base | PR #62 rebase-merged; verified `main` SHA `821beacdee235e9ed6e70781023126023120adef` | Continue CPU checkpoint coverage while TPU execution remains gated pending owner-run v5e-1 lifecycle and v5e-8 evidence |
| Legacy integration | PR #18 merged into `main` on 2026-10-28; merge commit `23959d29434fd2ae3f78b20469e1d61c6b1e18c9` | T0.1 inventory is complete; XLA-era PR cards are not the active queue |
| Earlier salvage | PR #16 was merged into PR #18, which is now merged into `main` | Inventory its code as part of the merged tree; do not merge it again |
| Existing TPU backend | Primarily PyTorch/XLA with historical measurements | Classify reusable vs XLA-specific components |
| Native TorchTPU | Explicit single-device software path exists; owner-reported standalone eager v5e-1 smoke passed; package compatibility and production training remain unverified | Continue only with explicit selection; verify compile/lifecycle on v5e-1 and collectives on v5e-8 before changing support claims |
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
- Use v5e-1 for initial single-device feasibility and record the exact image
  and software provenance. Use actual Kaggle v5e-8 evidence for gates that
  require multiple devices; CPU results are not TPU certification.

## Proposed T0–T11 milestones and acceptance gates

| Gate | Outcome | Required acceptance evidence |
|---|---|---|
| T0 | Architecture rebaseline and salvage | PR #16/#18 inventory; reusable/XLA-specific/uncertain classification; approved boundaries and PR mapping |
| T1 | Native TorchTPU utilities and environment | T1.1 inventory → T1.2 shared hardware/capability report → T1.3 verified dependency and process lifecycle → T1.4 multi-device entry probe. Record actual device properties, eager/compile results and reproducible setup; prefer v5e-1 for single-device and v5e-8 for collective evidence. |
| T2 | Python-first configs/API | Integrate shared discovery into auto backend selection, preflight and explain; verify HF contracts, CPU/CUDA compatibility and actionable errors without manual runtime setup. An explicit single-device prototype exists; automatic routing waits for T1/package and CPU evidence. |
| T3 | Single-device training/data | PR #28 merged the generic engine adapter and random-init GPT-2/TinyStories starter; PR #29 merged streamed, validated uint32 shards and `PackedBinDataset` integration; PR #30 merged CPU reference parity for a packed-data causal-LM update; PR #31 merged the CPU random-init train/eval/checkpoint/export lifecycle; PR #32 added exact packed-data continuation; PR #33 fixed notebook source freshness; PR #34 added a train-only BPE tokenizer; PR #35 builds GPT-2 from explicit config; PR #36 accepts pinned user corpora; PR #37 adds opt-in BF16; PR #38 adds optimizer schedule and seed controls; PR #39 adds opt-in activation checkpointing; PR #40 exposes per-device batch sizes; PR #41 adds supervised-token targets and actual token counts; PR #42 adds sparse logging and terminal loss evidence; PR #43 bundles the model export and evidence report for Colab/Kaggle. The current implementation verifies CPU reload and short generation from that export; this is not TPU generation evidence. TPU validation and convergence remain owner-run. |
| T4 | Eight-device data parallelism | Global valid-token-weighted gradients, unequal masks, accumulation and reference-parity proof |
| T5 | Checkpoint/resume and HF export | Model/tokenizer `save_pretrained` export and generic accelerator RNG capture/restore exist. PR #32 adds a fingerprinted data cursor for packed iterable sources and exact CPU continuation tests. Zero loader workers remain required; native TPU and clean-process owner validation remain pending. |
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

## T0.2 — accepted legacy-card mapping

The source review is recorded in [the salvage inventory](TORCHTPU_SALVAGE_INVENTORY.md).
The former 18-card XLA queue has an owner-approved T0–T11 disposition in
[`PR_DELIVERY_PLAN.md`](PR_DELIVERY_PLAN.md#t02-accepted-mapping-to-native-torchtpu-gates).
It is a mapping, not an approved implementation queue. Preserve old IDs for
traceability; create fresh sequential PR IDs only for an owner-approved scope.
T1 establishes package provenance and single-device behavior on v5e-1, then
uses v5e-8 for collectives and other requirements that need multiple devices,
before dependency or support claims change.

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
