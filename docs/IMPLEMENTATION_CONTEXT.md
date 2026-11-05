# TrainLM implementation context

> **Document status:** This file preserves detailed implementation history.
> For the live project state and sequence, use [the docs guide](README.md),
> [the handoff](HANDOFF.md), and [T0–T11](TORCHTPU_REBASELINE.md). The
> numbered M0–M14 notes below record the prior PyTorch/XLA implementation;
> they are not native TorchTPU milestones or certification.

## Current project snapshot

The [coding-agent guide](AI_AGENT_GUIDE.md) maps common pretraining tasks to
the current notebook, implementation, tests, and validation limits. Prefer it
for file discovery; this document remains the detailed implementation record.

The public `TrainLMPretrainingConfig` and `TrainLMTrainer.from_scratch()` API
accept raw text from a revision-pinned Hugging Face dataset, train a byte-level
BPE tokenizer, prepare validated packed shards, and initialize random weights
through a registered Transformers causal-LM config. GPT-2 remains the default;
`model_type` and JSON-compatible `model_config_overrides` select other
`AutoModelForCausalLM` architectures. The trained tokenizer supplies vocabulary
and special-token IDs. The from-scratch Colab/Kaggle notebook calls this API
with a concise token-budget and sequence-length setup; its downloadable schema
v6 report records requested and effective training settings, optimizer and
scheduler details, resolved backend/runtime, bounded TPU property discovery,
model config and parameter counts, tokenizer special-token IDs and artifact
hash, dataset revisions, runtime package versions, and the newest published
checkpoint path when available. The path is local to the notebook runtime, so
`RUN_DIR` must be on persistent storage to survive resets. Property-only
discovery does not allocate a device and is labeled unverified. Missing optional
runtime packages and unavailable hardware properties are recorded as null.
With a checkpoint, `resume.code` preserves every pretraining setting and an
executable example also displayed and bundled as `resume.py`. It targets the
recorded checkpoint with the same token budget, and retries only a previously
requested budget extension whose configuration publication is still pending.
The example checks for the original run files before constructing a trainer.
Restore the full run directory and match source/package versions; the model
export ZIP does not contain the optimizer checkpoints or packed data. If no
checkpoint is available or lookup fails, code is null and no resume script is
included. The
private notebook helper does not load checkpoint state while making the
report. Native TPU remains experimental; CPU/API evidence adds no hardware
validation.
The notebook's optional `RESUME_FROM_REPORT` path reads the saved config and
checkpoint directly. It checks schema, source revision, runtime package
versions, report directory, and required run artifacts before using the normal
training/evaluation/export flow; it does not execute code stored in the report.

The scratch notebook now uses a stable configurable `RUN_DIR` and exposes
`RESUME_FROM_CHECKPOINT`. The public trainer accepts the sentinel
`resume_from_checkpoint="latest"`, selecting the newest published local
checkpoint (`trainer_state.pt`) or native TPU checkpoint (`manifest.json`); a
missing checkpoint raises an actionable error. Notebook instructions describe
Colab Drive and Kaggle output persistence. CPU/API coverage verifies lookup and
resume behavior; native TPU checkpoint recovery remains owner-validated work.
The same output directory also caches preparation artifacts behind a source
fingerprint, tokenizer file hashes, and the packed-shard manifests. A cache hit
reuses them without importing `datasets`; changed dataset/tokenizer settings
fail before overwriting the prior run, while altered artifacts fail validation.
Each scratch run now also records its effective model configuration, tokenizer
file hashes, packed-shard manifests, training arguments, optimizer parameter
groups, scheduler horizon, and runtime identity in `run_config.json`; the same
fingerprinted record is embedded in every checkpoint. Resume compares it before
restoring model, optimizer, scheduler, RNG, or dataset state. Use a fresh output
directory after changing run settings, except for the explicit token-budget
extension path. `from_scratch(..., extend_training_budget=True)` permits only a
strictly larger `max_tokens` and the derived horizon/cadence fields. Once a
compatible checkpoint is restored, the API retains model and optimizer state,
starts a new cosine phase at the configured learning rate over the remaining
steps, and atomically updates the run manifest. This can raise the learning
rate, so it is a deliberate schedule restart, not exact scheduler continuation.
Scratch checkpoints created before this manifest was introduced are rejected
by `from_scratch()` resume because their full compatibility cannot be
established; non-scratch legacy checkpoints keep their existing compatibility.

Automatic checkpoint storage now has an opt-in `save_total_limit` in both
public argument classes. The library default remains unlimited; the notebook
uses three. New numbered local saves receive `.trainlm-checkpoint.json` only
after state and pending run metadata have been published. Cleanup retains the
new save plus the newest other owned saves, and excludes unmarked legacy
directories, temporary files, symlinks, exports, and separately named manual
backups. Filesystem cleanup errors warn without invalidating the new save.
Retention is deliberately excluded from both sections of the scratch run
fingerprint: it affects storage, not training, and older fingerprints still
match. CPU tests cover exact resumed-weight parity after pruning, failures
before publication, conservative exclusions, and budget-extension recovery.
Distributed runtimes and the legacy XLA coordinator reject this option until
coordinated retention is implemented. No TPU validation is claimed.
Focused local validation passed 89 tests covering retention, public lifecycle,
scratch resume, checkpoint contracts, roadmap consistency, and notebook cells.
When periodic checkpointing is enabled, the shared trainer now publishes the
final successful optimizer step if it falls between save intervals. It avoids
a duplicate when the cadence saved that step, skips final saves after failures,
and leaves automatic saving disabled when no interval is configured.
The public `trainer.latest_checkpoint` property returns the newest published
local checkpoint or committed native TPU checkpoint path, or `None`; passing
`resume_from_checkpoint="latest"` uses the same resolver. The notebook prints
this path for copying into run notes and includes it in failure output when
available.

PR #88 rebase-merged scratch budget extension at verified `main`
`ab11f59fc362684c3ba6da5111d7a30d2d77f8f2`, with both GitHub dependency CI
profiles passing. This retention implementation starts from that base.
PR #89 rebase-merged checkpoint retention at verified `main`
`a45432dfbee58713ee647aec8d3860dfd4fd885c`; both CI profiles passed 924 tests
and 41 subtests. PR #90 rebase-merged final-step checkpoint saving at verified
`main` SHA `5bab4b5c62ffe014b2700d2ea7157bb910d9f621` from head
`6f185a87a109140fb4aed62f60edb8542e152f86`; both CI profiles passed 928 tests
and 41 subtests. Focused CPU lifecycle tests passed 69 tests, the roadmap
consistency test passed 3 tests, and the changed trainer module compiled. No TPU
checkpoint recovery was run. PR #91 exposed `trainer.latest_checkpoint` and was
rebase-merged at verified `main` SHA
`2a68f781655d80ccbac91ed03e582e32539c0900` from head
`1c1a963486e9c42aff14f96d2ac78f1327a17a06`; both CI profiles passed 928 tests
and 41 subtests. Its focused CPU/API, notebook, and roadmap checks passed 52
tests. PR #92 added the checkpoint path to report schema v5 and rebase-merged
at verified `main` SHA `16984722897d6cf2b1f25b8548c4a1c6039ce94d` from head
`fccca1c9147580afa215456a86c32a8a11489e9d`; both dependency CI profiles passed.
Its local notebook/roadmap suite passed 24 tests and 14 subtests. PR #93 added
the schema-v6 resume script and rebase-merged at verified `main` SHA
`4de50666874592866fc50debd4261e465ffdaabe` from head
`4a089e373ec003be507cd4115165d78025cbf75a`; both CI profiles passed 940 tests
and 41 subtests. Its focused local suite passed 47 tests and 14 subtests across
resume/ZIP coverage, notebook contracts, the public pretraining API, and
roadmap consistency. It covered actual checkpoint restore, pending/committed
budget extensions, safe quoting, and refusal to construct a trainer when run
files are missing. No TPU runtime was used. The current report-driven notebook
recovery cell starts on `feat/94-notebook-report-recovery` from that verified
`main` SHA and adds no TPU validation.

PR #85's verified `main` base is `476bc68ff535cd49f3472316c5bc809bc38e7ae5`.
PR #86 added whole-run checkpoint compatibility validation and rebase-merged
at verified `main` SHA `32776292a7e6436a0b2a02fb2232ce0185d1b245` from head
`15670a2856affe641488f4b48addc9b6cf1cd52b`; both GitHub CI profiles passed.
Focused API/lifecycle/roadmap tests pass (50 tests),
and CPU/Gloo checkpoint plus token-weighting suites pass when run with
per-command network permission (22 tests). The initial full local test run had
885 passed and 13 failures. The six loopback-dependent failures passed in the
permitted rerun; remaining failures are the environment's missing `pip`, five
Transformers 5.0 Phi fixture errors (`pad_token_id` absent), and one existing
model reload mismatch. No TPU training or certification was performed.

PR #77
was rebase-merged to verified `main` at
`f844d048b17640d4d71a127c4a860dd1f6d7d52c`; the public API is in commit
`e340d7f604151ee869243d80bc5429c1bb008801`, and post-train evaluation support
is in `3fbac2a19209dfd035d021f7911f2fa0b4a2f91b`. The feature branch was
deleted after verifying the merge. The first PR #77 CI run (360) found four
failures across the new public/API contracts; the final run (362) passed all
890 tests and 41 subtests on Python 3.11. Local syntax, notebook, config-guard,
state-transition, and CLI-help checks pass. PR #79 adds explicit
`AutoModelForCausalLM` config selection and rebase-merged at verified `main`
SHA `ba5f4761c35c23403d35e44be5d564dbe7ffadb3`; CI passed 891 tests and 41
subtests. These are CPU/software checks, not TPU training results. No TPU run
was made. PR #80 added model config and parameter provenance to report schema
v2 and rebase-merged to verified `main` at
`e9b4a1bde4869d89c323ed457a6b6cebccb0ae40`; CI passed 891 tests and 41
subtests. No TPU run was made. The feature
is opt-in and does not change existing accelerator defaults or base dependency
bounds, so existing workflows can continue without using it. PR #81 added
tokenizer and runtime provenance in report schema v3 and was rebase-merged to
`main` at `4cfacae39e4e677f442208dfa098bb51009cfd46`; it adds no TPU training
evidence. PR #82 added requested/effective run context in report schema v4 and
rebase-merged at verified `main` SHA
`4107f2f2ab86c2e90d43428ad1a779a25fe86cac`; GitHub pytest passed (run
`41746021087`). No TPU training was run.

- PR #18, including PR #16, PR #20, and PRs #28–#43 are merged.
  PR #28
  merged the native single-device foundation at
  `860bb289bf35444d8d435cf41caf51bcf77afaaf`; PR #29 rebase-merged streaming
  packed-data support at verified `main`
  `f3df754ea98156b083801447b41540670134bf2e`; PR #30 rebase-merged packed
  causal-LM update parity at `2b5e0e9bef3b105d8ebeb40a0a9d04d1d4b7e748`;
  PR #31 rebase-merged the CPU scratch lifecycle at
  `a58a9e13d792c2468f4c20d6cdc7375f4bc63ffd`; PR #32 rebase-merged exact
  packed-data checkpoint recovery at `82ead5b72991fc154738bcd8334e477d563d7ea3`
  after passing CI; PR #33 rebase-merged current Colab source references at
  `83d4b08dd9c510a6edf6ae0a5dc7e2b739343eb6`; PR #34 rebase-merged the
  training-split BPE tokenizer at `7c75f6d0aa8ff75299328847e9fa7dcfcdf791e1`.
  PR #35 rebase-merged local GPT-2 configuration at
  `063a8b9ff9b42733f8b9f4720ba3307c2b2f4cdf`; PR #36 rebase-merged pinned
  custom corpus selection at `d075987574fe471d1b9159315a0688416664f2ba` after
  passing CI; PR #37 rebase-merged opt-in BF16 and precision reporting at
  `31aa3acd06bf89c9430615103ebd4dd0da46ae6c`; PR #38 rebase-merged optimizer
  schedule and seed controls at `357cac307bfaf94b4bcf4c55ed53c9fb99815cc8`;
  PR #39 rebase-merged opt-in gradient checkpointing at
  `df0a9ff562e48b67ae4ea48dbe10abe565137f4b` after CI passed, including a real
  GPT-2 CPU backward test. PR #40 rebase-merged per-device batch controls and
  nominal token reporting at verified `main` SHA
  `d7359bc20d5fa2af800c113b2e85779b6987f2f4`; its GitHub pytest workflow
  passed in 1m45s. PR #41 added supervised-token budgets at verified `main`
  SHA `a24514a7693e918c1c10c8a84c705399fa3b3bcf`; its GitHub pytest workflow
  passed in 2m10s, including the packed-training token-budget regression. PR
  #42 rebase-merged sparse logging and terminal loss snapshots at verified
  `main` SHA `2e90537371aa03c98f4fc395edb2b231727eaebb`; GitHub pytest passed in
  2m0s. PR #43 bundled the exported model/tokenizer with the JSON evidence
  report for Colab/Kaggle download at verified `main` SHA
  `73952a9724456390c6def5b74a474d4274dc7544`; GitHub pytest passed in 2m4s.
  PR #44 merged the CPU export reload/generation check into `main` at
  `fad1df41278d4ea904b4d26968d75ee97ae17923`. Its evidence report records
  `reload_generate`; this checks export usability on CPU and is not TPU
  generation, training, or convergence evidence. PR #45 rebase-merged
  agent-facing documentation and package metadata at verified `main` SHA
  `d983408221b92d631aea817bee7539e07de1bbf5`; it adds no runtime or TPU
  evidence. PR #47 aligned the live status and historical labels at verified
  `main` SHA `e109ad253754303e44eb6162aacb046c415b990b`; it also adds no runtime
  or TPU evidence. PR #48 rebase-merged distributed token correctness at
  verified `main` SHA `7ae837d085fbbaab9f6c2d3617ff129b89dd454e`; its full
  Python 3.11 GitHub CI run passed. PR #49 rebase-merged checkpoint integrity
  at verified `main` SHA `01a71f990fc5578b77be6e2e0a84473486fbe488`; CI runs
  #296 and #297 passed. PR #50 rebase-merged public trainer checkpoint
  preflight at verified `main` SHA `f57d6d71339753a6a3cee4082a43dfa2f5113017`;
  CI run #299 passed.
- T0.1 code inventory is recorded in `TORCHTPU_SALVAGE_INVENTORY.md`. The
  mapped legacy cards remain candidates, not an approved implementation queue.
- Current implementation backend: PyTorch/XLA. Target: native TorchTPU on
  Cloud TPU. No TorchTPU install profile or hardware certification is
  established. Hosted Kaggle documents v5e-8 and hosted Colab TPU v2; use a
  one-chip Cloud TPU v5e VM for the planned v5e-1 single-device evidence.
- PR #28's final GitHub CI run passed all 804 tests under its Python 3.11
  environment. The earlier 802-pass/2-fail run exposed two test-fixture
  assumptions that were corrected before merge. The older PR #18 result (700
  passed / 21 failed under Python 3.12.14, torch 2.9.0+cpu, Transformers 5.0.0,
  Hub 1.3.5, pytest 9.1.1) is historical, not the current baseline.
- T1 policy: use a one-chip Cloud TPU v5e VM first for single-device import,
  device operations, autograd, optimizer, and compile checks. Use v5e-8 for
  collective, DP8, sharding, distributed-checkpoint, or scaling requirements.
  An owner-run eager TPU smoke passed. The base workspace interpreter has no
  PyTorch; the current CPU checks ran in an isolated temporary environment.
- The current distributed-training slice implements global valid-token
  gradient normalization for backends that average gradients before clipping,
  and synchronizes distributed `max_tokens` stopping at a global microbatch
  boundary. DDP-style backends that declare pre-clipping gradient averaging
  reduce update losses and effective global batch size; callback progress
  counters are summed across distributed ranks. Two CPU/Gloo regressions cover
  accumulated unequal-mask parity, rank-consistent budget stopping, and global
  reporting. A temporary Python 3.12.14 environment with torch 2.9.0+cpu and
  pytest 9.2.0 passed the focused CPU/Gloo and related trainer/runtime/API suite
  (79 tests); PR #48's full GitHub CI also passed. `TrainerState.tokens_seen` and `samples_seen` remain
  per-rank; callback progress and the stopping threshold use global counts.
  The DDP-style token-indexed scheduler and `global_batch_size` use global
  counts too. The final microbatch can take the run past the requested budget.
  Native TorchTPU launch, distributed
  runtime, checkpointing, and v5e-8 training remain unimplemented or unverified.
- T5 private rank-sharded checkpoints now write schema-v3 SHA-256/size
  metadata and verify each rank's artifact and payload before state restore.
  Distributed shard-write, manifest-publication and resume-preflight failures
  are coordinated across ranks; legacy v1/v2 loading remains supported. PR
  #49 merged at `01a71f990fc5578b77be6e2e0a84473486fbe488`; CI runs #296 and
  #297 passed. Native TPU checkpoint validation remains pending.
- PR #50 added public trainer checkpoint preflight for schema, model
  keys/shapes, trainer fields, runtime compatibility, RNG state and data cursor
  before restoring model or optimizer state. It merged at
  `f57d6d71339753a6a3cee4082a43dfa2f5113017`; CI run #299 passed. PR #51
  rebase-merged at verified `main` SHA
  `c63e6903767f3a2aab2ff733e6248ff0c4e4bd3b`; GitHub CI run #302 passed. It
  records optimizer/scheduler class identity and checks optimizer group layout
  before restore, while accepting earlier schema-v1 checkpoints without the
  optional type metadata. PR #52 rebase-merged at verified `main` SHA
  `9cf8d4ba0de0b91cc52908d9d98f984b29873a33`; CI run #304 passed. It adds
  preflight checks for common optimizer slot shapes and scalar steps, plus
  scheduler payload fields, ranges, and optimizer-group counts. Branch
  PR #53 rebase-merged at verified `main` SHA
  `8ea810c2e5e54f492cb4e29e0844a74c9935f23e`; CI run #306 passed. It validates
  persisted counters and finite metrics before restore. PR #54 rebase-merged at
  verified `main` SHA `6ab2b8c9fcde5895c71d5f8450808b7f5756c520`; CI run #308
  passed. It preflights packed-data cursor identity. PR #55 rebase-merged at
  verified `main` SHA `ea9dbadfb86f17cd3713b2abcee808677c38e845`; GitHub CI
  passed (run ID `41578130077`). It restricts serialized trainer state to
  durable counters and metrics, rejecting transient lifecycle fields before
  restore. Its focused lifecycle tests passed (29 tests), and the full CPU
  suite passed (871 tests and 41 subtests) in Python 3.12.14 with torch
  2.9.0+cpu and pytest 9.2.0.
- PR #56 rebase-merged at verified `main` SHA
  `4aa3614858065c6255c4ecc2d0949c24ee715f70`; its GitHub pytest workflow passed
  in 2m24s. It documents and tests the optional custom-dataset
  `validate_state_dict` hook. PR #57 rebase-merged at verified `main` SHA
  `e62052ae39d6216bce0dd5aec86fc42d9f10a837`; GitHub pytest passed (run ID
  `41579944075`). It compares a two-step custom iterable checkpoint plus two
  resumed steps with uninterrupted four-step training and requires exact
  model-tensor equality.
- PR #58 rebase-merged at verified `main` SHA
  `e90697f99db29355e135161ea62e66d91bbc6047`; GitHub pytest passed (run ID
  `41581911584`). Commit `c45b201` rejects custom iterable checkpoints with a
  mismatched dataset ID or out-of-range cursor before model, optimizer,
  scheduler, trainer, or dataset state changes.
- PR #59 rebase-merged at verified `main` SHA
  `2985ef953546506f3cd6d84c8ff79739c6388a9b`; GitHub pytest passed (run ID
  `41582739096`). Commit `cc9ccd3` adds a CPU regression for stateful custom
  iterables that implement `state_dict()` and `load_state_dict()` but omit the
  optional `validate_state_dict()` hook. It requires exact model-tensor parity
  with uninterrupted training.
- PR #60 rebase-merged at verified `main` SHA
  `839d8fcc1f9320ff6193b7c54399c629881945f0`; GitHub pytest passed (run ID
  `41585383984`). Commit `fd9e61a` adds a CPU regression that removes the
  optional dataset cursor, data-loader RNG snapshot, and optimizer/scheduler
  type metadata from a schema-v1 checkpoint. Resume replays the saved micro-step
  count and matches uninterrupted model weights exactly. Its combined
  checkpoint/API lifecycle suites passed (39 tests); compileall and
  `git diff --check` passed. No full-suite rerun or TPU checkpoint validation
  is claimed for PR #60.
- PR #61 rebase-merged a documentation-only current-status correction at
  verified `main` SHA `448ea9dc3d5969aa88d612bdfbbf11af682bc78d`; GitHub pytest
  passed (run ID `41585945140`).
- PR #62 rebase-merged at verified `main` SHA
  `821beacdee235e9ed6e70781023126023120adef`; GitHub pytest passed (run ID
  `41594252050`). Commit `31742a5` removes a schema-v1 checkpoint's data-loader
  RNG snapshot and asserts that a RandomSampler resume rejects before model,
  optimizer, scheduler, trainer, or loader-generator state changes. The
  checkpoint/API lifecycle suites passed (40 tests); compileall and
  `git diff --check` passed. No full-suite rerun or TPU checkpoint validation
  is claimed for PR #62.
- PR #63 rebase-merged the status correction at verified `main` SHA
  `97a061df7bc96fc311d1bab341d91f92c8a43b61`; GitHub pytest passed (run ID
  `41594915207`).
- PR #64 rebase-merged at verified `main` SHA
  `dd3cf7bcea791f370ea0c58d3e863a1702dfc79f`; GitHub pytest passed (run ID
  `41595632995`). Commit `eaf7a88` adds an end-to-end CPU regression for a
  caller-provided `SequentialSampler` DataLoader without TrainLM-owned loader
  RNG. It removes the optional loader RNG checkpoint field, resumes from an
  intermediate checkpoint, and requires exact model-weight parity with
  uninterrupted training. The checkpoint/API lifecycle suites passed (41
  tests; 44 including the roadmap contract). No full-suite rerun or TPU
  validation is claimed for PR #64.
- PR #65 rebase-merged the status update at verified `main` SHA
  `908e21fa2fedd34a0de47203a0f12c5d4a473ef3`; GitHub pytest passed (run ID
  `41596201261`).
- PR #66 rebase-merged at verified `main` SHA
  `4a877768a1c8fcaae654be6c18e88ce374bdce25`; GitHub pytest passed (run ID
  `41596764595`). Commit `11f383a` extends the sequential-loader CPU regression
  with a checkpoint at the exact epoch boundary, after all four distinct rows.
  Both the within-epoch and boundary resumes match uninterrupted final model
  weights. The checkpoint/API suites passed 42 tests (45 including the roadmap
  contract). No full-suite rerun or TPU validation is claimed for PR #66.
- PR #67 rebase-merged the status update at verified `main` SHA
  `74ba1d1d8c05a782b29b18c31131347025b7fcef`; GitHub pytest passed (run ID
  `41597332118`).
- PR #68 rebase-merged at verified `main` SHA
  `f52ad5fbed7ab51335ff04079fafbd5939069ece`; GitHub pytest passed (run ID
  `41631351981`). Commit `728e787` adds CPU coverage for sequential resume with
  a six-row `DataLoader`, batch size two, and gradient accumulation two. The
  checkpoint records `micro_step=2` after optimizer step one; resume advances
  past both consumed batches and matches uninterrupted weights through the next
  epoch rollover. The checkpoint/API suites passed 43 tests (46 including the
  roadmap contract). No full-suite rerun or TPU validation is claimed for PR
  #68.
- PR #69 rebase-merged the status update at verified `main` SHA
  `bf57ebb20600eb69f4ea5f1666d427cf610d3870`; GitHub pytest passed (run ID
  `41632033098`).
- PR #70 rebase-merged at verified `main` SHA
  `2f0eafcbeb6d394496154fdddc22a55233aee5d0`; GitHub pytest passed (run ID
  `41632939362`). Commit `2383d35` adds CPU coverage for a five-row sequential
  `DataLoader` with batch size two, `drop_last=False`, and gradient accumulation
  two. Resume consumes the one-row final batch and the next epoch's first batch
  without dropping or repeating rows; final weights match uninterrupted
  training. The checkpoint/API suites passed 44 tests (47 including the roadmap
  contract). No full-suite rerun or TPU validation is claimed for PR #70.
- PR #71 rebase-merged the documentation status correction at verified `main`
  SHA `33094937a78e9a773ba8dbdd2007f73b10ec0a20`; GitHub pytest passed (run ID
  `41633758016`).
- PR #72 rebase-merged at verified `main` SHA
  `df9f232dff191354b976b0059a2a174777792a5f`; GitHub pytest passed (run ID
  `41634659366`). Commit `df9f232` adds CPU coverage for exact resume through
  TrainLM's default shuffled, TrainLM-owned random sampler. It checkpoints after
  the first update, restores sampler RNG/cursor state, crosses an epoch boundary,
  and matches uninterrupted model weights through step three. The focused API
  lifecycle, trainer, and roadmap-contract suites passed (57 tests). No
  full-suite rerun or TPU validation is claimed for PR #72.
- PR #73 rebase-merged the documentation status correction at verified `main`
  SHA `6090ba72e32d9d920499807f0c7b5a15e8ae5861`; GitHub pytest passed (run ID
  `41635317392`).
- PR #74 rebase-merged at verified `main` SHA
  `b0f81e91ad6c11aef865441efc05455202557677`; GitHub pytest passed (run ID
  `41636368557`). Commit `b0f81e9` extends the default shuffled-sampler exact
  resume regression to checkpoint both within an epoch and exactly after all
  four batches are consumed (`micro_step=4`). Both resumes reach step three
  with exact model-weight parity. The focused API lifecycle, trainer, and
  roadmap-contract suites passed (58 tests). No full-suite rerun or TPU
  validation is claimed for PR #74.
- The last full local suite, on PR #56's branch, produced 865 passed, 7 failed
  and 41 subtests: missing `pip` blocked the wheel smoke, five Phi cases hit the
  installed Transformers 5.0.0 `PhiConfig` missing `pad_token_id`, and a
  separate TrainLM model reload test had mismatched weights.
- [T1 v5e-1 probe guide](T1_V5E1_PROBE.md) and the
  [Colab-openable Kaggle/Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TorchTPU_T1_v5e1_Probe.ipynb) describe
  synthetic smoke. The latest owner run at source `d3dea896` passed allocation,
  forward, autograd and AdamW update on Python 3.13.18, torch 2.14.1+cpu,
  torch_tpu 0.2.0 and libtpu 0.0.50. Runtime properties reported device name
  `TPU v5e` and one visible device; the owner confirmed this was v5e-1. The
  shared notebook output did not show the VM image or topology; Colab may have
  hidden those details. Lifecycle and production integration remain
  unverified. Native `torch.compile` is now an opt-in on the one-device trainer
  path but remains unverified on TPU hardware. At the time of that report, the
  package profile flagged the old `<2.14` Torch bound and an IPython 7.34.0 →
  missing Jedi environment warning. The current compatibility change widens
  the core bound to `<2.15`, and parsed unrelated environment conflicts remain
  visible without blocking the TrainLM profile. The eager smoke is not CPU
  conformance or native trainer compatibility; the minimum/current CPU CI
  matrix is still pending.
- The [utility inventory and UX plan](TORCHTPU_UTILITIES.md) maps public upstream
  APIs to T1–T11 and reusable repository contracts. A source-level
  report now observes native TPU availability, device properties, versions and
  allocation in the probe process; CPU-only contract tests pass. The public
  trainer uses a non-allocating view for `preflight()` and optional
  `explain(include_hardware=True)`. A separate explicit native path is now
  described below; automatic TPU routing remains gated.
  The notebook runs a read-only metadata/profile check against TrainLM's
  source requirements, TorchTPU's installed requirements and `pip check`,
  then bounds eager/compile child processes. It does not alter packages;
  clean installation remains unverified. An opt-in T1 lifecycle harness can
  start two or three fresh single-device probe processes, bound each run,
  terminate its process group on timeout, and save sanitized per-run evidence.
  It can repeat eager or compiled smoke; compiled repeats require a compile-pass
  marker from every child and stay disabled until the standalone compile check
  passes. Source-level tests cover the mode guard; no compiled TPU
  repeat-run/teardown result exists.
  The opt-in compile probe now runs an eager reference beside the compiled tiny
  model on identical batches and checks forward, loss, gradients, and AdamW
  updates at each step (`rtol=1e-2`, `atol=1e-3`). Its pass marker is required
  by the evidence and lifecycle reports. This is synthetic numerical parity,
  not causal-LM training validation; no owner compiled run has been reported.
  `accelerator="auto"` selects CUDA/CPU only when native TPU is not detected;
  it now raises on detected TPU or broken TorchTPU discovery rather than
  silently falling back. Explicit `tpu` still uses the XLA coordinator.
  The successful standalone probe's Torch version prompted a separate core
  compatibility change: the current branch widens the source requirement to
  `torch<2.15` and adds constrained minimum/current CPU CI. This does not create
  a native TorchTPU install profile or certify native training.
- The native T1 single-device process can write a normalized JSON discovery
  artifact with `--report-json`. Trainer result handling accepts the same
  bounded `native_tpu_report` field from a worker summary and shows it in
  `preflight()` and opt-in `explain()`. The report is scoped to the device-owning
  process and does not certify training. The existing XLA worker does not
  produce this field; native worker launch and automatic routing remain gated.
- T1.4 has a single-host v5e-8 opt-in distributed probe and copyable notebook.
  Parent-side TorchTPU launch preparation precedes worker spawning; the probe
  checks topology structure and address count without displaying coordinator
  addresses, then reconciles per-rank `tpu_dist` sum all-reduce results.
  CPU contract tests and read-only inspection pass. No v5e-8 run, multi-host
  support, gradient-correct DP, or native trainer integration is claimed.
- T1 notebook usability/reliability checkpoint: direct Colab links, fresh
  detached source snapshots, source-change guards and a downloadable scoped
  JSON report are implemented. The v5e-1 notebook now runs the repository CPU
  test suite in a bounded subprocess before its eager TPU smoke, using the runtime's
  installed Python and PyTorch without installing packages. CPU suite status
  is recorded separately from native TPU evidence. Fatal teardown logs cannot
  pass on exit zero;
  timeout/interruption cleanup kills surviving child processes even after
  the launcher exits. The report keeps CPU, skipped and incompatible outcomes
  explicit and excludes raw logs, environment dumps, addresses and paths.
  Validation: 38 source-level tests pass on Python 3.12.14, including a real
  process-group cleanup test. Both notebooks' setup, metadata, expected
  missing-PyTorch failure/skip, download and changed-source guard paths were
  exercised using local test clones and a simulated Colab download callback.
  The owner ran the full CPU suite at source `08f02fb`: 746 passed, 17 failed
  under Python 3.13.18/Torch 2.14.1. Two facade tests incorrectly used auto
  device selection on the attached TPU; the wheel smoke assumed `venv` was
  available; and a roadmap test referenced a renamed historical heading.
  Those causes are corrected locally. The 12 loss-layout failures came from
  dividing by all shifted labels instead of excluding `ignore_index` targets;
  the loss now counts supervised targets even without an explicit mask. The
  scalar-materialization assertion expected one extraction although the
  logged loss and gradient norm are separately materialized; it now checks for
  exactly those two snapshots at the configured cadence. These fixes are not
  yet rerun on the owner runtime. PyTorch and pytest remain unavailable here,
  so rerun the full suite before changing the dependency range.
- The owner requested implementation to continue while hardware queues are
  long. An opt-in `accelerator="native_tpu"` backend now reuses the existing
  PyTorch engine for one visible TPU device; it checks native allocation and
  synchronizes through PyTorch's generic accelerator API. Existing
  `accelerator="tpu"` continues to mean the XLA coordinator, while `auto`
  still blocks on TPU. The copyable Colab notebook fine-tunes a small pretrained
  causal LM for three steps and exports its model and tokenizer. This is
  experimental code only: no native trainer run has been observed, and
  compile, multi-device, package compatibility, CPU conformance, and target
  performance remain unverified. The runtime now captures/restores accelerator
  RNG state through guarded generic APIs; an absent API rejects checkpointing
  or resume. Native resume is currently guarded to deterministic map-style
  data, zero loader workers, and a TrainLM-owned random sampler or sequential
  order. PR #32 adds a fingerprinted cursor for packed iterable data and wires
  it into trainer checkpoints; exact CPU resume parity passed CI. Native TPU
  validation remains pending.
  `torch_compile=True` opts into `torch.compile` for
  this native path and is rejected for other accelerators or injected runtimes.
  Eager remains the default; compile failures surface without fallback, and
  saved runtime state rejects resume if the compile setting changed. This has
  source-level coverage only; no v5e-1 compile result is claimed. T1 evidence
  gates still apply before widening the package range or enabling automatic
  routing.
  The native training notebook has an optional continuation comparison using
  the public API: save after three steps, rebuild and resume for step four,
  check model/tokenizer exports, then train a fresh uninterrupted four-step
  reference. Both paths use a constant scheduler and deterministic map-style
  data. Every final model tensor must match exactly; mismatches fail with a
  count, without exporting tensor values. The eight total optimizer steps
  default off. Notebook syntax is checked locally, but an owner TPU run and
  broader state/data validation remain necessary for exact-resume claims.
  A CPU regression test now mirrors the notebook's three-plus-one versus
  uninterrupted-four comparison using a tiny causal model, the same initial
  parameters and data, and a constant scheduler. It requires a real fourth
  update and exact final tensor equality. This workspace lacks PyTorch and
  pytest, so the new test awaits execution in a provisioned CPU environment.
  The native training notebook now delegates discovery, HF model loading,
  training, optional compile/resume, and export to
  `examples/torchtpu_native_trainer_smoke.py` in a bounded child process.
  The parent reuses the T1 process-group timeout/cleanup helper (900 seconds
  for eager, 1800 seconds for compile or resume), writes raw logs only to a
  local session directory, and prints bounded status fields. Source-level
  tests verify the parent does not import TorchTPU and that incompatible modes
  fail before hardware initialization. The child now emits a bounded summary;
  the parent validates its device, package versions, requested phases and pass
  markers against process completion before writing a shareable JSON report.
  A missing, malformed or duplicate summary cannot produce a pass. Raw logs
  and artifact paths stay local. This reporting path has source-level tests;
  no owner native trainer run has been reported.
  Before the HF model download, the notebook reuses the read-only T1 installed
  metadata checker and a trainer-specific classifier. Missing distributions,
  TorchTPU requirement mismatches, training-stack `pip check` conflicts and
  unclassified conflicts block the run and produce a shareable blocked report.
  In the earlier implementation, a source-Torch mismatch and unrelated
  IPython/Jedi conflict were both summarized as `experimental_mismatch` for
  the explicit smoke. The current compatibility change removes the observed
  Torch 2.14.1 mismatch from the core contract and classifies a parsed unrelated
  Jedi conflict as a nonblocking environment warning. Training-stack or
  unclassified conflicts remain blocking. This metadata change does not certify
  a native TorchTPU install or training run; the owner should refresh the
  Colab report after the branch is available.
  The native fine-tuning smoke pins its `distilbert/distilgpt2` model and
  tokenizer to Hub commit `2290a62682d06624634c1f46a6ad5be0f47f38aa`. The
  separate scratch path now trains a byte-level BPE tokenizer only on pinned
  training documents and builds a local GPT-2 config from explicit dimensions.
  This branch lets users select a pinned Hugging Face corpus, its distinct
  train/validation splits, and the text column.
  A separate [Colab notebook](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)
  constructs that GPT-2 architecture through `from_config`, so weights start
  random, then fits a byte-level BPE tokenizer on bounded TinyStories training
  text only. It reopens the pinned train/validation streams, writes packed
  shards, evaluates, saves periodic checkpoints, and exports through TrainLM.
  The dataset commit is `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; the
  package gate requires the optional `datasets` distribution before download.
  This source-level path has not run on TPU and does not establish convergence,
  throughput, or a production-scale recipe.
  PR #28's CI passed all 804 tests after a follow-up corrected two native-runtime
  test fixtures. The PR merged with merge commit
  `860bb289bf35444d8d435cf41caf51bcf77afaaf`; TPU execution and convergence
  remain owner-run validation.
- PR #29's GitHub CI passed and the branch was merged with Rebase & Merge.
  It adds a streaming writer for explicit uint32 token shards, SHA-256
  manifests, uint64 document offsets, and a from-scratch path that feeds the
  pinned TinyStories stream through `PackedBinDataset`. The notebook report
  records source revision, token/document counts, and packed-shard hashes.
- PR #30's CI passed its packed-data reference test and it was rebase-merged.
  PR #31's GitHub CI passed the random-init GPT-2 train/eval/checkpoint/export
  smoke; PR #32 passed exact packed-data checkpoint continuation and stale-data
  rejection; PR #33 passed active-notebook source checks; PR #34 passed the
  train-only BPE tokenizer tests; PR #35 passed explicit-config and evidence
  tests; PR #36 passed pinned custom-corpus tests; PR #37 passed BF16 option
  and precision evidence tests; PR #38 passed optimizer-config evidence tests;
  PR #39 passed checkpointed GPT-2 CPU forward/backward and full CI; PR #40
  passed CI after adding batch controls; PR #41 passed CI with the token-budget
  CPU integration regression; PR #42 passed full CI with sparse-logging and
  terminal-loss regressions; PR #43 passed full CI with notebook artifact
  packaging. PR #44 added the CPU export reload/generation check. This container lacks
  Torch/pytest, so full runtime tests are left to branch CI. No TPU run or
  convergence result is claimed.
- `trainlm doctor` adds a one-command read-only package/runtime report to the
  CLI, reusing the T1 profile and hardware discovery contracts. From a source
  checkout it records the Git revision; an installed wheel uses its declared
  requirements. It does not install packages or train; incompatibility remains
  visible for owner review. The v5e-1 notebook now invokes that same CLI from
  its fresh checkout without installing the package, so its actual Colab
  report can be compared with the separate eager-smoke evidence. The downloadable
  artifact now records whether the command ran, its profile and TPU statuses,
  and whether it used the same source SHA. Package reports include bounded
  parsed `pip check` requirement conflicts instead of only affected names.
  Because the notebook runs from a source checkout without installing TrainLM,
  the absent `trainlm` distribution is now reported as `source_checkout` and
  is not counted as a missing runtime dependency. Other dependency mismatches
  and `pip check` conflicts remain visible and can still make the profile
  incompatible. The pip conflict parser accepts both `has requirement` and
  `requires` output forms; the latest owner report identifies the unrelated
  IPython 7.34.0 → missing Jedi conflict. That captured doctor report predates
  the current `<2.15` core requirement and still records its original failed
  outcome; a refreshed report has not been supplied. Native discovery and
  eager smoke passed, while compile, lifecycle, distributed probes and full
  native trainer execution remain unrun.

- Current compatibility follow-up on `fix/83-torchtpu-package-profile` is based
  on fetched `main` SHA `4107f2f2ab86c2e90d43428ad1a779a25fe86cac`. Commit
  `e3b9f24` widens the core Torch range to `>=2.9,<2.15`, aligns the current
  Python 3.13 profile with Torch 2.14.1 / Transformers 5.20.2 / Hub 1.33.0,
  adds a minimum/current CI matrix, and lets the preflight report unrelated,
  fully parsed `pip check` warnings without blocking. Focused local source tests
  pass (26 unittest cases plus five dependency-matrix contract checks); Python
  and PyTorch are unavailable in this container, so the full CPU suite has not
  run locally. The first GitHub matrix run passed core-current; core-minimum at
  Python 3.10 / Torch 2.9 / Transformers 5.0 had 887 passed, six failed and 41
  subtests. Five failures hit the old PhiConfig's missing `pad_token_id`, and
  one was TrainLM model reload parity. Commit `bbf1b2e` raises the declared
  Transformers floor and minimum lock to 5.15.0, matching a previously passing
  full suite. Its first retry exposed that this Transformers version requires
  Hub 1.5.0; commit `cfc05ed` raises the project Hub floor and minimum lock to
  match. CI run `41752008510` passed 893 tests and 41 subtests in both profiles:
  Python 3.10 / Torch 2.9 / Transformers 5.15 / Hub 1.5, and Python 3.13 /
  Torch 2.14.1 / Transformers 5.20.2 / Hub 1.33. An owner-runtime metadata
  refresh also remains pending. This does not certify CUDA, native TorchTPU
  training, compile, lifecycle or distributed execution.

## Historical XLA development record

The sections below preserve the former M0–M14 implementation context, TPU
measurements, and technical investigations. Interpret backend-specific status
as applying to PyTorch/XLA only. Reuse decisions belong to the T0 inventory;
old XLA targets and PR cards do not automatically remain TorchTPU requirements.

## Sequential PR delivery policy

See [the current PR plan](PR_DELIVERY_PLAN.md) for the accepted T0.2 mapping
and [the roadmap](ROADMAP.md) for the historical M0–M17 specification.
Future implementation work follows the owner-approved T0–T11 sequence, with
small commits and verified `main` between PRs.

## Product intent

TrainLM is a Hugging Face-like trainer for decoder-only autoregressive models.
Users should provide a Hugging Face model ID/path (or a model object), a
tokenizer/processing class, a dataset or packed `.bin` source, and a small set
of familiar training arguments. TrainLM owns TPU/PJRT process launch, data
validation, model capability inspection, safe optimization transforms, XLA
compilation, checkpointing, logging, and export behind that surface.

The model implementation in Transformers remains the source of semantic
truth. TrainLM must not fork or edit every HF model family. Optimizations are
selected through capability inspection and optional, version-guarded adapters;
unsupported features remain on a correct fallback path and are reported by
`trainer.explain()`.

Initial scope is dense decoder-only AR language models. MoE and diffusion/
DLLM architectures are later extensions, not hidden promises of the current
API.

## User experience target

The documented workflow must look like a normal HF trainer, not like a TPU
debug notebook:

```python
from trainlm import TrainLMTrainer, TrainLMTrainingArguments

trainer = TrainLMTrainer(
    model="org/model-or-local-path",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=TrainLMTrainingArguments(
        output_dir="runs/example",
        max_steps=1000,
        sequence_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        accelerator="auto",
    ),
)
trainer.train()
```

The worker script, rank probe, preflight stage, cache directory, manifest
layout, and log parsing are internal implementation details. An equivalent
`trainlm train` CLI/config path is required for unattended jobs.

## Historical XLA repository state at the recorded milestone

- PR1 and PR2 are merged.
- PR3 (`milestone/m5-m7-xla-compatibility`) was rebase-merged into `main`,
  preserving its individual feature commits in linear mainline history.
- At that historical snapshot, PR4 was the active implementation branch:
  `milestone/m8-m9-optimization-core`.
- PR4 has been synchronized with the merged PR3 mainline; remote backup branches
  preserve the pre-synchronization PR3 and PR4 tips.
- Codex may inspect, commit, push story branches and open PRs in accordance
  with AGENTS.md and the sequential roadmap policy. Only the owner approves
  merges; do not create subsequent story branches before the current PR merges.
- Do not claim TPU validation from a local run. TPU validation is performed by
  the owner on Kaggle/Cloud TPU.

## Historical PyTorch/XLA TPU evidence

The current generic PyTorch/XLA path has completed a v5e-8 two-update smoke
and a 100-step baseline. The latest sparse-loss-materialization run reports
approximately **319,302 global supervised tokens/s** and **319,458 scheduled
tokens/s** over **311.824 s**. This is far below LaughLM's approximately 1M
tokens/s result and is a baseline, not a parity claim.

Current known bottlenecks are full-logit materialization, generic HF SDPA,
host-unrolled gradient accumulation, lack of fused QKV/MLP/RMSNorm kernels,
and incomplete TPU-side data/worker orchestration behind the public API.
Optimization targets (850K gate and LaughLM-class parity) must be measured in
matched runs; planning ranges are not guarantees.

### Evaluation partition API repair (2026-09-13)

The first public-notebook training attempt passed probe and model preflight
but failed when the worker prepared evaluation data: it accessed the removed
`BatchPartitionPlan.assignments` field. The worker now checks the current
rank-local `batch_indices` schedule, with a contract regression test covering
the boundary. This fixes evaluation setup only; the v5e-8 lifecycle and all
performance claims remain pending a fresh target run.

The validation notebook has since switched its model source to the pretrained
`HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint. Its Llama-family GQA shape
and 49,152 vocabulary exercise pretrained reconstruction and non-MHA head
geometry. The existing LaughLM `.bin` shards are intentionally retained only
for lifecycle testing because their tokenizer differs from SmolLM2; a valid
loss/quality experiment requires shards produced by the SmolLM2 tokenizer.
The first rerun uses one train shard and sequence length 128 to keep the
full-logit lifecycle compilation small. Model preflight separately caps its
diagnostic forward at 16 tokens: compiling the full requested shape in each of
eight ranks was redundant and could drive aggregate host compiler memory above
250 GiB before training. Probe and preflight also have bounded coordinator
inactivity limits that reclaim the complete process group on a PJRT/compiler
hang without killing a slow stage that continues to emit progress.
The matching YAML starter is `examples/smollm2_135m_tpu_pretraining.yaml`.

The notebook bootstrap now removes the common TensorFlow distribution
variants, Keras packages, and optional vision/audio wheels. It also removes an
orphaned `tensorflow/` package directory when no distribution metadata remains
and keeps the post-restart importability assertion as a hard setup gate.

The public notebook now bootstraps its source revision explicitly. Before
installation it clones the single
`milestone/m10-m12-kernels-parity` branch, enters `/kaggle/working/TrainLM`,
fetches `origin/<branch>`, and checks out a local branch at that remote ref.
It prints the short subject and full SHA so the run is reproducible. On a
rerun where the directory already exists, skip the clone cell and rerun the
fetch/checkout cell; this keeps the user-facing workflow current without
exposing private worker or topology commands.

## Implemented public API slice (M8-F0, in progress)

`src/trainlm/api.py` currently provides:

- `TrainLMTrainingArguments` with familiar batch, sequence, optimizer,
  scheduler, precision, logging, checkpoint, and accelerator fields;
- `TrainLMTrainer` accepting a model object, HF model ID/path, or
  `ModelSourceConfig`;
- map-style/iterable torch datasets and prebuilt `DataLoader` inputs;
- `train()`, `evaluate()`, `save_model()`, `save_state()`, `explain()`, and a
  lightweight `log_metrics()` hook;
- lazy HF model resolution so importing `trainlm` does not import
  Transformers/XLA before a model ID is actually used;
- delegation to the existing backend-neutral `training.Trainer` rather than
  a second training loop;
- private TPU coordination that defers model/runtime/optimizer construction to
  spawned workers, runs collective and model preflight stages, preserves the
  supported public training arguments in the worker request, and returns a
  structured coordinator summary;
- public `PackedBinDataset.from_directory()` and `.from_hub()` constructors
  that validate manifests and payload integrity eagerly, yield fixed-length
  causal-LM examples, and apply deterministic rank partitioning;
- local CPU/CUDA save and evaluation cadence, callback metric delivery, and
  versioned model/optimizer/scheduler/runtime/RNG/trainer-state restoration;
- structured TPU metric artifacts that the coordinator validates and delivers
  to public callbacks together with a normalized trainer-state snapshot;
- a family-neutral structural inspector used by `trainer.explain()` that
  reports only config/module/alias/signature-backed facts and leaves residual
  or custom semantics explicitly unknown.
- a deterministic optional model-adapter registry whose entries require exact
  model/config classes, inspected semantic capabilities, source providers, and
  explicit tested package versions; resolution is pure and records every
  rejection reason without importing or mutating a model.
- a pure optimization planner that selects declarative providers by backend,
  precision, inspected component kind, runtime requirements, and explicit
  request; it records portable fallbacks and blocks required unsupported paths
  before any model mutation.
- a transactional transform registry that captures rollback state before each
  mutation, validates inverse mappings and parameter aliases, and reverses the
  failing transform plus all prior transforms when application or downstream
  optimizer construction fails.
- a versioned state-dict layout converter for transformed resume and canonical
  Hugging Face export, with reversible concatenation/splitting plus strict key,
  shape, dtype, device, collision, and tied-alias validation.
- a versioned optimization explanation report with stable dictionary, JSON,
  and human-readable views covering capabilities, adapters, providers,
  transforms, fallbacks, backend/precision, graph evidence, limitations, and
  certification state; strict mode rejects unproven semantics before launch.

This slice is intentionally not complete. `accelerator="tpu"` now reaches the
private single-VM coordinator for reconstructible pretrained HF model sources
and a validated `PackedBinDataset`. Local lifecycle cadence and resume now use
the backend-neutral engine hooks. TPU callback metrics and return state no
longer require stage-log parsing. TPU worker checkpoint/evaluation parity and
transactional optimization application are next stories. Do not mark M8-F0
complete until TPU lifecycle behavior is validated on target hardware.

## Required next sequence

Implement one small commit/story at a time and update `docs/ROADMAP.md` and
this file in every turn:

1. **M8-F0 TPU checkpoint parity:** carry safe resume and save/evaluation
   cadence through worker coordination.
   The coordinator and worker now carry save cadence and resume paths. Each TPU
   rank atomically persists model, optimizer, scheduler, runtime, host/device RNG,
   and deterministic packed-data progress, and validates the committed topology
   before restoring. The tensor-aware implementation remains in the private
   `_tpu_checkpoint` boundary so `trainlm.checkpoint` contracts stay framework
   independent. TPU evaluation datasets now use the same validated packed-data
   boundary and scheduled engine evaluation, with structured callback metrics.
   Every replica currently evaluates the same stream to preserve exact global
   semantics until backend-level distributed evaluation reductions are added.
   Target-TPU lifecycle validation remains.
   Coordinator contract coverage constructs mismatched evaluation requests by
   replacing serialized optional fields, matching real request round trips.
2. **M9-F3 rematerialized chunks:** an explicit non-reentrant per-chunk
   checkpointing policy now preserves reference gradients. Compare chunk sizes
   2,048/4,096/8,192 and collect target-XLA HLO/HBM/throughput evidence before
   selecting a default or marking the story complete.
3. **M9-F4 TPU loss providers:** the pure catalog gates Pallas and Tokamax on
   explicit runtime and backward evidence, with portable chunked loss as the
   fallback. Implement and benchmark eligible providers before default use.
4. **M10-F3 XLA Pallas provider:** guarded HF attention/mask registration now
   requires canonical semantic compatibility and registers both interfaces
   under one key. Add the first version-guarded Pallas MHA provider with backward
   and target-HLO evidence while retaining the registered portable fallback.
   A dependency-free MHA bridge now gates construction on exact torch_xla
   versions and explicit backward evidence, uses an injected stable kernel
   adapter, and rejects unsupported dense masks/dropout. Keep M10-F3 open until
   numerical, gradient, and HLO/custom-call checks pass on target TPU hardware.
5. **M10-F4 GQA/MQA without KV repeat:** a logical head-owner mapping and
   guarded provider now preserve compact K/V inputs for 8/8, 8/4, and 8/1 and
   pass head geometry directly to the kernel adapter. Keep the story open until
   target-TPU correctness, gradient, and HBM evidence proves no hidden repeat.
6. **M10-F5 ALiBi/sliding window:** the guarded provider now passes an integer
   window to the compact kernel without materializing a dense mask and requires
   explicit model-supplied ALiBi slopes rather than deriving family semantics.
   Both paths remain gated until target-TPU boundary, gradient, and HBM evidence.
7. **M10-F6 attention autotuning:** a deterministic exact-key tuning cache now
   keys hardware, provider/version, dtype, batch/sequence/head geometry, mask,
   and window and uses stable candidate-ID tie breaking. Populate candidate
   ranges and measurements on v5e before selecting production defaults.
8. **M10-F7 loss/attention benchmark:** a matched-result gate now requires
   v5e-8 geometry, at least 850K global tokens/s, lower HBM, stable compilation,
   zero CPU fallback, eliminated full logits, named providers, and HLO evidence.
   Run synchronized fake- and real-data measurements on target hardware.
9. **M11-F1 reversible QKV packing:** a family-neutral descriptor now creates
   reversible state-dict mappings for separate MHA/GQA/MQA weights and optional
   biases using explicit head geometry and key names. Add partial-layout and
   transactional live-module transforms, then validate output/gradient/update.
10. **M11-F2 reversible gated-MLP packing:** explicit SwiGLU/GeGLU descriptors
    now create reversible gate/up weight and optional-bias mappings, while GELU
    remains an explicit no-transform path. Add transactional live transforms
    and validate output/gradient/update parity before completing the story.
11. **M11-F3 native fusion audit:** a deterministic audit now turns explicit
    HLO fingerprints and copy/transpose/materialization/custom-call/fallback
    observations into native, custom, or blocked decisions for norm, RoPE,
    residual, and MLP. Populate it from target-XLA captures and benchmark every
    custom candidate before implementing another kernel.
12. **M11-F4 decoder rematerialization:** explicit none/block/attention/MLP/
    loss-chunk policies now require pre-FSDP application, and selection uses
    measured gradient parity, graph stability, step slowdown, and peak HBM with
    deterministic ties. Populate target-XLA measurements before defaulting.
13. **M11-F5 XLA optimizer state:** a versioned AdamW policy now locks first/
    second moment precision, clipping, decoupled decay, and gradient reduction.
    Its gate requires update/resume parity, stable graphs, no CPU fallback,
    bounded step regression, and lower HBM. Populate target-XLA evidence.
14. **M11-F6 batch/prefetch tuning:** a deterministic production selector now
    requires matched token/update geometry, real data, stable graphs, zero CPU
    fallback, and HBM/input-idle budgets before ranking throughput. Measure
    MB2/GA32, MB1/GA64, safe alternatives, and prefetch near 16 on target TPU.
15. **M11-F7 final HLO/host closure:** a final gate now combines the 912.6K
    throughput and 47.8% MFU thresholds with graph, fallback, transpose/layout,
    host-sync, full-logits, input-idle, and collective evidence. Run the exact
    reference on v5e and resolve every reported reason before closing M11.
16. **M12-F1 numerical alignment:** a locked semantic comparator now covers
    initialization, residual/norm/position, shifted loss/z-loss, optimizer,
    schedule, and dtype fields and requires deterministic early-update errors
    within tolerance. Populate it from the exact reference and justify or remove
    every difference before certification.
17. **M12-F2 three-run benchmark:** a repeated-run evaluator now requires three
    matched warm-cache v5e-8 results with identical HLO, stable compilation,
    zero fallback, and per-run hard gates, then reports medians, spread, and max
    HBM. Populate it with three synchronized target runs.
18. **M12-F3 real-shard stability:** a structured gate now requires 200 updates
    on revision-pinned diverse shards with evaluation, resume, integrity,
    cursor continuity, finite loss/gradient ranges, stable compilation, zero
    fallback, and canonical export. Populate it from the target stability run.
19. **M12-F4 plain-HF export:** a structured gate now requires a clean
    Transformers-only environment, canonical keys, preserved aliases, no
    missing/unexpected keys, and logits/loss within tolerance. Run it against
    the optimized checkpoint without TrainLM installed.
20. **M12-F5 parity report:** a versioned report now records environment,
    reproduction commands, configuration/metric/profile/HLO artifacts,
    limitations, and all four M12 gate outcomes in stable JSON and Markdown.
    Populate it only with real target evidence before declaring certification.
21. **M13-F1 GPT-2/OPT mapping:** an explicit, version-guarded catalog now
    distinguishes GPT-2 fused QKV and learned absolute positions from OPT
    separate QKV and offset learned positions while mapping both to shared
    attention, QKV-layout, GELU-MLP, and linear causal-loss operations. Run
    output/gradient/update, graph, export, and target-TPU certification before
    advertising either family as optimized.
22. **M13-F2 GPT-NeoX/BLOOM mapping:** explicit adapters now preserve
    GPT-NeoX RoPE plus parallel residual semantics and BLOOM ALiBi plus serial
    residual semantics while sharing guarded MHA, fused-QKV, LayerNorm, GELU,
    and linear causal-loss operations. Complete shared target-TPU correctness,
    graph, export, and performance certification before enabling either path.
23. **M13-F3 Falcon/Phi mapping:** explicit variants now prevent Falcon MQA
    and GQA head ownership from being interchanged and preserve its fused-QKV
    parallel block. Phi separately requires MHA, partial RoPE, separate QKV,
    LayerNorm/GELU, and parallel residual semantics. Complete correctness,
    HBM, graph, export, and target-TPU certification before enabling them.
24. **M13-F4 RoPE gated families:** guarded mappings now distinguish Llama,
    Mistral sliding-window attention, Qwen2 biased QKV, Gemma scaled embeddings
    and GeGLU, and Gemma2 alternating windows plus attention soft-capping.
    Add QK-normalized variants where supported and complete shared correctness,
    graph, export, HBM, and target-TPU certification before advertising them.
25. **M13-F5 cross-family matrix:** a structured evaluator now rejects missing
    or duplicate advertised-family records, mismatched benchmark geometry,
    failed correctness/graph/export evidence, and full-attention results below
    45% architecture-adjusted MFU. Populate it with synchronized 135M target
    runs using family-specific FLOPs/token before completing the matrix.
26. **M13-F6 SPMD FSDP:** a backend-neutral policy now locks data/FSDP mesh
    dimensions, explicit decoder wrap classes, adapter-provided parameter
    partitions, sharded optimizer state, pre-wrap rematerialization, and
    topology-matched distributed checkpoints. Wire it into the XLA runtime and
    complete the 1.3B train/resume/export v5e-8 smoke before enabling it.
27. **M13-F7 1.3B benchmark:** a structured gate now compares measured FSDP
    evidence only against a review-locked target with exact workload, parameter,
    hardware, and mesh geometry. It also requires throughput/MFU parity, stable
    compilation, zero CPU fallback, correctness/resume/export, and collective
    plus HBM artifacts. Lock the comparison and populate it on target hardware.
28. **M14-F1 stable public API:** `TrainLMTrainer.from_pretrained()` and the
    versioned mapping/YAML `from_config()` path now construct explicit HF model
    sources and public training arguments while rejecting internal keys. The
    first deprecated YAML alias warns before removal. Add clean-environment and
    compatibility checks before declaring the release interface stable.
29. **M14-F2 secure packed-bin TPU guide:** the public tutorial now covers
    secret-managed `HF_TOKEN`, immutable model/data revisions, separate
    train/eval manifests, `explain()`, committed-checkpoint resume, private
    artifacts, and the honest current TPU export limitation. Run the guide in a
    clean target environment and implement canonical TPU export before closure.
30. **M14-F3 CI/hardware tiers:** a release gate now requires current passing
    CPU, CUDA, scheduled v5e correctness, and v5e performance/stability evidence
    for the exact release commit, with Tier 3 mandatory. Provision explicit
    hardware workflows and populate real Tier 1-3 artifacts next.
31. **M14-F4 preemption recovery:** TPU checkpoint shards now carry deterministic
    step/micro-step generations, committed destinations cannot be overwritten,
    manifest/shard progress must agree, and recovery discovers only complete,
    safe, committed topologies. Add real compute/staging/persistence process-kill
    tests and target-TPU cursor-continuity evidence before closure.
32. **M14-F5 support manifest:** the versioned machine manifest and development
    release notes now publish dependency ranges, hardware support, execution
    paths, providers, fallbacks, caveats, and deferred TorchTPU status. A pure
    validator prevents `trainer.explain()` from exceeding published backend or
    path support. Refresh the manifest from final certification evidence.
33. **CI contract regressions:** the core dependency assertion now includes the
    directly imported PyYAML runtime, and the FSDP mismatch fixture uses a valid
    16-device data=4/FSDP=4 mesh so mismatch rejection is exercised by the
    evaluator rather than failing during evidence construction.
34. **M14-F1 compatibility contract:** `support/public_api_v1.json` now records
    the versioned package-root symbols, accepted trainer configuration keys,
    and deprecated aliases. The release evaluator fails when an installed
    surface changes without an API-version update. The CI test suite now also
    builds and installs a wheel into an isolated environment, runs outside the
    source checkout, and verifies the documented package-root construction
    surface; M14-F1 is complete.
35. **M14-F4 process-kill recovery:** the private rank-local saver exposes a
    test-only stage hook at shard staging, shard publication, and manifest
    staging/publication boundaries. Spawned child processes are terminated at
    each pre-commit persistence boundary, and recovery continues to select the
    prior committed generation. Target-TPU cursor-continuity evidence remains.
36. **M14-F4 durable boundary:** spawned process termination now also covers
    compute before checkpointing and the instant after atomic manifest publish.
    Recovery rejects the former attempt and accepts the latter, establishing
    the manifest rename as the host-side durability boundary.
37. **M14-F2 immutable TPU model intake:** remote Hugging Face pretrained models
    now require lowercase 40-character commit revisions before the coordinator
    stages data or launches workers. Existing local model directories remain a
    supported offline snapshot path. Target tutorial smoke remains outstanding.
38. **M11-F1 live QKV transform:** an explicit adapter-selected wrapper path can
    now be replaced before optimizer construction by one packed linear QKV
    projection. The transform validates MHA/GQA/MQA geometry, bias layout,
    dtype/device agreement, returns compact query/key/value views, and restores
    the original wrapper on rollback. Partial source layouts and target-XLA
    update evidence remain outstanding.
39. **M11-F2 live gated-MLP transform:** an adapter-selected gate/up wrapper can
    now be replaced before optimizer construction by one packed linear call.
    Activation semantics remain explicit through a supplied callable rather
    than a family-name guess, GELU remains untransformed, and transactional
    rollback restores the original wrapper. Partial layouts and target-XLA
    update evidence remain outstanding.
40. **M11-F1 partial QKV layouts:** canonical sources with separate query and
    combined key/value parameters now have an explicit reversible state-dict
    descriptor. It preserves the source key boundary, compact KV geometry,
    optional paired biases, and canonical export instead of pretending the
    model exposes three separate projections. Already-packed sources now use an
    explicit validated no-transform descriptor with derived compact output
    geometry, completing separate/partial/packed layout representation. Target-
    XLA output, gradient, and update evidence remains outstanding.
41. **M11-F3 HLO text capture:** fusion observations can now be constructed
    reproducibly from normalized textual HLO with a SHA-256 fingerprint and
    exact copy, transpose, and custom-call opcode counts. Component-specific
    materializations require explicit regex patterns, and native fusion remains
    caller-supplied evidence because opcode text cannot prove semantics. Real
    target-XLA captures and matched provider benchmarks remain outstanding.
42. **M11-F4 live rematerialization:** an explicit policy can now install
    non-reentrant activation checkpointing on adapter-selected block, attention,
    or MLP paths through the transactional transform registry. Forward methods
    are patched in place so module structure, parameter aliases, and canonical
    state keys remain stable, and rollback restores the original callables.
    Loss chunks remain under the loss provider; target-XLA HBM and step-time
    measurements remain outstanding.
43. **M11-F5 worker policy integration:** the TPU worker now derives AdamW
    moment dtypes, non-fused execution, decoupled decay, and trainer clipping
    from one validated `XLAAdamWPolicy`. Materialization preserves the user's
    learning rate, betas, epsilon, and decay while preventing worker defaults
    from drifting away from the evidence contract. Target-XLA update/resume,
    graph, HBM, and step-time measurements remain outstanding.
44. **M11-F6 input-policy integration:** `BatchPrefetchGeometry` now validates
    both host and device prefetch depths, host-to-device transfer threads, and
    batches per execution, and materializes the exact `ParallelLoader` keyword
    arguments. The TPU worker builds the current production candidate through
    that contract so future measured selections have one application boundary.
    Real-data target-TPU geometry measurements remain outstanding.
45. **M11-F7 closure artifacts:** the final parity gate can now load a versioned
    benchmark result, graph diagnostics, and textual HLO from separate files,
    verify strict counter types, normalize and hash the HLO, and reject evidence
    whose declared fingerprint does not match the captured graph. Genuine v5e
    evidence satisfying every performance and graph gate remains outstanding.
46. **M12-F1 numerical artifacts:** reference and candidate semantic manifests
    can now be evaluated together with a strict versioned update-evidence file.
    The loader rejects unknown schema fields, malformed error sequences, and
    invalid justifications before applying the locked semantic and early-update
    tolerances. Exact deterministic target-run artifacts remain outstanding.
47. **M12-F2 repeated-run manifest:** repeated parity evidence can now be loaded
    from a strict versioned manifest containing exactly three distinct result
    artifacts and the complete hard/preferred threshold set. Result paths must
    remain relative to and confined beneath the manifest directory, preventing
    accidental evidence substitution. Three real synchronized v5e runs remain
    outstanding.
48. **M11-F1 portable numerical parity:** the live packed-QKV regression uses
    explicit float32 relative and absolute tolerances for output and gradient
    comparisons. This accounts for valid accumulation-order differences between
    three separate matrix multiplications and one packed matrix multiplication
    while retaining strict semantic parity coverage across CI CPU backends.
49. **M12-F3 stability artifacts:** real-shard lifecycle evidence now has a
    strict versioned JSON loader that requires the complete 200-update gate
    payload and explicit update/shard thresholds. Unknown, missing, malformed,
    and incorrectly typed evidence is rejected before evaluation. A real
    revision-pinned target run remains outstanding.
50. **M12-F4 plain-HF artifacts:** clean-process interoperability evidence now
    has a strict versioned JSON loader. It requires every environment, canonical
    layout, alias, reload-key, and numerical field; converts JSON key arrays to
    the immutable evaluation contract; and applies the recorded tolerance. A
    clean Transformers-only target execution remains outstanding.
51. **M12-F5 parity report bundle:** versioned report JSON can now be loaded back
    into the immutable report contract while rejecting unknown fields and
    malformed array shapes. Every configuration, metric, profile, and HLO path
    must be relative, confined beneath the report directory, and present, so a
    reviewer cannot unknowingly consume an incomplete evidence bundle.
52. **M13-F5 cross-family bundle:** the 135M family matrix now has a strict
    versioned manifest loader. It reconstructs complete typed family records,
    rejects schema drift, and confines every per-family evidence reference
    beneath the manifest directory before applying geometry, correctness,
    export, graph, and architecture-adjusted MFU gates.
53. **M13-F7 FSDP scaling bundle:** the review-locked 1.3B target and measured
    scaling evidence can now be loaded from one strict versioned manifest.
    Target/evidence schema drift is rejected, and collective plus HBM profile
    paths must be relative, confined beneath the manifest, and present before
    the performance, graph, correctness, resume, and export gates run.
54. **M14-F3 certification bundle:** Tier 0-3 evidence can now be loaded from a
    strict versioned release manifest that records the exact release commit,
    evaluation timestamp, and freshness window. Tier artifacts must be relative,
    confined beneath the manifest, and present before current-evidence
    certification is evaluated.
55. **M14-F5 strict support schema:** support-manifest loading now enforces the
    exact top-level, hardware, execution-path, provider, and TorchTPU field sets.
    Malformed JSON, incorrect arrays/objects, unknown claim fields, and boolean
    schema versions fail closed before public explanations are compared with
    published support levels.
56. **M13-F6 FSDP policy manifest:** the complete data/FSDP mesh, wrap classes,
    parameter partition rules, optimizer-state setting, and checkpoint contract
    now serialize and reconstruct through a strict versioned manifest. Unknown,
    missing, malformed, and incorrectly shaped nested settings fail before a
    runtime can consume the policy.
57. **M13-F6 XLA policy application:** `XlaRuntime` now consumes the validated
    policy directly, checks topology/rematerialization/module and parameter
    evidence before mutation, and annotates model parameters plus initialized
    optimizer tensors with the declared FSDP partitions. The Hugging Face model
    tree is not replaced. A real 1.3B v5e-8 train/resume/export smoke remains
    the acceptance gate.
58. **M13-F6 topology-safe checkpoint resume:** TPU checkpoint schema v2 binds
    each committed generation and rank payload to its logical mesh axes.
    Restore rejects same-world-size data/FSDP topology changes and discovery
    ignores manifests whose mesh product disagrees with their world size.
    Existing schema-v1 checkpoints remain readable; their mesh is recovered
    from rank runtime state and validated before any training state is mutated.
59. **M8-F0 distributed TPU evaluation:** validation data is now partitioned by
    worker rank instead of replicated. The causal task accumulates local loss
    numerators and token/sequence denominators on device, performs one backend
    sum reduction, and only then materializes global loss and perplexity.
60. **M3 resumable reference split:** the 28-train/2-validation shard fixture
    now proves disjoint exhaustive rank ownership and exact cursor restoration
    at every rank-local interruption point, including terminal token counts.
    M3 is complete; target TPU lifecycle checks remain tracked under M7/M8.
61. **M4-F4 WSD validation:** exact transition fixtures cover warmup, stable,
    linear decay, minimum learning rate, and post-horizon clamping. Restoring
    optimizer and scheduler snapshots mid-run reproduces every subsequent
    learning rate and scheduler state, closing M4-F4.
62. **M4-F5 streaming evaluation validation:** variable-mask causal batches
    match the token-weighted reference without retaining predictions or
    changing trainer progress, model parameters, training mode, or gradients.
    This closes M4-F5; callback synchronization and family overfit remain.
63. **M4-F6 synchronization-safe callbacks:** tensor scalar extraction is now
    instrumented across sparse logging. Non-logging steps produce no `.item()`
    calls or live callback tensors; exactly one loss is materialized at the
    requested boundary and callbacks receive host-only metric snapshots.
64. **M4-F7 family resume/export matrix:** every tiny dense-AR Transformers
    fixture now overfits for 30 updates, resumes model/optimizer/scheduler and
    trainer progress from update 15 to identical final parameters, and reloads
    the canonical export through `AutoModelForCausalLM` without family branches.
65. **M8-F1 explicit opaque-semantic evidence:** structural inspection now
    accepts narrowly scoped adapter evidence for residual and custom
    normalization semantics that module traversal cannot prove. Claims must be
    known, inferred, or unsupported and cite evidence; unknown or uncited
    overrides fail closed. Evidence also names the exact model class, config
    class, and source provider, and mismatched boundaries are rejected before
    inspection. This closes M8-F1 without family-name guessing.
66. **M8-F0 public CLI parity:** the optional `trainlm train` entry point accepts
    the versioned public YAML plus validated local train/evaluation manifest
    directories and an optional resume checkpoint. It constructs only public
    dataset/trainer objects, delegates to `TrainLMTrainer.train()`, and prints a
    structured result without exposing TPU worker or PJRT arguments. Dataset
    readers derive sequence geometry and seed from the same validated YAML
    training arguments passed to the trainer.
67. **M11-F1 live partial-QKV transform:** adapter-selected wrappers exposing a
    separate query linear and a combined key/value linear can now be replaced
    transactionally by one packed projection before optimizer construction.
    Geometry, bias, dtype, and device are validated; compact query/key/value
    views preserve output and gradient semantics, and rollback restores the
    original wrapper. Target-XLA update evidence remains outstanding.
68. **M11-F1 QKV trainability preservation:** live separate and partial QKV
    transforms preserve uniformly frozen weights and biases. Mixed
    `requires_grad` states fail before replacement because one packed parameter
    cannot represent independent source trainability without changing user
    fine-tuning semantics.
69. **M11-F1 QKV alias safety:** live QKV transforms inspect the complete model
    alias graph before replacement and reject internal or external aliases that
    concatenating source projections into one parameter cannot preserve. The
    failure occurs before mutation, leaving tied source parameters intact.
70. **M11-F2 gated-MLP semantic guards:** live gate/up packing now preserves
    uniformly frozen weights and biases and rejects mixed trainability. It also
    checks the full model alias graph before replacement, so tied gate/up or
    externally aliased parameters fail without mutating the source wrapper.
71. **M11-F2 already-packed gated MLP:** canonical SwiGLU/GeGLU sources that
    already expose one gate/up parameter now use an explicit validated
    no-transform descriptor with derived weight and optional-bias geometry.
    This preserves their source layout without inventing separate model keys.
72. **M11-F2 portable update parity:** biased and bias-free live gate/up
    transforms now compare one post-transform SGD update against the separate
    reference, including split weight/bias state and subsequent forward output.
    Target-XLA update parity remains the hardware acceptance gate.
73. **M11-F1 portable update parity:** separate and partial live QKV transforms
    now compare one SGD update against their source layouts for biased and
    bias-free projections. Split packed parameters and subsequent compact Q/K/V
    outputs match the untransformed reference within explicit float32 bounds;
    target-XLA parity remains outstanding.
74. **M8-F5 unambiguous alias manifests:** state-dict layout converters reject
    overlapping alias groups and aliases involving canonical or transformed
    mapping keys. This prevents order-dependent restoration and makes every
    declared tie either preserved unchanged or rejected before conversion.
75. **M8-F5 strict layout manifests:** layout and mapping loaders now reject
    unknown fields, non-mapping entries, and malformed key, shape, mapping, or
    alias-group containers. String values are never silently expanded into
    character tuples, and schema errors fail before tensor conversion.
76. **M8-F5 complete schema declarations:** state-dict layout manifests must
    explicitly provide the schema version, mappings, and alias groups, including
    empty collections. Boolean schema versions are rejected rather than being
    accepted through Python's integer equality rules.
77. **M8-F3 planner boundary validation:** the pure planner now rejects invalid
    capability objects, empty backend or precision names, unknown policies,
    malformed operation requests, and invalid adapter resolutions before
    hashing or provider selection.
78. **M8-F3 explicit fallback selection:** explicitly requesting an eligible
    provider marked for automatic fallback now produces a normal selected
    decision. Fallback status is reserved for automatic choice or substitution
    after an unavailable explicit request, so decision invariants remain valid.
79. **M8-F3 required-policy fallback guard:** an unqualified required operation
    now considers only non-fallback providers. If only portable fallbacks are
    eligible, planning blocks with explicit fallback-only evidence; explicitly
    requesting that same provider remains a valid, intentional selection.
80. **M8-F3 typed fallback declarations:** provider specifications reject
    integer, string, null, and other non-boolean fallback flags. Planner policy
    behavior can therefore depend on the flag without Python truthiness
    silently reclassifying a provider.
81. **M8-F3 provider-bound plan identity:** plan fingerprints now include the
    complete provider catalog in stable provider-ID order, including eligibility
    fields and declared transforms. Identical requests evaluated against
    different catalogs therefore cannot share an audit ID.
82. **M8-F3 immutable provider transforms:** provider specifications accept
    transformation declarations only as tuples. Mutable lists cannot be retained
    inside a frozen provider and changed after registration, preserving stable
    selection behavior and provider-bound plan fingerprints.
83. **M8-F3 component-bound provider transforms:** every provider transformation
    must name the provider's component as well as its provider ID. A provider
    selected for one inspected capability therefore cannot smuggle a mutation
    for an unrelated model component into an otherwise valid execution plan.
84. **M8-F3 catalog-wide transform identity:** provider construction rejects
    repeated transformation IDs, and planner registration rejects IDs already
    owned by another provider. Multi-operation selection therefore cannot fail
    late while constructing an execution plan with ambiguous transformations.
85. **M8-F4 strict execution-plan loading:** serialized plans require the full
    versioned schema and reject unknown keys, malformed collection containers,
    non-mapping decision/transform entries, and boolean schema versions. A
    checkpoint cannot silently default omitted plan evidence or accept typos.
86. **M8-F4 strict nested plan evidence:** serialized provider decisions and
    transformations require their complete exact field sets. Transformation
    layout-change flags must be booleans, preventing truthy strings or integers
    from altering checkpoint/export behavior after plan restoration.
87. **M8-F0 full-suite fixture isolation:** packed-reader fixtures size their
    synthetic vocabulary from the generated disjoint shard ranges. Sparse
    callback synchronization coverage uses a materialized batch iterable so
    PyTorch DataLoader's internal seed `.item()` is not misattributed to the
    trainer's loss-materialization boundary.
88. **M8-F4 execution-plan status invariants:** ready plans require at least one
    selected or fallback provider, while blocked plans cannot carry transforms.
    A selected decision with an explicit request must select that exact provider,
    preventing contradictory restored evidence from reaching execution.
89. **M8-F4 valid transform fixtures:** transactional transform, QKV, gated-MLP,
    and rematerialization tests now declare the selected fixture provider in
    every ready execution plan. The tests exercise the production invariant
    instead of relying on structurally invalid decision-free ready plans.
90. **M8-F4 decision-authorized transforms:** an execution plan accepts a
    transform only when a selected or fallback decision names the same component
    and provider. Tampered checkpoint plans cannot attach unrelated transforms
    to an otherwise legitimate provider decision.
91. **M8-F4 invariant error precedence:** blocked and no-op transformation
    violations are classified before ready-plan provider authorization. This
    preserves actionable lifecycle errors while retaining the authorization
    guard for executable plans.
92. **M8-F4 disjoint transform targets:** execution plans reject target paths
    claimed by more than one transform. Transaction capture and rollback cannot
    become order-dependent through overlapping mutations of the same module.
93. **M8-F4 serialized overlap fixture:** the tampered-plan regression extends
    the serialized transformation tuple immutably. It no longer fails in test
    setup by assuming `dataclasses.asdict()` changes tuples into lists.
94. **M8-F0 public dry run:** `trainlm train --dry-run` constructs the same
    versioned configuration, validates local packed datasets, and prints the
    trainer's dictionary explanation without calling `train()`. This is the
    final allocation-free command to run before the owner starts a TPU smoke.
95. **M8-F0 unambiguous dry-run resume boundary:** dry-run and
    `--resume-from-checkpoint` are mutually exclusive. The CLI never ignores a
    resume request or claims to validate checkpoint contents without executing
    the lifecycle path that owns full resume validation.
96. **M8-F0 public TPU validation notebook:** the target-hardware notebook now
    uses only `TrainLMTrainer`, `TrainLMTrainingArguments`, `PackedBinDataset`,
    and the installed public CLI. Its false-by-default execution gate separates
    host validation from paid TPU allocation, then covers the two-update smoke,
    scheduled evaluation and committed checkpoints, exact resume, and public
    evidence fields. Worker scripts, PJRT/rank setup, private coordinator types,
    and stage-log parsing are absent from the user workflow.
97. **M8-F0 runtime-owned TPU topology:** world size is discovered after
    Torch/XLA launches the worker group and is never a public input or a hidden
    fixed v5e-8 assumption. The collective probe derives its expected result
    from the discovered topology, while summaries may report the observed world
    size as evidence. The HF-like call remains `trainer.train()`.
98. **M8-F0 end-user notebook simplification:** the TPU notebook is no longer a
    multi-phase orchestration checklist. Its primary path contains only public
    packed datasets, familiar training arguments, trainer construction, and
    `trainer.train()`. Evaluation and checkpoint cadence happen within that
    call; explanation and resume are concise optional examples. Operational
    artifacts are mentioned only as outputs to archive for maintainers.
99. **M8-F0 ranged raw-bin Hub intake:** `PackedBinDataset.from_hub()` accepts a
    dataset repo plus an end-exclusive `(start, stop)` shard range, resolves a
    mutable selector once to its immutable Hub commit, downloads numbered
    LaughLM legacy `.bin` files through the standard cache, and streams each
    payload to derive checksum, token count, and bounds. Generated descriptors
    feed the existing deterministic reader, train/eval partitions, and private
    coordinator staging path; users do not create or publish sidecar manifests.
100. **M8-F0 Hub intake performance boundary:** Hub resolution and full token/
     checksum validation happen once during dataset construction, before TPU
     launch. Training reads the resulting local Hugging Face cache files through
     lazy memory maps, so it performs no network I/O or full-file rescans per
     batch. Token-bound scanning now uses vectorized Torch chunks instead of a
     Python loop. The TPU prefetch geometry also consumes the canonical worker
     `micro_batch_per_device` argument, fixing the end-to-end ranged-data path.
101. **M8-F0 executable ranged-data notebook:** the public TPU notebook now
     names an explicit local Hugging Face cache directory, imports every symbol
     it uses, passes that cache to both train and evaluation range downloads,
     and explains the lifecycle accurately: download and validation occur
     before launch, while `trainer.train()` consumes lazy memory-mapped local
     files without network access in the training loop.
102. **M8-F0 Kaggle parent-process isolation:** constructing the public trainer
     exposed Kaggle's preinstalled TensorFlow conflict through Torch/XLA and the
     notebook kernel terminated before `train()`. The installation section now
     removes TensorFlow and incompatible vision/audio wheels before installing
     the pinned TPU profile, requires exactly one session restart, and verifies
     TensorFlow is no longer importable before importing TrainLM. TPU worker
     launch remains private; this is environment preparation, not user-managed
     PJRT or topology configuration.
103. **M8-F0 interrupt-safe TPU ownership:** the private coordinator launches
     each stage in a new OS process group and streams output directly to its
     stage log. On worker failure, notebook interruption, or another wait error,
     it terminates the launcher and all spawned ranks, escalating to `SIGKILL`
     after a bounded grace period. Control returns to the public caller without
     orphaned ranks retaining TPU resources, so a corrected cell can run again.
104. **M8-F0 model revision resolution:** the validation notebook previously
     displayed a syntactically valid but nonexistent example commit, which
     reached worker preflight and produced a misleading missing-`model_type`
     error. The public TPU facade now accepts `main` or another user-friendly
     model selector, resolves it on the host to the actual immutable Hub commit,
     and sends only that commit to workers. The notebook uses `main` rather than
     encouraging users to paste a fake SHA.
105. **M8-F0 offline immutable model intake:** explicit commit SHAs pass through
     without a Hub metadata request, preserving offline/cache-only launches and
     deterministic coordinator tests. Mutable selectors such as `main` are
     resolved once before launch; users must supply a real commit when choosing
     the explicit immutable form.
106. **M8-F0 HBM-safe validation model:** the public TPU notebook now starts
     from the locked 135M Llama-shaped configuration (1024 hidden, 8 layers,
     8 heads, 2816 intermediate) instead of preflighting the 3.8B Phi-3.5
     checkpoint independently on every data-parallel replica. This validates
     the lifecycle within v5e-8 HBM before users attempt larger-model sharding.
107. **M8-F0 config-model worker reconstruction:** the coordinator now
     serializes the complete validated `ModelSourceConfig` for private workers
     instead of restricting launch to pretrained IDs. Config-initialized HF
     models therefore follow the same probe, preflight, train, evaluation, and
     checkpoint lifecycle without exposing worker arguments to notebook users.
108. **Consolidated continuation handoff:** `docs/HANDOFF.md` now records the
     public/data/runtime/optimization implementation state, uses distinct done,
     partial/evidence-pending, and planned marks, and carries the proposed
     parallelism, architecture-lab, telemetry, and closed-loop tuning backlog.
109. **M8-F0 bounded-memory TPU diagnostics:** model preflight now compiles a
     maximum 16-token forward rather than duplicating the full training-shape
     compile across all eight ranks. Probe/preflight have fifteen/thirty-minute
     inactivity limits with process-group reclamation, and the notebook starts its
     actual lifecycle at sequence length 128 until the chunked-loss path removes
     the full-logit training graph.
110. **M8-F0 notebook progress visibility:** setup/data/trainer/train/explain
     cells now emit numbered progress markers. During `trainer.train()`, the
     coordinator reports each private stage start/completion and a low-volume
     ten-second heartbeat containing the latest worker event, so a long XLA
     compile is visibly active without requiring users to open stage logs.
     Direct stage execution also creates its log directory before launch, and
     interrupt coverage exercises the heartbeat wait path.
111. **M8-F0 progress-aware diagnostic deadlines:** the diagnostic deadline is
     now based on time since the last new worker event rather than total stage
     duration. Cold TPU startup and compilation can therefore exceed the old
     five-minute probe wall-clock limit when making progress, while a silent
     probe is still reclaimed after fifteen inactive minutes.
   M9-F5 software conformance now covers representative hidden-output forms and
   tied/untied, biased/bias-free heads; M9-F3/F4 still require TPU measurements.

## Definition of done for the public surface

The public API is ready for optimization benchmarking only when all of these
are true:

- a user never writes `subprocess`, PJRT environment variables, rank probes,
  or worker command arguments;
- the same concise code works for CPU smoke and TPU execution;
- HF model output/labels/tokenizer conventions are preserved;
- `.bin` data is validated before workers start and partitioned without
  duplicate or missing tokens;
- `trainer.explain()` identifies selected kernels and every fallback;
- checkpoint/resume and export behavior are documented and tested;
- performance is reported as global supervised tokens/s with warm-up excluded
  and matched against LaughLM geometry.

## Commit convention for this branch

Use one feature/story per commit. Suggested next messages:

- `feat(api): hide TPU worker orchestration behind TrainLMTrainer`
- `feat(data): expose validated packed-bin dataset source`
- `feat(training): complete HF-like lifecycle and resume facade`
- `feat(optimization): add model capability inspector and plan report`
- `feat(optimization): add reversible kernel transform registry`

### 112. Scheduled TPU smoke evaluation is explicitly bounded

The Kaggle lifecycle smoke reached optimizer step 2 and then appeared idle because
`eval_steps=2` synchronously evaluated an entire roughly 250-million-token shard
before checkpointing. The public training arguments now expose
`max_eval_batches`; it is carried through the private coordinator/worker boundary
and enforced by the streaming trainer evaluation iterator. The SmolLM2 example
and validation notebook set the limit to one batch, while the default remains
`None` so production evaluation retains full-dataset semantics. Worker lifecycle
messages now distinguish evaluation start/completion and checkpoint
start/completion, making scheduled work after an optimizer step observable.

The trainer reads this optional policy compatibly: lightweight and legacy
configuration objects that do not define an `evaluation` section retain the
previous full-dataloader behavior. This matters for embedders and existing unit
fixtures as well as older serialized configurations.

The owner-run full-validation recommendation is deliberately not represented as
an unchecked M17 feature item: M17 is already complete, and roadmap status is
derived mechanically from the checklist within each milestone section.

### 113. Lifecycle smoke no longer compiles a second evaluation graph

The bounded one-batch evaluation still spent several minutes after
`evaluation_start`: on XLA the first eval-mode forward is a distinct compilation,
so limiting data volume does not bound first-graph compilation latency. The
public evaluation feature and `max_eval_batches` remain supported, but the
compile-small SmolLM2 notebook now validates train/checkpoint/resume without an
evaluation dataset. Evaluation is an explicit subsequent validation run.

A separate probe log showed `RAW: Dumping core` and `exit() hanging` while the
launcher nevertheless returned zero. The coordinator now treats those PJRT fatal
shutdown markers as failure and tells the user to restart the notebook session;
it no longer reports such a probe as completed or continues into model loading.

The fatal-marker regression belongs alongside—but not inside—the heartbeat
assertions; keeping those test scopes separate prevents the failure-path test
from referencing the heartbeat test's captured output.

### 114. Restore the proven 135M reference geometry for the first TPU run

The initial public-facade notebook again uses the shallow eight-layer Llama
reference that previously completed DP8 training: hidden size 1024, intermediate
size 2816, eight attention heads, vocabulary 32064, sequence length 2048,
per-replica batch two, and accumulation 32. The first run performs two training
updates only. Evaluation and checkpointing are separate lifecycle validations,
so their distinct compilations and synchronous I/O cannot be mistaken for a
training stall. The public call remains only `trainer.train()`; topology and
worker orchestration stay private.

### 115. Use one PJRT launch for the public training lifecycle

The public `trainer.train()` path no longer starts disposable probe and model
preflight process groups before starting the real training group. The training
worker already performs the collective probe before model construction, so the
extra launches duplicated initialization and, on Kaggle, failed during PJRT
shutdown after ranks had reported success. A training request now owns one PJRT
launch, one collective probe, one model construction, and one worker teardown.
Standalone probe and model-preflight modes remain private diagnostic tools, but
they are not prerequisites for normal training.

### 116. Fail closed on stale TPU notebook state

Immediately before allocating TPU workers, the validation notebook now compares
the constructed trainer against the exact 135M baseline geometry and rejects any
stale 128-token/SmolLM2 trainer retained by the notebook kernel. It also rejects
an evaluation dataset, evaluation cadence, or checkpoint cadence in the first
100-update baseline. The verified configuration is printed before the single
public `trainer.train()` call so the owner can confirm what will execute.

### 117. Measure the public baseline with the historical window

The notebook now runs the same 100-update reference geometry as the recorded
319K baseline, excludes the first five updates from timing, logs every ten
updates, and exposes the existing WSD scheduler through the public training
arguments. The earlier two-update result included compilation because it had no
excluded warmup window; its 29.9K tok/s value was a lifecycle measurement, not a
regression against the historical steady-state result. Evaluation and checkpoint
cadence remain absent from this matched performance run.

### 118. Owner-run public baseline reproduces the generic TPU result

The owner completed the guarded public-facade run on a Kaggle v5e-8 with
PyTorch/XLA 2.9.0. The single `trainer.train()` lifecycle finalized all 100
updates (3,200 rank-local microsteps), consumed 104,806,400 global supervised
tokens, and reported 315,512 global supervised tokens/s over the post-warmup
window. The corresponding scheduled-token rate was 315,667 tokens/s and the
slowest-rank measured window was 315.57 seconds.

This is a successful reproduction of the historical generic baseline, not a
LaughLM optimization result: it is 1.19% below the previously recorded 319,302
supervised tokens/s, a small run-to-run difference at otherwise matched
geometry. The loss moved from 10.47 at update 10 to 7.03 at update 100, and the
worker finalized normally with eight replicas. The WSD learning-rate values are
also expected to differ from the older manual transcript because the public run
uses a base learning rate of 3e-4 whereas that transcript used 2e-4.

The summary correctly remains `performance_certified: false`. It identifies the
remaining gap directly: training still uses `trainlm_xla_sdpa`,
`full_logits_causal_ce_z_loss`, host-unrolled gradient accumulation, and no
compiled training step. The ~850K and 912.6K LaughLM gates therefore require
wiring and validating the optimized attention/loss/update path; changing the
notebook timeout, scheduler, or lifecycle orchestration cannot produce those
gains.

### 119. First throughput candidate reduces host-unrolled accumulation

The next owner-run notebook experiment keeps the proven 1,048,576 scheduled
input tokens per optimizer update but changes the rank-local geometry from
microbatch 2 / accumulation 32 to microbatch 4 / accumulation 16. This halves
the number of Python-driven forward/backward microsteps and XLA graph fragments
per update without changing sequence length, replica count, or global batch.
The output directory is distinct from the 315.5K baseline so evidence cannot be
overwritten.

This is a measured experiment, not a claimed 500K result. The current worker
still materializes full logits, so MB4 raises peak per-replica logits storage and
may fail the HBM guard or show no speedup. The notebook therefore validates the
exact MB4/GA16 geometry immediately before launch and leaves evaluation and
checkpointing disabled. If it succeeds, its summary and XLA metrics decide
whether MB4/GA16 becomes the new base for chunked-loss and compiled-update work;
if it fails for memory, the established MB2/GA32 result remains the fallback.

### Public TPU logging and metadata repair

The public TPU facade now accepts `logging_verbosity="quiet"`, `"normal"`, or
`"verbose"`. Verbose coordinator heartbeats parse worker JSON events into compact
human-readable progress lines while the raw per-stage logs and structured JSON
artifacts remain available for debugging and benchmark analysis. The validation
notebook no longer renders the entire nested Python result after training; it
prints only the lifecycle and throughput fields needed at the notebook surface.

TPU metric snapshots now include global token count, global batch size, and world
size. The public `TrainerState` restores global batch size from the worker
summary, and the worker summary records its rank-specific cache directory and
writer rank. This metadata repair does not change device execution or throughput;
the next target run must verify that only rank zero owns shared artifacts and that
the persistent cache is reused as expected.

The TPU request also exposes an explicit `chunked_linear` loss implementation and
`logits_chunk_size`. When selected, the worker constructs a
`LinearCausalLMTrainingView` from inspected Hugging Face capabilities and an
explicit public `base_model` provider, then projects hidden states through the
output embedding in bounded chunks with per-chunk rematerialization. The existing
full-logit loss remains the default until numerical, gradient, HBM, and target-XLA
graph evidence is collected. The public option is currently guarded to TPU
execution; CPU/CUDA users retain the lower-level explicit training-view escape
hatch rather than silently falling back when they request it.

Verbose TPU heartbeats are event-oriented: unchanged worker lines are no longer
reprinted every poll, while a sparse liveness message remains available during
long quiet stages. Training-start events and summaries identify the current
topology explicitly as data parallel replicas and model-parallel degree one.

The worker now emits one rank-zero console metrics table modeled on the
frontier-style logger used by LaughLM. Each row includes step, progress, loss,
perplexity, gradient norm, learning rate, global tokens/sec, MFU estimates,
tokens seen, remaining tokens, ETA, and elapsed time. The same derived numeric
fields are written to `metrics.jsonl`; the coordinator forwards each changed
metric event once, while raw per-stage logs remain available for debugging.

The owner-run MB8/GA8 `chunked_linear` candidate exhausted v5e-8 HBM while
requesting another 88 MB with only 84.42 MB free. The throughput validation
notebook therefore defaults to MB4/GA16 with 1024-token chunks, the largest
known viable geometry on that TPU, while preserving 1,048,576 scheduled tokens
per update. The packed all-supervised path omits a per-microstep denominator
synchronization in the chunked loss.
The trainer also avoids a duplicate XLA `mark_step` boundary after each
optimizer update. These changes require a fresh v5e-8 run before any
throughput claim; they are not a local TPU certification.

The chunked output projection keeps the XLA tied vocabulary matrix in BF16;
only CPU/CUDA reference execution promotes it to FP32. This avoids the
additional head-sized HBM allocation that caused the MB4/GA16 launch to fail
with 3.15 MB free while requesting another 16 MB.

The validation notebook now checks the constructed trainer for the HBM-safe
MB4/GA16, BF16, 1024-token chunk, and verbose-console settings. It also checks
that the imported chunked-loss implementation contains the XLA model-dtype
projection before the single public `trainer.train()` call, so a stale Kaggle
editable install fails before allocating TPU workers.

The TPU `ParallelLoader` now groups up to eight microbatches per execution,
reducing per-batch `mark_step` overhead while keeping the graph bounded. The
selected value is recorded in the worker summary geometry for auditability.

The TPU launcher initializes each rank's persistent computation cache before its
collective probe. The worker runtime now receives an explicit
`cache_already_initialized` flag, reports the configured cache path without
calling `initialize_cache` a second time, and retains the normal runtime-owned
cache behavior for other callers. This fixes the v5e startup failure
`Computation cache has already been initialized` without changing cache policy.
