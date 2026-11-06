# TrainLM implementation handoff

## Current roadmap state

- **Direction:** native TorchTPU on Cloud TPU. Use v5e-1 for single-device
  feasibility and v5e-8 for collectives, DP8, sharding, and scaling. The
  explicit `accelerator="native_tpu"` path is experimental, not a production
  support claim.
- **Current gate:** T1 remains open. Owner-provided evidence confirms a
  standalone eager allocation/forward/backward/AdamW smoke on v5e-1. Package
  compatibility, TrainLM training on TPU, explicit compile, repeated lifecycle,
  and v5e-8 collective results remain unverified. CPU-verifiable implementation
  may continue behind explicit native selection while those hardware checks
  wait; do not enable automatic TPU routing or treat a core dependency change
  as native TorchTPU certification.
- **Core dependency follow-up:** PR #83 widens the Torch range to `>=2.9,<2.15`,
  raises the Transformers floor to `5.15` and Hub floor to `1.5`, aligns the
  Python 3.13 current profile to Torch 2.14.1, and adds constrained
  minimum/current CPU CI. The
  first core-current job passed. The first minimum run failed under Transformers
  5.0 (887 passed, six failed, 41 subtests); commit `bbf1b2e` raises the minimum
  to 5.15.0. Its retry exposed that Transformers 5.15 requires Hub 1.5.0;
  commit `cfc05ed` aligns the project floor and minimum lock. CI run
  `41752008510` passed 893 tests and 41 subtests in each profile. The old 2.14.1
  mismatch and unrelated parsed IPython-to-Jedi warning are handled by the
  branch. An owner-runtime report refresh is also pending. This does not
  establish a native TPU install profile or end-to-end training support.
- **Raw-text API:** `TrainLMPretrainingConfig` plus
  `TrainLMTrainer.from_scratch()` prepares pinned HF text, tokenizer, packed
  data, and a randomly initialized Transformers causal LM, then returns the
  ordinary trainer lifecycle. GPT-2 remains the default; `model_type` and
  `model_config_overrides` select another registered architecture. TrainLM
  derives the vocabulary and special-token IDs from its trained tokenizer.
  The notebook exposes common settings as touch-friendly Colab form controls
  and keeps ordinary Python assignments for Kaggle.
  The Colab/Kaggle notebook uses this public API and its downloadable schema v6
  report captures requested and effective training settings, optimizer and
  scheduler details, model/tokenizer provenance, runtime versions, selected
  backend, bounded TPU properties, and the newest checkpoint path when present.
  With a checkpoint, `resume.code` and the bundled `resume.py` preserve the
  full pretraining configuration and recorded checkpoint. Restore the original
  run directory and match source/package versions; the model export ZIP does
  not contain the optimizer checkpoints or packed data. The example keeps the
  same budget and only retries an already pending extension. Missing run files
  stop it before trainer construction. No checkpoint means no resume script.
  For a one-click notebook recovery, set `RESUME_FROM_REPORT = True` in the
  setup cell with the original `RUN_DIR` and run all cells. Setup pins the
  source revision from the report; the recovery cell restores its settings and
  checks installed package versions and saved run files before training.
  A package mismatch lists expected and installed versions and stops before
  training; use a fresh runtime with the matching accelerator/package profile.
  The focused notebook/recovery/roadmap CPU suite passes 39 tests and 14
  subtests, including the default model config and mismatch diagnostics. No TPU
  recovery is claimed.
  Property-only discovery does not allocate a device and is labeled unverified.
  Missing optional packages and properties are `null`. Native TorchTPU remains
  experimental; CPU/API work adds no TPU training evidence. Earlier continuation
  work added a stable notebook
  run directory, a `RESUME_FROM_CHECKPOINT` setting, and public
  `resume_from_checkpoint="latest"` lookup over completed checkpoints. Colab
  Drive and Kaggle output persistence are documented. Matching pinned data and
  tokenizer settings reuse validated preparation artifacts; changed settings
  require a fresh run directory. Hardware resume remains unverified. PR #79
  rebase-merged this
  configurable Transformers model path at verified `main` SHA
  `ba5f4761c35c23403d35e44be5d564dbe7ffadb3`; GitHub CI passed 891 tests and
  41 subtests. No TPU run was performed.
- PR #80 rebase-merged the model provenance report at verified `main` SHA
  `e9b4a1bde4869d89c323ed457a6b6cebccb0ae40`; CI passed 891 tests and 41
  subtests. It adds report schema v2; no TPU run was performed.
- PR #81 rebase-merged tokenizer and runtime provenance at verified `main` SHA
  `4cfacae39e4e677f442208dfa098bb51009cfd46`; its report schema is v3. No TPU
  run was performed.
- PR #82 rebase-merged effective run context in report schema v4 at verified
  `main` SHA `4107f2f2ab86c2e90d43428ad1a779a25fe86cac`; GitHub pytest passed
  (run `41746021087`). No TPU run was performed.
- PR #28–#44 established the experimental single-device foundation and the
  from-scratch CPU data/train/checkpoint/export workflow. PR #44's CPU reload
  and generation check does not establish TPU generation or convergence.
- PR #45 added agent-facing documentation and package metadata. It rebase-merged
  at `d983408221b92d631aea817bee7539e07de1bbf5`; it adds no runtime behavior
  or TPU evidence.
- PR #47 aligned the current status pages and historical labels. It rebase-merged
  at `e109ad253754303e44eb6162aacb046c415b990b`; it adds no runtime behavior
  or TPU evidence.
- PR #48 rebase-merged the T4 distributed token correctness slice at
  `7ae837d085fbbaab9f6c2d3617ff129b89dd454e`; its full Python 3.11 GitHub CI
  passed.
- PR #49 rebase-merged T5 rank-shard integrity at verified `main` SHA
  `01a71f990fc5578b77be6e2e0a84473486fbe488`; its Python 3.11 GitHub CI runs
  #296 and #297 passed.
- PR #50 rebase-merged public trainer checkpoint preflight at verified `main`
  SHA `f57d6d71339753a6a3cee4082a43dfa2f5113017`; GitHub CI run #299 passed.
- PR #51 rebase-merged optimizer/scheduler identity and parameter-group
  preflight at verified `main` SHA
  `c63e6903767f3a2aab2ff733e6248ff0c4e4bd3b`; CI run #302 passed.
- PR #52 rebase-merged optimizer slot and scheduler payload preflight at
  verified `main` SHA `9cf8d4ba0de0b91cc52908d9d98f984b29873a33`; CI run #304
  passed.
- PR #53 rebase-merged trainer progress metadata preflight at verified `main`
  SHA `8ea810c2e5e54f492cb4e29e0844a74c9935f23e`; CI run #306 passed.
- PR #54 rebase-merged packed-data cursor preflight at verified `main` SHA
  `6ab2b8c9fcde5895c71d5f8450808b7f5756c520`; CI run #308 passed.
- PR #55 rebase-merged transient trainer-field rejection at verified `main`
  SHA `ea9dbadfb86f17cd3713b2abcee808677c38e845`; GitHub CI passed (run ID
  `41578130077`).
- PR #56 rebase-merged custom dataset preflight coverage at verified `main`
  SHA `4aa3614858065c6255c4ecc2d0949c24ee715f70`; GitHub pytest passed in
  2m24s.
- PR #57 rebase-merged custom iterable resume parity at verified `main` SHA
  `e62052ae39d6216bce0dd5aec86fc42d9f10a837`; GitHub pytest passed (run ID
  `41579944075`).
- PR #58 rebase-merged custom iterable identity/cursor preflight at verified
  `main` SHA `e90697f99db29355e135161ea62e66d91bbc6047`; CI passed (run ID
  `41581911584`).
- PR #59 rebase-merged legacy custom iterable resume compatibility at verified
  `main` SHA `2985ef953546506f3cd6d84c8ff79739c6388a9b`; CI passed (run ID
  `41582739096`).
- PR #60 rebase-merged at verified `main` SHA
  `839d8fcc1f9320ff6193b7c54399c629881945f0`; CI passed (run ID
  `41585383984`). It tests exact resume from a schema-v1 checkpoint with
  optional dataset and newer metadata fields removed (39 focused checkpoint/API
  tests passed locally).
- PR #61 rebase-merged documentation status updates at verified `main` SHA
  `448ea9dc3d5969aa88d612bdfbbf11af682bc78d`; CI passed (run ID
  `41585945140`).
- PR #62 rebase-merged at verified `main` SHA
  `821beacdee235e9ed6e70781023126023120adef`; CI passed (run ID
  `41594252050`). It verifies rejection before mutation when a RandomSampler
  checkpoint lacks saved loader RNG state (40 focused checkpoint/API tests
  passed locally).
- PR #63 rebase-merged a documentation-only status correction at verified
  `main` SHA `97a061df7bc96fc311d1bab341d91f92c8a43b61`; CI passed (run ID
  `41594915207`).
- PR #64 rebase-merged at verified `main` SHA
  `dd3cf7bcea791f370ea0c58d3e863a1702dfc79f`; CI passed (run ID
  `41595632995`). It verifies exact resume with a caller-provided sequential
  loader and no saved loader RNG (41 focused checkpoint/API tests passed
  locally).
- PR #65 rebase-merged a documentation-only status correction at verified
  `main` SHA `908e21fa2fedd34a0de47203a0f12c5d4a473ef3`; CI passed (run ID
  `41596201261`).
- PR #66 rebase-merged at verified `main` SHA
  `4a877768a1c8fcaae654be6c18e88ce374bdce25`; CI passed (run ID
  `41596764595`). It tests exact sequential-loader resume both mid-epoch and
  after the epoch's final row (42 focused checkpoint/API tests passed locally).
- PR #67 rebase-merged a documentation-only status correction at verified
  `main` SHA `74ba1d1d8c05a782b29b18c31131347025b7fcef`; CI passed (run ID
  `41597332118`).
- PR #68 rebase-merged at verified `main` SHA
  `f52ad5fbed7ab51335ff04079fafbd5939069ece`; CI passed (run ID
  `41631351981`). It verifies exact resume with batch size two and two-step
  gradient accumulation (43 focused checkpoint/API tests passed locally).
- PR #69 rebase-merged a documentation-only status correction at verified
  `main` SHA `bf57ebb20600eb69f4ea5f1666d427cf610d3870`; CI passed (run ID
  `41632033098`).
- PR #70 rebase-merged at verified `main` SHA
  `2f0eafcbeb6d394496154fdddc22a55233aee5d0`; CI passed (run ID
  `41632939362`). It tests exact resume with a partial final batch and
  accumulation (44 focused checkpoint/API tests passed locally).
- PR #71 rebase-merged the documentation status correction at verified `main`
  SHA `33094937a78e9a773ba8dbdd2007f73b10ec0a20`; CI passed (run ID
  `41633758016`).
- PR #72 rebase-merged at verified `main` SHA
  `df9f232dff191354b976b0059a2a174777792a5f`; CI passed (run ID
  `41634659366`). It verifies exact resume with TrainLM's default random
  sampler across an epoch boundary (57 focused API/trainer/contract tests
  passed locally).
- PR #73 rebase-merged the documentation status correction at verified `main`
  SHA `6090ba72e32d9d920499807f0c7b5a15e8ae5861`; CI passed (run ID
  `41635317392`).
- PR #74 rebase-merged at verified `main` SHA
  `b0f81e91ad6c11aef865441efc05455202557677`; CI passed (run ID
  `41636368557`). It verifies exact resume both within an epoch and at the
  shuffled sampler's epoch boundary (58 focused API/trainer/contract tests
  passed locally).
- **Verified base for the current compatibility branch:** fetched `main` at
  `4107f2f2ab86c2e90d43428ad1a779a25fe86cac`; the compatibility work is under
  review in PR #83.
- **Probe:** [`T1 v5e-1 guide`](T1_V5E1_PROBE.md) and its
  [Kaggle/Colab notebook](../notebooks/TorchTPU_T1_v5e1_Probe.ipynb) test a tiny
  synthetic forward/backward and optimizer update, with optional `torch.compile`.
- **Evidence:** the owner reported an eager native TPU allocation,
  forward/backward and AdamW pass on torch_tpu 0.2.0 with torch 2.14.1+cpu.
  Hardware generation and explicit compile remain unverified.
- **Validation boundary:** PR #55 passed the full CPU suite (871 tests and 41
  subtests) in Python 3.12.14 / torch 2.9.0+cpu / pytest 9.2.0. PRs #56–#74
  passed GitHub pytest. PR #83 passed 893 tests and 41 subtests in both its
  minimum and current CPU profiles (run `41752008510`). The last full
  local suite reports 865 passed, 7 failed, and 41
  subtests. One failure is the wheel smoke blocked by missing `pip`; five Phi
  cases arise from the installed Transformers 5.0.0 missing
  `PhiConfig.pad_token_id`; a separate TrainLM model reload test has mismatched
  weights. TPU checkpoint validation remains pending. CPU tests, standalone
  device probes, and complete TPU training are separate
  evidence; no TPU generation, convergence, or performance result is claimed.
- **Current T5 checkpoint work:** PR #49 added schema-v3 shard checksums and
  coordinated distributed write/resume preflight; PR #50 added public trainer
  checkpoint preflight before state restore. PR #51 records optimizer/scheduler
  class identity and checks optimizer parameter-group compatibility before
  mutation; legacy schema-v1 checkpoints without class metadata remain
  loadable. PR #52 adds preflight checks for common optimizer slot shapes and
  scalar steps, plus scheduler payload structure and invariants. PR #53
  validates trainer progress counters and metrics before state mutation. PR
  #54 preflights packed-data cursors before restore. PR #55 rejects
  non-persisted lifecycle fields, PR #56 tests optional custom dataset
  `validate_state_dict` preflight, and PR #57 checks exact custom iterable
  continuation. PR #58 checks that identity/cursor corruption rejects before
  trainer mutation. Native TPU checkpoint validation remains pending.
- **Older open PR stack:** PRs #22–#27 remain open on the older M8 branch chain.
  They are more than 250 commits behind current `main`; PR #22 is merge-conflicted and
  its recorded CPU run had 335 passed / 6 failed. Keep them separate until
  rebased and reviewed against the current native TorchTPU roadmap.
- **History:** PR #18 (including PR #16) is merged. The old M0–M17 tables and
  P0/P1 XLA backlog below describe historical XLA implementation/evidence;
  use them for salvage, not as the current TorchTPU task list.

See the [docs guide](README.md), [T0–T11 plan](TORCHTPU_REBASELINE.md),
[salvage inventory](TORCHTPU_SALVAGE_INVENTORY.md), and [T0.2 mapping](PR_DELIVERY_PLAN.md#t02-accepted-mapping-to-native-torchtpu-gates)
for the active planning sources.

This handoff retains implementation and target-hardware notes from the legacy
XLA work. Each feature status below refers to the backend named in its row; it
does not transfer to native TorchTPU.

## Historical CPU baseline (PR #18 snapshot)

The available failure run is from the PR #18 snapshot, not a rerun on merged
`main`:
700 passed and 21 failed under Python 3.12.14, torch 2.9.0+cpu, Transformers
5.0.0, huggingface-hub 1.3.5, and pytest 9.1.1. Failures: five Phi
configuration cases, one checkpoint reload, twelve loss-layout parity cases,
one scalar-materialization assertion, one stale roadmap contract, and one
wheel smoke test blocked by missing `pip` in the venv. This is a CPU-only
baseline, not TPU evidence.

## Status legend

| Mark | Meaning |
| --- | --- |
| `[x]` | Implemented and covered by repository tests/contracts. |
| `[~]` | Partially implemented, or implemented in software but still awaiting required v5e evidence. |
| `[ ]` | Planned; no production implementation exists yet. |

An `[x]` never implies TPU performance certification unless the row explicitly
says that target-v5e evidence passed.

## Product boundary

> `trainlm.optimize(model)` must preserve the supplied Hugging Face model's
> semantics. Architecture-changing work belongs to the future TrainLM Lab API.

Current production scope is dense decoder-only autoregressive training. The
public workflow supplies a model/config, train and evaluation datasets, and
familiar training arguments. Worker launch, PJRT/ranks, topology discovery,
manifest staging, process cleanup, and stage-log parsing remain private.

## Implemented public training and data path (historical XLA-era status)

| Status | Feature | Current state |
| --- | --- | --- |
| `[x]` | HF-like trainer facade | `TrainLMTrainer`, training arguments, `from_pretrained`, `from_config`, train/eval, callbacks, explanation, local save/state, CLI, and YAML entry points exist. |
| `[x]` | Packed local `.bin` dataset | Manifest and payload validation, fixed-length causal samples, lazy memory mapping, deterministic ordering, and rank partitioning exist. |
| `[x]` | Hugging Face shard ranges | `PackedBinDataset.from_hub()` downloads end-exclusive numbered shard ranges to a local cache, resolves mutable dataset revisions, validates once, and performs no download in the training loop. |
| `[x]` | Train/eval data integration | Disjoint packed shard ranges can be staged for training and evaluation through the same reader and coordinator boundary. |
| `[x]` | Private TPU coordinator | Probe, model preflight, training, structured result ingestion, logs, and private worker launch are hidden behind `trainer.train()`. |
| `[x]` | Failure cleanup | Every stage owns a process group; error or interruption terminates the launcher and spawned ranks so notebook cells can be rerun. |
| `[x]` | Visible lifecycle progress | Notebook cells print numbered start/completion markers; coordinator stages print start, ten-second heartbeat with the latest worker event, and completion messages. |
| `[x]` | Model-source reconstruction | Both pretrained HF sources and config-initialized HF models are serialized and reconstructed independently inside workers. |
| `[x]` | HBM-safe validation notebook | The notebook uses the pretrained `HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint with a 128-token lifecycle geometry instead of replicating a 3.8B checkpoint on every TPU worker. Model preflight is independently capped at 16 tokens. |
| `[x]` | Model revision handling | Mutable model selectors are resolved before TPU launch; explicit commit SHAs support offline/cache-only launches. |
| `[~]` | End-to-end TPU lifecycle | Software plumbing exists, but the latest 135M notebook path still needs an owner-run Kaggle v5e-8 completion after the config-source coordinator fix. |

## Implemented runtime, checkpoint, and optimization foundations (historical XLA-era status)

| Status | Feature | Current state / remaining evidence |
| --- | --- | --- |
| `[x]` | Scheduled evaluation and saves | Local engine and TPU request/worker plumbing carry evaluation and checkpoint cadence. |
| `[x]` | Rank-local TPU checkpoints | Atomic staging, committed manifests, topology validation, discovery, RNG/runtime/trainer/data-position state, and resume contracts exist. |
| `[~]` | Exact distributed resume | Contracts and worker implementation exist; exact target-v5e interruption/resume evidence remains. |
| `[x]` | Structural model inspection | Family-neutral config/module/signature inspection reports capabilities without family-name guesses. |
| `[x]` | Adapter/provider registry | Exact class/capability/version guards and deterministic rejection reasons exist. |
| `[x]` | Pure optimization planner | Declarative provider selection, fallbacks, requirements, serialization, stable identity, and explanation exist. |
| `[x]` | Transactional transforms | Reversible application and rollback on transform or optimizer-construction failure exist. |
| `[x]` | State-dict layout conversion | Strict reversible concatenate/split conversion and tied-alias validation exist. |
| `[~]` | Packed QKV and partial Q+KV | Reversible descriptors/handlers and parity tests exist; live target-XLA fusion and performance evidence remain. |
| `[~]` | Packed SwiGLU/GeGLU | Reversible descriptors/handlers and parity tests exist; target-XLA fusion and performance evidence remain. |
| `[~]` | Chunked linear causal CE | Reference forward/backward and rematerialized chunks exist; production TPU provider selection and HBM/HLO evidence remain. |
| `[~]` | Training view without full logits | Guarded view and software parity exist; target TPU proof that full logits are absent remains. |
| `[~]` | Pallas attention family | Canonical specs, guarded bridges, MHA/GQA/MQA/window/ALiBi contracts, and tuning cache exist; real kernels and target evidence are not certified. |
| `[~]` | Selective rematerialization | Policy and evidence-based selector exist; target measurements are required before a default is enabled. |
| `[~]` | XLA AdamW policy | State/update policy and evidence gate exist; target update/resume/HBM proof remains. |
| `[~]` | FSDP policy | Mesh and sharding contracts plus XLA annotations exist; full ZeRO-3/FSDP scaling and distributed checkpoint evidence remain. |
| `[~]` | Batch/prefetch tuning | Deterministic selector exists; real-data v5e measurements are required. |
| `[x]` | Benchmark/release contracts | Parity, numerical alignment, stability, HLO, scaling, export, repeated-run, and support-manifest validators exist. |
| `[ ]` | Performance certification | No current result may be marked certified until the locked v5e gates and required evidence artifacts pass. |

## Immediate execution-preserving backlog (priority order)

> **Historical XLA backlog:** this is not the active TorchTPU priority order.

| Priority | Status | Task | Completion gate |
| --- | --- | --- | --- |
| P0 | `[~]` | Complete the 135M public notebook lifecycle | Probe, preflight, six finite steps, evaluation, committed save, exact resume, no orphan process, and archived artifacts on v5e-8. |
| P0 | `[~]` | Chunked/rematerialized CE production path | No `[B,S,V]` materialization, gradient parity, stable graph, lower HBM, and measured throughput. |
| P0 | `[~]` | TPU-native Pallas attention | MHA/GQA/MQA plus causal/window/ALiBi cases pass output, gradient, mask, HLO/custom-call, and fallback gates. |
| P0 | `[~]` | Prove QKV/MLP/native fusion | Target HLO shows intended fusion with no harmful copy/transpose regressions and matched updates. |
| P0 | `[~]` | Runtime cleanup | Static shapes, compile-count tracking, host-sync removal, persistent workers, asynchronous prefetch, and input-idle evidence. |
| P0 | `[~]` | Checkpoint closure | Async I/O without step stalls, canonical HF export, and exact distributed resume on target TPU. |
| P1 | `[ ]` | Vocabulary and loss parallelism | Distributed max/sum CE, sharded LM head/tied embeddings, numerical parity, and lower per-device HBM. |
| P1 | `[~]` | State sharding abstraction | Finish `none / optimizer / optimizer_gradient / full` semantics and prove full state ownership at 1.3B+. |
| P1 | `[ ]` | Tensor + sequence parallelism | Semantic linear-role sharding, legal-layout planner, collective correctness, and TP×FSDP measurements. |
| P1 | `[ ]` | Communication overlap | Measured all-gather/reduce-scatter/TP overlap improves the matched workload without extra graphs. |
| P1 | `[ ]` | XLA layer scan and compiled accumulation | Smaller stable graphs and favorable wall time versus unrolled layers/microsteps. |
| P1 | `[~]` | Budget-aware rematerialization | Select policy from measured activation bytes, recompute cost, HBM budget, and graph stability. |
| P1 | `[ ]` | Muon/hybrid optimizer policy | Semantic parameter groups and better wall-clock/tokens-to-target-loss than the AdamW baseline. |
| P2 | `[ ]` | Context parallelism | Long-context distributed attention correctness and communication/HBM evidence at 16K+. |
| P2 | `[ ]` | Pipeline parallelism | 1F1B/interleaved schedule with an explicit bubble and memory model. |
| P2 | `[ ]` | Weight-rematerialized TP | Research prototype only after conventional TP×FSDP is stable. |

## Architecture-changing backlog (TrainLM Lab, not optimizer transforms)

> **Deferred research:** this section is outside the active T0–T11 queue.

| Priority | Status | Feature | Direction |
| --- | --- | --- | --- |
| P1 | `[ ]` | Architecture IR | Semantic nodes for token/channel mixers, memory, router, residual topology, normalization, position, objective, and sharding constraints. This is the prerequisite for TrainLM Lab. |
| P2 | `[ ]` | Multi-token prediction | Explicit objective with matched-token and matched-compute evaluation; never enabled as a transparent optimization. |
| P2 | `[ ]` | MLA | Native architecture spec and reference implementation first, followed by Pallas/TP/CP providers; never auto-convert MHA/GQA. |
| P2 | `[ ]` | Sparse attention | Unified dense/window/block/local-global/indexed patterns backed by genuinely sparse kernels, not dense masks. |
| P2 | `[ ]` | MoE and expert parallelism | Router, dispatcher/combine, grouped GEMM, AllToAll, capacity, balancing, placement, overlap, and expert replication. |
| P3 | `[ ]` | Hybrid recurrent/attention mixers | Generalize attention to `TokenMixerSpec` and evaluate recurrent/state-space/global-attention schedules. |
| P3 | `[ ]` | Conditional memory | Explore predictable host-memory lookup and asynchronous transfer as a separate capacity axis. |
| P3 | `[ ]` | Adaptive depth and output gating | Explicit architecture features with stability and quality evaluation. |
| P3 | `[ ]` | Residual topology IR | Serial, parallel, gated, multi-stream, and hyper-connection semantics. |
| P3 | `[ ]` | Unified adaptive computation | One learned event/difficulty signal coordinating memory access, expert routing, global interaction, and depth. |
| P3 | `[ ]` | Structured architecture search | Controlled design spaces, staged 10–30M → 100–300M → 1B → 3B–7B experiments, and Pareto scoring. |
| P3 | `[ ]` | Hardware-aware co-design | Optimize quality together with compute, memory, collectives, runtime, compilation, and decode costs. |

## Telemetry and self-tuning backlog

| Priority | Status | Feature | Direction |
| --- | --- | --- | --- |
| P1 | `[~]` | Structured metrics | Worker metric JSONL and coordinator aggregation exist; formal run/environment/provider/event schemas remain. |
| P1 | `[ ]` | Rank-0 phase-aware dashboard | Workers measure, coordinator reports; normal output is deduplicated and only rank divergence expands. |
| P1 | `[ ]` | Low-sync metric collection | Device-side accumulation with separate collect, aggregate, render, and persist cadences. |
| P1 | `[ ]` | Async writer | Bounded priority queue; training never waits on rendering, JSON, disk, or network sinks. |
| P1 | `[ ]` | Trends and adaptive cadence | Short/long EMA, slope, variance, and wall-clock display cadence based on stability/anomaly state. |
| P1 | `[ ]` | Structured warnings and dedupe | Stable warning codes, severity/component/rank/fallback/count fields, and distributed coalescing. |
| P1 | `[ ]` | Compile/recompile and host-sync detection | Attribute unexpected graph creation, shape changes, and synchronization sources. |
| P1 | `[ ]` | Runtime-share observability | Compute, collective, input, host, and idle timing with worker imbalance detection. |
| P2 | `[ ]` | Anomaly diagnostics | Correlate throughput, input wait, HBM, collectives, compile, fallback, checkpoint, and host-sync signals. |
| P2 | `[ ]` | Circular diagnostic buffer | Persist high-resolution windows only around NaN/OOM/recompile/desync/throughput anomalies. |
| P2 | `[ ]` | Triggered profiling | Short representative or anomaly windows, never continuous profiling by default. |
| P2 | `[ ]` | Multiresolution retention | Exact recent points and aggregated older history with min/max/mean/std/count/EMA. |
| P2 | `[ ]` | Run fingerprint and baseline comparison | Bind model/data revisions, software, hardware, plan, mesh, precision, and compare matched runs. |
| P2 | `[~]` | Performance regression CI | Contract gates exist; scheduled TPU runners and fresh real evidence remain. |
| P2 | `[ ]` | Telemetry overhead budgets | Measure and enforce fast/normal/expert overhead tiers. |
| P3 | `[ ]` | Runtime recommendations | Turn diagnosed HBM/input/communication gaps into benchmarkable planner candidates. |
| P3 | `[ ]` | Closed-loop autotuning | Analyzer → plan → run → telemetry → candidate benchmark → retained plan, with rollback and evidence. |

## Next owner-run TPU procedure

> **Historical XLA validation procedure:** this does not describe TorchTPU validation.

1. Use the owner-approved validation revision from that XLA-era procedure
   (ideally the latest merged `main` after PR #18) in a fresh Kaggle TPU
   session; do not treat an old milestone branch as current.
2. Run the install cell, restart the session exactly once, then run the
   remaining notebook top-to-bottom.
3. Confirm `probe`, `model_preflight`, and `train` complete for the pretrained
   SmolLM2-135M-Instruct configuration.
4. Confirm finite loss, evaluation at steps 2/4/6, committed checkpoints at
   steps 2/4/6, and successful resume from `checkpoint-4`.
5. Archive `request.json`, `coordinator_summary.json`, `summary.json`,
   `metrics.jsonl`, stage logs, checkpoint manifests/shards, and XLA metrics.
6. Record the outcome in `docs/IMPLEMENTATION_CONTEXT.md`; do not mark a TPU or
   performance item complete from software tests alone.

## First continuation task after lifecycle success

Close the dense execution gap before starting architecture research: integrate
the guarded chunked-loss and attention providers into the actual worker plan,
capture target HLO/HBM/compile/fallback evidence, then validate reversible QKV
and MLP transforms through complete optimizer updates and canonical export.

## Scheduled evaluation stall diagnosis

The observed pause after step 2 was scheduled evaluation, not another training
step: the smoke config used `eval_steps: 2` with a roughly 250-million-token
validation shard and no evaluation limit. `max_eval_batches` is now wired across
the public API, coordinator, worker, and trainer. The notebook and SmolLM2 smoke
example use `max_eval_batches: 1`; remove that limit only for deliberate full-set
validation. Evaluation and checkpoint phase transitions are printed so future
runs identify this work directly.

## TPU smoke and first-evaluation compilation

`max_eval_batches=1` bounds evaluation execution but cannot bound the first XLA
eval-mode graph compilation. The validation notebook therefore omits evaluation
from its initial train/checkpoint/resume smoke. Add an evaluation dataset and
cadence only in a separate run after lifecycle success. If a probe log contains
`RAW: Dumping core` or `exit() hanging`, the TPU runtime is unhealthy; the
coordinator now rejects the false-success launcher result and requests a full
notebook session restart.
