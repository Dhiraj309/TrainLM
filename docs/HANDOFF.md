# TrainLM implementation handoff

This is the concise continuation guide for the current
`milestone/m10-m12-kernels-parity` branch. It separates code that exists from
code that has been proven on target hardware, and it keeps execution-preserving
work separate from architecture-changing research.

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

## Implemented public training and data path

| Status | Feature | Current state |
| --- | --- | --- |
| `[x]` | HF-like trainer facade | `TrainLMTrainer`, training arguments, `from_pretrained`, `from_config`, train/eval, callbacks, explanation, local save/state, CLI, and YAML entry points exist. |
| `[x]` | Packed local `.bin` dataset | Manifest and payload validation, fixed-length causal samples, lazy memory mapping, deterministic ordering, and rank partitioning exist. |
| `[x]` | Hugging Face shard ranges | `PackedBinDataset.from_hub()` downloads end-exclusive numbered shard ranges to a local cache, resolves mutable dataset revisions, validates once, and performs no download in the training loop. |
| `[x]` | Train/eval data integration | Disjoint packed shard ranges can be staged for training and evaluation through the same reader and coordinator boundary. |
| `[x]` | Private TPU coordinator | Probe, model preflight, training, structured result ingestion, logs, and private worker launch are hidden behind `trainer.train()`. |
| `[x]` | Failure cleanup | Every stage owns a process group; error or interruption terminates the launcher and spawned ranks so notebook cells can be rerun. |
| `[x]` | Model-source reconstruction | Both pretrained HF sources and config-initialized HF models are serialized and reconstructed independently inside workers. |
| `[x]` | HBM-safe validation notebook | The notebook uses the from-scratch 135M Llama-shaped reference instead of replicating a 3.8B checkpoint on every TPU worker. |
| `[x]` | Model revision handling | Mutable model selectors are resolved before TPU launch; explicit commit SHAs support offline/cache-only launches. |
| `[~]` | End-to-end TPU lifecycle | Software plumbing exists, but the latest 135M notebook path still needs an owner-run Kaggle v5e-8 completion after the config-source coordinator fix. |

## Implemented runtime, checkpoint, and optimization foundations

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

1. Pull the latest `milestone/m10-m12-kernels-parity` branch into a fresh
   Kaggle TPU session.
2. Run the install cell, restart the session exactly once, then run the
   remaining notebook top-to-bottom.
3. Confirm `probe`, `model_preflight`, and `train` complete for the 135M config.
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
