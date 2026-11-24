# TrainLM Roadmap

**Document-Isolated Varlen Sequence Packing (2026-11-23):** on branch
`perf/58-varlen-document-packing` (from `main` `daef79c`): upgraded
`_PackedTokenDataset` and `_maybe_pack_token_dataset` in `src/trainlm/api.py`
to emit per-document `position_ids` resets (`0..L_i-1`), mask cross-document
boundary targets (`labels == -100` at `position_ids == 0` after index 0), and
provide `compute_varlen_cu_seqlens` for `flash_attention_2` varlen kernels.

**CI Speedup (`uv` + CPU PyTorch + `pytest-xdist`) (2026-11-23):** on branch
`ci/57-fast-uv-cpu-torch-and-xdist` (from `main` `1766f4b`): updated
`.github/workflows/ci.yml` to use `astral-sh/setup-uv@v5` with dependency
caching, CPU-only PyTorch wheels (`https://download.pytorch.org/whl/cpu`),
and parallel test execution (`pytest -n auto`).

**Liger-Kernel Auto-Wiring & `torch_compile_mode` (2026-11-23):** on branch
`perf/56-liger-kernel-and-compile-mode` (from `main` `c554749`): added
`use_liger_kernel: bool | Literal["auto"] = "auto"` and
`torch_compile_mode` (`"default"`, `"reduce-overhead"`, `"max-autotune"`,
`"max-autotune-no-cudagraphs"`) to `TrainLMTrainingArguments`, implemented
`apply_liger_kernels` in `src/trainlm/optimization/liger.py` (auto-applied on
CUDA when `liger-kernel` is installed, skipping `pack_gated_mlp` when Liger's
fused SwiGLU is active), and wired `compile_mode` through `TorchRuntime`.
CPU/mocked tests added; real GPU validation and hosted CI remain pending.

**Built-In Console Step Logging (2026-11-23):** on branch
`feat/55-builtin-console-step-logging` (from `main` `c76b758`): added
`_ConsoleLoggingCallback` to `TrainLMTrainer` (`src/trainlm/api.py`) so
`trainer.train()` automatically prints `step X/N | loss=... | ... tok/s | MFU ...% | wait ...%`
at `logging_steps` intervals whenever `logging_verbosity != "quiet"`, without
requiring custom callback classes in user notebooks.

**Restore Compiled TPU Optimizer Performance & Init Order (2026-11-23):**
on branch `fix/54-restore-tpu-40pct-mfu` (from `main` `04b5be6`): removed
`@_compiler_disable` from `AdamWStateDtype.step` (which forced eager
single-tensor execution in `STRICT` mode for every step, regressing
`SmolLM2-135M` eager/compiled to `9.94%` / `16.12%` MFU) and scoped
`_place_model_on_runtime_device()` to CUDA only so `torchtpu` keeps its
`453a1c0` init order (`39.46%` MFU on `SmolLM2-360M`).

**Fix Compiled Backward Graph Break Regression (2026-11-23):** on branch
`fix/53-remove-compiler-disable-from-backward` (from `main` `2472116`):
removed `@_compiler_disable` from `TorchRuntime.backward`, `optimizer_step`,
and `zero_grad` in `src/trainlm/runtime/runtime.py` (which caused
`TPU v5e-1` compiled MFU on `2472116` to fall back to eager backward and drop
from `39.46%` -> `18.06%` on `SmolLM2-360M`), keeping `@_compiler_disable`
only on `AdamWStateDtype.step`.

**Fix MFU Notebook Accelerator Routing (2026-11-23):** on branch
`fix/52-notebook-accelerator-routing` (from `main` `8abbf3b`): updated
`notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb` to pass explicit
`accelerator="cuda"` (on CUDA), `"native_tpu"` (when `torch_tpu` is installed),
or `"cpu"` instead of `accelerator="auto"` (which keeps automatic TPU routing
disabled by policy), and added a contract guard in
`tests/contracts/test_v5e1_mfu_benchmark_notebook.py`.

**Fix Dynamo `lr`/Token-Count Recompiles, CUDA Optimizer Placement, & Steady-State MFU (2026-11-23):**
on branch `fix/51-compile-step-recompiles-and-cuda-device-placement` (from
`main` `c515ca1`): Colab `TPU v5e-1` run on `453a1c0` achieved `39.46%`
non-emb MFU (`44.19%` logits-incl) on `SmolLM2-360M` and `38.17%` non-emb MFU
(`48.25%` logits-incl) on `Qwen2.5-0.5B`, while revealing `AdamWStateDtype.step`
hitting Dynamo `recompile_limit (8)` on Python float `lr` updates (`11,998.7`
tok/s in Window 2 -> `61,588.1` tok/s after fallback). Audit also identified
redundant `zero_grad` and `loss*T`/`grad/T` scaling inside `_single_step_tensors`,
optimizer construction running before model device placement for CUDA `fused=True`,
missing `compile_training_step` on CUDA, and cold Window 1 skewing
`performance_summary()`. Fixed all five and updated
`TrainLM_v5e1_MFU_Benchmark.ipynb` to print `steady_state` metrics. CPU/mocked
tests updated; Colab rerun pending.

**Simplify MFU Benchmark Notebook UX (2026-11-23):** on branch
`refactor/50-simplify-mfu-benchmark-notebook` (from `main` `453a1c0`):
removed custom `Dataset`, `TrainerCallback`, `_summarize`, and HBM/TPU
boilerplate from `notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb` in favor of
direct HF-like `TrainLMTrainer(...)` + `trainer.train()` +
`trainer.performance_summary()` calls with `accelerator="auto"`.

**Opt-In 8-Bit Compute Pretraining (`fp8` / experimental `int8`) (2026-11-23):**
on branch `feat/49-opt-in-8bit-compute-pretraining` (from `main` `1ca87b5`):
added `quantized_training: Literal["none", "fp8", "int8"] = "none"` to
`TrainLMTrainingArguments`, implemented `apply_quantized_training` in
`src/trainlm/optimization/quantized_training.py` (using optional
`torchao.float8.convert_to_float8_training` for `fp8` and
`torchao.prototype.quantized_training.Int8MixedPrecisionTrainingConfig` for
`int8`, keeping `lm_head`/`embed_tokens` and non-16-aligned layers unquantized,
and warning on `int8` with `hidden_size < 768`), and wired it before
`prepare_model`/`_make_optimizer` in `TrainLMTrainer`. CPU/mocked tests added;
real H100/L4/A100 hardware validation and hosted CI remain pending.

**8-Bit Optimizer States & PEFT/LoRA Integration (2026-11-23):** on branch
`feat/48-8bit-optimizer-and-peft` (from `main` `9cd8202`): added
`optim` (`"adamw"`, `"adamw_torch"`, `"adamw_8bit"`, `"paged_adamw_8bit"`,
`"torchao_adamw_8bit"`) to `TrainLMTrainingArguments` and `OptimizerConfig`
(using optional `bitsandbytes` / `torchao` with clear install errors when
absent), filtered `OptimizerFactory.create` to trainable parameters when
frozen parameters exist, and added `peft_config` support on `TrainLMTrainer`.
CPU/mocked tests added; real GPU/TPU hardware validation and hosted CI remain
pending.

**Kernel & Packing Auto-Wiring (2026-11-23):** on branch
`perf/47-kernel-and-packing-autowire` (from `main` `23b0c8c`): added
`attn_implementation` (`"auto"` selects `"flash_attention_2"`/`"sdpa"` on CUDA,
and `"splash"` wires `TorchTPUSplashAttentionRuntime`), `pack_gated_mlp`
(default auto-enabled on CUDA/TPU via `apply_torchtpu_kernel_plan`, with
weight-preserving rollback in `TorchTPUKernelPlanResult.rollback` and
`TrainLMTrainer.save_model`), and `packing: bool = False` for constant-length
token sequence packing in `_maybe_pack_token_dataset`. CPU tests added;
real GPU/TPU hardware validation and hosted CI remain pending.

**Native CUDA Parity & Speed Pack (2026-11-23):** on branch
`perf/46-cuda-compile-fused-adamw-and-mfu` (branched from `main` `54b0909`):
enabled `torch_compile` on `accelerator="cuda"` / `"auto"`, wired `fused=True`
AdamW on CUDA (with a Python-loop fast path in `AdamWStateDtype.step` for
fused FP32 moments), added `float32_matmul_precision("high")` TF32 lifecycle,
`non_blocking=True` H2D transfers, `GradScaler` for `fp16` on CUDA, BF16/FP16
`chunked_linear` Tensor Core projection on CUDA, and documented NVIDIA GPU peak
FLOPs (`H100`, `A100`, `L40S`, `L4`, `T4`, `V100`) in `performance.py`.
CPU/mocked-CUDA tests added; real GPU hardware validation and hosted CI remain
pending.

**Fix compiled-step device transfer & `chunked_loss` graph break (2026-11-23):**
owner Colab `TPU v5e-1` run (`db9eb94`) improved `SmolLM2-135M` eager (`DeferAndFuse`)
to `14.35%` MFU (`27,723.9 tok/s`), and uncovered why all three `compiled`
runs failed (`expected tensor to be on tpu, got cpu` because `prepare_batch` ran
inside `compile_training_step`) plus a Dynamo graph break on
`denominator.item()` in `chunked_loss.py`. On branch
`fix/45-tpu-compiled-step-args-and-graph-break`: moved
`runtime.prepare_batch(task_batch)` before `step_fn(task_batch, counts)` in
`Trainer._train_step`, skipped no-op `.to(device)` in `_move_to_device`, and
guarded `denominator.item()` with `not torch.compiler.is_compiling()`.

**Fix MFU notebook `nbformat` cell ID (2026-11-23):** on branch
`fix/44-notebook-cell-id-nbformat`, renamed cell ID `qwen2.5-0.5b` to
`qwen2_5-0_5b` in
[`notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb`](../notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb)
to match `nbformat`'s `^[a-zA-Z0-9-_]+$` requirement and added a regex guard
to `tests/contracts/test_v5e1_mfu_benchmark_notebook.py`.

**Split MFU benchmark into per-model cells (2026-11-23):** updated
[`notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb`](../notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb)
on branch `refactor/43-split-mfu-notebook-cells` so setup, config, helpers,
`SmolLM2-135M`, `SmolLM2-360M`, `Qwen2.5-0.5B`, and summary/download each live
in separate runnable cells (7 code cells total). Contract tests pass on CPU;
Colab v5e-1 MFU rerun remains pending.

**Tier 4 — HF-simple auto-UX for chunked linear loss (2026-11-23):**
implemented on branch `feat/42-tier4-hf-simple-auto-ux` after PR #162 merged
(`30f2e57`). `TrainLMTrainer` now auto-resolves `LinearCausalLMTrainingView`
for standard HF `PreTrainedModel` causal LMs via `_auto_training_view(model)`
(`model.base_model` + `ForwardBatchDispatcher`) whenever
`loss_implementation="chunked_linear"` is selected without a manual
`training_view_factory`. Simplified
[`notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb`](../notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb)
to drop manual `LinearCausalLMTrainingView`/`inspect_dense_causal_lm`
boilerplate, updated contract tests, and added CPU lifecycle tests. Hosted CI
and Colab v5e-1 MFU rerun remain pending.

**Tier 3 — Native BF16 parameters, BF16 first-moment policy, and MXU precision controls (2026-11-23):**
implemented on branch `perf/41-tier3-torchtpu-memory-precision-and-remat` after
PR #161 merged (`f597e35`). `TorchTPURuntime.prepare_model` now places models in
`torch.bfloat16` when `precision == "bf16"` (eliminating per-layer FP32→BF16
`_to_copy` casts and halving weight HBM); `on_train_begin`/`on_train_end`
manage `float32_matmul_precision("medium")` and optional
`allow_excess_precision`; `TrainLMTrainer._optimizer_config` selects
`mu_dtype="bfloat16", nu_dtype="float32"` on `torchtpu + bf16`; and
`AdamWStateDtype._cast_moment_state` skips redundant `.to()` casts when dtypes
already match. CPU unit tests added; hosted CI and Colab v5e-1 MFU rerun remain
pending. Tier 4 (HF-simple auto-UX) follows on the next branch after merge.

**Tier 2 — Native TorchTPU Splash Attention, SwiGLU packing, and loss provider catalog (2026-11-23):**
implemented on branch `perf/40-tier2-torchtpu-kernels-and-packing` after
PR #160 merged (`eccef6f`). Added guarded native Splash Attention adapter
(`src/trainlm/optimization/torchtpu_attention.py`), reversible
`apply_torchtpu_kernel_plan` with `PackedGatedMLPBlock`
(`src/trainlm/optimization/torchtpu_kernels.py`), and `"torchtpu"` backend
support in `causal_loss_provider_specs`
(`src/trainlm/optimization/loss_providers.py`). CPU unit tests added in
`tests/optimization/test_torchtpu_kernels.py`; hosted CI and Colab v5e-1 MFU
rerun remain pending. Tier 3 (BF16 optimizer states, selective
rematerialization, MXU precision/bucketing) and Tier 4 (auto-wired UX) follow
in separate branches.

**Tier 1 — Whole-step compile, DeferAndFuse, and foreach grad scaling (2026-11-22):**
implemented on branch `perf/39-tier1-whole-step-compile-and-defer-fuse` after
PR #159 merged (`9cd249f`). `TorchTPURuntime.compile_model` and
`compile_training_step` now compile with `backend="tpu", dynamic=False`;
`TorchTPURuntime.on_train_begin`/`on_train_end` enable `DeferAndFuse` eager
mode when `compile_enabled=False`; `TorchRuntime.scale_gradients` uses
`torch._foreach_mul_` to eliminate per-parameter Python loops; and
`Trainer._train_step` routes single-microbatch host-prepared steps through
`runtime.compile_training_step`. CPU unit tests were added; hosted CI and
Colab v5e-1 MFU rerun remain pending. Tier 2 (Pallas Splash Attention, Fused
CE, QKV/SwiGLU packing) and Tier 3/4 follow in separate branches.

**TorchTPU compile, BF16 chunked-loss, and WSD fixes (2026-11-22):** following
owner-run Colab `TPU v5e-1` MFU benchmark output (`SmolLM2-135M` eager `13.68%`
MFU, `SmolLM2-135M` compiled failure in `ForwardBatchDispatcher`,
`SmolLM2-360M` `15.04%` MFU, `Qwen2.5-0.5B` `9.26%` MFU, `wait=0.1%`),
implemented the PR A fixes on branch `fix/38-torchtpu-compile-and-bf16-chunked-loss`:
unwrapped `_orig_mod` in `ForwardBatchDispatcher.from_model`, routed
`LinearCausalLMTrainingView` hidden-state extraction through `torch.compile`
when the trainer's model is compiled, kept `chunked_loss._chunk_terms` `F.linear`
in BF16/FP16 on `device.type == "tpu"` (not only `"xla"`), and populated
`horizon_tokens` (and warmup/stable fractions) for `lr_scheduler_type="wsd"` in
`TrainLMTrainer`. CPU unit tests were added; hosted CI and owner v5e-1 MFU
rerun remain pending. Whole-step compilation (`compile_training_step`) and
Pallas/Splash kernels remain next in PR B/C.

**Dedicated v5e-1 LLM MFU benchmark notebook (2026-11-22):** added
[`notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb`](../notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb)
as a short 5-cell notebook dedicated solely to training `SmolLM2-135M`,
`SmolLM2-360M`, and `Qwen2.5-0.5B` on `TPU v5e-1` with live MFU, cold-vs-steady
window separation, HBM stats, and `torch.profiler` bottleneck reporting.
Notebook JSON, syntax, and catalog contracts pass on CPU; owner v5e-1 baseline
run completed (see above).

**Colab v5e-1 full validation notebook (2026-11-22):** added
[`notebooks/TrainLM_Colab_v5e1_Full_Validation.ipynb`](../notebooks/TrainLM_Colab_v5e1_Full_Validation.ipynb)
to group all owner-run single-device v5e-1 checks into one toggleable notebook.
An initial owner run on Colab `TPU v5e` (`torch==2.14.1+cpu`, `torch-tpu==0.2.0`,
`libtpu==0.0.50`, `transformers==5.22.1`, source `a091ce4`) passed 15 checks:
`t1_install_preflight`, `cli_doctor`, `t1_eager_probe`, `t1_compile_probe`,
`t1_lifecycle_probe` (2 fresh child processes), `hf_matrix_gpt2`,
`hf_matrix_llama`, `cli_pretrain_and_finetune` (reporting `mfu_status="measured"`),
`vm_replacement_recovery_and_extend`, all four `run_finetuning` checks (finite
Hub, local JSONL epoch/best/early-stop, pretokenized `vocab_sha256`, and
`streaming=True` cursor resume), and `packed_epoch_and_chunked_linear`.
`pretrain_options_bf16` failed because `TrainLMTrainer._make_scheduler` does
not yet pass `horizon_tokens` for `lr_scheduler_type="wsd"` (tracked for the
next fix PR). The notebook now uses `cosine` in `pretrain_options` and adds
Section 8 (`RUN_MFU_BASELINE`) to measure steady-state MFU and bottlenecks
across `SmolLM2-135M` (eager + `torch_compile`), `SmolLM2-360M`, and
`Qwen2.5-0.5B` (`chunked_linear`). Notebook JSON, syntax, and contract tests
pass on CPU; the MFU baseline and `pretrain_options` rerun on v5e-1 remain
pending.

**Measured training performance (2026-11-22):** ordinary training runs now
collect sparse synchronized wall-clock windows and report observed supervised
tokens/second plus dataloader wait fraction. Native single-device TPU v5e BF16
runs also report standard non-embedding and logits-inclusive MFU using the
documented 197 TFLOPs/chip peak. Unknown peak/model/precision combinations
return MFU unavailable instead of substituting a guessed value. This provides
input-pipeline evidence; device-kernel timeline profiling and owner-run TPU
measurements remain outstanding.

PR #156's hosted CI runs caught scratch-notebook contract mismatches and stale
tests expecting exact iterable resume without a verifiable cursor/source
identity. The notebook now keeps fresh-run controls and exact-resume/budget-
extension choices within its concise layout. The resume tests now verify that
unverifiable readers are rejected before state changes. Its intro now clearly
labels the short run as a workflow check with no useful model output. Hosted
CI is the validation gate before merging.

**Pinned Hub streaming fine-tuning (2026-11-22):**
`TrainLMFineTuningConfig(streaming=True)` resolves one immutable Hub revision
and loads raw-text splits lazily. TrainLM fingerprints the source, splits,
preprocessing, tokenizer mapping and sequence length for resume. Finite steps,
bounded evaluation, zero workers and step schedules are required. CPU suites
pass (84 tests), including exact stateful iterable resume parity and the
mocked Hub loader/restore contract. Live Hub and TPU execution remain
unverified; tokenized streaming, epoch budgets and high-level pretraining
streams remain future work. HF-06 remains partial.

**HF workflow implementation follow-up (2026-11-22):** fine-tuning resume now
restores the resolved immutable Hub dataset revision and records/compares
fingerprints for caller-provided HF Dataset splits; a provided split without a
stable fingerprint is rejected for exact resume. Existing local fine-tuning
file hashes continue to be checked during restore. `bundle_pretraining_artifacts`
has an explicit `include_resume_state=True` option that packages generated
resume files, checkpoint, prepared tokenizer/shards and report for recovery;
it is opt-in because archives may be large and contain private prepared corpus
data. The scratch/CPT notebook now separates raw Hub input from tokenized HF
Dataset input, reloads the pinned splits for resume, and has a private Hub
publishing cell disabled by default. The support matrix now lists runtime,
dtype/compile and evidence status for each workflow. `TrainLMTrainer` also
gates iterable checkpoint/resume on zero workers, a state cursor, and a stable
dataset identity or an explicit state validation hook, then checks available
identity before restoring model or optimizer state. `run_finetuning` now
exposes the contract for raw-text streams; `run_pretraining` remains finite-
preparation based. The fine-tuning and trainer CPU suites pass (84 tests).
No TPU execution, replacement-VM resume or authenticated Hub upload has been
run. HF-03, HF-06 and HF-07 remain in progress; HF-08 still needs the owner-run
hardware and Hub evidence described in [the matrix](HF_SUPPORT_MATRIX.md).
Replacement-VM recovery remains unverified. Provided
pretraining Datasets can declare an HF source/revision for provenance; TrainLM
resolves it and fingerprints the supplied objects but cannot independently
prove their origin.

**Preview an HF pretraining run.** The [scratch/CPT notebook](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) calls `config.training_plan(inspect_model=True)` to show the architecture, parameter count and equivalent FP32 parameter bytes before training. It uses the HF configuration and a PyTorch meta model. The normal plan stays offline. When training a new BPE tokenizer, the estimate assumes the requested vocabulary capacity. Activations, optimizer state and allocator overhead require separate memory measurements.

**Check pretokenized input.** Corpus producers can record `trainlm.tokenizer_vocab_sha256(tokenizer)` alongside their token rows. Pass that digest as `expected_tokenizer_vocab_sha256` in scratch, CPT or fine-tuning configs to reject a different token-to-ID mapping before training. The comparison includes added vocabulary entries. Its scope is vocabulary identity; formatting, token boundaries and label alignment are separate contracts. The check is optional for existing corpora.



**Single-device HF acceptance.** PR #145 merged at `638c6643af0a6db19b491622805b74db19fcb09f`. The [HF support matrix](HF_SUPPORT_MATRIX.md) records four live Hub CPU runs covering random weights and CPT with raw reviews and tokenized JSONL. The v5e-1 validation notebook runs the same acceptance checks on native TorchTPU. Hardware results remain outstanding.


Four CPU end-to-end cases exercise random initialization and CPT on raw text and token IDs from a fixture stream, followed by Transformers generation. A single-pass token budget also checks the actual supervised targets remaining after masking.

The [HF workflow support matrix](HF_SUPPORT_MATRIX.md) links the scratch, CPT and tokenized corpus recipes to CPU and TPU evidence.

**Finite packed training (HF-06).** The public trainer counts epochs for a zero-worker PackedBinDataset. The v5e-1 notebook uses epoch budgets and checks the checkpoint format written by the selected backend. CPU coverage exercises two epochs and continuation from the first checkpoint. Hardware validation is pending. Scratch and CPT can stream token rows from a pinned Hub split or paired JSON/Parquet files, and repeat finite prepared data with `repeat_train_data=True`. The CPU CPT regression preserves tokenizer provenance, reaches the budget across passes, and generates from the export. Streamed Hub slices are applied to the named split using offsets or published split sizes. A CPU run resumed from the first checkpoint directory after moving the full run and increasing its token budget. Use `restore_pretraining_report(..., restored_output_dir=...)` for a changed mount path.

**Current product delivery plan (2026-11-19):**
[Hugging Face user workflows, HF-01–HF-08](HF_USER_WORKFLOWS.md).
The owner's priority is reusable HF models/tokenizers/datasets through
scratch or continued pretraining, save, optional Hub publication and standard
inference. Verified merged `main` is
`7138cc04b26a0eab48e12fc86c6787cb38741e5d`; HF-01–HF-03 are complete through
PR #140, HF-05's publishing API is merged by PR #141, and PR #142 merged the
HF-04 training-plan lifecycle and finite TPU epoch budgets. Live Hub upload
remains unverified; the connector could create and inspect repos but provided
no upload operation, and remote HF Jobs returned 402. HF-01 resolves independent
fine-tuning model/tokenizer IDs and revisions, preserves same-source snapshot
pinning, checks known token-ID bounds
and static context limits, and pins resume to resolved source commits. HF-02
connects scratch pretraining to an HF model config and tokenizer: random
initialization or pretrained CPT, with an explicit BPE option and uint32 token
IDs. Independent tokenizer meaning remains the user's responsibility. HF-03
merged on `feat/20-hf-dataset-sources` through PR #140: streamed raw Hub data
supports configs, friendly revisions, and non-overlapping split slices. The
pretrained fine-tuning helper also accepts prepared HF Dataset rows through
`data_format="tokens"`, preserving causal labels/masks and padding ragged
batches. The current HF-03 branch also accepts raw-text and tokenized HF
Dataset objects in `run_pretraining` / `TrainLMTrainer.from_scratch`, with
fingerprinted preparation inputs and resume checks. For token inputs, TrainLM
adds EOS when absent, preserves optional aligned labels and loss masks in
checksummed sidecars, derives labels from `input_ids` when absent, and applies
one causal shift. Selected tokenizer provenance is recorded while compatibility
remains caller responsibility. Raw multi-column datasets support a named
`formatting_func` whose identity participates in preparation and checkpoint
resume. Local raw inputs support paired text, CSV, JSON/JSONL, and Parquet
files; extension-based or explicit parser selection, source-column selection,
and named row formatting are recorded in cache identity and reports. Unified
cross-workflow provenance remains incomplete. HF-04 epoch evaluation and
checkpoint scheduling merged through PR #143 on
`feat/23-tpu-epoch-eval-save`; HF-06–HF-08 have partial implementation and evidence; notebook
presets do not complete those workflows. HF-04 adds offline `training_plan()`
methods that report the budget and user-visible training controls, including
optimizer settings, logging, save/evaluation cadence and best-model behavior.
The named `TrainLMPretrainingConfig.continued_pretraining(...)` factory makes
CPT selection explicit while reusing the same run lifecycle.
Pretraining plans now distinguish fresh scratch/CPT runs, exact checkpoint
resume, and strict token budget extension; they never inspect checkpoint files.
Fine-tuning plans now show all configured step/token/epoch stop conditions
and the first-limit stopping rule. Token-step counts are one-device
dense-target estimates. The legacy XLA coordinator forwards global
supervised-token targets and supports finite packed-data epoch budgets. The
worker resolves epochs from each rank's finite partition and flushes partial
gradient accumulation at epoch boundaries; it preserves the established first
epoch order for checkpoint compatibility, then deterministically reshuffles
shards and batches from the run seed and reconstructs that order when resuming.
PR #143 added epoch-based evaluation and checkpoint saving, flushing partial
accumulation at each boundary. The focused coordinator, worker CLI, lifecycle,
roadmap, trainer, and notebook-contract suites pass 140 CPU tests. Regressions
verify partial accumulation flush, evaluation-before-checkpoint ordering, and
independent evaluation-only/checkpoint-only schedules. The new v5e-1 Colab
validation notebook runs those CPU contracts plus combined and independent
epoch schedules and checkpoint resume using local model/data fixtures. It
prefers native TorchTPU when available and can explicitly run the legacy XLA
coordinator; its report distinguishes the two. It requires Colab connected to
the owner's v5e-1 Cloud TPU VM; no TPU run has been performed for this branch.
The from-scratch notebook now previews both fresh-run settings and the selected
resume/extension mode in short cells, including an exact-resume option.
Epoch-based fine-tuning estimates require a caller-supplied finite dataset.
Parameter count, memory, runtime selection and wall time remain unresolved.
HF-05's opt-in
`push_to_hub()` is
merged: successful pretraining and fine-tuning results publish allowlisted
model/tokenizer artifacts and a model card, default new repos to private,
check existing visibility, and return the immutable Hub commit. Mocked CPU
coverage and hosted CPU CI pass; live Hub round-trip validation remains
unverified. PR #141's first hosted run found that the
installed-wheel smoke test's explicit `__all__` allowlist omitted the new
`HubPublishResult`; the contract now includes it. Seven focused tests pass
locally, and a locally built wheel imports the public symbol; the full wheel
contract requires hosted CI because local build isolation cannot fetch its
build dependency. The corrected `core-minimum` and `core-current` hosted
profiles both pass on `59e17f87eedc637d34b299c75efe152aaf27c1d5`. PR #140
passed both hosted CPU profiles at `0200e0f19069611b95bc2daebabfe57f26837e3a`.
The previous
token-input slice passed 54 local tests plus 14 subtests, and CI for
`e4e1c9f621148a76cf10efb6d00ceb5769d4e5ce` passed both core profiles. This
follow-up passed 78 focused local tests plus 14 subtests. PR #140's hosted
`core-minimum` and `core-current` jobs then exposed a schema contract gap:
the v1 manifest schema did not accept the new nullable or populated supervision
sidecar fields. The schema now covers both forms, with sidecars remaining
optional for legacy manifests; the affected CPU suite passed 82 tests plus 14
subtests, and both hosted CI profiles passed on correction commit
`c4822048162d84b01348ebdd2c61bf9770070357`. The multi-column row formatter
passes 99 focused CPU tests plus 14 subtests, and both hosted core profiles
pass on `594ec2d75bd8c721847f6350be611642a5debe4d`. The local structured-file
follow-up passes 43 focused CPU tests; it adds no live Hub or TPU evidence.
The checkpoint-compatibility fix omits the default `local_data_format="auto"`
from new resume fingerprints so pre-field manifests remain compatible; explicit
parser choices are retained. Its focused CPU regression passes.
The fine-tuning quickstart now bounds its Hub train/validation slices before
tokenization and avoids duplicate scheduled-plus-final evaluation for its
three-step smoke run; larger workflows can widen the slices.
Fine-tuning config accepts Hugging Face split-slice selectors, and tokenized
train/evaluation rows are checked after mapping in bounded batches so cached
`.map()` output cannot skip structural or model-embedding-bound validation.
Causal-LM supervision now masks targets whose preceding input position is
padding, including the first real token of left-padded rows, for both TrainLM
and model-owned loss paths.
Scratch runs with packed labels or loss masks derive the step/scheduler horizon
from observed supervised-token density, while the live counter still stops on
actual supervised targets and may pass the requested budget by a final
microbatch.
The local model export now has a separate-process CPU inference example and a
regression check that loads both tokenizer and model and generates text; no
accelerator export claim is made.
T0–T11 remains the runtime architecture/evidence plan; the M0–M17
specification below remains historical.

> **Status: historical M0–M17 specification.** This document preserves the
> detailed PyTorch/XLA-era roadmap and evidence for traceability. Its milestone
> checkboxes and old XLA PR cards do not describe the active native TorchTPU
> queue. For current work, start at [the docs guide](README.md), then follow
> the [coding-agent guide](AI_AGENT_GUIDE.md), [T0–T11](TORCHTPU_REBASELINE.md),
> the [salvage inventory](TORCHTPU_SALVAGE_INVENTORY.md),
> and the [accepted T0.2 mapping](PR_DELIVERY_PLAN.md#t02-accepted-mapping-to-native-torchtpu-gates).

Historical status labels below refer to software and evidence in the backend
that existed at the time, primarily PyTorch/XLA. They do not imply native
TorchTPU support. “Implemented” means present in that historical code path;
“partial” means validation remained; “deferred” means it was not delivered.

**Current user-workflow update (2026-11-16):** merged `main` exposes
`run_pretraining(TrainLMPretrainingConfig(...))` for from-scratch training and
`run_finetuning(TrainLMFineTuningConfig(...))` for pretrained causal LMs. These
public APIs own preparation, training, evaluation, export, and run summaries.
The quick-start notebooks and Python examples are intentionally short: setup,
a few editable values, one API call, and the saved result. Longer schedules,
custom data, model options, and checkpoint recovery are documented in the
guides. The scratch notebook's 10,000-token default is a workflow smoke test
and explicitly selects the experimental native TorchTPU path. Fine-tuning's
quick example targets CPU/CUDA; native TPU fine-tuning is unvalidated. Neither
notebook establishes model quality, convergence, or TPU performance. See the
[coding-agent guide](AI_AGENT_GUIDE.md) for the example, implementation, test,
and evidence map. After scratch training, the notebook reloads the local
export with standard Transformers classes and shows a short CPU generation
preview; the smoke budget does not produce a useful model.

**Merged scratch-workflow update (PR #112, 2026-11-16):** the from-scratch API
accepts local text files or the pinned Hub source, and exposes the existing
`constant`, `linear`, `cosine`, and `wsd` schedules plus configurable warmup.
The notebook identifies its randomly initialized GPT-2 model and opens in
Colab or Kaggle. Reports record requested and effective schedule settings.
Budget extension remains limited to cosine with automatic warmup. Both hosted
CPU suites passed; PR #112 did not include TPU training.

**Merged runtime update (PR #113, 2026-11-16):** the owner’s Colab TPU run
failed at its first scheduled checkpoint because the native runtime queried
the wrong RNG API. TorchTPU exposes capture through
`torch.accelerator.random.get_rng_state()` and restore through
`torch.tpu.set_rng_state()`. The adapter now uses those paths with
compatibility fallbacks. A rerun on the owner’s TPU is still needed to validate
saving and resume; this fix does not imply TPU certification.

**Merged notebook runtime preflight update (PR #118, 2026-11-17):** before
native TPU training, the from-scratch notebook setup cell reports whether
`torch-tpu` and `libtpu` metadata are available and suggests selecting a runtime
that provides them if either is missing. It does not install packages or replace
the accelerator stack; TrainLM's preflight still checks compatibility. This
adds no TPU training evidence.

**Notebook UX guidance update (PR #119, 2026-11-17):** the scratch notebook
now points users to its setup-cell package report, and the native single-device
smoke explains how to use a blocked report to select a compatible TorchTPU
runtime. These wording changes do not alter runtime behavior or TPU evidence.

**Owner-run native pretraining and resume (2026-11-17):** the owner reports a
successful fresh `run_pretraining()` run on a Colab TPU previously identified
as v5e-1: 79 optimizer steps, 10,033 tokens against a 10,000-token target,
evaluation loss 9.237371, and an exported 8,325,120-parameter GPT-2. Setup
reported `torch-tpu==0.2.0` and `libtpu==0.0.50`. In the same live runtime, the
notebook then restored `checkpoint-79`, continued from step 80, and completed
at step 95 with 12,065 tokens against a 12,000-token target and evaluation
loss 9.243080. It wrote a separate resume report and model export. The owner
repeated restore from `checkpoint-79` after restarting the notebook session,
again completing steps 80–95. The report records source revision
`9593ff67f7afb744907c252813349e7a737e74fb`, Python 3.13.18, torch 2.14.1+cpu,
Transformers 5.22.1, TorchTPU 0.2.0, and libtpu 0.0.50. Discovery reported one
visible device named TPU v5e, world size 1, rank 0, fp32, and
`torch_compile_enabled=false`. Device properties are `available_unverified`,
property allocation was not run, and topology/HBM are null. This verifies one
eager fresh run and checkpoint recovery after a Python session restart with
the run files available. Recovery after VM replacement, exact
uninterrupted-trajectory parity, model quality, performance, compile, repeated
lifecycle, and distributed support remain unverified. TorchTPU warned that its
RNG restore API is experimental, but restoration completed.

**Progress-output follow-up:** the owner logs also showed progress events
without a materialized loss and repeated optimizer-step numbers. The callback
now suppresses those events and reports each step at most once. This is a
console-only fix. PR #130 rebase-merged at verified `main` SHA
`5f1ecd1bff5d2615a54b1007d43f5c0efabb0890`; both core-current and core-minimum
CPU CI profiles passed. No owner TPU rerun was performed, so this adds no TPU
execution evidence.

The notebook now records the installed Git commit in the run report and
prints it with runtime package versions and accelerator discovery fields as
JSON. Capture this output with the next v5e-1 resume attempt; it improves
provenance but adds no TPU execution evidence by itself.

The scratch notebook now separates the fresh run from an optional continuation
cell. Fresh training uses `FRESH_MAX_TOKENS=10_000` in a new default run
directory, and exposes `FRESH_MAX_TRAIN_EXAMPLES` for expanding its text cap
alongside a larger token target. `FRESH_MAX_TOKENS` maps to the config's
`max_tokens` value. After it succeeds, set `RUN_RESUME=True` and
run only the resume cell. The cell defines its own run directory, so an owner
can attempt recovery after restarting the Python kernel/process while keeping
the same VM, package versions, and full run directory. It validates the saved
report, restores its
config/checkpoint and adds 2,000 tokens to its saved target, so the same resume
step works after a larger fresh budget. It writes a separate resume
report/export using the supported cosine/automatic-
warmup extension. This preserves the baseline artifacts. The owner reports
that continuation succeeded both in the same live runtime and after a Python
session restart; recovery after VM replacement remains unverified.

The notebook's fresh-run cell now uses one `MODEL_SIZE` setting to choose a
randomly initialized GPT-2 shape: `tiny` (4 layers, width 256, 4 heads) or
`small` (6 layers, width 384, 6 heads). The default remains the approximately
8.3M-parameter tiny model and 10,000-token workflow smoke. This is a notebook
configuration convenience; Python callers continue to set model dimensions
directly, and no TPU fit or quality claim is implied for the larger preset.
The run setup also provides a `longer` profile (1,000,000 token target, 20,000
training-example cap, and 100-step logging) while keeping `smoke` as the
default. Resume extends the saved target by 20%, with a 2,000-token minimum,
so longer runs do not receive only a smoke-sized continuation. The profile is
an exploratory training budget, not a convergence or quality claim.

**Current public API update:** scratch pretraining accepts Transformers-style
`warmup_ratio` in the inclusive range 0–1 and rounds it up to a whole optimizer
step. It cannot be combined with `warmup_steps`. Leaving both unset preserves
the existing automatic warmup. Optional `save_steps` and `eval_steps` expose
independent optimizer-step cadence controls; unset values preserve the current
automatic cadence. Scratch runs can select the lowest `eval_loss` checkpoint
and stop after a configured validation plateau. The result exposes the selected
checkpoint and marks a valid early stop explicitly; TPU validation remains
outstanding.

The same config now exposes AdamW `betas` and `eps`, plus `max_grad_norm`
(`0` disables clipping). Requested and effective optimizer settings are stored
in the run report. Native TPU behavior remains experimental. The owner-run
fresh training and same-runtime plus kernel-restart resumes validate a small
eager single-device path; VM-replacement recovery and the other pending runtime
gates still need owner validation. The notebook follow-up keeps its existing presentation
contracts: at most five cells, at most 30 lines in a code cell, and the literal
10,000-token starter marker. Hosted CPU CI exposed these constraints during
review; notebook-only JSON and syntax checks also pass locally.

### API and notebook implementation history

The API implementation history follows. Its branch names, notebook controls,
and intermediate test counts are historical records rather than the current
user interface.

The fine-tuning API introduced a task-focused `run_finetuning` workflow for a
pretrained causal LM and separate train and validation splits. It loads the
model/tokenizer and either reads a pinned Hugging Face text-dataset revision
or accepts caller-provided Hugging Face `Dataset` objects. Dataset identity is
optional for caller-provided data; the Hub-loading path requires a pinned ID
and immutable revision. The API tokenizes one text field by default, trains,
evaluates, and exports locally. It
does not upload artifacts or create a validation split. Callers can provide
preloaded Hugging Face `Dataset` splits, a row formatter, tokenizer, and custom
collator for prompt formatting and label construction. Caller-provided data can
omit Hub identity. `TrainLMTrainingArguments.num_train_epochs` now provides a
finite map-style single-device epoch budget; it respects per-device batch size
and flushes partial gradient accumulation at epoch boundaries. Its summary
preserves the requested model revision and
records resolved model and tokenizer commits and whether they match when
available. The focused fine-tuning API, roadmap, and notebook-contract suite
passes 15 tests under Python 3.12.14, torch 2.9.0+cpu, and pytest 9.1.1. No
live Hub or TPU run was performed. The epoch-budget follow-up passes 116
focused API, lifecycle, config, trainer, and notebook-contract tests under the
same CPU environment; native TPU epoch training remains unverified.
The API follow-up added `eval_strategy` and `save_strategy` values
`"no"`, `"steps"`, and `"epoch"`. Epoch actions flush partial accumulation at
finite map-style dataset boundaries. Epoch scheduling requires a single device
and is not implemented in the legacy XLA coordinator. The focused CPU suite for
this change passed 151 API, lifecycle, config, engine, notebook, TPU coordinator,
and packed-scratch tests. No accelerator run was performed. At that point,
model loading was pinned to the tokenizer's resolved commit and a detected
mismatch stopped before training. HF-01 now retains that behavior only for a
shared source/revision contract; separately hosted tokenizers resolve without
coupling their commit IDs. Each successful run writes a JSON summary beside the export with
the arguments, data splits, revisions,
and metrics. Failed runs record the phase, error type, and last known
step/checkpoint before re-raising the original exception; the exception text is
not saved. Local sources may not expose commit metadata. PR #109 later merged
the API to `main`. Its follow-up added `load_best_model_at_end`, metric
direction, and `early_stopping_patience` with an improvement threshold.
Evaluation improvements force a same-step
checkpoint; local retention protects the selected checkpoint and newest save.
The patience counter is checkpointed and restored on resume. These features
currently require a single-device runtime and are not implemented by the legacy
XLA coordinator. The paired API, notebook, retention, and engine regression
suite passed 177 tests under Python 3.12.14, torch 2.9.0+cpu, and pytest 9.1.1,
including same-step saving, best-checkpoint retention, threshold behavior, and
patience resume. No TPU or live Hub run was performed. The focused CPU/API suite passes
73 tests in
the local Python
3.12.14, torch 2.9.0+cpu, pytest 9.1.1 environment, whose Transformers 5.0.0
is below the declared minimum; those tests mock Hub/model loading. A separate
one-step CPU fine-tuning run passed with a locally created GPT-2 model and
tokenizer under Transformers 5.20.2, using an in-memory dataset fixture.
Ten notebook/roadmap contract tests pass. Real Hub and TPU fine-tuning
remain unverified. `notebooks/TrainLM_Text_Fine_Tuning.ipynb` provides the API in
short, copyable Colab/Kaggle cells. It uses a tiny three-step smoke default,
creates a local model bundle, and installs TrainLM without replacing Torch.
Its source ref must be changed to `main` after merge before deleting the
feature branch. The fine-tuning summary and failure-report follow-up passes 17
focused API, model-provider, notebook, and roadmap tests.
The notebook now exposes a periodic save cadence and a
`RESUME_CHECKPOINT="latest"` control. It explains raising `MAX_STEPS` and
keeping the output directory on persistent storage for cross-session recovery.
The API's final-step checkpoint is available when the configured cadence falls
between the last periodic save and successful completion. The focused CPU/API,
checkpoint lifecycle, notebook, and roadmap suite for this follow-up passed 64
tests under Python 3.12.14, torch 2.9.0+cpu, and Transformers 5.20.2. This is
not Hub or TPU training evidence.
The follow-up adds public `restore_finetuning_config(summary_path, max_steps=...)`.
It restores model, data, and training settings from the fine-tuning summary,
anchors the run directory to the summary location, and rejects a target step
that does not exceed the recorded step. The notebook uses the helper for
`RESUME_CHECKPOINT="latest"` so users only provide the new step budget.
The focused API, checkpoint lifecycle, notebook, roadmap, and installed-wheel
suite passes 66 tests under Python 3.12.14, torch 2.9.0+cpu, and Transformers
5.20.2. Real Hub and TPU training remain unverified.
The notebook also handles missing or incompatible resume summaries with a
recovery message and skips training and bundling until the setup is corrected.
This prevents the previous model export from appearing as a successful resumed
run. The notebook and roadmap contract tests pass 10 checks.
The restore helper checks for a published checkpoint before the fine-tuning
notebook loads the model or dataset. A missing checkpoint prompts the user to
restore the full run directory or start a new run.
The 66-test API, checkpoint lifecycle, notebook, roadmap, and installed-wheel
suite now includes the missing-checkpoint guard. It passes under Python
3.12.14, torch 2.9.0+cpu, and Transformers 5.20.2; no Hub or TPU run was
performed.
Resume setup now rejects an unreadable local checkpoint archive and checks
native TPU manifest/shard publication before model or dataset loading. It does
not unpickle the checkpoint during this early check; model/state compatibility
is still checked by the trainer during restore. No Hub or TPU validation was
performed for this guard. The focused API, notebook, roadmap, and installed-wheel
suite passes 66 tests; the CPU/Gloo checkpoint-worker suite passes 20 tests with
command-scoped local socket permission.
The convenience helper now also accepts paired preloaded Hugging Face
`Dataset` splits with a row formatter, optional tokenizer, and custom collator.
Original columns are retained for custom label construction. The run report
records its input mode and formatter name; resume callers must provide those
runtime objects again. The focused suite covers both Hub-loaded and caller-owned
dataset paths; live Hub and TPU validation remain pending.
The package-root helpers are additive to the existing `api_version: "1"`
configuration schema; the compatibility contract requires the v1 exports while
allowing new public helpers without forcing a configuration-version change.
The report records requested/effective settings, optimizer/scheduler details,
selected backend, bounded runtime TPU discovery, model/tokenizer provenance,
package versions, training metrics, and `checkpoint.latest_path` when a
published checkpoint exists. Property-only discovery is labeled unverified
and does not allocate a device. The checkpoint path refers to the run
directory, so persistent storage is required to survive a reset.
When a checkpoint exists, `resume.code` captures all pretraining settings and
an executable example that is also displayed and bundled as `resume.py`.
The example targets the recorded checkpoint with the same token budget and
only retries a previously requested extension while its publication remains
pending. It checks for the original run files before constructing a trainer.
Restore the full run directory with matching source and package versions;
the model export ZIP alone cannot restore optimizer or packed-data state.
The notebook can load this report directly: keep the original `RUN_DIR`, set
`RESUME_FROM_REPORT = True`, and run all cells. It restores the recorded source
revision and pretraining config, checks packages and run files, and continues
through train/evaluate/export before refreshing the report. A package mismatch
prints expected and installed versions and stops before training; use a fresh
runtime with the matching accelerator/package profile instead of changing
Torch/TorchTPU packages in a live TPU session. Colab shows common token, data,
run-directory, and resume settings as form controls; Kaggle retains ordinary
Python assignments.

The from-scratch notebook now keeps its run in an editable stable directory and
passes an optional `RESUME_CHECKPOINT` value through the public trainer
API. `train(resume_from_checkpoint="latest")` resolves the newest published
CPU or native TPU checkpoint and rejects a missing checkpoint instead of
silently starting over. For Colab reset recovery, users can mount Drive and
point `RUN_DIR` to persistent storage; Kaggle users can retain `/kaggle/working`
outputs by saving a notebook version. CPU tests cover committed-checkpoint
selection and resume; native TPU checkpoint recovery remains unvalidated.
An explicit `extend_training_budget=True` option now permits resuming with a
strictly larger `max_tokens` value while requiring all model, data, optimizer,
precision, and runtime settings to match. After checkpoint validation, it
retains model and optimizer state, updates the run manifest, and starts a new
cosine schedule from the configured learning rate across the remaining steps.
This opt-in schedule restart can raise the learning rate; CPU tests cover the
transition, while native TPU checkpoint recovery remains unvalidated.
The public trainer and scratch config now accept `save_total_limit`: `None`
keeps all saves, while a positive limit keeps the new automatic checkpoint and
the newest other owned saves. The notebook selects three. Cleanup happens only
after checkpoint and run metadata publication; failed saves leave previous
checkpoints intact. Unmarked legacy checkpoints, incomplete saves, symlinks,
manual backups under separate names, and directories with extra files are
preserved. Retention is excluded from training fingerprints so existing scratch
checkpoints and changes to storage policy remain compatible. This is a
single-device CPU/CUDA/explicit native TPU feature; distributed/XLA coordinator
retention is rejected and TPU hardware validation remains pending. See the
[retention and resume evidence](AI_AGENT_GUIDE.md#evidence-and-limits).
When periodic checkpointing is enabled, a successful run also saves its final
optimizer step if the configured cadence missed it. The save occurs before
train-end callbacks; runs that fail do not publish this final save, and
`save_steps=None` keeps automatic saves disabled.
The public `trainer.latest_checkpoint` property returns the newest committed
checkpoint path or `None`, using the same lookup as
`train(resume_from_checkpoint="latest")`. The notebook prints the path after
training so users can copy it or select `latest` on the next run.
Matching pinned dataset/tokenizer settings now reuse the fingerprinted,
integrity-checked tokenizer and packed shards under the same output directory.
The scratch workflow also writes `run_config.json` and embeds its full
fingerprint and configuration in each checkpoint. Resume rejects changed model,
tokenizer, packed data, optimizer, schedule, precision, or runtime settings
before restoring training state, except when the opt-in budget-extension path
validates a larger token target and its derived schedule horizon. Other changed
settings require a fresh output directory; older scratch checkpoints without
this record cannot be resumed through the guarded scratch API. Generic legacy
checkpoint behavior is unchanged. A cache hit avoids importing `datasets` or
reopening the Hugging Face stream.

PR #85 added the integrity-checked preparation cache and was rebase-merged to
verified `main` at `476bc68ff535cd49f3472316c5bc809bc38e7ae5`.
PR #86 added whole-run compatibility checks to scratch checkpoints and was
rebase-merged at verified `main` SHA
`32776292a7e6436a0b2a02fb2232ce0185d1b245` from head
`15670a2856affe641488f4b48addc9b6cf1cd52b`. Both GitHub CI profiles passed.
Focused API, lifecycle, and roadmap tests pass (50 tests);
the CPU/Gloo checkpoint and token-weighting suites pass with per-command network
permission (22 tests). The full local run reports 885 passed and 13 failures;
six loopback-dependent failures passed in the permitted rerun, while the
remaining environment baseline includes missing `pip`, Transformers 5.0 Phi
fixtures lacking `pad_token_id`, and an existing model reload mismatch. No TPU
training was run.

PR #88 added opt-in scratch budget extension and was rebase-merged at verified
`main` SHA `ab11f59fc362684c3ba6da5111d7a30d2d77f8f2`. Both dependency CI
profiles passed after restoring the notebook's Google Drive instructions. The
retention work builds on that verified base; it adds no accelerator evidence.
PR #89 then added configurable checkpoint retention and rebase-merged at
verified `main` SHA `a45432dfbee58713ee647aec8d3860dfd4fd885c`; both CI profiles
passed 924 tests and 41 subtests. PR #90 added final-step checkpoint saving
and rebase-merged at verified `main` SHA
`5bab4b5c62ffe014b2700d2ea7157bb910d9f621`; both CI profiles passed 928 tests
and 41 subtests. PR #91 exposed the public latest-checkpoint path and
rebase-merged at verified `main` SHA
`2a68f781655d80ccbac91ed03e582e32539c0900`; both CI profiles passed 928 tests
and 41 subtests. PR #92 added the checkpoint path to report schema v5 and
rebase-merged at verified `main` SHA
`16984722897d6cf2b1f25b8548c4a1c6039ce94d`; both dependency CI profiles passed.
PR #93 added the schema-v6 `resume.py` artifact and rebase-merged at verified
`main` SHA `4de50666874592866fc50debd4261e465ffdaabe`; both profiles passed
940 tests and 41 subtests. Its focused CPU suite passed 47 tests and 14
subtests. PR #94 added direct report-driven notebook recovery and rebase-merged
at verified `main` SHA `4ce8deec5e61075c8c3c294584ae691f3253554b` from head
`716625c87b77f46566a582cde7fcd636f8e487fb`; both CI profiles passed. PR #95
clarified package-mismatch recovery guidance and added executable CPU coverage.
It rebase-merged at verified `main` SHA
`4b986a7e296cb77a12091bc691fe67c13a322068` from head
`62261e6f0e8aca40dfa4a9dcceddd7a1d1ee6d8f`; its focused notebook/recovery/
roadmap suite passed 39 tests and 14 subtests locally, and both CI profiles
passed. No TPU validation is claimed. PR #96 corrected the post-merge status
record and rebase-merged at verified `main` SHA
`2e154afd90df9f65b29c394b045ad06b6a6c6259` from head
`d87c195af6a23beaece94189aac1b19c1df36491`; both CI profiles passed. PR #97
added the one-call run API and simplified the notebook, then rebase-merged at
verified `main` SHA `1828e6ad079057fd775e00b3ea11683ba7c237c6` from head
`553f602d426924a70f9b3876f05d4935040d4a9e`. Both CI dependency profiles passed;
the focused local CPU suite passed 55 tests and 14 subtests. No TPU training
or hardware evidence is claimed.

The notebook uses this API; native TPU remains experimental and gains no
end-to-end evidence from CPU/API tests. PR #81 added tokenizer and runtime
provenance to report schema v3 and was rebase-merged to `main` on 2026-11-03
at `4cfacae39e4e677f442208dfa098bb51009cfd46`. PR #77 was rebase-merged to verified `main` at SHA
`f844d048b17640d4d71a127c4a860dd1f6d7d52c`. The rebase-preserved API commit
is `e340d7f604151ee869243d80bc5429c1bb008801`; the post-train evaluation fix
is `3fbac2a19209dfd035d021f7911f2fa0b4a2f91b`. PR #79 added configured
Transformers causal-LM architectures and rebase-merged to verified `main` at
`ba5f4761c35c23403d35e44be5d564dbe7ffadb3`; CI passed 891 tests and 41
subtests. This is CPU software evidence only. The feature branch was deleted
after verification. PR #80 added model provenance to report schema v2 and
rebase-merged at verified `main` SHA
`e9b4a1bde4869d89c323ed457a6b6cebccb0ae40`; CI passed 891 tests and 41
subtests. CI run 360 first found four failures; final run 362 passed
890 tests and 41 subtests on Python 3.11. Local syntax, notebook, config-guard,
state-transition, and CLI-help checks pass. PR #82 added the effective run
context to report schema v4 and rebase-merged at verified `main` SHA
`4107f2f2ab86c2e90d43428ad1a779a25fe86cac`; GitHub pytest passed (run
`41746021087`). No TPU training was run. See the
[coding-agent guide](AI_AGENT_GUIDE.md) for the entry point and limits.

- **Historical product:** Hugging Face-native dense autoregressive training
- **Historical TPU backend:** PyTorch/XLA
- **Current target:** Native TorchTPU on Cloud TPU; one small v5e-1 workflow is
  owner-reported, while production support remains unverified
- **Current gate:** T0 is closed; PR #20 merged the T0.2 mapping. PR #28 merged
  the native single-device software foundation and starter pretraining notebook
  with merge commit `860bb289bf35444d8d435cf41caf51bcf77afaaf`. PR #29
  rebase-merged streaming packed data at
  `f3df754ea98156b083801447b41540670134bf2e`; PR #30 rebase-merged one-update
  packed causal-LM reference parity at
  `2b5e0e9bef3b105d8ebeb40a0a9d04d1d4b7e748`; PR #31 rebase-merged the
  random-init packed train/eval/checkpoint/export CPU lifecycle at
  `a58a9e13d792c2468f4c20d6cdc7375f4bc63ffd`; PR #32 rebase-merged exact
  packed-data checkpoint recovery at verified `main`
  `82ead5b72991fc154738bcd8334e477d563d7ea3` after passing CI; PR #33
  rebase-merged Colab notebook source fixes at verified `main`
  `83d4b08dd9c510a6edf6ae0a5dc7e2b739343eb6`; PR #34 rebase-merged the
  training-split BPE tokenizer at verified `main`
  `7c75f6d0aa8ff75299328847e9fa7dcfcdf791e1`. The active
  `063a8b9ff9b42733f8b9f4720ba3307c2b2f4cdf` after PR #35 added a local GPT-2
  config; PR #36 added pinned custom corpus selection at verified `main`
  `d075987574fe471d1b9159315a0688416664f2ba`; PR #37 added opt-in BF16 at
  `31aa3acd06bf89c9430615103ebd4dd0da46ae6c`; PR #38 added optimizer settings
  at `357cac307bfaf94b4bcf4c55ed53c9fb99815cc8`; PR #39 added activation
  checkpointing at `df0a9ff562e48b67ae4ea48dbe10abe565137f4b`. PR #40 merged
  per-device batch sizes at `d7359bc20d5fa2af800c113b2e85779b6987f2f4`; PR #41
  merged supervised-token budgets at
  `a24514a7693e918c1c10c8a84c705399fa3b3bcf`, with CI passing. PR #42
  rebase-merged configurable sparse logging and terminal loss snapshots at
  `2e90537371aa03c98f4fc395edb2b231727eaebb`, with GitHub pytest passing in
  2m0s. PR #43 packages the exported Hugging Face model/tokenizer and evidence
  report for Colab/Kaggle download at `73952a9724456390c6def5b74a474d4274dc7544`;
  GitHub pytest passed in 2m4s. PR #44 merged a CPU reload/generation check at
  `fad1df41278d4ea904b4d26968d75ee97ae17923`; the report records that phase
  separately from TPU validation. PR #45 rebase-merged agent-discoverability
  documentation at verified `main` SHA
  `d983408221b92d631aea817bee7539e07de1bbf5`; it adds no accelerator
  evidence. PR #47 aligned current status pages at verified `main` SHA
  `e109ad253754303e44eb6162aacb046c415b990b`; it adds no runtime behavior.
  PR #48 rebase-merged distributed token correctness at verified `main` SHA
  `7ae837d085fbbaab9f6c2d3617ff129b89dd454e`; its full Python 3.11 GitHub CI
  run passed.
  PR #49 rebase-merged T5 checkpoint integrity at verified `main` SHA
  `01a71f990fc5578b77be6e2e0a84473486fbe488`; CI runs #296 and #297 passed.
  PR #50 rebase-merged public trainer checkpoint preflight at verified `main`
  SHA `f57d6d71339753a6a3cee4082a43dfa2f5113017`; CI run #299 passed.
  CPU/Gloo regressions and backend-gated logic now cover global valid-token
  gradient weighting, synchronized distributed `max_tokens` stopping, and
  global loss/progress reporting. The focused CPU/Gloo and related
  trainer/runtime/API suite passed locally in an isolated Python 3.12.14,
  torch 2.9.0+cpu, pytest 9.2.0 environment (79 tests); PR #48's GitHub CI
  passed. T5 rank-sharded checkpoints now use schema-v3 shard digests and
  coordinated CPU/Gloo write/resume preflight; native TPU validation remains
  unverified. Public trainer checkpoint preflight is merged. PR #51
  rebase-merged at verified `main` SHA
  `c63e6903767f3a2aab2ff733e6248ff0c4e4bd3b`; CI run #302 passed. It records
  optimizer/scheduler class identity and checks optimizer group compatibility
  before state restore; old schema-v1 checkpoints without type metadata remain
  accepted. PR #52 rebase-merged at verified `main` SHA
  `9cf8d4ba0de0b91cc52908d9d98f984b29873a33`; CI run #304 passed. It validates
  optimizer slot shapes and scalar steps plus scheduler payload invariants
  before restore. PR #53 rebase-merged at verified `main` SHA
  `8ea810c2e5e54f492cb4e29e0844a74c9935f23e`; CI run #306 passed. It validates
  saved progress counters and metrics before mutation. PR #54 rebase-merged at
  verified `main` SHA `6ab2b8c9fcde5895c71d5f8450808b7f5756c520`; CI run #308
  passed. It preflights packed-data cursor identity before trainer restore.
  PR #55 rebase-merged at verified `main` SHA
  `ea9dbadfb86f17cd3713b2abcee808677c38e845`; GitHub CI passed (run ID
  `41578130077`). It rejects transient trainer fields in saved checkpoint
  state. PR #56 rebase-merged at verified `main` SHA
  `4aa3614858065c6255c4ecc2d0949c24ee715f70`; GitHub pytest passed in 2m24s.
  It documents and tests pre-restore validation for custom stateful datasets.
  PR #57 rebase-merged at verified `main` SHA
  `e62052ae39d6216bce0dd5aec86fc42d9f10a837`; GitHub pytest passed (run ID
  `41579944075`). Its custom iterable continuation test requires exact equality
  between resumed and uninterrupted model weights. PR #58 rebase-merged at
  verified `main` SHA `e90697f99db29355e135161ea62e66d91bbc6047`; GitHub pytest
  passed (run ID `41581911584`). It rejects a mismatched custom dataset ID or
  out-of-range cursor before state mutation. PR #59 rebase-merged at verified
  `main` SHA `2985ef953546506f3cd6d84c8ff79739c6388a9b`; GitHub pytest passed
  (run ID `41582739096`). It proves legacy stateful custom iterables still
  resume without the optional `validate_state_dict` hook. PR #60 rebase-merged
  at verified `main` SHA `839d8fcc1f9320ff6193b7c54399c629881945f0`; GitHub
  pytest passed (run ID `41585383984`). It removes optional dataset cursor,
  loader RNG and optimizer/scheduler type fields from a schema-v1 checkpoint
  and verifies exact resume parity. Its combined checkpoint/API lifecycle
  suites passed (39 tests). PR #61 rebase-merged status-document updates at
  `448ea9dc3d5969aa88d612bdfbbf11af682bc78d`; GitHub pytest passed (run ID
  `41585945140`). PR #62 rebase-merged at verified `main` SHA
  `821beacdee235e9ed6e70781023126023120adef`; GitHub pytest passed (run ID
  `41594252050`). Its CPU regression rejects a RandomSampler checkpoint
  without saved loader RNG before state mutation. The combined checkpoint/API
  lifecycle suites passed (40 tests). The last full local suite had 865 passed,
  7 failed and 41 subtests:
  missing `pip` blocked wheel smoke, five Phi cases hit a Transformers 5.0.0
  config issue, and a TrainLM model reload test had mismatched weights. No full-suite rerun or
  TPU checkpoint validation is claimed for PR #62.
  PR #63 rebase-merged the status correction at verified `main` SHA
  `97a061df7bc96fc311d1bab341d91f92c8a43b61`; GitHub pytest passed (run ID
  `41594915207`). PR #64 rebase-merged at verified `main` SHA
  `dd3cf7bcea791f370ea0c58d3e863a1702dfc79f`; GitHub pytest passed (run ID
  `41595632995`). Its CPU regression supplies a SequentialSampler DataLoader
  without TrainLM-owned loader RNG, removes the optional checkpoint field, and
  verifies exact weights after resuming from an intermediate checkpoint. The
  checkpoint/API lifecycle suites passed (41 tests; 44 including the roadmap
  contract). No full-suite or TPU validation is claimed for PR #64.
  PR #65 rebase-merged the status update at verified `main` SHA
  `908e21fa2fedd34a0de47203a0f12c5d4a473ef3`; GitHub pytest passed (run ID
  `41596201261`). PR #66 rebase-merged at verified `main` SHA
  `4a877768a1c8fcaae654be6c18e88ce374bdce25`; GitHub pytest passed (run ID
  `41596764595`). Its CPU regression covers both within-epoch resume and a
  checkpoint exactly after the four-row sequential loader is exhausted;
  resumed weights match uninterrupted training in both cases. The checkpoint/API
  suites passed 42 tests (45 including the roadmap contract). No full-suite or
  TPU validation is claimed for PR #66.
  PR #67 rebase-merged the status update at verified `main` SHA
  `74ba1d1d8c05a782b29b18c31131347025b7fcef`; GitHub pytest passed (run ID
  `41597332118`). PR #68 rebase-merged at verified `main` SHA
  `f52ad5fbed7ab51335ff04079fafbd5939069ece`; GitHub pytest passed (run ID
  `41631351981`). Commit `728e787` adds a six-row sequential-loader case with
  batch size two and gradient accumulation two. Its step-one checkpoint records
  two consumed microbatches; resumed weights match uninterrupted training
  through the next epoch rollover. The checkpoint/API suites passed 43 tests
  (46 including the roadmap contract). No full-suite or TPU validation is
  claimed for PR #68.
  PR #69 rebase-merged the status update at verified `main` SHA
  `bf57ebb20600eb69f4ea5f1666d427cf610d3870`; GitHub pytest passed (run ID
  `41632033098`). PR #70 rebase-merged at verified `main` SHA
  `2f0eafcbeb6d394496154fdddc22a55233aee5d0`; GitHub pytest passed (run ID
  `41632939362`). Commit `2383d35` covers a five-row sequential loader with
  batch size two, `drop_last=False`, and gradient accumulation two. Resume
  consumes the one-row tail and the following epoch's first batch exactly; final
  weights match uninterrupted training. The checkpoint/API suites passed 44
  tests (47 including the roadmap contract). No full-suite or TPU validation is
  claimed for PR #70.
  PR #71 rebase-merged the documentation status correction at verified `main`
  SHA `33094937a78e9a773ba8dbdd2007f73b10ec0a20`; GitHub pytest passed (run ID
  `41633758016`). PR #72 rebase-merged at verified `main` SHA
  `df9f232dff191354b976b0059a2a174777792a5f`; GitHub pytest passed (run ID
  `41634659366`). Commit `df9f232` adds exact-resume coverage for the default
  TrainLM-owned shuffled sampler. It checkpoints after the first update,
  resumes across an epoch boundary, and matches uninterrupted model weights
  through step three. The focused API lifecycle, trainer, and roadmap-contract
  suites passed 57 tests. No full-suite or TPU validation is claimed for PR #72.
  PR #73 rebase-merged the documentation status correction at verified `main`
  SHA `6090ba72e32d9d920499807f0c7b5a15e8ae5861`; GitHub pytest passed (run ID
  `41635317392`). PR #74 rebase-merged at verified `main` SHA
  `b0f81e91ad6c11aef865441efc05455202557677`; GitHub pytest passed (run ID
  `41636368557`). Commit `b0f81e9` extends default shuffled-sampler exact-resume
  coverage to checkpoints both within an epoch and exactly after all four
  batches are consumed (`micro_step=4`). Both resumes reach step three with
  exact model-weight parity. The focused API lifecycle, trainer, and
  roadmap-contract suites passed 58 tests. No full-suite or TPU validation is
  claimed for PR #74.
  BF16 and checkpointing remain unvalidated on TPU. T1 source-level
  probes and the explicit `native_tpu` path are implemented; owner-run compile,
  lifecycle and v5e-8 collective evidence remain pending. Hosted Kaggle currently documents
  v5e-8 and hosted Colab TPU v2, so neither is the v5e-1 target; use a
  one-chip Cloud TPU v5e VM. Use v5e-8 for multi-device checks.
  The [v5e-1 probe notebook (open in Colab)](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TorchTPU_T1_v5e1_Probe.ipynb) and
  [run guide](T1_V5E1_PROBE.md) have an owner-reported eager TPU pass:
  allocation, forward, backward and AdamW update succeeded. Hardware generation
  and explicit compile remain unverified. The current [utility inventory](TORCHTPU_UTILITIES.md)
  adds shared discovery/reporting, dependency/lifecycle verification and
  distributed entry checks as T1.1–T1.4. T1.2 now has a source-level report
  for TPU availability and properties, used by the standalone probe. A
  provisional single-device runtime is available only through explicit
  `native_tpu`; production validation and auto-routing remain pending. T1.3 now has a
  read-only metadata/profile check (source and TorchTPU requirements plus
  `pip check`), bounded probe execution, and an opt-in repeat-process lifecycle
  harness in the notebook; clean-install and accelerator lifecycle results
  remain owner-run work.
  T1.4 now has an opt-in single-host v5e-8 launch/collective probe with
  topology and rank reconciliation; only its CPU-side contracts have run.
  The actual multi-device result remains owner-run evidence.
  A guarded trainer preflight now exposes non-allocating native TPU discovery
  through `preflight()` and opt-in `explain(include_hardware=True)`. `auto`
  rejects a detected native TPU rather than silently selecting CPU while the
  native training route is unverified; explicit `tpu` remains the XLA path.
  The native single-device probe can now persist a bounded device-owned report;
  trainer diagnostics accept such a report from a future native worker result.
  The current XLA worker does not emit native evidence.
  Both T1 notebooks now offer direct Colab entry and a downloadable evidence
  JSON tied to a fresh source SHA. The v5e-1 notebook runs the repository CPU
  suite before the TPU probe and records it as a separate conformance result;
  it installs no packages and a CPU pass does not certify TPU support. The
  owner run at source `08f02fb` had 746 passes and 17 failures. Two facade tests
  selected auto on a TPU host, the wheel smoke required `venv`, and a roadmap
  check used a stale heading; these harness failures are corrected locally.
  Twelve loss-layout cases were caused by ignored labels in the mean-loss
  denominator; that is fixed in the current tree. The scalar test now counts
  both loss and gradient-norm snapshots at the logging cadence. This 746/17
  result is historical; the new minimum/current CI matrix is the next full CPU
  conformance check for the expanded Torch range.
  Process
  cleanup and fatal teardown checks
  are covered on CPU; the export preserves skipped checks and package
  mismatches. Owner-run hardware evidence still gates the dependency profile
  decision and T2 integration scope.
  The latest owner eager run at source `d3dea896` passed allocation, forward,
  autograd and AdamW update with a runtime-reported `TPU v5e` device and one
  visible device. The owner confirmed the hardware was v5e-1; the shared Colab
  output did not show image or topology details. The package profile correctly
  reports `source_checkout`. Its captured Torch 2.14.1 mismatch predates the
  current `>=2.9,<2.15` core contract. IPython 7.34.0's missing Jedi remains a
  visible but nonblocking environment warning when fully parsed; training-stack
  or unclassified conflicts still block. An opt-in `torch_compile=True`
  argument now routes the native runtime through `torch.compile`; it defaults
  off, is limited to `accelerator="native_tpu"`, and resume rejects a changed
  compile mode. This software path has not been validated on TPU hardware.
  The opt-in compile probe now compares each step's output, loss, gradients,
  and AdamW update against an eager copy of the same tiny model, using the
  same inputs. A compiled pass requires this numerical check; hardware evidence
  remains pending.
  The bounded lifecycle harness can now repeat either eager or compiled smoke
  processes, and the notebook records which mode it ran. Eager/compiled
  lifecycle and T1.4 collective remain unrun. The expanded core range has a
  constrained CPU CI matrix pending; the native TorchTPU package profile
  remains unverified pending owner runtime and trainer evidence.
  Software follow-up adds `trainlm doctor` as one read-only report for package
  compatibility and native TPU discovery. It does not install packages or
  train; dependency and hardware support remain evidence-gated.
  Installed wheels use their declared package requirements; editable/source
  installs compare against the checked-out project files. The v5e-1 notebook
  invokes it directly from its source snapshot, without an installation step;
  a missing installed `trainlm` distribution is identified as source-checkout
  mode rather than a dependency gap. Other package incompatibilities remain
  visible. Pip conflict details are parsed from both common `pip check` output
  forms; the latest owner report identifies the IPython-to-Jedi conflict. The
  compatibility PR #83 aligns core-current to Torch 2.14.1, raises the
  Transformers minimum to 5.15.0 and Hub minimum to 1.5.0, and adds
  minimum/current CPU CI. The first current-profile run passed; the initial
  Transformers 5.0 minimum run had 887 passes and six failures. Its first retry
  exposed the old Hub lock, so commits `bbf1b2e` and `cfc05ed` align the minimum
  package pair. CI run `41752008510` then passed 893 tests and 41 subtests in
  both minimum and current profiles. No refreshed owner package report is
  claimed in this historical CI entry; the later owner-run pretraining result
  below passed package preflight in the Colab runtime. A separate clean-install
  profile remains unverified.
  The 18 old PR cards are candidates, not the active queue.

- **Current implementation continuation (experimental):** the owner asked to
  advance software while TPU hardware slots are delayed. The public API now
  has an explicit `accelerator="native_tpu"` path for one visible TPU device,
  backed by the existing backend-neutral PyTorch trainer. The legacy
  `accelerator="tpu"` XLA coordinator is preserved; `auto` still refuses a
  detected TPU. A Colab-openable, copyable notebook exercises a tiny real HF
  causal LM fine-tune and export. The later owner run below confirms one small
  native pretraining workflow, but does not establish package behavior across
  runtimes, compile, resume, repeated lifecycle, or multi-device support. Native
  checkpoint state now includes an optional device RNG capture/restore contract;
  runtimes missing those APIs fail checkpoint/resume rather than implying
  exact continuation. PR #32 adds exact zero-worker packed-source recovery
  using a fingerprinted dataset cursor; worker prefetch state remains
  unsupported. Native model
  compilation is exposed as `torch_compile=True`; it has no eager fallback,
  is checkpoint-mode guarded, and remains disabled pending a v5e-1 owner run.
  The lifecycle harness also supports a compiled repeat pass after the one-shot
  compile check succeeds; each fresh child must emit compile evidence, and the
  normalized report records the selected mode. This harness extension has only
  source-level validation so far.
  The native training notebook now offers an optional resume comparison:
  save after three steps, resume a fresh trainer for step four, verify
  model/tokenizer export, then train an uninterrupted four-step reference.
  A constant learning-rate schedule and deterministic data keep the paths
  comparable; every final model tensor must match exactly. The eight total
  optimizer steps are opt-in. No TPU result or full resume certification is
  claimed until an owner run supplies evidence.
  A CPU regression now compares the same three-step save and one-step resume
  against an uninterrupted four-step run with identical initial weights,
  deterministic data order, and a constant learning rate. It checks that the
  fourth step changes weights and that all final tensors match exactly; the
  target TPU comparison remains owner-run.
  The notebook now launches its native trainer smoke in a child process with
  a 900-second eager or 1800-second compile/resume limit. The existing process
  harness reaps the process group on timeout or interruption, rejects fatal
  teardown markers, and keeps raw logs in a local session directory. The
  parent notebook does not initialize TorchTPU. It now writes a downloadable
  JSON report with source revision, package versions, normalized device facts,
  and phase outcomes; process exit, child summary, and phase markers must agree
  before the report passes. Raw logs remain local. At the time of this harness
  change, the owner had not yet run the native trainer smoke; later evidence is
  recorded below.
  The notebook now runs the read-only T1 package checker before downloading
  `distilgpt2`. It blocks missing or conflicting training-stack dependencies
  and preserves a downloadable blocked report. The earlier Colab Torch 2.14
  source-range mismatch is addressed by the current core requirement; a fully
  parsed unrelated IPython/Jedi conflict remains visible but nonblocking.
  Training-stack and unclassified conflicts still block. This gate had
  source-level tests; the later owner run passed this package preflight in one
  Colab runtime, while a clean-install profile remains pending.
  The trainer smoke pins `distilbert/distilgpt2` and its tokenizer to Hub commit
  `2290a62682d06624634c1f46a6ad5be0f47f38aa`; the downloadable evidence records
  that model revision for repeatable runs. No TPU trainer result is claimed.
  A separate Colab notebook trains random model weights and a byte-level BPE
  tokenizer from the pinned TinyStories training split, then evaluates on the
  separate validation split. PR #35 builds its GPT-2 config locally from
  explicit dimensions; PR #36 allows other pinned HF text datasets; PR #37
  exposes opt-in BF16; PR #38 adds optimizer schedule and seed controls; PR #39
  adds opt-in activation checkpointing; PR #40 adds per-device batch sizes; PR
  #41 adds supervised-token targets and reports actual consumption; PR #42
  exposes logging cadence and preserves the final loss snapshot; PR #43 bundles
  the model export and evidence report for download. The current notebook
  iteration verifies that export reloads and generates tokens on CPU. This is
  not a TPU generation or convergence result. The
  default recipe targets 1,000,000 next-token targets, with gradient
  accumulation, periodic checkpoints, evaluation, and HF export. It checks for
  the optional `datasets` package before fetching data. This is a small-scale
  pretraining implementation; native TPU execution and convergence remain
  unverified. PR #28's final CI run passed all 804 tests after a follow-up
  corrected two native-runtime test fixtures; the PR was merged with the SHA
  above. PR #29's CI passed and its T3 data implementation is merged: pinned
  streaming text is written to validated uint32 packed shards and fed through
  `PackedBinDataset`. PR #30's GitHub CI verified one public-trainer update
  against direct causal-LM reference loss, clipping, gradients and parameter
  updates. PR #31's CI verified the random-init train/eval/checkpoint/export
  path; PR #32 verified exact packed-data continuation and stale-data rejection;
  PR #33 verified notebook source freshness; PR #34 verified train-only BPE
  fitting and reporting; PR #35 verified local model config and report
  provenance. The owner later reported one successful small v5e-1 training
  workflow; resume, larger-run convergence, and performance remain pending.
  Continue only behind explicit experimental selection until CPU and owner-run
  TPU checks pass.
  The public from-scratch API now checks installed metadata for core and
  pretraining dependencies before reading or preparing the corpus. Explicit
  `native_tpu` runs additionally check TorchTPU's installed distribution and
  declared requirements. Merged PR #121 makes a blocked `run_pretraining()`
  call preserve its metadata report and summarize each missing or mismatched
  package in `TrainLMPretrainingError`; the API gate itself never installs or
  changes packages. Both GitHub CPU dependency profiles passed. A later owner
  run passed package preflight and completed a 10,000-token training smoke on
  the Colab TPU previously identified as v5e-1; exact source SHA and full
  package report were not included in the pasted output. A clean-install
  profile remains unverified. The local
  `.venv` reports incompatible Hub/Transformers versions and missing `datasets`,
  so real local scratch training remains unavailable. The focused CPU/API
  suite passes with the package profile mocked for training-path tests and
  isolated tests verifying mismatch handling. This metadata gate does not
  certify general TPU support or convergence; the 10,000-token result is only
  a smoke. See the [profile implementation](../src/trainlm/runtime/pretraining_profile.py),
  [profile tests](../tests/test_pretraining_profile.py), [API tests](../tests/test_pretraining_api.py),
  and the
  [scratch notebook](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb).
  The notebook setup installs only missing `datasets`/`tokenizers` extras;
  incompatible installed versions stop with a clear error, and PyTorch/TorchTPU
  are not installed or replaced by this setup step.
  `TrainLMPretrainingConfig.gradient_checkpointing` is an opt-in memory control
  for Hugging Face models with the standard checkpointing hook. Training turns
  KV cache off and model export restores the original setting. Its setting is
  included in run fingerprints when enabled; the false default is omitted to
  keep older run manifests resumable. A tiny CPU training/export test covers
  the behavior. The notebook exposes it as an advanced option; TPU memory and
  performance effects remain unverified.

## Historical native-backend transition note

The [T0–T11 plan](TORCHTPU_REBASELINE.md) is the current target architecture.
Existing M0–M14 work and measurements are salvage material, not TorchTPU
certification. PR #18 is merged, T0.1 inventory is complete, and PR #20
merged the T0.2 mapping. The owner approved v5e-1 for single-device T1
feasibility, with v5e-8 reserved for checks that require multiple devices.
The old XLA cards are still not an approved implementation queue.

## Historical M0–M17 product goal and acceptance gates

TrainLM will load an ordinary dense autoregressive Hugging Face
`AutoModelForCausalLM`, train it efficiently on Google TPUs through TrainLM's
own engine, and export a checkpoint loadable by plain Transformers.

V1 makes two independent promises:

1. **Universal dense-AR compatibility:** every in-scope model trains,
   evaluates, checkpoints, resumes, and exports through the generic path
   without a family adapter.
2. **Certified TPU optimization:** advertised optimized models pass semantic,
   graph, HBM, throughput, MFU, resume, and export gates. The exact 135M
   reference must reach at least 90% of LaughLM throughput; 95% is preferred.

The normative model and support definition is in [`SCOPE.md`](SCOPE.md).
Merely executing is not optimization, and making only Llama-style models fast
is not a successful framework.

## Historical roadmap operating rules

Status flags: `[ ]` not started, `[~]` in progress or awaiting validation,
`[x]` complete, and `[d]` intentionally deferred.

- Only one **new story PR** may be active at a time. Legacy PR #18 is merged; complete the post-merge audit before starting the first numbered story branch.
- Every progress change must update the milestone overview, PR overview, and
  detailed milestone status in the same commit.
- Every future **PR** is one cohesive deliverable containing multiple small,
  traceable implementation, test, fix, regression and documentation commits.
  Maintain multiple sequential PRs across the roadmap, each based on the
  verified `main` produced by the immediately preceding merged PR.
- Do not squash implementation PRs; feature commits are review and rollback
  boundaries.
- TPU-dependent validation is performed on the target TPU environment, not the
  local Windows development machine.
- An optimization defaults on only after semantic and hardware evidence.
- A completed milestone gate cannot be weakened without a decision record.

Every implementation change includes a focused diff, static review, relevant
non-TPU tests, an exact TPU command when required, saved evidence, and a keep,
revise, or revert decision.

## Historical XLA-era user experience and acceptance boundary

The user-facing workflow must remain Hugging Face-like: load a model or HF ID,
supply a tokenizer/processing class and train/eval datasets, set familiar
training arguments, call `TrainLMTrainer.train()`, and receive understandable
progress, evaluation, checkpoints, resume and portable `save_pretrained` export.
An optional `trainlm train` CLI/config entry point must use the same execution
contracts. No user should manually launch TPU workers, set PJRT ranks, parse
stage logs, construct manifests for ordinary Hub-packed sources, or manage
internal caches. `trainer.explain()` must expose active providers and honest
fallbacks without leaking internal orchestration into the public API.

**Constraints:** Preserve HF model semantics and backward compatibility,
train on CPU in Codex only, keep TPU/CUDA certification tied to actual owner-run
hardware evidence, never silently fall back when strict optimization was
requested, keep transforms reversible, and never claim unmeasured speedups.
For each story, write a user-visible behavior or justified internal benefit,
a demonstrable acceptance test, explicit non-goals, and rollback criteria.

## Non-negotiable contracts

### Hugging Face remains the public model contract

- Load through `AutoConfig` and `AutoModelForCausalLM` from config or pretrained
  weights.
- Do not edit installed Transformers source.
- Apply reversible in-memory transformations before optimizer construction.
- Export canonical HF state loadable without TrainLM.
- Default to `trust_remote_code=False`; remote code is best-effort until
  explicitly certified.

### TrainLM owns the performance-sensitive engine

- `TrainLMTrainer`, not HF `Trainer`, owns optimized training.
- TrainLM owns task semantics, data, loss, accumulation, runtime, checkpointing,
  telemetry, and optimization planning.
- No tensor-to-host synchronization is allowed in the hot path.
- HF Trainer interoperability may exist but carries no TPU parity guarantee.

### Optimization is capability-driven

- The core accepts generic `torch.nn.Module` and HF contracts.
- Optional adapters map family structure to reusable capabilities; adapters are
  never required for generic training.
- Unsupported optimizations fall back explicitly in compatibility mode and fail
  before allocation in strict mode.
- `trainer.explain()` reports capabilities, providers, transformations,
  fallbacks, checkpoint conversions, and certification.

### Runtime backends are replaceable

```text
HF model -> capability plan -> prepared model -> TrainLMTrainer
                                                   |
                                            ExecutionBackend
                                         /        |         \
                                       CPU      CUDA    PyTorch/XLA
                                                             |
                                                         TorchTPU later
```

Core trainer, task, data, and planner code must not import `torch_xla`.
Compilation, device placement, sharding, synchronization, checkpoint I/O,
profiling, and backend kernels live behind interfaces.

### Support levels are public

| Level | Meaning |
|---|---|
| Compatible | Correct generic training and plain-HF round trip |
| Optimized | Compatible structural optimizations are active |
| Certified | Correctness, HBM, graph stability, throughput, and MFU passed on hardware |

Only Certified may be called fully supported or optimally trainable.

## Dense-AR V1 scope

V1 covers decoder-only causal Transformers with MHA/GQA/MQA, full causal and
safe sliding-window attention, learned positions/RoPE/ALiBi, LayerNorm/RMSNorm,
GELU/SwiGLU/GeGLU, serial/parallel residuals, separate/partial/packed QKV,
tied/untied heads, packed binary data, DP, and a larger-model FSDP path.

Required representative clusters:

| Capability cluster | Families |
|---|---|
| Learned positions + LayerNorm + GELU | GPT-2, OPT |
| RoPE/parallel residual + ALiBi/packed QKV | GPT-NeoX/Pythia, BLOOM |
| MQA/GQA + non-Llama dense blocks | Falcon, dense Phi |
| RoPE + RMSNorm + gated MLP | Llama, Mistral, dense Qwen, dense Gemma |

Deferred: encoder/encoder-decoder models, MoE, DLLMs, SSM/recurrent/hybrid
models, multimodal models, quantized pretraining, and CUDA-only training paths.

## Performance contract

The locked LaughLM geometry is vocab 32,064; hidden 1,024; 8 layers; 8 Q/8 KV
heads; MLP 2,816; sequence 2,048; RoPE; pre-RMSNorm; SwiGLU; tied embeddings;
fused QKV; SplashAttention; BF16 compute; FP32 parameters/output; microbatch 2;
GA 32; DP 8; 1,048,576 tokens/update; chunk size 4,096; z-loss `1e-4`; AdamW;
WSD; native memmap; prefetch 16; persistent cache; async checkpoint; unscanned
layers.

Validated reference: `1.014M` global tok/s, `53.1%` non-embedding MFU,
`5.73 GiB` peak HBM, and `1.034 s` total step.

| Stage | Required result | Failure response |
|---|---:|---|
| Generic HF + stable XLA | `>= 600K tok/s` | Pause expansion; diagnose runtime/graph/sync |
| Optimized loss + attention | `>= 850K tok/s` | Pause adapters; inspect HLO/kernels/GA/HBM |
| Hard V1 parity | `>= 912.6K tok/s`, `>= 47.8%` MFU | Do not claim LaughLM-class performance |
| Preferred release | `>= 963.3K tok/s`, `>= 50.4%` MFU | Document hard-gate release if used |

Release evidence uses steady-state medians from three matched runs plus one
200-update real-shard stability run.

## Historical branch and PR policy

### Transition from historic milestone PRs

Legacy [PR #18](https://github.com/Dhiraj309/TrainLM/pull/18) merged to
`main` on 2026-10-28 at `23959d29434fd2ae3f78b20469e1d61c6b1e18c9`. Its
pre-existing `milestone/*` branches are historical; do not derive new PRs
from them. T0.1's source classification is in
[`TORCHTPU_SALVAGE_INVENTORY.md`](TORCHTPU_SALVAGE_INVENTORY.md); T0.2's
accepted mapping is in [`PR_DELIVERY_PLAN.md`](PR_DELIVERY_PLAN.md#t02-accepted-mapping-to-native-torchtpu-gates).
Owner approval of a concrete scope is required before activating a new
implementation branch.

### Multiple meaningful PRs, each with multiple meaningful commits

The canonical detailed plan is
[`docs/PR_DELIVERY_PLAN.md`](PR_DELIVERY_PLAN.md). The plan has a **series
of multiple separately reviewable PRs**, not one mega-PR for the roadmap
and not one PR per tiny commit. Each PR includes **several logically
coherent commits**: implementation, focused tests, discovered fixes,
regression coverage, and docs. The branch, PR title, PR description, and
commits must describe the same deliverable.

**Finish PR N and merge it into `main` → verify remote `main` → only then
create PR N+1 from the newly merged `main`.** No stacked PRs, parallel
implementation PRs, unrelated commit mixtures or pre-created future branches.
A blocked PR remains active until its blocker is resolved or the owner
explicitly approves a documented change of plan. PR numbers are assigned
by GitHub; the roadmap numbers below are *planning IDs only*.

Future branch format is `<type>/<NN>-<deliverable>`, starting at `01`
after the post-merge audit is approved, e.g. `perf/03-chunked-loss-no-full-logits`.
Use `fix`, `feature`, `perf`, `benchmark`, `validation`, `docs`,
`ci`, `release` according to the predominant deliverable. Preserve
M0–M17 milestone IDs as lineage in descriptions, not legacy branch names.
The final planned queue must be audited against post-merge `main` before
activation; already completed capabilities should be marked satisfied rather
than reimplemented, and numbers reassigned to actual approved future work.

Prefer **merge commit or rebase merge** to preserve distinct commit
boundaries. Squash only when explicitly approved by the owner. Keep each
commit focused and understandable; no artificial limit, but avoid giant
unreviewable branches. Follow the PR title and content format in
`.github/pull_request_template.md`.

### Candidate sequential PR roadmap

| Planned slot | Candidate branch | Proposed PR title | Related milestone |
|---:|---|---|---|
| 01 | `fix/01-cpu-regressions-test-baseline` | fix(test): establish a clean, reproducible CPU regression baseline | M2-F5, M13-F3, M14-F3 |
| 02 | `feature/02-tpu-trainer-lifecycle-resume` | feat(tpu): complete public trainer lifecycle, checkpoints and resume | M8-F0, M7-F1-F5 |
| 03 | `perf/03-chunked-loss-no-full-logits` | perf(loss): integrate chunked causal loss without full logits | M9-F1-F5 |
| 04 | `perf/04-compiled-gradient-accumulation` | perf(xla): reduce host-unrolled accumulation and recompilation | M5-F4-F5, M11-F7 |
| 05 | `perf/05-pallas-mha-attention` | perf(attention): add guarded TPU-native Pallas MHA | M10-F1-F3 |
| 06 | `perf/06-compact-gqa-window-alibi` | perf(attention): support compact GQA/MQA, windows and ALiBi | M10-F4-F6 |
| 07 | `perf/07-reversible-qkv-mlp-fusion` | perf(model): activate reversible QKV and gated-MLP fusion | M8-F4-F5, M11-F1-F3 |
| 08 | `perf/08-optimizer-remat-memory-policy` | perf(train): tune XLA optimizer state and rematerialization | M11-F4-F5 |
| 09 | `perf/09-data-prefetch-cache-graph-stability` | perf(input): overlap packed data, prefetch and XLA compilation cache | M5-F3, M11-F6 |
| 10 | `perf/10-async-telemetry-xla-profile` | perf(telemetry): add low-overhead TPU profiling and async reporting | M7-F4, M11-F7, telemetry backlog |
| 11 | `benchmark/11-optimized-loss-attention-850k-gate` | perf(benchmark): certify the optimized loss and attention 850K gate | M10-F7, M11-F7 |
| 12 | `validation/12-numerical-resume-hf-export-parity` | test(parity): validate numerical equivalence, resume and HF export | M12-F1, M12-F4, M7-F1-F5 |
| 13 | `benchmark/13-laughlm-135m-three-run-certification` | perf(parity): run 135M LaughLM-class three-run certification | M12-F2-F3-F5 |
| 14 | `validation/14-dense-family-conformance-matrix` | test(compat): certify dense HF family support and fallbacks | M6-F1-F4, M13-F1-F5 |
| 15 | `feature/15-fsdp-sharding-1p3b-scale` | feat(scale): validate FSDP sharding and 1.3B training | M13-F6-F7 |
| 16 | `feature/16-hf-like-cli-config-user-journey` | feat(ux): polish HF-like trainer, CLI and secure TPU workflows | M8-F0, M14-F1-F2 |
| 17 | `ci/17-release-ci-preemption-evidence` | ci(release): enforce reproducibility, hardware gates and recovery | M14-F3-F4 |
| 18 | `release/18-dense-ar-v1-support-certification` | release(v1): publish dense-AR support, limitations and evidence | M14-F5 |

Each entry has its intended outcome, commit-by-commit development outline,
scope/non-goals, acceptance/verification gates and fallback in
[`docs/PR_DELIVERY_PLAN.md`](PR_DELIVERY_PLAN.md#detailed-pr-cards).
The same plan also defines the full [PR title/body contract](PR_DELIVERY_PLAN.md#exact-pr-title-description-and-commit-trace-contract)
and [additional measured-only optimization experiments](PR_DELIVERY_PLAN.md#additional-engineering-optimizations-to-evaluate-not-assumed-wins).
**This historical candidate list is not the active queue. Use the T0.2 mapping and obtain owner approval before creating a branch.**

### Lifecycle and enforcement

1. **Before PR creation:** Confirm the predecessor is actually merged,
   verify live `origin/main`, reconcile prior delivery/evidence,
   select exactly the next approved PR card, and create the named branch
   from `origin/main` (not another feature/milestone branch).
2. **During PR development:** Create small, purpose-specific commits and
   tests as features/fixes land. Keep user-facing HF semantics, API,
   checkpoints, reversible transforms and fallbacks intact.
3. **Before merge:** Require review, relevant passing CPU tests, correct
   GitHub base/head, per-commit clarity, accurate status/doc updates,
   evidence for any hardware-performance claim, and a bounded rollback.
   An experimental TPU capability may merge guarded/off-by-default when
   hardware certification is still pending *if its PR only claims an
   implemented guarded path and the reviewer approves the explicitly
   deferred hardware gate*.
4. **After merge:** Confirm GitHub reports merged, verify actual new
   `main` SHA and file/docs presence, then mark the PR delivered and
   unlock exactly the next one. No automatic merges.
5. **On failure:** Keep the current PR active; report checks, conflicts,
   hardware evidence gaps and sanitized network/auth failures. Never
   change the managed proxy or leak credentials; supported approved
   per-command network permission may be requested for GitHub operations.

### Product UX and hard technical boundaries

The normative product requirement remains HF-like: users supply model,
datasets, processing class/tokenizer and familiar training arguments;
`TrainLMTrainer.train()` and optional `trainlm train` privately handle
PJRT, worker launch, validation, caches and logs. Maintain canonical
Transformers model semantics, generic dense-AR fallback, capability-selected
reversible optimizations, plain HF export, checkpoint safety and accurate
`trainer.explain()`. TPU validation is owner-run, not claimed from local
CPU tests. An optimized default is earned by semantic, HLO/HBM and matched
hardware throughput evidence—not model-family naming or a speculative
speedup.

### Historical Dense-AR V1 milestone PR overview (legacy, not the future queue)

| Status | PR | Branch | Milestones | Commits | Merge gate |
|---|---|---|---|---:|---|
| [x] | PR1 | `milestone/m0-m2-foundation` | M0-M2 | 15 | Contracts and generic HF CPU conformance |
| [~] | PR2 | `milestone/m3-m4-data-trainer` | M3-M4 | 18 | Merged; post-merge data/trainer validation remains tracked by validation gates |
| [~] | PR3 | `milestone/m5-m7-xla-compatibility` | M5-M7 | 16 | Implementation merge-ready; DP8 launcher/data/attention/optimizer fixes landed; v5e-8 baseline is 319,302 tok/s after sparse-loss fix, performance work pending |
| [~] | PR4 | `milestone/m8-m9-optimization-core` | M8-M9 | 11+ | HF-like public trainer facade, reversible planner, and optimized loss |
| [~] | PR5 | `milestone/m10-m12-kernels-parity` | M10-M12 | 19 | Canonical attention contract started; 850K and hard LaughLM parity pending |
| [~] | PR6 | `milestone/m13-m14-family-release` | M13-M14 | 12 | Cross-family certification and V1 release |

Future tracks M15 TorchTPU, M16 MoE and M17 DLLM remain deferred; each must be
split into story-sized PRs under the sequential policy if ever activated.
Legacy milestone branches may exist but are not the base for new development.

### Current branch handoff

PR2 (`milestone/m3-m4-data-trainer`) and the PR3 implementation branch
(`milestone/m5-m7-xla-compatibility`) are merged. PR3 delivered 16 feature
commits; its DP8 probe, HF model preflight, data path, two-update optimizer
smoke, and 100-update measured baseline are validated in Kaggle logs.
The framework-independent round-trip and precision-safe telemetry corrections
passed CI before merge. The measured v5e-8 baseline is complete; PR4
(`milestone/m8-m9-optimization-core`) begins with the public trainer facade
before wiring optimization passes into the user path.

Validation checkpoint (2026-09-06): the staged v5e-8 smoke and 100-update
baseline are complete; M6 cross-family and M7 reliability gates remain
tracked validation work. Do not run the full 20B-token workload yet; use the
evidence to decide whether each optimization is ready for TPU measurement.

Validation artifact: `notebooks/TrainLM_TPU_Validation.ipynb` now exercises the
public facade from a fresh TPU session with the same concise workflow proposed
to end users: provide immutable model and validated packed-data inputs, create
familiar training arguments, and call `trainer.train()`. Evaluation and save
cadence are demonstrated in the same run, with one optional public resume
example. It must be run on TPU; no local execution is implied. The notebook
never invokes worker scripts, selects a world size, configures PJRT/ranks,
imports private coordinator types, or parses stage logs.

The notebook includes a corrected Kaggle/Colab bootstrap and consolidated
import order. Its first cells clone
`milestone/m10-m12-kernels-parity`, enter `/kaggle/working/TrainLM`, and fetch
the matching `origin/<branch>` ref before installation. This keeps a fresh
session on the requested revision and prints both the subject and full commit
SHA for evidence. Re-running in the same session should skip the clone cell
when the directory already exists, then use the refresh cell to select the
latest fetched commit; the notebook never asks users to resolve a detached or
stale checkout manually.

The notebook downloads the supplied `LaughTaleAI/LaughLM-Tokenized-Fine`
shards at a resolved immutable Hub revision, or accepts mounted raw files and
existing local manifests. The raw path requires explicit dtype/header/vocab
information, checks file geometry against the supplied token count, and scans
the entire payload for bounds and SHA-256 before creating validated in-memory
manifest descriptors. No sidecar upload is required. Tokenizer identity
(`microsoft/Phi-3.5-mini-instruct` in the supplied metadata) must also match
the model vocabulary mapping before parity results are meaningful.
The first public lifecycle smoke now uses the pretrained
`HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint rather than loading the full
3.8B Phi checkpoint on every replica. SmolLM2 is a dense Llama-family decoder
with 30 layers, hidden size 576, 9 query heads, 3 KV heads, and a 49,152-token
vocabulary. The notebook uses a 128-token lifecycle-smoke geometry. The
current LaughLM binary shards remain suitable for lifecycle/API testing only:
their tokenizer vocabulary is different, so loss and quality are not valid
SmolLM2 training evidence until SmolLM2-tokenized shards are supplied. The
private coordinator serializes the pretrained HF source so each TPU worker
reconstructs the same model for probe, preflight, training, evaluation, and
resume.

The matching starter configuration is
`examples/smollm2_135m_tpu_pretraining.yaml`. The notebook deliberately uses
one shard, sequence length 128, BF16 checkpoint parameters, and six steps for a
memory-safe lifecycle check. This
configuration is not a throughput benchmark and must not be used to claim
SmolLM2 quality when the input shards were produced by another tokenizer.

The TPU model-preflight stage is a capability diagnostic, not a duplicate
training-shape benchmark. It therefore caps its forward at 16 tokens while
preserving batch, model, precision, attention, and device paths. The coordinator
terminates probe after fifteen inactive minutes or preflight after thirty
inactive minutes so a compiler/PJRT hang cannot indefinitely retain eight
CPU-heavy ranks and grow host RAM. New worker events reset the inactivity
deadline, so cold startup or compilation is not rejected merely for exceeding
a total wall-clock duration. The actual training shape is still compiled in
the train stage.
The notebook prints numbered cell markers, while the coordinator prints stage
start/completion messages and a ten-second heartbeat with the latest bounded
worker-log line. Long compilation is therefore distinguishable from a silent
stall without exposing subprocess commands or requiring manual log parsing.
The stage boundary creates its output directory before opening the log and
preserves keyboard-interrupt process-group cleanup while heartbeat polling.

Kaggle installation checkpoint (2026-09-05): the editable install completed
with the pinned Torch 2.9.0, Torch/XLA 2.9.0, Transformers 5.15.0, and libtpu
profile. Kaggle reported pre-existing `torchvision`/`torchaudio` 2.8 versus
Torch 2.9 and `google-tunix` versus `safetensors` 0.8 conflicts; these do not
block TrainLM's TPU path unless those unrelated packages are imported. Restart
the kernel before the XLA environment check.

Kaggle XLA API correction (2026-09-05): Torch/XLA 2.9 uses
`torch_xla.runtime.world_size()` and `global_ordinal()`; the notebook and
`XlaRuntime` now use those APIs with a legacy `xm` fallback for older releases
and injected test doubles. The old `xm.xrt_world_size()` call must not be used
in TPU setup code.

Kaggle PJRT launch correction (2026-09-05): the notebook kernel must stay free
of Torch/XLA, JAX, TensorFlow, model objects, and device queries. Its environment
report reads package metadata only. A fresh script parent clears stale
single-VM topology overrides before importing XLA, then uses
`torch_xla.launch(..., start_method="spawn")`. Spawned workers retain PJRT's
assigned rank/chip settings. This runner is for one v5e-8 VM, not multi-host
launch, and has no automatic single-device fallback.

Reference-notebook findings (2026-09-05): the previous HF Trainer workflow
achieved DP8 by placing the complete model/data/training construction inside a
worker function and invoking `torch_xla.launch(train_fn, args=())`. Reusable
patterns are setting `PJRT_DEVICE=TPU` before imports, disabling TF/Flax
integration, clearing stale TPU topology variables, using
`DistributedSampler.set_epoch`, wrapping loaders with `MpDeviceLoader`, doing
XLA optimizer steps with `barrier=False`, logging/checkpointing on ordinal
0, and rendezvousing before exit. Its raw memmap reader (1024-byte header,
`uint16`) and ~338K tok/s result are useful diagnostics, but are not a
manifest-validated TrainLM run or a comparable LaughLM 1M tok/s measurement.

DP8 validation entry point: `scripts/trainlm_tpu_worker.py` owns launch and a
real all-reduce probe (ranks 0-7 sum to 36). `trainlm_tpu_data.py` validates
shards once on the host; `trainlm_tpu_training.py` builds models only inside
workers. The notebook's in-process training cells have been removed.

Native allocation abort follow-up (2026-09-05):

- [x] Remove parent device ownership and duplicate in-process model allocation.
- [x] Configure per-rank caches before the first tensor computation.
- [x] Move HF models to XLA and restore ties before optimizer construction.
- [x] Prepare labels/masks and exact counts on CPU before asynchronous XLA
  transfer; keep bounded prefetch and one microbatch per graph execution.
- [x] Replace dynamic z-loss selection with a fixed-shape masked reduction.
- [x] Advance WSD by global tokens for equal-count replicated DP batches.
- [x] Save synchronized slowest-replica timing, both scheduled and supervised
  token throughput, stage logs, and XLA metrics; expose full-logits loss honestly.
- [x] Add regression tests and statically validate source/notebook syntax.
- [ ] Run regression tests in CI (not executed on the local workstation).
- [x] On a fresh Kaggle kernel, complete the DP8 collective probe and all-rank
  HF model preflight; rank-sum 36 and logits shape checks passed.
- [x] On a fresh Kaggle kernel, complete the two-update smoke with the AdamW
  XLA dtype bridge: all eight ranks reached finite step 1/2 and finalized
  `summary.json`.
- [x] Complete the measured baseline after the smoke; record steady-state
  throughput, token rates, and XLA metrics. MFU/HBM certification remains
  pending because the current summary does not provide a trustworthy FLOP
  denominator for every HF family.

Measured baseline evidence (2026-09-06, before sparse-loss materialization):
The completed 100-step summary reports `99,566,080` measured global supervised
tokens over `311.197` seconds: `319,945` global supervised tok/s and `320,101`
scheduled tok/s. The step 30-40 slice is consistent with this result: it
advanced `1,310,080` supervised tokens per rank, or `10,480,640` global tokens
across DP8 in roughly 32-33 seconds. The rank-local calculation is therefore
about `40K`, but it must not be presented as cluster throughput. This is about
3.17x slower than the LaughLM reference. The primary avoidable synchronization
found in this path was
`loss.item()` after every optimizer update even when logging every 10 steps.
TrainLM now exposes `trainer.materialize_loss_every_steps` (default `1` for
compatibility) and the TPU worker aligns it with `--log-every-steps`.

Post-fix rerun (2026-09-06): with `materialize_loss_every_steps: 10` active on
all ranks, the measured summary reported `99,566,080` global supervised tokens
over `311.824` seconds: `319,302` global supervised tok/s and `319,458`
scheduled tok/s. This is within `0.20%` of the pre-fix `319,945` result, so
sparse loss materialization is correct but not a material throughput lever.
The remaining gap is dominated by device-side work and graph structure, not
the per-update host loss readback.

Decision after baseline (2026-09-06): the functional and measured v5e-8 run
completed with `all_workers_finished` and a finalized summary. Stop/release
the TPU environment after preserving `request.json`, probe/model-preflight/
smoke/baseline logs, `baseline/summary.json`, and `baseline/xla_metrics.txt`.
Do not spend TPU quota on another unchanged baseline; proceed with PR4's
public HF-like facade, then run each M9 optimization as an HLO-backed A/B
benchmark against this recorded 319K baseline.

The observed code-path defects are resolved for the validated 135M Llama
geometry. Full logits, cross-family coverage, and uncertified
attention/kernel performance remain open. Sparse loss materialization is
implemented and measured; it changed throughput by less than 0.20%, so the
next gains must come from graph, memory, and kernel work.

PR3 handoff metadata: title `feat(m5-m7): add PyTorch/XLA compatibility and
TPU runtime foundation`; branch `milestone/m5-m7-xla-compatibility`.

## Milestone overview

| Status | Milestone | Outcome |
|---|---|---|
| [x] | M0 | Scope, parity manifest, metrics, and dependencies |
| [x] | M1 | Backend-neutral framework contracts |
| [x] | M2 | Universal HF dense-causal CPU path |
| [x] | M3 | Packed data validates, partitions, streams, and resumes exactly |
| [~] | M4 | F1-F7 implemented; validation pending |
| [~] | M5 | F1-F7 implemented; DP8 launch, host data preparation, timing, expected-world-size gates, and two-update v5e-8 smoke passed; certification remains pending |
| [~] | M6 | F1-F4 positional, attention, block-layout, and TPU round-trip coverage implemented; Llama preflight passed, cross-family validation pending |
| [~] | M7 | F1-F5 distributed resume, async lifecycle, canonical HF export, telemetry, and integrity gates implemented; v5e-8 smoke and measured baseline passed, reliability certification pending |
| [~] | M8 | HF-like public trainer facade plus reversible capability optimization engine |
| [~] | M9 | Memory-efficient causal loss |
| [~] | M10 | TPU attention and 850K gate |
| [~] | M11 | Projection, optimizer, remat, and HLO tuning |
| [~] | M12 | Exact 135M parity certification |
| [~] | M13 | Cross-family certification and 1.3B scaling |
| [~] | M14 | Dense-AR V1 release |
| [d] | M15 | TorchTPU migration |
| [d] | M16 | MoE extension |
| [d] | M17 | DLLM extension |

## M0 — Reproducibility and release contract

**Status:** [x] Complete

**Goal:** Freeze scope, geometry, metrics, and dependency expectations before
runtime or kernel work.

- [x] **M0-F1 — Dense-AR product scope**
  `docs(scope): define dense causal LM support contract`
  Define inclusions, exclusions, support levels, remote-code policy, issue
  labels, release terminology, and representative families.
  **Acceptance:** One normative scope document is referenced by tests, issue
  configuration, release notes, and README.

- [x] **M0-F2 — LaughLM parity manifest**
  `test(benchmark): lock LaughLM 135M parity manifest`
  Lock model, initialization, loss, optimizer, schedule, batch, precision,
  data, runtime, hardware, reference commit, and evidence.
  **Acceptance:** Validate `135,611,392` parameters and `1,048,576`
  tokens/update; a same-version field change fails the lock test.

- [x] **M0-F3 — Benchmark schema and MFU calculator**
  `feat(benchmark): add throughput and MFU result schema`
  Standardize global/device throughput, step/compile time, HBM, FLOPs, MFU,
  input idle, and collectives with real token counts and device waits.
  **Acceptance:** Reproduce saved LaughLM metrics within rounding tolerance.

- [x] **M0-F4 — Dependency matrix**
  `build(deps): define reproducible framework compatibility matrix`
  Separate core, CUDA, XLA, Pallas, profiling, and dev extras; pin TPU stacks
  and tested Transformers v5 ranges.
  **Acceptance:** Clean CPU and pinned TPU environments resolve reproducibly.

### Exit gate

- [x] Scope, parity data, metric formulas, and dependency profiles are reviewed
  and versioned.

## M1 — Framework contracts and backend boundary

**Status:** [x] Complete

**Goal:** Remove custom-model, XLA, and task assumptions from core TrainLM.

- [x] **M1-F1 — Configuration ownership**
  `refactor(config): separate model training runtime and optimization config`
  HF `PretrainedConfig` owns architecture; TrainLM owns training/runtime policy.
  **Acceptance:** `TrainLMConfig` is built only when explicitly selected.

- [x] **M1-F2 — Execution backend protocol**
  `feat(runtime): define replaceable execution backend protocol`
  Define device, precision, compile, mesh, shard, sync, data, checkpoint,
  diagnostics, and lifecycle hooks.
  **Acceptance:** Trainer-facing packages contain no `torch_xla` import.

- [x] **M1-F3 — Language-model task protocol**
  `feat(tasks): define causal language modeling task protocol`
  Own dispatch, label shift, ignored tokens, normalization, counting, and eval.
  **Acceptance:** Trainer code contains no causal-label logic.

- [x] **M1-F4 — Capability and plan schemas**
  `feat(optimization): define capability and execution plan schemas`
  Model attention, position, norm, MLP, residual, projections, head, embedding,
  checkpoint, and provider decisions.
  **Acceptance:** Plans serialize and explain without mutating a model.

- [x] **M1-F5 — Checkpoint/export contracts**
  `feat(checkpoint): define internal resume and HF export contracts`
  Version canonical/sharded state, transforms, optimizer, RNG, scheduler,
  tokens, data cursor, atomicity, and incomplete-save behavior.
  **Acceptance:** Both formats are fixed before implementations depend on them.

### Exit gate

- [x] CPU boundary contracts prove
  trainer, task, optimization, checkpoint, and runtime interfaces have no family
  or XLA dependency.

## M2 — Universal Hugging Face dense-causal intake

**Status:** [x] Complete

**Goal:** Every representative HF causal LM works through the unchanged generic
path before optimization adapters exist.

- [x] **M2-F1 — Generic HF model provider**
  `feat(models): add generic HF causal LM provider`
  Support `AutoConfig`, `from_config`, and `from_pretrained`; preserve revision,
  dtype, metadata, ties, and local loading.
  **Acceptance:** No TrainLM model class is required.

- [x] **M2-F2 — Forward-aware batch dispatch**
  `feat(models): add forward signature aware batch dispatch`
  Filter optional inputs while preserving masks, positions, cache position, and
  supported family fields.
  **Acceptance:** Tiny representative models receive only valid inputs.

- [x] **M2-F3 — Generic output/loss path**
  `feat(models): add generic causal LM output and loss protocol`
  Use valid model loss or correctly shifted CE normalized by supervised tokens.
  **Acceptance:** CPU loss and gradients match direct HF execution.

- [x] **M2-F4 — Plain-HF round trip**
  `test(models): certify generic HF save and reload round trip`
  Train one update, save, reload without TrainLM model code, compare tied and
  untied state/output.
  **Acceptance:** Every representative family passes.

- [x] **M2-F5 — Dense-AR CPU matrix**
  `test(models): add dense AR architecture conformance matrix`
  Cover tiny GPT-2, OPT, GPT-NeoX, BLOOM, Falcon, Phi, Llama, Mistral, Qwen2,
  and Gemma through construct/forward/loss/backward/update/export/overfit.
  **Acceptance:** No optimization adapter is used.

- [x] **M2-F6 — Compatibility explanation**
  `feat(models): report generic model capabilities and fallbacks`
  Report known/unknown capabilities, support level, and selected generic path.
  **Acceptance:** Every model has stable human and machine-readable output.

### Exit gate

- [x] All representative clusters are Compatible on CPU without an adapter.

## M3 — Production packed-binary data

**Status:** [x] Complete

**Goal:** Provide safe, distributed, fixed-shape, exactly resumable `.bin` data.

- [x] **M3-F1 — Binary manifest**
  `feat(data): define packed binary shard manifest`
  Define header, dtype, endian, count, vocab bounds, checksum, documents, and
  existing 1,024-byte-header `uint16` compatibility.
  **Acceptance:** Corrupt metadata, size, checksum, and token range fail early.

- [x] **M3-F2 — HF shard source**
  `feat(data): add revision-pinned HF binary shard source`
  Resolve repo/revision/path/IDs/cache/offline reuse with standard `HF_TOKEN`.
  **Acceptance:** Mocked source resolution is deterministic.

- [x] **M3-F3 — Batched memmap reader**
  `feat(data): add contiguous packed memmap batch reader`
  Read complete `[batch, sequence]` regions and fixed-shape inputs/labels/masks.
  **Acceptance:** Valid bytes match the reference reader.

- [x] **M3-F4 — Deterministic partitioning**
  `feat(data): add deterministic shard shuffle and host partitioning`
  Define seeded order, rank ownership, cross-shard, and validation behavior.
  **Acceptance:** Simulated hosts cover intended samples exactly once.

- [x] **M3-F5 — Backend-aware prefetch**
  `feat(data): add backend-aware asynchronous prefetch queue`
  Keep reading backend-neutral; expose transfer wrappers, depth, and timing.
  **Acceptance:** Ordering/backpressure pass; TPU begins tuning at depth 16.

- [x] **M3-F6 — Resumable cursor**
  `feat(data): make packed data iteration exactly resumable`
  Save revision, permutation, shard, offset, epoch/tokens, and RNG.
  **Acceptance:** Interrupted and uninterrupted next batches match.
  The complete 28-train/2-validation reference split is exercised with eight
  training ranks and two validation ranks. Rank ownership covers every retained
  batch exactly once, and every possible rank-local interruption point resumes
  to the identical remaining token windows and final token count.

### Exit gate

- [x] The 28/2 reference split validates, streams, partitions, and resumes.

## M4 — Correct backend-neutral trainer

**Status:** [~] M4-F1-F7 implemented; validation pending

**Goal:** Complete correct training independently of TPU optimization.

- [x] **M4-F1 — Lifecycle/state machine**
  `feat(training): implement trainer lifecycle and control state`
  Prepare, train, evaluate, save, resume, stop, finalize, and callback order.
  **Acceptance:** Normal, failure, stop, and resume flows pass.

- [x] **M4-F2 — Token-correct accumulation**
  `feat(training): implement token-normalized gradient accumulation`
  Handle ignored-token variation and separate microstep/update/token counters.
  **Acceptance:** Accumulation matches an equivalent large CPU batch.

- [x] **M4-F3 — Optimizer/state dtype factory**
  `feat(optim): add backend-neutral AdamW state policy`
  AdamW, decay policy, clipping, independent parameter/moment dtype, and
  backend option validation.
  **Acceptance:** Match PyTorch AdamW under identical policy.

- [x] **M4-F4 — Token-based WSD**
  `feat(scheduler): add token-based WSD schedule`
  Warmup/stable/decay/min-LR/horizon with token-based resume.
  **Acceptance:** Boundaries and resume match the locked schedule.
  Boundary fixtures now cover the token immediately before, at, and after each
  warmup/stable/decay transition plus post-horizon clamping. Optimizer and
  scheduler snapshots restored mid-schedule produce the identical remaining
  learning-rate sequence and serialized scheduler state.

- [x] **M4-F5 — Streaming evaluation**
  `feat(evaluation): add token-weighted causal LM evaluation`
  Compute loss/perplexity without retaining predictions; isolate eval state.
  **Acceptance:** Match reference without mutating training.
  Variable-mask causal batches now match the task's token-weighted reference
  loss and perplexity while preserving trainer state, model mode, parameters,
  and empty gradients. Evaluation remains streaming and retains no predictions.

- [x] **M4-F6 — Sync-safe callbacks**
  `feat(training): separate host callbacks from compiled metrics`
  Callbacks consume sparse materialized metrics, never live hot-path tensors.
  **Acceptance:** Backend mock detects no hidden scalar extraction.
  Scalar-extraction instrumentation over a three-update run now proves that
  non-logging updates perform no tensor `.item()` calls. Step hooks receive
  `None` for unmaterialized loss, and the sole metrics callback receives only
  immutable host floats at the requested logging boundary.

- [x] **M4-F7 — Multi-family overfit**
  `test(training): overfit dense AR conformance models`
  Check falling loss, finite gradients, resume, and export across tiny families.
  **Acceptance:** Trainer contains no family branch.
  Every dense-AR fixture now trains through 30 updates with falling finite loss
  and finite gradients, captures an exact update-15 model/optimizer/scheduler/
  trainer checkpoint, resumes to the identical update-30 parameters, and
  reloads its canonical export through standard Transformers APIs.

### Exit gate

- [ ] CPU training and CUDA smoke are correct; the custom model is only one HF
  reference model.

## M5 — PyTorch/XLA runtime and accumulation feasibility

**Status:** [~] M5-F1-F7 implemented; target TPU validation pending

**Goal:** Establish stable DP8 execution before specialized TPU kernels.

Current validation repair: one coordinator-only notebook and native PJRT
launcher, host-validated raw shards, host-prepared causal batches, correct
model/optimizer placement order, and saved synchronized baseline evidence.
Use `notebooks/TrainLM_TPU_Validation.ipynb` from a fresh Kaggle kernel; retain
`probe.log`, `smoke.log`, `baseline.log`, `request.json`, and the baseline
`summary.json`/`xla_metrics.txt`. CI and target hardware gates remain open.

- [~] **M5-F1 — Optional XLA backend**
  `feat(runtime): add pinned PyTorch XLA backend`
  Contain initialization, BF16 policy, device/ordinal, versions, and guarded
  Pallas/JAX imports.
  **Acceptance:** Core TrainLM imports without TPU extras.

- [~] **M5-F2 — SPMD data-parallel mesh**
  `feat(runtime): add SPMD data parallel mesh`
  Implement DP8 replicated parameters/sharded batch and validate logical axes.
  **Acceptance:** v5e-8 shapes and gradient reduction are correct.

- [~] **M5-F3 — Cache and recompile guard**
  `feat(runtime): add XLA cache and recompile guard`
  Initialize the official persistent cache, fingerprint the compilation
  contract, and reject changes to batch, sequence, mask, or accumulation
  structures after warmup. The trainer registers accumulation steps before
  backend initialization; the guard remains backend-owned and model-agnostic.
  **Acceptance:** No compile beyond the allowed warmup graph set; cache and
  guard state are visible in runtime diagnostics.

- [~] **M5-F4 — Compiled training operations**
  `feat(runtime): compile XLA training operations`
  Add an explicit backend hook around a complete device-step callable, using
  `torch_xla.compile` when enabled and leaving model-only compilation disabled.
  Keep data I/O, host token accounting, logging, callbacks, and checkpointing
  outside the compiled boundary; defer trainer integration until the
  accumulation strategy spike selects the stable update shape.
  **Acceptance:** A v5e-8 update matches reference semantics after the
  accumulation strategy is selected; the hook is covered by backend tests.

- [~] **M5-F5 — Accumulation strategy spike**
  `perf(runtime): select v5e gradient accumulation strategy`
  Add an evidence-gated selector for compiled unrolled, XLA loop/scan, native,
  and safe microstep paths at MB2/GA32/S2048. Optimized paths require complete
  compile, dispatch, and HBM evidence; unsupported explicit requests fall back
  or fail according to policy.
  **Acceptance:** Strategy plans are deterministic and explain fallbacks; the
  v5e-8 benchmark comparison remains required before any optimized default.

- [~] **M5-F6 — XLA diagnostics**
  `feat(runtime): expose XLA metrics HLO and profile diagnostics`
  Add opt-in, bounded collection of compile/execute metric samples, `aten::`
  fallback counters, short/full reports, optional HLO, and optional XProf
  metadata. Keep collection at explicit host boundaries and expose snapshots
  through runtime state and diagnostics.
  **Acceptance:** TPU evidence proves graph count and fallback status; fake
  metrics coverage is included without making XLA a core dependency.

- [~] **M5-F7 — Generic 135M baseline**
  `perf(benchmark): record generic HF 135M XLA baseline`
  Add locked-workload loading and evidence evaluation for exact geometry with
  generic HF loss/attention on fake and real data. Require synchronized
  steady-state results, no unexpected post-warmup compile, and report fallback
  counters without fabricating TPU measurements.
  **Acceptance:** Target TPU run records stable finite updates and `>= 600K
  tok/s`; otherwise pause.

### Exit gate

- [ ] PyTorch/XLA is replaceable/correct and the first go/no-go gate passes.

## M6 — Universal dense-AR TPU compatibility

**Status:** [~] M6-F1-F4 implemented; target TPU validation pending

**Goal:** Prove the framework is not Llama-specific before parity optimization.

- [~] **M6-F1 — Positional semantics**
  `test(tpu): cover learned RoPE and ALiBi causal models`
  Add conservative, family-neutral detection for learned positions, RoPE, and
  ALiBi, plus the five-update TPU conformance procedure and static coverage.
  Run learned-position, RoPE, and ALiBi models for five finite updates.
  **Acceptance:** Stable post-warmup graph for all three.

- [~] **M6-F2 — Attention layouts**
  `test(tpu): cover dense MHA GQA and MQA layouts`
  Add family-neutral head-count and QKV projection-layout detection with
  static coverage for MHA, GQA, MQA, packed, and separate projections.
  Cover MHA/GQA/MQA and separate/packed projections on the generic path.
  **Acceptance:** No adapter or unexpected CPU fallback.

- [~] **M6-F3 — Block/MLP layouts**
  `test(tpu): cover norm MLP and residual layouts`
  Add family-neutral normalization, MLP activation, and residual-layout
  detection with static coverage for LayerNorm/RMSNorm, GELU/gated MLP, and
  serial/parallel blocks.
  Cover LayerNorm/RMSNorm, GELU/gated MLP, serial/parallel residual.
  **Acceptance:** All variants pass the same finite-update/graph gate.

- [~] **M6-F4 — Generic TPU round trip**
  `test(tpu): certify generic dense AR save resume and export`
  Add an evidence-boundary comparison helper for resumed state, next-update
  equivalence, and canonical Hugging Face logits/state reloads.
  Verify equivalent next-update resume and clean HF reload across the matrix.
  **Acceptance:** Internal and canonical state match defined tolerances.

### Exit gate

- [ ] Every V1 cluster trains generically on TPU without adapter, recompile, or
  fallback and round-trips through HF.

## M7 — Checkpointing, observability, and integrity

**Status:** [~] M7-F1-F5 implemented; target TPU validation pending

The staged validation runner is available at
[`notebooks/TrainLM_TPU_Validation.ipynb`](../notebooks/TrainLM_TPU_Validation.ipynb).
It exercises the M5 runtime, M6 generic HF path, and M7 evidence boundaries on
v5e-8 without claiming that exact-resume persistence is implemented by the
trainer itself.

**Goal:** Make long TPU training recoverable and measurable before mutation.

- [~] **M7-F1 — Distributed resume state**
  `feat(checkpoint): add sharded distributed resume checkpoints`
  Add a backend-neutral DP resume plan that validates complete state ownership,
  sharded model/optimizer artifacts, per-rank RNG, and exact data cursors before
  distributed I/O.
  Save model, optimizer, scheduler, trainer, RNG, tokens, cursor, plan, and
  manifest versions with direct shard I/O where supported.
  **Acceptance:** Interrupted DP8 resumes to an equivalent next update.

- [~] **M7-F2 — Async checkpoint lifecycle**
  `feat(checkpoint): add asynchronous checkpoint lifecycle`
  Add a thread-safe lifecycle state machine for in-flight, committed, failed,
  retention, and shutdown states; durability is reported only after publication.
  Surface in-flight, completion, failure, retention, and shutdown semantics.
  **Acceptance:** Compute overlaps persistence; durability is never reported early.

- [~] **M7-F3 — Canonical HF export**
  `feat(checkpoint): export canonical Hugging Face checkpoints`
  Add a validated canonical-export plan requiring safetensors, reversed runtime
  transforms, tied-weight aliases, and no training-only artifacts.
  Export config, safetensors, generation config, tokenizer, ties, and shard index.
  **Acceptance:** Plain Transformers reload matches outputs.

- [~] **M7-F4 — Sync-safe telemetry**
  `feat(monitoring): add TPU throughput memory and MFU telemetry`
  Add synchronized host-boundary samples and bounded snapshots for throughput,
  compile, HBM, input idle, collectives, and fallback counters.
  Measure synchronized throughput, compile, HBM, idle, and collectives sparsely.
  **Acceptance:** No `.item()` or tensor print in the hot path; overhead bounded.

- [~] **M7-F5 — Integrity gates**
  `feat(monitoring): add configurable training integrity checks`
  Add sparse finite loss/gradient/parameter, update norm, token, and optional
  cursor-continuity checks with injected-corruption coverage.
  Check finite loss/grad/parameters, update size, tokens, and data continuity.
  **Acceptance:** Sparse checks detect injected corruption.

### Exit gate

- [ ] A 200-update run resumes, reports trusted metrics, and detects corruption.

## M8 — Capability planner and reversible optimization

**Status:** [~] In progress — the coordinator, packed-data adapter, local
lifecycle facade, TPU metric/state bridge, structural inspector, and guarded
adapter registry, pure provider planner, transactional transform registry, and
reversible state-dict conversion and the complete explain report are
implemented; TPU checkpoint parity remains.

**Goal:** Provide a minimal Hugging Face-like trainer surface while transforming
loaded HF models safely without family logic in core.

- [~] **M8-F0 — Public TrainLM trainer facade**
  `feat(api): add HF-like TrainLMTrainer and training arguments`
  Expose `model`, `args`, `train_dataset`, `eval_dataset`,
  `processing_class`/tokenizer, callbacks, `train`, `evaluate`, save methods,
  and `explain`. Hide worker launch, PJRT setup, probes, cache, manifests,
  preflight, and optimization internals; keep a CLI/config entry point aligned
  with the same API.
  **Acceptance:** A concise HF-style example trains on CPU and reaches the TPU
  coordinator without users constructing subprocess commands or parsing stage
  logs; raw validation remains an internal implementation detail.
  The TPU branch now reaches a private coordinator that owns probe, model
  preflight, worker launch, logs, and structured summaries for pretrained HF
  sources plus validated `PackedBinDataset` inputs. Local and revision-pinned
  Hub constructors validate shard integrity and deterministically partition
  examples by rank. CPU/CUDA runs honor save/evaluation cadence, forward
  callback metrics, and restore versioned training state. TPU worker summaries
  expose normalized public state and callback metrics without log parsing. TPU
  checkpoint/evaluation parity and target-hardware validation remain.
  The private request/CLI boundary transports save cadence and resume paths;
  rank-local atomic shards preserve model, optimizer, scheduler, runtime,
  trainer, host/device RNG, and deterministic packed-data progress. TPU
  evaluation uses a separately validated packed-data source, deterministic
  rank partition, scheduled engine evaluation, and structured callback
  metrics. Causal evaluation now reduces the loss numerator and normalization
  denominator across replicas before producing global loss and perplexity.
  Target-hardware lifecycle validation remains. Tensor serialization
  is isolated in a private TPU module, preserving the framework-independent
  checkpoint contracts.
  Coordinator tests cover serialized evaluation-field mismatch rejection.
  The optional `trainlm train`/`python -m trainlm train` entry point now builds
  validated local packed datasets and delegates YAML construction, resume, and
  execution to the same public facade. It emits one structured result and does
  not expose worker commands, PJRT settings, or coordinator log parsing. Packed
  readers inherit sequence length and seed from the validated training arguments
  instead of introducing a second CLI-only geometry configuration.
  The 28/2 shard lifecycle fixture now derives a valid synthetic vocabulary for
  its disjoint per-shard token ranges, and callback synchronization tests isolate
  trainer loss extraction from DataLoader's private iterator-seeding scalar read.
  `trainlm train --dry-run` performs the same YAML and packed-manifest validation
  and emits the public optimization explanation without launching training,
  providing a safe final host-side check before TPU allocation.
  Dry-run rejects a resume argument instead of silently implying that checkpoint
  contents were validated without entering the training lifecycle.
  The TPU validation notebook now presents one uncluttered end-user run rather
  than a coordinator test harness: construct datasets and familiar arguments,
  then call `trainer.train()`. Scheduled evaluation/checkpoint cadence is part
  of that run, and exact resume is a separate optional public call.
  TPU topology is runtime-discovered inside launched workers. Neither the public
  arguments, coordinator request, CLI, nor notebook asks users to predict a
  world size; collective validation derives its expected rank sum from the
  actual topology selected by Torch/XLA.
  Public Hub data intake now accepts a repository and end-exclusive numeric
  shard range. LaughLM legacy `.bin` files are downloaded through the Hub cache,
  pinned to one resolved commit, fully scanned and validated, then passed into
  the same reader, dataloader, train/eval partition, and coordinator staging
  paths as manifest-backed data. Sidecar authoring is no longer user work.
  Download and full-file validation are startup costs only: training uses local
  cached files through lazy memory maps and never downloads during iteration.
  Validation scans token bounds in vectorized chunks, and the TPU prefetch
  geometry now uses the same canonical microbatch argument forwarded by the
  public trainer.
  The notebook exposes one local cache directory for both disjoint train and
  evaluation ranges and makes the startup boundary explicit. Its training cell
  remains only `trainer.train()`; downloaded files are reused locally and no
  network operation is part of batch iteration.
  A Kaggle construction attempt emitted Torch/XLA's TensorFlow-conflict warning
  and killed the notebook kernel before `trainer.train()`. The setup cell now
  removes the unused TensorFlow and mismatched vision/audio installations,
  installs the pinned TPU profile, and requires one explicit session restart.
  A pre-import assertion prevents users from continuing in the contaminated
  parent process and receiving the same opaque kernel death.
  Coordinator stages now have explicit process-group ownership. Failure or
  notebook interruption terminates every spawned rank, with bounded TERM/KILL
  escalation, before control returns to `trainer.train()`. Stage output is
  written directly to disk so cleanup does not depend on draining a blocked
  output pipe and rerunning a corrected cell does not inherit orphaned workers.

- [x] **M8-F1 — Structural inspector**
  `feat(optimization): inspect dense causal LM capabilities`
  Prefer public HF contracts and expose unknown semantics explicitly.
  **Acceptance:** Reports match family fixtures; no name-only semantic guesses.
  Implemented through config, module, parameter-alias, and forward-signature
  evidence. Opaque residual and custom-normalization semantics remain unknown
  unless a guarded adapter supplies an explicit known, inferred, or unsupported
  capability with a non-empty evidence citation. Unknown or uncited overrides
  are rejected, preserving the no-family-name rule. Evidence is also bound to
  the exact model class, configuration class, and source provider so a valid
  claim for one adapter boundary cannot be replayed against another model.

- [x] **M8-F2 — Adapter registry**
  `feat(optimization): add optional model adapter registry`
  Resolve explicit model/class adapters with semantic and version guards.
  **Acceptance:** Removing adapters preserves M6 compatibility.
  The registry is declarative and optional: exact model/config class matches
  are additionally gated by inspected component kinds, source provider, and
  allow-listed tested package versions. Resolution has stable priority/ID
  ordering, records every rejection reason, and never imports or mutates a
  model, so an empty registry preserves the generic compatibility path.

- [x] **M8-F3 — Pure planner**
  `feat(optimization): select kernels transforms and fallbacks`
  Match capability to provider by backend, shape, dtype, mask, and backward
  support under auto/required/disabled/explicit policies.
  **Acceptance:** Deterministic snapshots explain every decision.
  Provider specifications declare backend, precision, capability kind, runtime
  requirements, fallback status, and reversible transformations. The pure
  planner deterministically handles disabled, auto, required, and explicit
  provider requests, producing a blocked plan before mutation when necessary.
  Its public planning boundary validates capability, backend, precision,
  policy, request, and optional adapter-resolution types before selection. A
  compatible provider explicitly requested by ID is recorded as selected even
  when it is designated as the automatic fallback, avoiding contradictory
  fallback metadata where selected and requested IDs are identical. Conversely,
  an unqualified `required` request rejects fallback-only candidates rather than
  reporting a portable fallback as satisfying an optimized-path requirement.
  Provider declarations require a real boolean fallback flag, preventing
  truthy strings or integers from changing required-policy semantics. Plan IDs
  include the complete sorted provider catalog, so distinct eligible providers
  or transform declarations cannot produce the same audit identity. Transform
  declarations must use immutable tuples, preventing catalog mutation after a
  provider has been registered or a plan ID has been recorded. Every declared
  transform must also target the provider's own component, preventing a
  provider selected for one capability from mutating an unrelated component.
  Transformation IDs are unique within each provider and across the registered
  catalog, so independently selected operations cannot produce an invalid plan
  with ambiguous transform identity.

- [x] **M8-F4 — Transactional transforms**
  `feat(optimization): apply validated reversible model transforms`
  Plan before mutation, transform before optimizer, preserve aliases, and roll
  back failure without global patches or hot-path hooks.
  **Acceptance:** Injected failure leaves the original model usable.
  Transform handlers capture rollback state before mutation, validate declared
  inverse IDs, apply plans in order, and roll back the failing transform plus
  all earlier transforms in reverse order. Undeclared alias changes, missing
  handlers, blocked plans, and downstream context failures cannot commit.
  Serialized execution plans require every schema field, reject unknown fields
  and malformed nested entries, and do not accept booleans as schema versions.
  Nested provider decisions and transformations also require complete exact
  schemas, including a real boolean parameter-layout flag. Ready plans must
  select a provider, blocked plans cannot retain transforms, and an explicit
  selected decision must name the provider that was actually requested.
  Transform, QKV, MLP, and rematerialization fixtures construct valid ready
  plans with explicit provider decisions rather than bypassing that invariant.
  Every executable transform must match both the component and provider of a
  selected or fallback decision, preventing unauthorised restored mutations.
  Status-specific validation runs before transform authorization so malformed
  blocked/no-op plans retain their actionable lifecycle error classification.
  Distinct transforms cannot claim the same target path in one plan, avoiding
  order-dependent capture, apply, and rollback behavior.
  The overlapping-target regression preserves the tuple container emitted by
  `to_dict()`, so it reaches the intended plan invariant in the full CI suite.

- [x] **M8-F5 — State-dict conversion**
  `feat(optimization): add reversible parameter layout mappings`
  Pack/split maps, aliases, dtype/shape validation, transformed resume, and
  canonical import/export.
  **Acceptance:** Train/save/reload matches canonical HF state.
  Versioned layout manifests now validate and reversibly concatenate/split
  parameter groups with exact key, shape, dtype, device, collision, and alias
  checks. The converter preserves unrelated state and restores declared tied
  keys as shared tensor objects for canonical export. Alias groups must be
  disjoint and cannot include keys consumed or produced by a layout mapping,
  preventing ambiguous restoration or silently lost ties. Manifest loading
  rejects unknown fields and malformed mapping, key, shape, or alias-group
  containers instead of coercing strings or accepting misspelled contracts.
  All schema fields are mandatory, and booleans cannot masquerade as integer
  schema versions.

- [x] **M8-F6 — Explain report**
  `feat(optimization): expose model optimization explanation`
  Human/JSON capabilities, providers, transforms, fallbacks, backend, graph,
  and certification.
  **Acceptance:** Strict mode fails before TPU allocation when unsupported.
  A versioned `OptimizationExplanation` now combines inspected capabilities,
  adapter resolution, provider decisions, reversible transforms, fallbacks,
  backend/precision, graph evidence, limitations, and certification state in
  stable dictionary, JSON, and human-readable views. Strict validation rejects
  blocked plans and unknown or unsupported capabilities before execution.

### Exit gate

- [ ] No-op and fixture transforms are transactional, explainable,
  checkpoint-safe, and backend-independent.

## M9 — Memory-efficient causal-LM loss

**Status:** [~] In progress — the portable reference chunked linear causal
cross-entropy is implemented; optimized hidden-state access, rematerialization,
TPU providers, and multi-family integration remain.

**Goal:** Remove the full FP32 logits bottleneck without changing semantics.

- [x] **M9-F1 — Reference chunked linear CE**
  `feat(loss): add reference chunked linear causal cross entropy`
  Consume hidden state, output weight, labels, optional bias; support shift,
  ignore index, tied/untied, FP32 reduction, chunks, and z-loss.
  **Acceptance:** Loss and hidden/head gradients match full-logits reference.
  The FP32 reference projects flattened shifted tokens in bounded chunks,
  supports optional bias, masks/ignore index, tied or untied output weights,
  supervised-token normalization, and z-loss. Tests compare loss and hidden,
  head, and bias gradients against full-logits cross entropy.

- [x] **M9-F2 — Optimized training view**
  `feat(loss): bypass full HF logits during optimized training`
  Safely obtain final hidden state/output weight while preserving HF export.
  **Acceptance:** HLO contains no full `[B,S,V]` tensor.
  A non-mutating training view now accepts an explicit hidden-state provider,
  requires inspected linear-head evidence, obtains weights through the public
  HF output-embedding contract, and feeds hidden states directly to chunked
  loss without calling the model's full-logits forward. Target-XLA HLO evidence
  remains required before enabling this path automatically.

- [~] **M9-F3 — Rematerialized chunks**
  `feat(loss): add rematerialized chunked loss backward`
  Compare chunks 2,048/4,096/8,192 and bound HBM.
  **Acceptance:** Select size from matched throughput/HBM evidence.
  An explicit `per_chunk` policy now uses non-reentrant activation checkpointing
  around each linear-CE chunk and preserves reference loss/z-loss/gradient
  semantics. Chunk-size selection and HBM/throughput evidence remain pending on
  target TPU hardware, so this story is not complete.

- [~] **M9-F4 — TPU loss providers**
  `perf(loss): evaluate native XLA Pallas and Tokamax loss providers`
  Compare pure XLA, Pallas bridge, and Tokamax where versions/backward permit.
  **Acceptance:** Record correctness, HBM, speed, compatibility, and fallback.
  The pure provider catalog keeps Pallas and Tokamax gated on explicit runtime
  and backward evidence and otherwise selects portable chunked loss as an
  explained fallback. Provider implementation and matched TPU correctness,
  HBM, and throughput measurements remain pending.

- [x] **M9-F5 — Multi-family loss adapters**
  `test(loss): certify chunked loss across dense output-head layouts`
  Cover tied/untied heads and representative backbone outputs.
  **Acceptance:** Loss/gradient/export parity passes without full logits.
  Explicit output adapters now normalize tensor, mapping, tuple/list, and
  attribute-based hidden-state results. The conformance matrix covers tied and
  untied embeddings, biased and bias-free heads, rematerialized chunked loss,
  parameter gradients, alias preservation, and canonical state-dict parity.

### Exit gate

- [ ] Reference geometry avoids full logits and the selected provider has TPU
  correctness and performance evidence.

## M10 — TPU attention provider family

**Status:** [~] In progress — the canonical attention semantics contract and
guarded HF interface integration are implemented; TPU providers, layout
optimizations, autotuning, and benchmark evidence remain.

**Goal:** Correct memory-efficient attention across the full V1 semantic surface.

- [x] **M10-F1 — Canonical attention spec**
  `feat(attention): define canonical causal attention specification`
  Represent layouts, head geometry, scale, masks, segments, ALiBi, dropout,
  soft-cap, QK norm, and output without model-name conditionals.
  **Acceptance:** Every V1 family maps to the specification.
  A versioned model-family-neutral schema now validates MHA/GQA/MQA head
  geometry, scale, full/sliding causal masks, segment IDs, learned/RoPE/ALiBi
  positions, dropout, soft-cap, QK normalization, and BSH output. Capability
  mapping uses inspected facts only and rejects unknown semantics.

- [x] **M10-F2 — HF attention/mask integration**
  `feat(attention): integrate TrainLM attention with HF interfaces`
  Use public attention/mask interfaces when sufficient and adapters otherwise.
  **Acceptance:** Causal leakage tests detect missing/wrong mask registration.
  A guarded adapter now registers attention and mask callables under the same HF
  interface key only after the canonical spec passes layout, position, segment,
  dropout, soft-cap, and QK-normalization checks. Reference visibility tests
  detect future-token, sliding-window, and cross-segment leakage.

- [~] **M10-F3 — XLA Pallas provider**
  `feat(attention): add XLA Pallas causal attention provider`
  Wrap supported Pallas attention with import/version guards and backward.
  **Acceptance:** MHA matches reference and emits expected HLO/custom call.
  A dependency-free bridge now requires an exact tested torch_xla version and
  explicit backward evidence before constructing an MHA-only provider. It uses
  an injected public kernel adapter, rejects dense masks and dropout, and keeps
  causal masking internal to avoid quadratic mask materialization. The story
  remains open pending target-TPU numerical, backward, and HLO/custom-call
  evidence.

- [~] **M10-F4 — GQA/MQA without KV repeat**
  `feat(attention): support grouped and multi query TPU attention`
  Support 8/8, 8/4, and 8/1 without silent K/V HBM expansion.
  **Acceptance:** Correctness and HBM tests pass or explicit fallback is used.
  A compact logical query-to-KV mapping now covers 8/8, 8/4, and 8/1 without
  repeating K/V objects. The guarded provider forwards compact head geometry to
  an explicit kernel adapter and rejects GQA/MQA unless runtime evidence exists.
  Target-TPU numerical, gradient, and HBM evidence remains pending.

- [~] **M10-F5 — ALiBi/sliding window**
  `feat(attention): add ALiBi and sliding window capabilities`
  Provide optimized support or efficient explicit XLA fallback.
  **Acceptance:** Boundary-token, mask, and gradient fixtures pass.
  The guarded provider now forwards a scalar sliding-window boundary without a
  dense mask and forwards explicit model-supplied ALiBi slopes unchanged. Both
  paths require independent runtime evidence and otherwise fail explicitly.
  Target-TPU boundary-token, numerical, gradient, and HBM evidence remains.

- [~] **M10-F6 — Attention autotuning**
  `perf(attention): tune v5e attention provider configurations`
  Tune/cache tiles by hardware, shape, dtype, mask, and provider version.
  **Acceptance:** Selection is deterministic and reproducible.
  A pure exact-key cache now includes hardware, provider/version, dtype, batch,
  head geometry, sequence length, mask, and window. Candidate measurement order
  and equal-score selection are deterministic, and versioned results round-trip
  through stable JSON. Target-v5e measurements and candidate ranges remain.

- [~] **M10-F7 — Loss/attention benchmark**
  `perf(benchmark): certify optimized loss and attention stage`
  Benchmark exact 135M geometry on fake and real data with HLO evidence.
  **Acceptance:** `>= 850K tok/s`, lower HBM, stable graph, no fallback.
  A pure matched-result evaluator now enforces v5e-8 geometry, the 850K gate,
  lower HBM, stable post-warmup compilation, zero CPU fallback, no full logits,
  named loss/attention providers, and an HLO fingerprint. The story remains open
  until real synchronized fake- and real-data measurements satisfy the gate.

### Exit gate

- [ ] MHA/GQA/MQA paths are correct, semantic fallbacks are explicit, and the
  second go/no-go gate passes.

## M11 — Projection, optimizer, rematerialization, and HLO tuning

**Status:** [~] In progress — reversible separate and partial QKV weight/bias
packing plus explicit live wrapper transforms are implemented; target update
evidence remains.

**Goal:** Close the parity gap one measured bottleneck at a time.

- [~] **M11-F1 — Reversible QKV packing**
  `feat(optimization): pack compatible QKV projections`
  Handle separate/partial/packed layouts, bias, heads, aliases, load/export.
  **Acceptance:** Output/gradient/update and round-trip parity pass.
  A family-neutral geometry descriptor now builds versioned reversible mappings
  for separate MHA/GQA/MQA Q/K/V weights and optional biases. Optimized resume
  packs along output features and canonical export restores the original keys.
  An explicit pre-optimizer transform now replaces an adapter-selected wrapper
  exposing separate linear projections with one packed linear call, preserves
  compact GQA/MQA splits, validates geometry/bias/device/dtype, and rolls back
  transactionally. A second reversible descriptor covers canonical layouts with
  a separate query projection and an already-combined key/value projection
  without inventing model keys. Already-packed canonical projections now use a
  validated no-transform descriptor that preserves compact GQA/MQA geometry.
  Partial query/combined-KV wrappers can now be transactionally replaced by a
  single packed projection, preserving compact K/V outputs and restoring the
  original wrapper on rollback. Separate and partial transforms preserve a
  uniform frozen/trainable state and reject mixed source states that one packed
  parameter cannot represent without changing fine-tuning semantics. They also
  reject internal or external source-parameter aliases before replacement,
  rather than silently breaking tied-parameter behavior.
  Portable CPU regression coverage compares packed and separate float32 matrix
  multiplications with explicit numerical tolerances for their valid
  accumulation-order differences, including output and gradient parity.
  Portable SGD update parity covers separate and partial layouts with and
  without bias. Target-XLA output/gradient/update parity remains.

- [~] **M11-F2 — Reversible gated-MLP packing**
  `feat(optimization): pack compatible gate and up projections`
  Pack compatible SwiGLU/GeGLU paths and preserve GELU paths.
  **Acceptance:** Output/gradient/update and export parity pass.
  An explicit SwiGLU/GeGLU descriptor now packs equal-shape gate/up weights and
  optional biases through the reversible state-dict converter, while GELU is an
  explicit no-transform path. An adapter-selected wrapper can now be replaced
  transactionally by one packed linear call using an explicit model-owned
  activation callable; rollback restores the original modules. Packing
  preserves uniform frozen/trainable state and rejects mixed trainability or
  source aliases that a single parameter cannot represent. Already-packed
  SwiGLU/GeGLU sources now have a validated no-transform descriptor, avoiding
  invented separate keys or needless repacking. Portable SGD update parity now
  covers biased and bias-free live packed layouts; target-XLA update parity
  remains pending.

- [~] **M11-F3 — Native fusion audit**
  `perf(hlo): audit norm RoPE residual and MLP fusion`
  Inspect copies, transposes, materialization, custom calls, and layout before
  writing kernels.
  **Acceptance:** Every candidate has evidence-backed native/custom decision.
  A versioned pure audit now consumes HLO fingerprints plus copy, transpose,
  materialization, custom-call, and fallback observations for normalization,
  RoPE, residual, and MLP components. It retains proven clean native fusion,
  identifies custom-kernel candidates, and blocks unknown/fallback cases.
  A text-capture constructor now normalizes and fingerprints HLO, counts exact
  copy/transpose/custom-call opcodes, and accepts explicit component-specific
  materialization patterns without pretending text alone proves fusion.
  Target-XLA captures and matched benchmarks remain pending.

- [~] **M11-F4 — Decoder rematerialization**
  `feat(optimization): add structural decoder rematerialization policies`
  None/block/attention/MLP/loss-chunk with correct FSDP ordering.
  **Acceptance:** Selected policy is justified by step/HBM evidence.
  Explicit none/block/attention/MLP/loss-chunk policies now enforce pre-FSDP
  ordering. A deterministic selector rejects gradient mismatches, unstable
  graphs, and policies outside a configured slowdown budget, then chooses the
  lowest measured HBM with stable ties. Adapter-selected block, attention, or
  MLP paths can now be patched transactionally with non-reentrant activation
  checkpointing without changing module structure, parameter aliases, or state
  keys; loss-chunk rematerialization remains owned by the loss provider.
  Target-XLA measurements remain pending.

- [~] **M11-F5 — XLA optimizer-state path**
  `feat(optim): optimize XLA AdamW state and update graph`
  Evaluate BF16 first moment, FP32 second, clip, decay, reduction, and resume.
  **Acceptance:** Correct update with measured HBM/step impact.
  A versioned policy now fixes BF16-or-FP32 first moment, FP32 second moment,
  clipping, decoupled decay, and reduction semantics. A matched evidence gate
  requires update/resume parity, stable graphs, zero CPU fallback, bounded step
  regression, and lower HBM. The TPU worker now materializes its optimizer
  configuration and trainer clipping value from that policy instead of
  duplicating moment and clipping defaults. Target-XLA measurements remain
  pending.

- [~] **M11-F6 — Batch/prefetch tuning**
  `perf(runtime): tune microbatch accumulation and prefetch geometry`
  Re-test MB2/GA32, MB1/GA64, safe alternatives, and prefetch near 16.
  **Acceptance:** One synchronized real-data production geometry is selected.
  A deterministic selector now compares only equal token/update geometries and
  requires real data, stable graphs, zero CPU fallback, bounded HBM, and bounded
  input idle. It selects throughput first with stable HBM/idle/ID ties. The TPU
  worker now constructs its `ParallelLoader` arguments from the validated
  geometry contract rather than an unrelated hard-coded call site. Target
  measurements for MB2/GA32, MB1/GA64, and prefetch near 16 remain pending.

- [~] **M11-F7 — Final HLO/host closure**
  `perf(hlo): remove residual parity path bottlenecks`
  Resolve proven transpose, graph, fallback, collective, sync, or input issues.
  **Acceptance:** Exact reference reaches the hard 90% thresholds.
  A final evidence gate now requires the 912.6K throughput and 47.8% MFU hard
  thresholds, stable compilation, zero fallback, no unresolved transpose/layout
  copies, no unplanned host sync, no full logits, and bounded input/collective
  idle. A strict artifact loader now binds the benchmark JSON and graph counters
  to the exact normalized HLO capture by SHA-256 before evaluation. The story
  remains open until target-v5e evidence satisfies every gate.

### Exit gate

- [ ] Transforms are reversible/correct and the graph reaches `912.6K tok/s`
  plus `47.8%` MFU.

## M12 — Exact LaughLM 135M parity certification

**Status:** [~] In progress — the required semantic comparison and deterministic
early-update evidence gate are implemented; target-run alignment remains.

**Goal:** Convert the fast path into repeatable release evidence.

- [~] **M12-F1 — Numerical alignment**
  `test(parity): align 135M initialization and loss semantics`
  Match initialization, residual scale, eps, RoPE, shift, z-loss, optimizer,
  schedule, and dtype; compare early deterministic updates where possible.
  **Acceptance:** Every semantic difference is removed or justified.
  A locked-path comparator now covers position/norm/residual, initialization,
  shifted loss/z-loss, optimizer, schedule, and dtype semantics. Differences
  must be explicitly justified, and at least one deterministic early-update
  error must remain within tolerance. A strict artifact loader now combines
  reference and candidate configuration JSON with versioned update evidence and
  rejects unknown evidence fields. Target parity evidence remains pending.

- [~] **M12-F2 — Three-run benchmark**
  `perf(parity): certify repeated v5e 135M throughput`
  Report matched runs, dispersion, cache, HBM, HLO, compiles, idle, and MFU.
  **Acceptance:** Every run clears hard thresholds; preferred median clears 95%.
  A repeated-run evaluator now requires exactly three matched warm-cache v5e-8
  results, identical HLO fingerprints, stable graphs, zero fallback, and every
  run above the hard throughput/MFU thresholds. It reports median throughput/
  MFU, spread, and max HBM, with separate preferred-median status. Real runs
  remain pending. A versioned manifest loader now requires three distinct,
  confined result artifacts and explicit hard/preferred thresholds before
  evaluation.

- [~] **M12-F3 — Real-shard stability**
  `test(parity): validate 200 update real data stability`
  Run pinned diverse shards with eval, integrity, checkpoint, resume, and graph.
  **Acceptance:** Finite/stable, continuous data, no recompile/fallback, export.
  A structured gate now requires at least 200 updates across diverse
  revision-pinned shards, scheduled evaluation, checkpoint resume, integrity
  checks, cursor continuity, finite loss/gradient ranges, stable compilation,
  zero fallback, and canonical export. The real target run remains pending.
  A strict versioned artifact loader now validates the complete lifecycle and
  numerical evidence schema together with explicit update/shard requirements
  before evaluating a captured run; unknown or missing fields are rejected.

- [~] **M12-F4 — Plain-HF export certification**
  `test(parity): certify optimized HF checkpoint interoperability`
  Reverse internal layouts and reload in clean Transformers-only environment.
  **Acceptance:** Logits/loss match without TrainLM installed.
  A structured gate now requires a clean environment without TrainLM,
  canonical-only state-dict keys, preserved tied aliases, zero missing/unexpected
  keys, and logits/loss within tolerance after plain Transformers reload.
  A strict versioned artifact loader now ingests the clean-process evidence,
  requires the complete schema, normalizes JSON reload-key arrays, and applies
  the artifact's explicit numerical tolerance without importing Transformers.
  Execution in a clean target environment remains pending.

- [~] **M12-F5 — Parity report**
  `docs(benchmark): publish LaughLM TrainLM parity report`
  Publish environment, commands, configs, metrics, limits, profile, and HLO.
  **Acceptance:** Reviewer can reproduce without hidden configuration.
  A versioned report schema now requires environment versions, reproduction
  commands, configuration/metric/profile/HLO artifacts, limitations, and the
  pass state of numerical, repeated-run, stability, and plain-HF gates. Stable
  JSON and reviewer-facing Markdown are available. A strict bundle loader now
  reconstructs the immutable report and requires every configuration, metric,
  profile, and HLO reference to be relative, confined, and present. Real
  evidence remains.

### Exit gate

- [ ] Exact 135M parity is reproducible and clears the hard gate; no result is
  generalized to other families yet.

## M13 — Cross-family certification and larger-model scaling

**Status:** [~] In progress — learned-position family mappings are declarative; target certification remains

**Goal:** Prove TrainLM is a framework, not one optimized Llama geometry.

- [~] **M13-F1 — GPT-2/OPT mapping**
  `feat(adapters): optimize learned-position dense causal models`
  Reuse QKV, GELU/LayerNorm, loss, and attention capabilities.
  Explicit GPT-2 and OPT adapters now require exact model/config classes,
  inspected MHA, learned-position, LayerNorm, GELU, residual, projection, and
  LM-head semantics, plus the tested Transformers version. The catalog maps
  both families to shared attention, QKV-layout, GELU-MLP, and chunked-loss
  operations without importing Transformers or mutating models. Output,
  gradient, update, graph, export, and target-TPU evidence remain required.
  **Acceptance:** Both pass output/gradient/update, graph, and export.

- [~] **M13-F2 — GPT-NeoX/BLOOM mapping**
  `feat(adapters): optimize parallel residual and ALiBi models`
  Preserve parallel residual, RoPE/ALiBi, and projection semantics.
  Version-guarded mappings now keep GPT-NeoX's RoPE and parallel residual
  operations distinct from BLOOM's ALiBi and serial residual operations while
  requiring explicit MHA, LayerNorm, GELU, fused-QKV, and linear-head evidence.
  Shared TPU output/gradient/update, graph, export, and performance evidence
  remains required.
  **Acceptance:** Shared TPU certification passes.

- [~] **M13-F3 — Falcon/Phi mapping**
  `feat(adapters): optimize MQA GQA and nonstandard dense blocks`
  Preserve head geometry, parallel blocks, projections, and activations.
  Explicit mappings now separate Falcon MQA and GQA eligibility and preserve
  its fused-QKV parallel block, while Phi requires MHA, partial RoPE,
  separate QKV, LayerNorm/GELU, and parallel residual semantics. Correctness,
  HBM, graph, export, and target-TPU evidence remain required.
  **Acceptance:** Correctness, HBM, graph, and export pass.

- [~] **M13-F4 — Llama/Mistral/Qwen/Gemma mapping**
  `feat(adapters): certify RoPE gated dense model families`
  Share structure while preserving windows, activations, scaling, soft-cap,
  QK norm, and family differences.
  Version-guarded mappings now keep plain Llama RoPE, Mistral sliding windows,
  Qwen2 biased QKV, Gemma GeGLU/scaled embeddings, and Gemma2 alternating
  windows/attention soft-capping explicit. Common certification and additional
  QK-normalized family variants remain outstanding.
  **Acceptance:** Difference fixtures and common certification pass.

- [~] **M13-F5 — 135M cross-family matrix**
  `perf(certification): benchmark dense AR capability families`
  Report matched raw throughput and architecture-adjusted MFU; dense full
  attention targets 45% MFU absent a reviewed roofline.
  A pure evidence evaluator now requires one unique support/performance record
  per advertised family, matched workload/device/update geometry, correctness,
  graph and export gates, and computes MFU from family-specific FLOPs/token.
  A strict versioned matrix loader now reconstructs every record and requires
  each family evidence path to be relative, confined beneath the manifest, and
  present before applying the advertised-family and MFU gates.
  Real synchronized 135M measurements remain outstanding.
  **Acceptance:** Every advertised family has a support/performance record.

- [~] **M13-F6 — SPMD FSDP**
  `feat(runtime): add backend-neutral FSDP mesh policy`
  Begin data=4/FSDP=2 with correct wrapping, remat, state sharding, checkpoint.
  A backend-neutral policy now describes explicit wrap classes, adapter-owned
  parameter partition rules, optimizer-state sharding, a data/FSDP logical
  mesh, rematerialization-before-wrap ordering, and topology-matched rank-shard
  checkpoints. The complete nested mesh, wrapping, parameter, optimizer-state,
  and checkpoint policy now round-trips through a strict versioned manifest so
  worker/runtime boundaries cannot silently drop or invent sharding settings.
  Runtime application and the 1.3B target smoke remain outstanding.
  `XlaRuntime.apply_fsdp_policy()` now validates topology, rematerialization
  order, explicit wrap-class evidence, parameter-rule coverage, tensor ranks,
  and optimizer-state geometry before emitting any SPMD annotations. It marks
  model parameters and matching optimizer tensors on the data/FSDP mesh while
  leaving the Hugging Face module tree intact. A real 1.3B target smoke,
  distributed checkpoint proof, and canonical export remain outstanding.
  Rank-sharded TPU checkpoint schema v2 now records the logical mesh
  axes in both the committed manifest and every rank payload. Resume requires
  exact axis-name/size equality, rather than accepting a different data/FSDP
  topology merely because its total world size happens to match.
  The loader remains backward-compatible with schema-v1 checkpoints: it
  derives their topology from the rank runtime state and still requires an
  exact match before restoring model or optimizer tensors.
  **Acceptance:** Correct 1.3B train/resume/export smoke on v5e-8.

- [~] **M13-F7 — 1.3B benchmark**
  `perf(benchmark): certify 1.3B FSDP scaling path`
  Compare matched LaughLM approximately `41.9K tok/s`, `42.6%` MFU with
  collective/HBM analysis.
  A pure evidence gate now requires a review-locked target, exact workload,
  parameter-count, v5e-8 and data/FSDP geometry matching, throughput/MFU gates,
  stable compilation, zero CPU fallback, lifecycle parity, and named
  collective/HBM artifacts. The matched target and real run remain outstanding.
  A strict versioned bundle loader now reconstructs the review-locked target and
  measured evidence together, rejecting schema drift and requiring collective
  and HBM profiles to be relative, confined, and present before evaluation.
  Its mismatch regression fixture now remains internally valid (16 devices,
  data=4/FSDP=4), leaving target mismatch rejection to the evidence evaluator.
  **Acceptance:** A matched target is locked before certification.

### Exit gate

- [ ] Advertised clusters are Certified or explicitly lower-level and the 1.3B
  FSDP path works.

## M14 — Dense-AR V1 release

**Status:** [~] In progress — versioned code/YAML construction exists; release certification remains

**Goal:** Deliver a safe, documented, reproducible user release.

- [x] **M14-F1 — Stable public API**
  `feat(api): finalize TrainLM dense AR pretraining interface`
  Code/YAML from-config and from-pretrained workflows with deprecation policy.
  The public trainer now has explicit `from_pretrained()` and versioned
  mapping/YAML `from_config()` constructors, rejects worker/internal config
  keys, and warns for one-version key aliases. The example imports only the
  package root, and the dependency contract includes the directly imported
  PyYAML runtime. A machine-readable v1 compatibility contract now freezes the
  root exports, accepted configuration keys, and deprecated aliases, with an
  evaluator that detects unversioned breaking changes. The CI test suite now
  builds the wheel, installs it into an isolated environment, changes outside
  the checkout, and smoke-tests only the documented package-root surface.
  **Acceptance:** Examples use no internal packages/custom model class.

- [~] **M14-F2 — Secure `.bin` TPU guide**
  `docs(tutorial): add HF model and binary shard TPU pretraining guide`
  Document token secret, revisions, splits, explain, resume, and export.
  A public-surface-only guide now covers secret-manager token handling,
  immutable model/data revisions, disjoint train/eval shard lists, preflight
  explanation, committed-checkpoint resume, and explicitly documents that TPU
  canonical export remains unavailable. The facade resolves user-friendly
  remote model revisions to immutable Hub commits before coordinator launch.
  Explicit commit SHAs remain usable without a metadata request for offline,
  cache-only execution.
  Target clean-environment smoke remains.
  **Acceptance:** Secret scan and clean-environment smoke pass.

- [~] **M14-F3 — CI/hardware tiers**
  `ci(test): add dense AR release certification tiers`
  Tier 0 CPU, Tier 1 CUDA, Tier 2 scheduled v5e correctness, Tier 3 release
  performance/stability.
  A commit-specific freshness gate now requires unique passing Tier 0-3
  artifacts, including mandatory current Tier 3
  evidence. CUDA/v5e runners, schedules, and real evidence remain outstanding.
  A strict versioned release manifest now binds all tier artifacts to the
  release commit, evaluation timestamp, and freshness window while requiring
  each referenced artifact to be relative, confined, and present.
  **Acceptance:** Release requires current Tier 3 evidence.

- [~] **M14-F4 — Preemption recovery**
  `test(checkpoint): validate preemption and incomplete save recovery`
  Kill during compute, staging, and persistence; accept only durable state.
  Rank shards are now generation-named, committed destinations are immutable,
  shard progress must match the atomic manifest, and latest-checkpoint discovery
  ignores temporary, incomplete, malformed, unsafe, and missing-shard attempts.
  Spawned-process tests now terminate during compute and after shard staging,
  shard publish, manifest staging, and manifest publish. Discovery retains the
  prior generation before the atomic manifest commit and accepts the new one
  after it. Target-TPU recovery and cursor-continuity tests remain outstanding.
  **Acceptance:** No partial checkpoint or silently skipped data.

- [~] **M14-F5 — Support manifest/release notes**
  `docs(release): publish dense AR support and certification matrix`
  Publish versions, hardware, providers, caveats, fallbacks, TorchTPU status.
  A versioned machine manifest and development status page now publish these
  fields without certification overclaims, and a validator checks that public
  explanation backend/path/certification values do not exceed the manifest.
  Manifest loading now rejects top-level and nested schema drift, malformed
  JSON, incorrect container types, and boolean schema versions instead of
  silently accepting unsupported release-claim fields.
  Populate final release evidence and revalidate all claims before closure.
  **Acceptance:** Machine manifest agrees with `trainer.explain()`.

### Exit gate

- [ ] A new user reproduces a certified v5e-8 run and plain-HF export through
  documented public APIs.

## M15 — TorchTPU backend migration

**Status:** [d] Future; starts only when required public APIs exist

**Goal:** Add native TorchTPU through existing contracts, without trainer rewrite.

- [d] **M15-F1 — Compatibility ADR**
  `docs(runtime): map TorchTPU APIs to TrainLM backend contracts`
  Map device, compile, sync, distributed, kernels, checkpoint, profiler, cache.
  **Acceptance:** Every backend method maps to public API or blocking gap.

- [d] **M15-F2 — Native backend**
  `feat(runtime): add native TorchTPU backend`
  Implement existing `ExecutionBackend`, not a second trainer.
  **Acceptance:** Existing contracts and dense causal smoke pass by selection.

- [d] **M15-F3 — Native kernel providers**
  `feat(kernels): register TrainLM kernels with TorchTPU`
  Reuse kernel mathematics and replace bridge/registration details.
  **Acceptance:** Attention/loss forward-backward parity passes capture.

- [d] **M15-F4 — Correctness/checkpoint parity**
  `test(runtime): certify TorchTPU dense AR compatibility`
  Run unchanged M6/M7 suites on the backend.
  **Acceptance:** Both suites pass.

- [d] **M15-F5 — Performance parity**
  `perf(runtime): compare TorchTPU and PyTorch XLA backends`
  Compare exact 135M/1.3B manifests, compile, kernels, HBM, and MFU.
  **Acceptance:** TorchTPU meets active thresholds before becoming preferred.

### Exit gate

- [d] TorchTPU meets or beats XLA correctness/performance. XLA deprecation is a
  separate decision.

## M16 — MoE extension

**Status:** [d] Future; starts after dense-AR V1 release

**Goal:** Add efficient autoregressive MoE without weakening dense paths.

- [d] **M16-F1 — MoE capabilities**
  `feat(moe): define router expert and auxiliary loss capabilities`
  Top-k, capacity, balancing, shared experts, grouped weights, checkpoints.
  **Acceptance:** Selected HF MoE structures require no family names in core.

- [d] **M16-F2 — Expert kernels**
  `feat(moe): add backend grouped expert kernel providers`
  Evaluate Pallas/Tokamax/native including empty/imbalanced groups and backward.
  **Acceptance:** Selected provider and fallback pass parity.

- [d] **M16-F3 — Expert parallel mesh**
  `feat(moe): add expert parallel routing and collectives`
  Expert axis, all-to-all, overlap, source order, load/communication telemetry.
  **Acceptance:** Tokens and gradients return correctly across devices.

- [d] **M16-F4 — MoE checkpoint/export**
  `feat(moe): add expert-sharded checkpoint conversion`
  Equivalent sharded resume and canonical HF MoE export.
  **Acceptance:** Resume matches uninterrupted and plain HF reloads.

- [d] **M16-F5 — MoE certification**
  `perf(moe): certify representative HF MoE families`
  Correctness, routing, balance, communication, HBM, MFU, resume, export.
  **Acceptance:** Each advertised family has a current model-specific target.

### Exit gate

- [d] Representative HF MoE causal LMs are Certified without dense regression.

## M17 — Diffusion language model extension

**Status:** [d] Future; starts after dense-AR V1 release

**Goal:** Add DLLM semantics on shared infrastructure without contaminating
causal contracts.

- [d] **M17-F1 — Diffusion task**
  `feat(dllm): add diffusion language modeling task`
  Timestep/noise, corruption, targets, token weights, eval, resumable RNG.
  **Acceptance:** Deterministic task fixtures pass independently of causal LM.

- [d] **M17-F2 — Diffusion masks**
  `feat(dllm): add noncausal and mixed mask specifications`
  Keep causal and bidirectional/mixed semantics impossible to confuse.
  **Acceptance:** Leakage/boundary fixtures pass; causal cannot be accidental.

- [d] **M17-F3 — Diffusion loss providers**
  `feat(dllm): add optimized diffusion token losses`
  Chunk/fuse only where mathematically valid across timesteps/parameterizations.
  **Acceptance:** Reference/optimized loss and gradients match.

- [d] **M17-F4 — Compiled denoising update**
  `feat(dllm): compile diffusion training updates`
  Keep host randomness/timestep variation from causing unbounded graphs.
  **Acceptance:** Bounded warmup graph set and no step-dependent compile.

- [d] **M17-F5 — DLLM certification**
  `perf(dllm): certify representative HF diffusion language models`
  Quality, correctness, HBM, graph, throughput, resume, and export.
  **Acceptance:** Each advertised DLLM has a task-specific certification target.

### Exit gate

- [d] Representative DLLMs are Certified under a distinct task contract and
  dense causal gates remain unchanged.

## Cross-cutting validation

Every optimization passes applicable gates before becoming an `auto` default.

### Semantic correctness

- Original HF forward/output reference.
- Loss, label shift, mask, z-loss, ignore index, and token reduction.
- Input and transformed-parameter gradients.
- One-update and multi-update parity.
- Tied aliases and state-dict key/shape/dtype preservation.
- Pretrained load and canonical HF export/reload.
- Documented numerical differences from operation reordering.

### Compilation health

- Fixed expected graph count after warmup.
- No unexpected `aten::` CPU fallback or scalar extraction in the hot path.
- No data-dependent recompilation.
- Cache keyed by dependency/provider/shape/config fingerprint.
- Custom providers support backward and active capture.

### Performance evidence

- Separate cold compile, warm cache, and steady-state timing.
- Device synchronization before timed windows end.
- Median and dispersion, never best sample only.
- Fake-data compute ceiling and real-data end-to-end result.
- HBM, input idle, collectives, HLO fingerprint, and profile summary.
- Identical model, data, precision, batch, seed, and hardware for A/B.

### Regression policy

- No provider becomes `auto` without a certification record.
- Transformers/backend upgrades rerun affected capability suites.
- A changed parity HLO fingerprint triggers performance review.
- The generic path is never removed because current adapters pass.

## Critical path and stop conditions

```text
M0 -> M1 -> M2 -> M4 -> M5 -> M6 -> M7 -> M8 -> M9 -> M10 -> M11 -> M12 -> M13 -> M14
       \-> M3 ---------^

After V1: M15 TorchTPU | M16 MoE | M17 DLLM
```

- M3 may proceed alongside M2-M4 after M1 contracts land.
- M6 passes before family-specific optimization dominates work.
- M9/M10 may prototype in parallel only after M8 stabilizes.
- M13 starts after M12 hard parity except small M6 correctness work.
- M16/M17 start only after dense-AR V1 release.
- Below 600K at M5: stop kernels and fix runtime/graph behavior.
- Below 850K at M10: stop adapters and diagnose loss/attention/HLO/GA.
- Below 912.6K at M12: do not claim LaughLM-class performance.

## Primary risks

| Risk | Required mitigation |
|---|---|
| HF internals change | Public interfaces, thin versioned adapters, multi-version CI |
| Transform changes semantics | Transactional plan, forward/gradient/mask tests, reversible export |
| Full logits remain hidden | HLO shape assertion and HBM gate |
| Pallas/JAX mismatch | Optional pinned extra, import/version guard, native fallback |
| GA creates oversized graph | Decide strategy in M5 before kernel investment |
| Layer scan conflicts with Pallas | Isolate until support is proven |
| Metrics/checkpoints synchronize | Sparse metrics and async backend lifecycle |
| “Any AR” becomes Llama-only | M6 before parity; M13 framework certification |
| Fallback marketed as fast | Public Compatible/Optimized/Certified levels |
| TorchTPU obsoletes XLA code | Backend/provider boundary; no XLA in core |
| Remote code executes Python | `trust_remote_code=False` default |
| Hub credentials leak | Standard `HF_TOKEN`, secret scans, no literals |

## Reference implementations

References guide design; they are not blanket dependencies.

- [Hugging Face attention interfaces](https://huggingface.co/docs/transformers/attention_interface)
- [Hugging Face kernels](https://huggingface.co/docs/transformers/kernels)
- [PyTorch/XLA](https://docs.pytorch.org/xla/)
- [PyTorch/XLA SPMD](https://docs.pytorch.org/xla/master/spmd.html)
- [Pallas custom kernels](https://docs.pytorch.org/xla/release/r2.6/features/pallas.html)
- [OpenXLA HLO dumps](https://openxla.org/xla/hlo_dumps)
- [XProf](https://openxla.org/xprof)
- [Tokamax](https://github.com/openxla/tokamax)
- [TorchTPU announcement](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)

Use public HF extension points where sufficient, but retain TrainLM's
capability/transformation layer for complete TPU coverage and hidden-state-based
linear CE. Use PyTorch/XLA now behind interfaces. Do not invent TorchTPU APIs.

## Dense-AR V1 definition of done

V1 is complete only when:

- every representative HF causal LM loads from config/pretrained without
  TrainLM code changes;
- every cluster passes CPU/TPU generic training, resume, and plain-HF export;
- `trainer.explain()` never hides a fallback;
- exact 135M reaches `>= 912.6K tok/s` and `>= 47.8%` MFU in three matched runs,
  preferably `>= 963.3K tok/s` and `>= 50.4%` MFU;
- a 200-update real-shard run is stable, resumable, and graph-clean;
- advertised families have current certification records;
- transformed resume and canonical HF export both pass;
- core has no PyTorch/XLA dependency and accepts a future TorchTPU backend;
- secure `.bin` workflows are documented.

The finished product keeps Hugging Face compatibility at the surface, applies
explainable reversible optimization in memory, and runs on a replaceable
high-performance TPU backend.
Observed Kaggle result (2026-09-05): the eight-rank probe passed, including the
rank-sum all-reduce (36), and host memory was healthy (about 390 GB available).
The two-update smoke then aborted while importing `torchvision/extension.py`
through Transformers' optional vision path. Kaggle's installed Torch 2.9.0
and the pre-existing vision/audio packages were not a matched text-training
dependency set. The notebook now removes `torchvision` and `torchaudio`
before any Transformers import and requires one kernel restart when removal
occurs. This is a dependency failure, not evidence of a TPU HBM or model
allocation failure. Re-run the probe after the restart, then the smoke.

Final pre-rerun cross-check (2026-09-05): the DP8 entrypoint still has no
single-process fallback, validates raw shards once before spawning, pins the
Hub revision, and performs the rank-sum collective before model loading. The
training body keeps packed batches flat for `ParallelLoader` and avoids a
per-microbatch device-to-host token-count reduction on the fully supervised
raw-stream path. A successful probe is necessary but not sufficient: smoke
must still verify HF model construction, XLA graph compilation, AdamW update,
and finalization on all eight ranks.
The current validation intentionally exercises the generic unrolled GA32
update; it is a functional gate, not the final microstep/XLA-loop performance
implementation.
The validation notebook now inserts a model-preflight gate between the
collective probe and smoke: every rank must construct the selected HF causal
LM and complete one XLA forward before optimizer/data execution begins.
The first preflight exposed an HF rotary/SDPA dtype mismatch on TPU
(Q/K=float32, V=bf16). TrainLM now registers a reversible
`trainlm_xla_sdpa` adapter through Transformers' public attention and mask
registries, normalizing Q/K/V to one dtype before delegating to HF SDPA.
Boolean masks remain boolean; only additive masks are cast to the selected
attention dtype.
The subsequent preflight reached all eight `model_preflight_passed` stages;
the notebook failure was only a log-format mismatch (Python repr versus JSON),
which is now corrected at the producer.
The first smoke update then reached AdamW and exposed an XLA-only in-place
moment dtype requirement. The optimizer now temporarily aligns moments with
the current gradient dtype for the update and restores the configured
bf16/fp32 state policy afterward.
The notebook gate accepts both JSON and legacy Python-repr stage records so a
successful preflight from an older worker cannot be discarded.
The notebook also counts successful probe and preflight events in the saved
logs, so a partial-rank success cannot be mistaken for a valid DP8 gate.
Its environment cell now repairs a missing editable `trainlm` distribution
record with a local `pip install --no-deps -e` and treats only the TPU runtime
packages as required; optional `libtpu` metadata reporting is non-fatal.
The setup path is now aligned with the full pinned editable install
`pip install -e ".[tpu-xla]" -c constraints/tpu-xla-2.9.txt`, including when
the checkout exists but its distribution metadata is absent.
The launcher now accepts `--expected-world-size 1` for a v5e-1 functional
smoke and keeps the default `8` for the v5e-8 parity gate. A v5e-1 pass proves
HF/XLA/model/trainer compatibility only; it is not evidence of DP8 scaling or
LaughLM throughput parity. The notebook names evidence by world size and
asserts the actual expected rank count, preventing a v5e-1 run from being
mistaken for the v5e-8 certification.

## v5e-8 rerun readiness (2026-09-06)

The v5e-8 functional rerun passed with the current worker, HF attention
adapter, pinned editable TPU install, and AdamW XLA dtype bridge. All eight
ranks completed model preflight, raw-shard validation, two finite optimizer
updates, and finalization. The completed gate is:

1. `probe_passed` on exactly eight ranks and rank-sum 36;
2. `model_preflight_passed` on exactly eight ranks;
3. two finalized optimizer updates with finite loss and a written summary;
4. next, a measured run for throughput/MFU evidence.

This baseline still uses full-vocabulary logits and host-unrolled GA32, so a
successful smoke demonstrates correctness and graph formation, not LaughLM
throughput parity. Chunked logits, device-side accumulation, and fused TPU
attention remain the performance phase before any parity claim.

## Next validation after sparse-loss fix (2026-09-06)

### TPU smoke repair (2026-09-13)

The first public-notebook run reached the training worker and failed during
evaluation setup because `scripts/trainlm_tpu_training.py` still referenced
the removed `BatchPartitionPlan.assignments` field. The current partition
contract exposes rank-local work through `batch_indices`; the worker now uses
that field and a contract test prevents the stale name from returning. This
is a functional compatibility repair, not a performance result. Rerun the
six-step notebook lifecycle on a fresh v5e-8 session before collecting any
throughput evidence.

The bootstrap also removes common TensorFlow distribution variants and
optional Keras/vision/audio packages. Some Kaggle images leave an orphaned
`tensorflow/` directory after metadata removal, so the install cell removes
only that exact site-packages directory before the required kernel restart.
The importability assertion remains an intentional post-restart gate.

The branch bootstrap is part of the public notebook contract: clone the
single branch first, then run `git fetch origin "$BRANCH"` and
`git checkout -B "$BRANCH" "origin/$BRANCH"`. This avoids the stale local
branch/reset pattern that previously caused notebook runs to miss fixes.

Run the pretrained SmolLM2-135M-Instruct smoke on a fresh v5e-8 session with
the current source and pinned editable install. First repeat the two-update smoke
(`--log-every-steps 1`) to guard the optimizer/lifecycle path, then repeat the
100-update measured run (`--log-every-steps 10`). Confirm every `train_start`
record reports `materialize_loss_every_steps: 10` for the measured run and use
`summary.json`'s slowest-replica throughput as the comparison value. The first
comparison target is whether sparse materialization improves on the prior
`319,945` global supervised tok/s; it is not yet a parity claim. Preserve the
baseline log, summary, and `xla_metrics.txt` for the subsequent chunked-loss
and fused-kernel work.

## Optimization stack for the 319K baseline

The next implementation work is optimization of the existing dense-AR path,
not additional model-family features. Every pass is selected from inspected
HF capabilities, backend/dtype/shape/mask evidence, and reversible state-dict
maps; unsupported models retain the generic path or fail in strict mode.

1. **Memory-efficient loss (highest priority):** project hidden states to the
   vocabulary in fixed chunks, accumulate FP32 cross-entropy and z-loss, and
   rematerialize chunks during backward. This removes the full `[B,S,V]`
   logits tensor while preserving shift, ignore-index, tied/untied heads, and
   HF export semantics.
2. **Compiled accumulation:** replace host-unrolled GA32 with a supported
   device/XLA-loop or complete compiled update, keeping static shapes and
   token-normalized gradients. Select only after compile/HBM evidence.
3. **TPU attention providers:** retain the HF AttentionInterface contract but
   add a TPU-native provider (Pallas/custom call where supported), with causal
   masks, MHA/GQA/MQA, sliding-window/ALiBi semantics, and no silent KV repeat.
4. **Projection/block fusion:** reversible QKV packing, gated-MLP packing,
   and native/fused RMSNorm, RoPE, residual, and SwiGLU paths where HLO proves
   a gain. Preserve aliases and canonical HF state-dict conversion.
5. **Optimizer and rematerialization tuning:** replace the correctness-only
   AdamW dtype bridge with an XLA-efficient state/update path; benchmark BF16
   first moment, FP32 second moment, clipping, decay, and block/attention/MLP
   rematerialization.
6. **Input and cache tuning:** benchmark mmap-to-device copies, worker and
   prefetch geometry, persistent compilation cache, and static batch layouts;
   keep this behind the compute optimizations because the current gap is
   primarily device-side.

The order is intentional: prove chunked loss and compiled accumulation first,
then add attention and block fusion. Each pass requires semantic parity, HLO
evidence, HBM, and a matched v5e-8 throughput comparison before becoming the
default.

## Performance forecast (non-binding)

The measured starting point is `319K` global supervised tok/s. Chunked loss
alone should primarily reduce HBM and enable better geometry; it does not
remove vocabulary projection FLOPs, so a fixed-geometry speedup is expected to
be modest (roughly 0-25%). TPU-native attention, compiled accumulation, and
projection/block fusion are the potential large gains, but their effects are
not additive and depend on HLO fusion, layouts, and provider support.

| Optimization state | Planning range | Meaning |
|---|---:|---|
| Current generic HF/XLA path | `~319K` | Observed baseline |
| Chunked loss + stable graph | `320-420K` | HBM/graph improvement; speed uncertain |
| Add compiled accumulation and TPU attention | `450-700K` | Provider-dependent intermediate target |
| Add projection/block fusion and tuning | `650-900K` | Strong optimized outcome |
| LaughLM-class parity | `>=912.6K` | Must be measured and repeated; not guaranteed |

The `650-900K` range is engineering planning, not a promise. Reaching the
hard `>=912.6K` or preferred `>=963.3K` thresholds requires critical providers
to be active without fallback and three matched TPU runs to confirm the result.

## Public UX gate before optimization release

The notebook and `scripts/trainlm_tpu_worker.py` are internal validation
surfaces only. They must not be the documented user workflow. Before calling
the optimized path usable, expose one HF-Trainer-like `TrainLMTrainer` API and
one optional `trainlm train` CLI. Users provide only the model source (or HF
model), tokenizer/data source, output directory, and essential training
geometry; TrainLM owns worker launch, PJRT environment, rank probe, model
preflight, capability planning, transforms, cache, data validation, logging,
checkpointing, and export.

The public API must support both a concise path and an advanced escape hatch:

```python
trainer = TrainLMTrainer(
    model="org/model-or-local-path",
    train_dataset=PackedBinDataset.from_hub(
        repo_id="org/dataset", pattern="train/*.bin", tokenizer=tokenizer
    ),
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

An equivalent CLI/config path is required for non-notebook jobs. Advanced
users may supply explicit manifests, provider policies, cache paths, or
optimization requests, but validation internals and raw worker arguments stay
private. A TPU model source must be reconstructible inside each worker; live
model objects remain supported for local backends and are serialized or
recreated through an explicit model factory for TPU.

### HF-like surface contract

`TrainLMTrainer` is TrainLM's own trainer, not a wrapper around
`transformers.Trainer`, but its common surface should be recognizable:

- constructor fields: `model`, `args`, `train_dataset`, `eval_dataset`,
  `processing_class`/tokenizer, `data_collator`, and callbacks;
- lifecycle: `train(resume_from_checkpoint=...)`, `evaluate()`,
  `save_model()`, `save_state()`, and `log_metrics()`;
- familiar arguments: `output_dir`, `max_steps`, `per_device_*_batch_size`,
  `gradient_accumulation_steps`, `learning_rate`, `weight_decay`, `bf16`,
  `logging_steps`, `save_steps`, `eval_steps`, and `report_to`;
- standard HF model outputs, tokenizer conventions, Hub revisions, and
  `save_pretrained` export;
- automatic accelerator/world-size/dtype/cache selection with an explicit
  `trainer.explain()` report for every optimization or fallback decision.

TrainLM-specific arguments are additive: packed-binary source specifications,
token-normalized loss, optimizer-state precision, strict optimization policy,
and TPU diagnostics. A compatibility adapter may accept
`transformers.TrainingArguments`, but TrainLM owns execution and does not
promise every HF Trainer feature or callback side effect on TPU.

For cross-session implementation continuity, see
[`docs/IMPLEMENTATION_CONTEXT.md`](IMPLEMENTATION_CONTEXT.md). It records the
current measured baseline, public-API boundary, explicit non-goals, and the
ordered next stories for this PR branch.

### TPU lifecycle smoke follow-up (completed)

- Completed: diagnose the apparent post-step-2 stall as an unbounded scheduled
  evaluation over the full validation shard, rather than an optimizer hang.
- Completed: add a public `max_eval_batches` argument and carry it through the private
  TPU coordinator into `EvaluationConfig.max_batches`.
- Completed: preserve unbounded evaluation for legacy/lightweight trainer configs that
  do not provide an `evaluation` section.
- Completed: bound the SmolLM2 notebook/example smoke evaluation to one batch and emit
  explicit evaluation/checkpoint lifecycle messages.
- Follow-up: run an intentionally full validation pass only with a suitably
  sized validation shard and record target-hardware timing evidence. This is
  validation guidance rather than an M17 completion criterion.

### TPU smoke reliability correction

- Completed: separate train/checkpoint/resume smoke from the independently
  compiled XLA evaluation graph; keep evaluation available as a subsequent
  opt-in validation run.
- Completed: reject fatal PJRT core-dump/shutdown markers even when the launcher
  incorrectly exits with a successful status, with explicit session-restart
  guidance.
- Completed: keep heartbeat-output assertions scoped to the heartbeat contract
  and fatal-marker assertions scoped to the false-success failure contract.

### Public TPU baseline geometry correction

- Completed: restore the previously successful 135M eight-layer Llama reference
  geometry for the first public-facade TPU smoke, including sequence 2048,
  per-replica batch two, and accumulation 32.
- Completed: keep evaluation and checkpointing out of the first two-update run;
  validate those lifecycles separately after baseline training succeeds.
- Follow-up: collect target-v5e throughput, HBM, compilation, and fallback
  evidence before making any performance or certification claim.

### Single-launch public TPU lifecycle

- Completed: remove disposable probe and model-preflight launches from the
  normal public training request; the training workers perform their existing
  collective probe before model construction in the same PJRT lifetime.
- Completed: retain standalone diagnostic modes for maintainers without making
  users pay for repeated PJRT initialization and teardown.
- Follow-up: verify the single-launch lifecycle on a fresh Kaggle v5e-8 session
  and retain the resulting train log and structured summary.

### TPU notebook stale-state guard

- Completed: verify and print the exact model-independent training geometry
  immediately before the first TPU allocation.
- Completed: fail before launch when a stale notebook kernel retains evaluation,
  checkpoint, SmolLM2, or 128-token baseline state.
- Follow-up: capture the first clean 100-update result from the verified 135M
  configuration on a fresh Kaggle v5e-8 session.

### Public 319K baseline reproduction

- Completed: expose the existing WSD scheduler through the HF-like training
  arguments so the public path can reproduce the recorded baseline policy.
- Completed: configure the validation notebook for 100 updates, five excluded
  warmup updates, and ten-update metric materialization at the locked 135M
  geometry.
- Follow-up: compare the fresh target-v5e result with the recorded ~319K generic
  baseline before evaluating the 850K and 912.6K optimization gates.

### Public 319K baseline evidence

- Completed: owner-run Kaggle v5e-8 validation finalized 100 updates through the
  public single-launch `trainer.train()` path with all eight replicas.
- Completed: the post-warmup window measured 315,512 global supervised tokens/s
  (315,667 scheduled tokens/s), within 1.19% of the historical 319,302 generic
  baseline.
- Completed: classify this result as lifecycle and generic-baseline evidence,
  not performance certification; the summary correctly reports
  `performance_certified: false`.
- Follow-up: wire the real worker to the optimized attention, memory-bounded
  loss, and compiled update path before testing the ~850K and 912.6K LaughLM
  gates.

### MB4/GA16 throughput candidate

- In progress: benchmark microbatch 4 / accumulation 16 at sequence length 2048
  and DP8, preserving exactly 1,048,576 scheduled input tokens per update.
- Acceptance: complete 100 updates, preserve token/loss accounting, avoid graph
  instability or HBM failure, and improve the matched post-warmup throughput
  over the owner-run 315,512 supervised tokens/s baseline.
- Constraint: treat the result as a geometry experiment only; 500-600K still
  requires the memory-bounded loss and compiled/native update path if this
  reduced-accumulation run does not independently reach that range.

### MB8/GA8 chunked-loss throughput candidate

- Rejected by owner-run v5e-8 validation: MB8/GA8 chunked loss exhausted HBM
  (`88.00M` requested with only `84.42M` free).
- Retain the candidate in the tuning matrix for larger-memory TPU versions,
  but do not use it as the default v5e-8 geometry.
- Completed: remove the duplicate XLA graph flush at the end of each optimizer
  update and avoid a host scalar synchronization for dense packed chunked-loss
  denominators.
- Completed: align the device loader's `batches_per_execution` with the
  accumulation window (capped at eight) instead of flushing the lazy graph for
  every microbatch.

### MB4/GA16 chunked-loss fallback

- In progress: use MB4/GA16 with a 1024-token chunk as the largest known viable
  v5e-8 candidate while preserving 1,048,576 scheduled tokens/update.
- Completed: keep the XLA output projection in BF16 instead of duplicating the
  tied vocabulary matrix in FP32, removing the HBM allocation failure observed
  during the first chunked-loss launch.
- Completed: make the validation notebook assert `bf16=True`, verbose metrics,
  and the BF16 XLA projection implementation before calling `trainer.train()`;
  stale Kaggle package state now fails before TPU allocation.

### Public TPU progress reporting and state metadata

- Completed: add `quiet`, `normal`, and `verbose` coordinator logging modes;
  verbose mode renders worker JSON events as compact stage, step, loss, LR, and
  token progress while preserving the full `train.log` and JSON artifacts.
- Completed: stop the validation notebook from displaying the entire nested
  Python result object; it now prints a compact throughput summary.
- Completed: carry global batch size and global token counters into TPU metric
  snapshots and public trainer state, and expose the rank-specific cache path
  with its summary writer rank.
- Completed: expose an explicit `chunked_linear` TPU loss option with configurable
  vocabulary projection chunks and an inspected Hugging Face base-model adapter;
  the generic full-logit path remains the default pending target validation.
- Follow-up: verify rank-0-only artifact ownership and warm-cache behavior on
  the next v5e-8 run, then benchmark chunk sizes 1024/2048/4096 for HBM and
  throughput.

### Verbose TPU heartbeat de-duplication

- Completed: suppress unchanged verbose worker events so a metric snapshot is
  printed once instead of repeating on every five-second poll.
- Completed: emit sparse liveness messages during long quiet stages and include
  explicit `data_parallel`/`model_parallel` metadata in training events and the
  worker summary.

### Frontier-style TPU console metrics

- In progress: replace document rendering with one rank-zero console table
  containing step, progress, loss, perplexity, gradient norm, LR, tokens/sec,
  MFU estimates, tokens seen, remaining tokens, ETA, and elapsed time.
- Completed: keep structured `metrics.jsonl` snapshots synchronized with the
  console rows while suppressing duplicate polling output from the coordinator.

### TPU cache initialization repair

- Completed: distinguish launcher-initialized XLA caches from runtime-owned
  caches so the worker does not initialize the same PJRT computation cache twice.
- Completed: preserve cache path and initialized status in runtime diagnostics;
  added a regression contract for the externally initialized case.
- Follow-up: rerun the MB4/GA16 candidate on v5e-8 and confirm worker startup
  reaches the collective probe and training stages.

**HF usability follow-up.** The first reporter integration accepts `report_to="jsonl"` and appends training/evaluation metrics to the run directory. TensorBoard/W&B adapters remain queued. This keeps metrics visible without adding third-party dependencies.

**Fine-tuning source parity.** Fine-tuning now accepts friendly HF dataset revisions and paired local CSV/JSONL/Parquet sources. Content fingerprints for local fine-tuning sources remain planned.

**Fine-tuning training controls.** Gradient checkpointing now uses the HF model hook, and the facade passes through the requested model/causal loss choice. Local source content hashes are included in fine-tuning summaries and validated by the restore helper. Native TPU hardware behavior remains unverified.

**Chunked causal-LM loss.** The Python facade accepts `chunked_linear` when callers supply an explicit `LinearCausalLMTrainingView` factory. It uses capability inspection, checks model identity and forwards the configured chunk size. Model-specific hidden-state adapters and hardware performance evidence remain required.

**Metric visibility.** JSONL, TensorBoard, and W&B reporter callbacks now consume existing host-materialized metric snapshots. Optional packages are loaded only when selected. Pretraining plans and run summaries explicitly state the configured row ceiling so the 2,000-example default is visible before a run.

**High-level CLI workflows.** `trainlm pretrain` and `trainlm finetune` now
load YAML configurations and call the same public run APIs as Python users.
Both expose allocation-free `--dry-run` plans and checkpoint resume; pretraining
also exposes budget extension. Copy-ready TinyStories configs are in `examples/`.
The CLI supports Hub and local-file configuration, while Python-only datasets,
formatters, and callbacks remain Python API features.

**CLI Hub publishing.** Both high-level commands can explicitly publish a
successful run with `--push-to-hub REPO_ID`; new repositories default to
private, and public visibility requires `--public`. Dry-run and publication
options are mutually exclusive. The CLI prints the upload receipt without
exposing credentials. Live Hub upload remains owner-validated.

**Notebook Hub publishing check.** The short scratch/CPT notebook now includes
an optional private upload cell after a successful local export. It is disabled
by default, uses Hugging Face's standard login flow, and prints only the
repository URL and commit SHA. A real upload still must be run and reviewed by
the owner in Colab or Kaggle.


**Zero-recompile compiled optimizer and fused `F.rms_norm` (Step 8).** `Trainer`
now keeps `group["lr"]` as a persistent 0-D `torch.Tensor` updated in-place via
`.fill_()` (and enables `capturable=True` on `torch.optim.AdamW`) whenever
compiled single-step execution is active, eliminating per-step Dynamo `lr`
recompiles. `src/trainlm/optimization/rmsnorm.py` adds reversible `F.rms_norm`
patching (`apply_fused_rms_norm` / `restore_rms_norm`) for standard HF RMSNorm
modules when Liger-Kernel RMSNorm is not active, controlled by
`TrainLMTrainingArguments.use_fused_rms_norm` (default `True`). Hardware TPU/CUDA
validation remains owner-run.


**TorchTPU kernel probe, Splash auto-wiring, and 128-alignment advisor (Step 9).**
`probe_torchtpu_kernels()` / `TorchTPUKernelInventory` now probe `torch.tpu` and
`torch_tpu` kernel namespaces, and `TrainLMTrainer` auto-wires Splash Attention
when `attn_implementation="auto"` on TPU (if a Splash kernel is present and
`packing=False`). `src/trainlm/optimization/alignment.py` adds
`inspect_model_alignment` and `pad_vocab_to_multiple`, exposed via
`TrainLMTrainingArguments.pad_vocab_to_multiple` (`None`, `"auto"`, or integer).
Hardware TPU/CUDA validation remains owner-run.


**Multi-device DDP overlap, FSDP2 sharding, and `world_size > 1` MFU (Step 10).**
`src/trainlm/runtime/distributed_overlap.py` adds `prepare_distributed_model`,
`apply_distributed_data_parallel` (`bucket_cap_mb`, `gradient_as_bucket_view`,
`static_graph`), and `apply_fsdp2_fully_shard`, exposed via
`TrainLMTrainingArguments.distributed_strategy` (`"auto"`, `"ddp"`, `"fsdp2"`,
`"none"`) and `ddp_bucket_cap_mb`. `Trainer` and `build_performance_summary`
now report cluster-wide MFU for `world_size > 1` when `global_tokens_per_second`
is measured. Multi-device hardware (`v5e-8`, multi-GPU) validation remains
owner-run.
