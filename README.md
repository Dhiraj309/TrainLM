# TrainLM

TrainLM is a Hugging Face-native framework for efficient language-model
pretraining, with Google TPU execution provided through replaceable runtime and
kernel backends.

The product goal is to bring an HF model, tokenizer, and raw or tokenized
dataset; train from random weights or continue pretrained weights; then save,
optionally publish, and use the result through Transformers. The
[HF user-workflow roadmap](docs/HF_USER_WORKFLOWS.md) records what works today,
the remaining gaps, and the next implementation order. HF-02 is currently
under implementation; its example below shows the new HF config and tokenizer
options while the hosted notebook still demonstrates the legacy default.

## Start here: pretrain a language model from scratch

Open the [from-scratch notebook in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)
or [import it into Kaggle](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Dhiraj309/TrainLM/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb).
It uses one high-level call to initialize random GPT-2 weights, prepare pinned
raw text, train, evaluate, and export a Hugging Face model. The notebook keeps
the first run to a few cells; advanced controls are in the [pretraining
guide](docs/tutorials/TPU_PACKED_BIN_PRETRAINING.md). Native TorchTPU remains
experimental, and the smoke run does not establish convergence or performance.

The notebook runs a 10,000-token workflow smoke test to check setup, training,
evaluation, and export; it is not useful-model training. Increase `max_tokens`
in the config for a longer run; token count alone does not guarantee
convergence. Advanced model dimensions and budgets belong in the guide so the
first notebook path stays short.

See the [notebook catalog](notebooks/README.md) for the roles of the training
quickstarts, TPU diagnostics, and historical XLA notebook.

For a simple Python run, `run_pretraining` owns the complete lifecycle and
returns a result with the report and export paths:

```python
from trainlm import TrainLMPretrainingConfig, run_pretraining

config = TrainLMPretrainingConfig(
    output_dir="runs/tinystories",
    max_tokens=10_000,
    sequence_length=128,
    model_embedding_size=256,
    model_layers=4,
    model_attention_heads=4,
    accelerator="native_tpu",  # Explicit, experimental single-device path.
)
result = run_pretraining(config, show_progress=True)
print(result.summary())
print(result.report_path, result.export_path)
```

`result.parameter_count` reports the exact total parameter count measured from
the instantiated model; `result.summary()` includes the same count. Progress
updates use the configured `logging_steps` cadence. Omit `show_progress=True`
for a quiet library call.

Training progress also reports synchronized supervised tokens/second and
observed dataloader wait. `result.report["training"]["performance"]` contains
the aggregate measurements. Native single-device TPU v5e BF16 runs include
standard MFU values when the model geometry is supported; CPU runs and unknown
hardware/precision combinations return MFU as unavailable. See the
[measurement definition and limits](docs/benchmarks/README.md#user-run-measured-training-performance).

With no model or tokenizer source configured, the original convenience path
builds a small GPT-2 config and trains a byte-level BPE tokenizer on the
training split. To use an architecture and tokenizer already hosted on the HF
Hub, set `model_name_or_path`; TrainLM loads its config and tokenizer, then
creates random model weights without downloading pretrained weights:

```python
config = TrainLMPretrainingConfig(
    output_dir="runs/gpt2-from-config",
    model_name_or_path="openai-community/gpt2",
    model_revision="main",
    model_initialization="random",
    max_tokens=10_000,
    sequence_length=128,
    accelerator="native_tpu",
)
result = run_pretraining(config, show_progress=True)
```

For continued pretraining (CPT), use the same config with
`model_initialization="pretrained"`; TrainLM loads the existing weights and
defaults to the tokenizer in that model repository. Set
`tokenizer_name_or_path` and `tokenizer_revision` to select a separately hosted
tokenizer. TrainLM checks known token-ID bounds and context length, never
resizes pretrained embeddings or remaps IDs, and reports that semantic token
compatibility for an independently selected tokenizer remains your
responsibility. Hub `revision` values may be branch or tag names; TrainLM pins
model config and weights to the resolved snapshot recorded in the run report.

Set `tokenizer_mode="train_bpe"` to explicitly use the corpus-trained BPE
tokenizer with a random model. It cannot be combined with pretrained weights.
For a locally cached source without Hub access, set `local_files_only=True`.
Use `trust_remote_code=True` only when the selected repository requires its
own model or tokenizer implementation. Architecture fields from the selected
HF config are preserved; `model_config_overrides` applies explicit changes.

To use your own corpus instead of the default pinned Hub dataset, provide
separate UTF-8 text files for training and validation. Each non-empty line is
one text example:

```python
config = TrainLMPretrainingConfig(
    output_dir="runs/my-corpus",
    train_file="data/train.txt",
    validation_file="data/validation.txt",
    max_tokens=1_000_000,
    sequence_length=512,
    model_type="gpt2",
    accelerator="native_tpu",
)
result = run_pretraining(config, show_progress=True)
```

TrainLM records both file hashes and refuses to reuse prepared data if either
file changes under the same `output_dir`; choose a fresh run directory after
changing the corpus.

The scratch API defaults to cosine decay with automatic warmup (floor of 10%
of the step budget, capped at 100 steps). Choose `lr_scheduler_type` from
`"constant"`, `"linear"`, `"cosine"`, or `"wsd"`. Set either `warmup_ratio`
from 0 to 1 or a non-negative integer `warmup_steps` to override automatic
warmup; the options are mutually exclusive. A ratio rounds up to a whole
optimizer step, matching Transformers:

```python
config = TrainLMPretrainingConfig(
    output_dir="runs/my-corpus",
    train_file="data/train.txt",
    validation_file="data/validation.txt",
    max_tokens=1_000_000,
    lr_scheduler_type="cosine",
    betas=(0.9, 0.95),
    eps=1e-8,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    save_steps=1_000,
    eval_steps=500,
    load_best_model_at_end=True,
    early_stopping_patience=3,
    accelerator="native_tpu",
)
```

Use `warmup_steps=500` instead when you want an exact number of warmup steps.
By default, TrainLM saves and evaluates roughly every 10% of the planned update
budget. `save_steps` and `eval_steps` override those intervals independently;
each is a positive number of optimizer steps. The report records both
requested and effective intervals.

AdamW uses the familiar `betas` and `eps` settings. `max_grad_norm` controls
gradient clipping (default `1.0`); set it to `0` to disable clipping. These
values are recorded in the run report and checked when resuming a run.

To export the weights from the best validation checkpoint, set
`load_best_model_at_end=True`. Scratch pretraining currently selects the lowest
`eval_loss`. `save_steps` must be a multiple of `eval_steps`, and at least one
evaluation must fit inside the planned run. Add `early_stopping_patience` to
stop after that many consecutive evaluations without a significant loss
improvement; `early_stopping_threshold` sets the minimum loss improvement.
The selected step and checkpoint are available as
`result.best_model_selection`. A valid early stop is reported as a successful
run and marked by `result.training["early_stopping_triggered"]`. The exported
model uses the best weights while the latest checkpoint remains available for
resume. This feature has no native TPU hardware validation.

Extending an existing run's token budget remains limited to runs whose
effective schedule, warmup, save interval, and evaluation interval match the
automatic cosine defaults; custom settings need a fresh run directory.

For a short copyable script, see [`examples/pretrain_from_scratch.py`](examples/pretrain_from_scratch.py).
The [examples index](examples/README.md) also separates user quickstarts from
runtime diagnostics.

If setup, training, evaluation, or export fails, `TrainLMPretrainingError`
points to the saved report. For a custom lifecycle, the lower-level
`TrainLMTrainer.from_scratch(config)` API remains available.

`save_total_limit` also works in `TrainLMTrainingArguments` for single-device
CPU, CUDA, and explicit native TorchTPU training. The API default, `None`, keeps
all saves. Cleanup runs after a successful save and
preserves the new checkpoint. When `save_steps` is configured, TrainLM also
saves a final checkpoint after a successful run if its last step falls between
periodic save steps. Unmarked legacy checkpoints, incomplete saves,
symlinks, separately named manual backups, and folders with exports or user
files are excluded, so the directory may contain more than the configured
limit. The limit can change on resume without changing training compatibility.
Distributed retention and the legacy XLA coordinator are not supported yet;
native TPU behavior still requires hardware validation.

After training, `trainer.latest_checkpoint` returns the newest published
checkpoint as a `Path`, or `None` if no checkpoint exists. To resume in a new
session, create a trainer with the same `output_dir` and call
`trainer.train(resume_from_checkpoint="latest")`.

For a different built-in Transformers causal-LM architecture, set `model_type`
and pass its architecture-specific fields. TrainLM derives `vocab_size` and
special-token IDs from the tokenizer it trained on your corpus:

```python
config = TrainLMPretrainingConfig(
    output_dir="runs/tinystories-llama",
    max_tokens=10_000,
    sequence_length=128,
    model_type="llama",
    model_config_overrides={
        "hidden_size": 512,
        "intermediate_size": 1_376,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
    },
    accelerator="native_tpu",
)
trainer = TrainLMTrainer.from_scratch(config)
```

The default `model_embedding_size`, `model_layers`, and
`model_attention_heads` settings apply to GPT-2. For other architectures,
provide the relevant Transformers config fields in `model_config_overrides`.
Only model types supported by `AutoModelForCausalLM` are suitable; TrainLM does
not enable custom remote model code unless `trust_remote_code=True` is
explicitly selected. Dense causal-LM compatibility and accelerator support
still depend on the model implementation and runtime.

`run_pretraining` writes a versioned report containing the effective settings,
model and tokenizer details, selected backend, package versions, and training
metrics. `bundle_pretraining_artifacts(result.report_path)` packages the
report, a resume helper when a checkpoint is available, and the model export
when complete. Keep the full run directory to resume; the model ZIP does not
contain optimizer state or packed data. The [pretraining guide](docs/tutorials/TPU_PACKED_BIN_PRETRAINING.md)
covers report recovery and longer runs.

## Fine-tune a pretrained model

For a text dataset on the Hub, `run_finetuning` loads the pretrained model and
tokenizer, tokenizes the selected text column, trains against a separate
validation split, evaluates, and exports the model and tokenizer locally.
Pin the dataset revision to an immutable commit; pin the model revision when
reproducibility requires it.

```python
from trainlm import TrainLMFineTuningConfig, TrainLMTrainingArguments, run_finetuning

result = run_finetuning(TrainLMFineTuningConfig(
    model_name_or_path="sshleifer/tiny-gpt2",
    dataset_id="roneneldan/TinyStories",
    dataset_revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    args=TrainLMTrainingArguments(
        output_dir="runs/tinystories-finetune",
        max_steps=3,  # Workflow smoke only; increase for real training.
        sequence_length=128,
    ),
))
print(result.summary())
```

This path keeps validation data separate and writes the export to
`runs/tinystories-finetune/model` plus a machine-readable run summary to
`runs/tinystories-finetune/finetuning-summary.json` (`result.summary_path`). The
summary records training arguments, data splits, resolved model/tokenizer
revisions, and evaluation metrics. If a run fails, the same file records the
failed phase, error type, and last known step/checkpoint; the original
exception is still raised. It does not upload to the Hub. Install the
optional dataset dependency with `pip install 'trainlm[finetuning]'`. The
standard `TrainLMTrainer` remains available when you already have a model and
tokenized datasets or need a custom lifecycle. Native TorchTPU fine-tuning is
experimental and has not been certified by this API example.

If your tokenizer is hosted separately, set `tokenizer_name_or_path` and
`tokenizer_revision` on `TrainLMFineTuningConfig`. TrainLM resolves that source
independently, checks known token-ID bounds and context length, and reports
token-semantic compatibility as unverified. Choose a tokenizer whose token
meanings match the model; TrainLM does not resize or remap pretrained weights.

For custom formatting, pass preloaded Hugging Face `Dataset` train and
validation splits, a `formatting_func` that returns one text string per row,
and optionally a tokenizer and custom `data_collator`. In this mode,
`dataset_id` and `dataset_revision` are optional; provide them when you want
the report to identify a source, but no placeholder Hub ID or revision is
needed for local or transformed data. Both remain required when TrainLM loads
the splits from the Hub. See the
[fine-tuning guide](docs/tutorials/TEXT_FINETUNING.md) for this pattern and
resume requirements.

`TrainLMTrainingArguments` accepts either a step budget or
`num_train_epochs` (`max_steps=None`). Epoch budgets count finite map-style
data, respect batch size and gradient accumulation, and currently require a
single-device runtime. `eval_strategy` and `save_strategy` schedule evaluation
and training checkpoints by steps or at epoch boundaries; epoch schedules
also require finite map-style data and flush partial accumulation at each
boundary. Set `load_best_model_at_end=True` to export the weights from the
best validation metric; evaluation and saving must use matching strategies,
and step-based save intervals must be multiples of the evaluation interval.
The default selection metric is the lowest `eval_loss`. See the
[fine-tuning guide](docs/tutorials/TEXT_FINETUNING.md#restore-the-best-evaluation-checkpoint)
for metric selection, retention, resume behavior, and current limitations.
Set `early_stopping_patience` to stop after consecutive validation checks fail
to improve the selected metric by `early_stopping_threshold`; it requires
`load_best_model_at_end=True` and uses the same evaluation/save schedule.

To resume from the latest checkpoint, restore the saved run configuration and
set a larger step budget. The helper reloads model, dataset, and training
settings from the summary, anchors the output directory to its location, and
pins resolved model/tokenizer commits rather than mutable branches. If the
summary marks `tokenizer_object_required`, pass the original tokenizer object
again; TrainLM checks its recorded commit when available:

```python
from trainlm import restore_finetuning_config, run_finetuning

config = restore_finetuning_config(
    "runs/qwen-tinystories/finetuning-summary.json",
    max_steps=2000,
)
result = run_finetuning(config, resume_from_checkpoint="latest")
```

See the [fine-tuning guide](docs/tutorials/TEXT_FINETUNING.md) for setup
requirements and limitations. The [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Text_Fine_Tuning.ipynb)
and [`examples/finetune_text.py`](examples/finetune_text.py) show the short
first-run path. The notebook targets CPU or GPU; native TPU fine-tuning has not
been validated.

The quick-start configs use a pinned TinyStories revision. In a standard Python
environment, install the package with `pip install 'trainlm[pretraining]'` for
from-scratch runs or `pip install 'trainlm[finetuning]'` for fine-tuning.
The notebook setup cells install TrainLM without dependency resolution, then
install its Transformers/data-library requirements separately. They do not
request PyTorch or TorchTPU, so those runtime packages stay in place.

- [Open the notebook in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)
- [Read the notebook source](notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)
- [Inspect the public pretraining config and raw-text preparation](src/trainlm/pretraining.py) · [one-call run API](src/trainlm/pretraining_run.py)
- [Inspect the public trainer API](src/trainlm/api.py)
- [Review the public API's CPU lifecycle coverage](tests/test_pretraining_api.py)
- [Review one-call run API coverage](tests/test_pretraining_run.py)
- [Inspect saved resume generation](src/trainlm/_notebook_resume.py)
- [Review executable resume and ZIP tests](tests/test_notebook_resume.py)
- [Find files and current evidence for coding agents](docs/AI_AGENT_GUIDE.md)

## Project status and roadmap

The current development direction is native TorchTPU on Kaggle TPU v5e-8.
The established backend is PyTorch/XLA. A single-device native TorchTPU path is
implemented behind explicit selection, but remains experimental and is not
hardware-certified. Existing XLA code is being classified for reuse,
replacement, or preservation as an optional backend.

- [Current roadmap and documentation guide](docs/README.md)
- [T0–T11 native TorchTPU roadmap](docs/TORCHTPU_REBASELINE.md)
- [T0.1 code inventory](docs/TORCHTPU_SALVAGE_INVENTORY.md)
- [Accepted T0.2 work mapping](docs/PR_DELIVERY_PLAN.md#t02-accepted-mapping-to-native-torchtpu-gates)
- [T1 v5e-1 probe and run guide](docs/T1_V5E1_PROBE.md)
- [Kaggle/Colab-style T1 probe notebook](notebooks/TorchTPU_T1_v5e1_Probe.ipynb)
- [Experimental native single-device training notebook](notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb)
- [Native TorchTPU from-scratch pretraining notebook](notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)

The M0–M17 material below and in the detailed roadmap records the earlier
PyTorch/XLA development. Treat its completed, partial, and deferred labels as
XLA-era status; they do not certify TorchTPU. Before relying on a model-support
or performance claim, read the [dense-AR support contract](docs/SCOPE.md).

Model architecture and training-policy boundaries are defined by the
[configuration ownership contract](docs/configuration/OWNERSHIP.md).

Device-specific execution is isolated behind the
[execution backend protocol](docs/runtime/BACKEND_PROTOCOL.md).
The optional PyTorch/XLA implementation follows the
[XLA runtime contract](docs/runtime/PYTORCH_XLA.md).
Its logical data-parallel mesh follows the
[XLA SPMD mesh contract](docs/runtime/XLA_SPMD.md).
Persistent compilation caching and fixed-shape enforcement follow the
[XLA cache and shape-guard contract](docs/runtime/XLA_CACHE_GUARD.md).
Accumulation choices and evidence-gated fallbacks follow the
[XLA accumulation strategy contract](docs/runtime/XLA_ACCUMULATION.md).
Compile counters, fallback reports, HLO attachments, and XProf metadata follow
the [XLA diagnostics contract](docs/runtime/XLA_DIAGNOSTICS.md).

Next-token shifting, masking, normalization, and accounting are fixed by the
[causal language-model task contract](docs/tasks/CAUSAL_LM.md).

Optimization discovery and provider selection exchange immutable
[capability and execution-plan schemas](docs/optimization/SCHEMAS.md).

Exact resume state and plain Transformers output follow separate
[checkpoint and Hugging Face export contracts](docs/checkpoint/CONTRACTS.md).
Generic TPU save/resume/export comparisons follow the
[round-trip contract](docs/checkpoint/TPU_ROUND_TRIP.md).
Distributed checkpoint state ownership follows the
[distributed resume contract](docs/checkpoint/DISTRIBUTED_RESUME.md).
Asynchronous publication follows the
[checkpoint lifecycle contract](docs/checkpoint/ASYNC_LIFECYCLE.md).
Canonical model publication follows the
[Hugging Face export contract](docs/checkpoint/CANONICAL_HF_EXPORT.md).
Synchronized throughput, HBM, compile, and fallback metrics follow the
[TPU telemetry contract](docs/monitoring/TELEMETRY.md).
Sparse numerical and data-continuity checks follow the
[training-integrity contract](docs/monitoring/INTEGRITY_GATES.md).

Generic dense causal models are acquired through the
[Hugging Face causal-model provider](docs/models/HUGGINGFACE_PROVIDER.md).
Model batches follow the model's own
[forward-signature dispatch contract](docs/models/BATCH_DISPATCH.md).
Trainer lifecycle, stop, evaluation, and checkpoint hook ordering follow the
[backend-neutral lifecycle contract](docs/training/LIFECYCLE.md).
Microbatch reduction and token accounting follow the
[token-normalized accumulation contract](docs/training/ACCUMULATION.md).
Optimizer construction and moment precision follow the
[optimizer state policy](docs/optimization/OPTIMIZER_POLICY.md).
Token-indexed warmup/stable/decay scheduling follows the
[WSD scheduler contract](docs/training/TOKEN_SCHEDULER.md).
Streaming, token-weighted evaluation follows the
[streaming evaluation contract](docs/evaluation/STREAMING_EVALUATION.md).
Host callbacks consume immutable, materialized metrics through the
[sync-safe callback contract](docs/training/CALLBACKS.md).
Generic dense-AR trainability is checked across representative HF families by
the [multi-family overfit matrix](docs/training/MULTI_FAMILY_OVERFIT.md).
Updated models must pass the
[plain Hugging Face round-trip contract](docs/models/PLAIN_HF_ROUNDTRIP.md).
Representative non-Llama families are exercised by the
[dense-AR CPU conformance matrix](docs/models/DENSE_AR_CPU_MATRIX.md).
Every loaded model has a stable
[generic compatibility explanation](docs/models/COMPATIBILITY_EXPLANATION.md)
with explicit capabilities, providers, and fallbacks.
Positional encoding detection and the TPU conformance procedure are defined in
the [positional-semantics contract](docs/models/POSITION_SEMANTICS.md).
Attention head and QKV projection coverage follows the
[attention-layout contract](docs/models/ATTENTION_LAYOUTS.md).
Normalization, MLP, and residual coverage follows the
[transformer-block contract](docs/models/BLOCK_LAYOUTS.md).

Packed `.bin` inputs are accepted only through the versioned
[binary shard manifest and integrity contract](docs/data/PACKED_BINARY_MANIFEST.md).
Remote shards use the
[revision-pinned Hugging Face dataset source](docs/data/HUGGINGFACE_SHARD_SOURCE.md).
Validated payloads are exposed through the
[contiguous fixed-shape batch reader](docs/data/CONTIGUOUS_BATCH_READER.md).
Shard order and rank ownership follow the
[deterministic packed-data partition contract](docs/data/DETERMINISTIC_PARTITIONING.md).
Packed reads can overlap training through the bounded, backend-aware
[asynchronous prefetch contract](docs/data/ASYNC_PREFETCH.md).
Exact next-batch restart state follows the
[resumable cursor contract](docs/data/RESUMABLE_CURSOR.md).
The [historical PyTorch/XLA validation notebook](notebooks/TrainLM_TPU_Validation.ipynb)
records an earlier proposed end-user workflow. For the current native
TorchTPU direction, use the linked native notebooks above. TrainLM's target
workflow is to construct familiar arguments and call
`trainer.train()`. TrainLM keeps topology discovery, worker orchestration,
preflight, evaluation cadence, checkpointing, and resume behind that call. The
[secure packed-bin TPU guide](docs/tutorials/TPU_PACKED_BIN_PRETRAINING.md)
covers secret handling, immutable revisions, train/eval splits, explanation,
resume, and the current canonical-export boundary.

TrainLM distinguishes models that are **Compatible**, **Optimized**, and
hardware **Certified**. Generic execution is never presented as TPU performance
certification.
See the [Dense-AR V1 development support status](docs/release/DENSE_AR_V1_STATUS.md)
and its [machine-readable manifest](support/dense_ar_v1.json) for current
versions, hardware paths, providers, fallbacks, and caveats.

## Public trainer (M8-F0 in progress)

The intended user workflow is a small Hugging Face-like surface. TrainLM owns
the backend, validation, optimization planning, logging, and checkpoint
lifecycle; users do not launch the TPU worker or parse its logs.

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

The same public facade supports a versioned YAML workflow. Dataset objects stay
in user code while model acquisition and familiar training arguments live in
configuration:

```python
from trainlm import TrainLMTrainer

trainer = TrainLMTrainer.from_config(
    "examples/dense_ar_pretraining.yaml",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train(resume_from_checkpoint="runs/dense-ar/checkpoint-500")
```

Use `TrainLMTrainer.from_pretrained("org/model", revision="...")` for the
equivalent code-first pretrained workflow. Public config files carry an
`api_version`; renamed keys emit `DeprecationWarning` for one public API version
before removal.

The equivalent packed-data command-line path delegates to the same public
trainer and dataset adapters:

```bash
trainlm train \
  --config examples/dense_ar_pretraining.yaml \
  --train-manifest-dir data/packed/train \
  --eval-manifest-dir data/packed/validation \
  --resume-from-checkpoint runs/dense-ar/checkpoint-500
```

The CLI prints one structured JSON training result. TPU worker commands, PJRT
configuration, coordinator stage logs, and raw manifests remain private.
Packed readers use the YAML `sequence_length` and training seed; validation
always uses its deterministic validation partition semantics.

Numbered Hub `.bin` shards can be selected directly without downloading or
authoring manifests by hand:

```bash
trainlm train --config examples/dense_ar_pretraining.yaml \
  --dataset-repo LaughTaleAI/LaughLM-Tokenized-Fine \
  --dataset-revision main \
  --train-shard-start 0 --train-shard-stop 8 \
  --eval-shard-start 8 --eval-shard-stop 9
```

Step and epoch save/evaluation strategies run through the same backend-neutral
training lifecycle. Epoch actions require a finite map-style single-device
loader and flush partial gradient accumulation at its boundary. A local run
can continue from a TrainLM training checkpoint with
`trainer.train(resume_from_checkpoint="runs/example/checkpoint-100")`.

Custom stateful datasets can include their position in a checkpoint by
implementing `state_dict()` and `load_state_dict()` with
`dataloader_num_workers=0`. They can also implement
`validate_state_dict(state)` to reject incompatible saved state during
preflight, before TrainLM restores model, optimizer, scheduler, or trainer
state. This validator must not change the active data position. Without the
optional validator, the dataset is still loaded during restore, so an error in
`load_state_dict()` can occur after the trainer state has been applied.
See the [custom iterable resume regression](tests/test_packed_dataset_resume.py)
for a CPU example that compares resumed and uninterrupted model weights.

Packed token shards can be downloaded by end-exclusive numeric range. TrainLM
resolves the requested Hub revision to an immutable commit, uses the standard
Hugging Face cache, validates every downloaded token payload, and partitions
examples deterministically:

```python
from trainlm import PackedBinDataset

train_dataset = PackedBinDataset.from_hub(
    "LaughTaleAI/LaughLM-Tokenized-Fine",
    revision="main",
    shard_range=(0, 4),  # downloads 00000 through 00003
    sequence_length=2048,
)
```

The facade delegates local runs to the portable CPU/CUDA engine and routes
`accelerator="tpu"` through a private single-VM coordinator. The initial TPU
bridge accepts a reconstructible pretrained Hugging Face model source and a
local directory of validated shard manifests; public packed-binary dataset
construction, TPU checkpoint parity, and automatic capability-based kernel
planning remain in progress. Structured worker metrics are delivered through
public callbacks; worker scripts and stage logs are internal details.

## Native `TorchTPU` + Pallas Benchmarks (`TPU v5e-1`, `seq=2048`, `torch-tpu==0.2.0`)

`LaughLM-Loop` combines a shallow Prelude + shared 4-layer Recurrent Core (looped `R=4` times for 20 effective layers with truncated BPTT) + Coda over 128-aligned geometry (`hidden_size=896`, `head_dim=128`, `vocab_size=49152`), using our custom VMEM-fused Pallas kernel `fused_recurrent_inject_rmsnorm` (`RMSNorm(h_t + sigmoid(gate) * h_0)`) and `pallas_splash_attention` to avoid the un-fused norm overhead observed on `Qwen3-0.6B`.

Measured on **Cloud TPU v5e-1** (`16.0 GiB` HBM, `197 TFLOP/s` BF16 peak, `torch==2.14.1+cpu`, `torch-tpu==0.2.0`, `libtpu==0.0.50`, `transformers==5.22.1`, `loss_implementation="chunked_linear"`, `optim="adamw_bf16"`, steady-state Windows 2+, `wait=0.0%`):

| Model | Mode | Batch Size | Throughput | Non-Logit MFU | Total MFU | Host Wait |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `SmolLM2-135M` | Eager | 8 | `28,881.3 tok/s` | `15.57%` | `18.06%` | `0.0%` |
| `SmolLM2-135M` | Compiled | 8 | `54,645.4 tok/s` | `29.46%` | `34.17%` | `0.0%` |
| `SmolLM2-360M` | Compiled | 4 | `23,549.2 tok/s` | `31.59%` | `34.98%` | `0.0%` |
| `Qwen2.5-0.5B` | Compiled | 2 | `24,782.6 tok/s` | `33.66%` | `43.94%` | `0.0%` |
| `Qwen3-0.6B` | Compiled | 2 | `11,560.0 tok/s` | `19.64%` | `25.12%` | `0.0%` |
| `LaughLM-Loop-135M` | Compiled (`Pallas Off`, `eff_L=20`) | 4 | `62,873.3 tok/s` | `46.13%` | `54.56%` | `0.0%` |
| `LaughLM-Loop-135M` | Compiled (`Pallas On`, `eff_L=20`) | 4 | **`62,947.8 tok/s`** | **`46.18%`** | **`54.63%`** | **`0.0%`** |
| `LaughLM-Loop-135M` | Compiled (`Pallas On`, `eff_L=20`) | 8 | `61,004.9 tok/s` | `44.76%` | `52.94%` | `0.0%` |

## Stage-1 Pilot & Held-Out Loop Scaling (`TPU v5e-1`)

End-to-end from-scratch Stage-1 pretraining pilot published at [`Dhiraj45/laughlm-loop-135m-pilot`](https://huggingface.co/Dhiraj45/laughlm-loop-135m-pilot) (model card source: [laughlm-loop-135m-pilot.md](docs/model_cards/laughlm-loop-135m-pilot.md)) on [`Dhiraj45/laughlm-135m-shards-v2`](https://huggingface.co/datasets/Dhiraj45/laughlm-135m-shards-v2):

### Stage-1 Pilot Run Summary (`131.07M` Tokens on `TPU v5e-1`)
- **Command / Config:** `stage1_base_30b.py --loop-preset pilot-1b --arch laughlm_loop` (`1,000` optimizer steps, `micro_batch_size=4`, `grad_accum=16`, `global_batch=64`, `seq_len=2048` = `131,072` tokens/step; `131,072,000` total tokens on `stream_stage1_w00_shard_00000.bin`).
- **Sustained Training Throughput:** **`73,240.1 tok/s`** | **`53.73%` non-logit MFU** (**`~63.6%` total MFU**) | **`0.0%` host wait**.
- **XLA Compilation Cache:** `99.9%` hit rate (`17,145 / 17,157` cache requests, `12` total compilations).
- **Convergence:** Cross-entropy loss dropped from `10.9748` (`58,384` PPL at step 5) to **`4.2332` (`68.93` PPL at step 1,000; lowest 5-step window `3.9422` / `51.53` PPL at step 885)**.

### Held-Out Test-Time Loop Scaling (`stream_stage1_w00_shard_00029.bin`, `131,072` Tokens)
| Recurrent Loops ($R$) | Effective Layers | Held-Out Loss | Perplexity ($\text{PPL}$) | $\Delta$ PPL vs. $R=1$ | Eager Eval Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$R = 1$** *(warmup + eval)* | 8 | `4.8129` | `123.09` | Baseline | `2,414.0 tok/s` *(includes initial XLA trace)* |
| **$R = 2$** | 12 | `4.7035` | `110.33` | `-10.4%` | `44,089.2 tok/s` |
| **$R = 4$** *(trained depth)* | **20** | **`4.2198`** | **`68.02`** | **`-44.7%`** | **`30,042.5 tok/s`** |
| **$R = 6$** *(extrapolated)* | 28 | `4.2892` | `72.91` | `-40.8%` | `22,368.5 tok/s` |
| **$R = 8$** *(extrapolated)* | 36 | `4.2511` | `70.18` | `-43.0%` | `16,652.9 tok/s` |

### Qualitative Pilot Generations (`R=4, eff_L=20`, `0.37%` of 35B Target)
Representative completions from the `131.07M`-token pilot (`TTFT ~69–89 ms`, `8.0–9.9 tok/s` eager CPU decode):
- **Narrative (`Once upon a time, in a quiet village by the sea`):**
  > `Once upon a time, in a quiet village by the sea, I can see you as an ocean turtle. We visited a 31-mile ship that was built in the 1980s, and the 2024-mile ship that was built in the 1970s. We were also known to have a very large colony` (`TTFT=76.2 ms`, `8.85 tok/s`)
- **Dialogue / Explanatory (`The key difference between a star and a planet is`):**
  > `The key difference between a star and a planet is that the universe is not an absolute element of the mind. “We are a complex human and are at the center of the earth,” said Dr. Gaudimrabran, the world's first leader and co-author. “I am able to prove myself as true in this world.”` (`TTFT=88.8 ms`, `8.00 tok/s`)
- **Academic (`In 1969, engineers at NASA`):**
  > `In 1969, engineers at NASA in the National Academy of Engineering (2013). By the University of New York (2014), with its own technical language, it is possible to find out more about the science of the study. The researchers have demonstrated that researchers in the United States have been able to produce the data on the` (`TTFT=80.5 ms`, `9.40 tok/s`)
