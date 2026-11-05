# TrainLM

TrainLM is a Hugging Face-native framework for efficient language-model
pretraining, with Google TPU execution provided through replaceable runtime and
kernel backends.

## Start here: pretrain a language model from scratch

The [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)
uses TrainLM's public from-scratch API. By default it initializes random GPT-2
weights, learns a tokenizer from pinned raw text, prepares packed data, trains,
evaluates, and exports a Hugging Face model. You can also select a registered
Transformers causal-LM architecture and pass its config fields directly. The
native TorchTPU path is experimental; the starter run does not establish
convergence or TPU performance.

```python
from trainlm import TrainLMPretrainingConfig, TrainLMTrainer

config = TrainLMPretrainingConfig(
    output_dir="runs/tinystories",
    max_tokens=10_000,
    sequence_length=128,
    save_total_limit=3,  # Keep up to three automatic training checkpoints.
    accelerator="native_tpu",  # Explicit, experimental single-device path.
)
trainer = TrainLMTrainer.from_scratch(config)
trainer.train()
metrics = trainer.evaluate()
trainer.save_model("runs/tinystories/model")
```

`save_total_limit` also works in `TrainLMTrainingArguments` for single-device
CPU, CUDA, and explicit native TorchTPU training. The API default, `None`, keeps
all saves; the notebook selects `3`. Cleanup runs after a successful save and
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
`trainer.train(resume_from_checkpoint="latest")`; the notebook prints the
resolved path for easy copying.

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
not load custom remote model code in this scratch path.

The notebook's downloadable ZIP includes a version 6 evidence report with the
requested and effective training settings, optimizer and scheduler details,
selected backend, runtime TPU discovery and device properties, model config,
tokenizer vocabulary and special-token IDs, tokenizer artifact hash, parameter
counts, source and dataset revisions, package versions, and training metrics.
The top-level `checkpoint.latest_path` field records the newest published
checkpoint path, including on failed runs when a prior checkpoint remains.
This path belongs to the notebook runtime; keep `RUN_DIR` on persistent storage
such as Google Drive if it must remain available after a reset.
When a checkpoint is available, `resume.code` captures a runnable Python
example with every pretraining setting and the recorded checkpoint path. The
download cell displays it and includes the same code as `resume.py` in the ZIP.
Restore the full original run directory and match the report's source revision
and package versions before running it. The ZIP contains a model export, not
the optimizer checkpoints or packed data needed for training recovery. The
example keeps the saved token budget; use the notebook's explicit extension
settings to add more training to a completed run. Without a checkpoint,
`resume.code` is `null` and no resume script is included.
To recover directly in the notebook, restore the same `RUN_DIR`, set
`RESUME_FROM_REPORT = True`, and run all cells. The notebook checks the report,
restores its TrainLM source revision and pretraining settings, and verifies the
recorded package versions before starting. It then resumes, evaluates, exports,
and refreshes the report without copying the settings into a separate script.
Optional distributions and unavailable hardware properties are recorded as
`null`. Property discovery does not allocate a device; the report labels its
unverified query separately from the active runtime's status.

The config defaults to a pinned TinyStories revision. In a standard Python
environment, install the package with `pip install 'trainlm[pretraining]'`.
The notebook installs only missing `datasets` and `tokenizers` helpers; it
does not reinstall the Torch/TorchTPU packages supplied by the TPU runtime.

- [Open the notebook in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)
- [Read the notebook source](notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)
- [Inspect the public pretraining config and raw-text preparation](src/trainlm/pretraining.py)
- [Inspect the public trainer API](src/trainlm/api.py)
- [Review the public API's CPU lifecycle coverage](tests/test_pretraining_api.py)
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

On CPU and CUDA, `save_steps` and `eval_steps` are handled by the same
backend-neutral lifecycle used for training. A local run can continue from a
TrainLM training checkpoint with
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
