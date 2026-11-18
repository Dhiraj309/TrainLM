# TrainLM: Hugging Face user workflows and delivery backlog

**Priority reset requested by the owner on 2026-11-17.** This is the active
product delivery plan. T0–T11 remains the architecture and accelerator evidence
plan. Story labels below distinguish merged work, the current implementation
branch, and work that remains planned.
The baseline audit is merged `main` at
`9591d8b9823c9a8757630d1b79e724e3590d7028` (PR #138). The latest merged
`main` includes HF-02 at `c9ece9a2e9057260586636a428dfd7679c42e8fa` (PR #139).

## The experience we are building

A user should be able to choose an HF model, tokenizer, and dataset, choose
random initialization or continued pretraining (CPT), train, save, optionally
publish to the Hub, and load the result with Transformers. They should reuse
the HF resources and objects they already have. They should not need to learn
TrainLM's internal packed-file format, rebuild an architecture by hand, train
a tokenizer unnecessarily, write training loops, or manage TPU workers.

The next implementation priority is this complete workflow. More notebook
presets and longer smoke budgets are not substitutes for missing API behavior.
Notebooks should demonstrate the shared public API; preparation, validation,
source resolution, and runtime setup belong in the library.

Three starting points must stay distinct:

| User intent | Model and tokenizer | Training state |
| --- | --- | --- |
| Train from scratch using an HF architecture | Load the HF config, create random weights, normally reuse its compatible HF tokenizer; architecture overrides are optional | New optimizer, scheduler, RNG and data position |
| Continue pretraining an existing HF model | Load pretrained weights and preserve the tokenizer's token meanings | New training run by default; a new corpus and schedule are allowed |
| Resume an interrupted TrainLM run | Restore the checkpoint's model and compatible tokenizer/data configuration | Restore optimizer, scheduler, RNG, progress and data position under the supported resume contract |

CPT uses the causal next-token objective over ordinary corpus text. It is not
the same operation as checkpoint resume or instruction tuning with selective
label masking. The existing fine-tuning implementation already provides some
of this causal-LM behavior; reuse it instead of building a parallel engine.

“Any HF model/dataset” is an interoperability goal, not a universal support
claim. The initial contract covers compatible dense decoder-only causal LMs,
text or token columns, and models that fit the selected runtime. HF hosting or
`AutoModelForCausalLM` registration alone does not prove native TPU support.
MoE, multimodal models, diffusion language models, and unsupported custom code
remain separate scopes. Unsupported cases need an actionable early error.

## What exists and what is missing

“Present” below means source code exists; accelerator validation is separate.

| User need | Audited baseline | Gap to close |
| --- | --- | --- |
| Bring an HF model object and tokenizer | `TrainLMTrainer(model=..., processing_class=..., ...)` accepts them | Make this a first-class documented route, including prepared datasets and custom collators |
| Random initialization from a Hub config | `TrainLMPretrainingConfig.model_name_or_path` loads an HF config and initializes random weights through `from_config`; model weights are not fetched | HF-02 is merged; live Hub and TPU execution remain unverified |
| Start from pretrained weights for CPT | `TrainLMPretrainingConfig.model_initialization="pretrained"` loads the HF checkpoint; the same pretraining lifecycle accepts raw text | HF-02 is merged; prepared-token dataset input and public upload remain later work |
| Reuse an HF tokenizer | Scratch defaults to the model repository's tokenizer or accepts independent `tokenizer_name_or_path` and `tokenizer_revision`; `tokenizer_mode="train_bpe"` retains corpus training for random initialization | HF-02 fingerprints tokenizer mapping and resolved source; independently hosted token meaning remains the user's responsibility |
| Independently hosted tokenizer | `run_finetuning` accepts separate model/tokenizer IDs and revisions, resolves them independently, and checks known token-ID bounds and static context limits | Token meanings cannot be proven from Hub IDs or vocabulary sizes; the report marks them unverified and an explicit user policy remains |
| Raw HF datasets | Scratch streams Hub text with a dataset config, named revision, selected split or non-overlapping train/eval slices; fine-tuning accepts HF `Dataset` splits and row formatters | Scratch pretraining still needs direct `Dataset` inputs, arbitrary data files, and a shared text formatter contract |
| Pretokenized HF datasets | `run_finetuning(..., data_format="tokens")` accepts HF token rows with `input_ids` and optional labels/masks, validates shapes and token ranges, and pads ragged batches | Apply the same source and token contract to random-init/CPT pretraining; record tokenizer provenance and data identity consistently across workflows |
| Modern HF vocabularies | Packed writer and scratch config support uint32 IDs; scratch derives required capacity from the largest tokenizer ID, including sparse/added IDs | HF-02 removed the 65,536 ceiling; a CPU round-trip covers token ID 70,000 |
| Train/evaluate/save/resume | Public trainer and helper lifecycles exist; scratch has strong prepared-data identity checks | Share contracts across HF data paths; include tokenizer/data identity and declare which streams can resume exactly |
| Standard HF model export | `save_model` uses `save_pretrained` and saves the processing class when supplied | Verify a complete export independently of TrainLM, including tokenizer and applicable generation configuration |
| Publish to HF Hub | Users can use HF tools themselves; TrainLM has no public `push_to_hub` operation | Add explicit publication of the selected export, model card/provenance, and a returned immutable revision |
| Inference | Scratch notebook reloads an export using Transformers | Provide a short local/Hub inference recipe and a fresh-process round trip; other engines require their own compatibility evidence |

Evidence paths:

- [Model acquisition policy](../src/trainlm/config/model.py) and
  [HF provider](../src/trainlm/model/huggingface.py): reuse these loaders.
- [Public trainer](../src/trainlm/api.py): `from_pretrained`, constructor,
  `save_model`, checkpoint identity and lifecycle.
- [Scratch preparation](../src/trainlm/pretraining.py): source fields,
  tokenizer fitting, config construction, packing and vocabulary validation.
- [Fine-tuning workflow](../src/trainlm/finetuning.py): `run_finetuning`,
  `_CausalLMCollator`, raw-text mapping and model/tokenizer revision coupling.
- [Packed writer](../src/trainlm/data/writer.py) and
  [run reports/recovery](../src/trainlm/pretraining_run.py).
- Existing [provider tests](../tests/model/test_huggingface_provider.py),
  [fine-tuning tests](../tests/test_finetuning_api.py),
  [trainer lifecycle tests](../tests/test_api_lifecycle.py),
  [pretraining workflow tests](../tests/test_pretraining_run.py), and
  [resume tests](../tests/test_notebook_resume.py) are starting points, not
  proof that the missing workflows already work.

The owner has verified one small eager GPT-2 scratch run and retained-file
resume across a Python session restart on v5e-1. Native TPU CPT, new HF
architectures/tokenizers, a 1M-token run, VM-replacement recovery, compile,
and multi-device behavior are not established by that result.

## Public API direction

Keep `TrainLMTrainer` and familiar training arguments as the common lifecycle.
Existing `run_pretraining` and `run_finetuning` helpers should use the same
model acquisition, data preparation, reporting, and export components. Keep
existing calls working; do not create a separate CPT training engine.

Users who prefer ordinary HF objects can already express the model choice
through upstream APIs:

```python
# HF construction patterns; choose IDs/revisions appropriate to the run.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)

# Random weights using the selected HF architecture:
config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_config(config)

# Or load existing weights for CPT:
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
```

The two model-loading alternatives above are alternatives, not instructions
to allocate both models. Use the model's existing tokenizer by default for
CPT. An alternative tokenizer must preserve token semantics; equal vocabulary
length or equal commit strings do not establish compatibility.

**Unifying API target sketch:** the current specialized fine-tuning helper
supports `data_format="tokens"`; a single model-object trainer interface and
publication method remain future work. HF loading supplies `model`,
`tokenizer`, and dataset splits.

```python
trainer = TrainLMTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    args=TrainLMTrainingArguments(output_dir="my-run", max_steps=1000),
    data_format="text",  # "tokens" for already prepared input_ids.
    text_field="text",
)
trainer.train()
trainer.save_model("my-model")
# Optional, explicitly requested publication:
trainer.push_to_hub("my-account/my-model", private=True)
```

For current fine-tuning, use `run_finetuning` with `data_format="tokens"` for
prepared HF Dataset rows; token IDs pass through without decoding or
re-tokenizing. The default collator preserves caller labels and masks while
padding variable-length rows, and a caller-provided collator remains supported.
Unifying tokenized inputs with random-init/CPT pretraining and defining its data
fingerprints and resume contract remain outstanding. Raw text should be
tokenized/packed internally according to an explicit policy.
Convenience calls should also accept source IDs/paths plus separate revisions,
using the existing loaders. Final names must align with the existing API;
`TrainLMTrainer.from_config` currently means TrainLM configuration, so do not
silently reinterpret it as a Transformers `PretrainedConfig` factory.

## Delivery order and acceptance criteria

HF-01 and HF-02 are merged through PR #139 at
`c9ece9a2e9057260586636a428dfd7679c42e8fa`. HF-03 is in progress on
`feat/20-hf-dataset-sources`; HF-04–HF-08 remain planned. IDs
HF-01–HF-08 identify outcomes, not Git branch sequence numbers. Deliver each
outcome in a focused PR with meaningful
implementation/regression/docs commits. Start each new branch after the
preceding PR merges and `main` is verified. Keep the requested co-author
trailer and use rebase-and-merge first, merge commits second; squash only on an
explicit request. The planning PR changes documentation only.

### HF-01 — Independent model and tokenizer sources (P0, complete)

**Progress:** PR #137 and PR #138 are merged to `main` at
`9591d8b9823c9a8757630d1b79e724e3590d7028`. `run_finetuning` loads and records model and tokenizer IDs and
requested/resolved revisions independently. A separate tokenizer commit no
longer selects the model snapshot or causes a false SHA mismatch. The
same-source, same-revision path still pins both loads to one immutable commit.
`restore_finetuning_config` pins to resolved model/tokenizer commits instead
of replaying movable branch names. A tokenizer object with no reproducible
source must be supplied again on resume; its recorded commit is checked when
available.
Before training, it checks known tokenizer IDs against model embedding rows and
the configured sequence length against a known static model context limit; it
records `unknown` when these values cannot be determined or rope scaling makes
a static limit ambiguous. It never resizes embeddings or remaps IDs. Token
meaning compatibility is not machine-verifiable and is marked
`not_verified_by_trainlm`; the user must choose a tokenizer compatible with
the model. The explicit policy is caller responsibility for semantic
compatibility, with structural checks and unknowns recorded; TrainLM does not
add a second acknowledgement switch for this user-selected source. Arbitrary
dataset stream/cursor identity remains part of HF-06.

**User outcome:** choose a model and compatible HF tokenizer without manually
rebuilding their configs or making the repositories share a commit.

- Resolve and record model/config and tokenizer IDs, paths, requested revisions
  and resolved commits independently. Reuse same-repository snapshot pinning
  only when the sources actually share that repository/revision contract.
- Fix fine-tuning's tokenizer-SHA-as-model-SHA behavior. Preserve supplied
  tokenizer objects and distinguish known provenance from unknown metadata.
- Reuse standard HF authentication, cache, offline/local and gated-access
  behavior. Accept a friendly revision when loading, resolve once and record
  the immutable source; require the recorded identity on strict resume.
- Validate embedding bounds, tokenizer token meanings/provenance, special
  tokens and context limits. Never silently retrain a CPT tokenizer, remap
  token IDs or resize/reinitialize pretrained embeddings.
- Acceptance: same-repository moving refs remain consistent; different valid
  repositories with different SHAs work; incompatible/unknown token semantics
  receive an explicit compatibility policy; no token enters logs or reports.

The implementation commits are `aee61b6` (independent source resolution) and
`31a581d` (resume source identity).

### HF-02 — HF architecture, tokenizer reuse, and CPT initialization (P0, complete)

**Progress:** PR #139 extended
`TrainLMPretrainingConfig` with a Hub/local HF model source, separate source
revisions, `random` or `pretrained` initialization, tokenizer reuse and an
explicit `train_bpe` option. Random initialization loads the HF config and
calls the provider's `from_config` path; pretrained initialization pins model
weights to the config's resolved Hub commit. Existing source tokenizers are
cached by a tokenizer-mapping fingerprint and requested/resolved provenance.
TrainLM derives the packed vocabulary capacity from the highest token ID,
including sparse and added tokens. Pretrained embeddings are not resized; an
out-of-range tokenizer fails before training. Known context limits are checked.
Focused offline CPU tests cover both initialization paths and uint32 storage;
real Hub and TPU execution are still unverified.

**User outcome:** name an HF model architecture and train new weights with an
existing HF tokenizer, using one ordinary TrainLM workflow.

- Expose Hub/local config sources and random/pretrained initialization through
  the convenient path, reusing `ModelSourceConfig` and the HF provider.
- Preserve HF config fields; allow explicit architecture overrides and report
  the effective parameter count/config. Do not reconstruct families from a
  short list of guessed dimension names.
- Reuse an existing tokenizer by default when a model source is supplied.
  Keep current corpus-trained BPE available as an explicit compatible option.
- Remove the obsolete scratch uint16 ceiling where uint32 storage applies;
  validate maximum emitted token IDs, sparse/added IDs and embedding size.
- Acceptance: random mode never loads pretrained weights; CPT loads them;
  both can reuse a tokenizer without fitting it; a vocabulary above 65,536
  round-trips safely; old scratch calls and saved-run compatibility are covered.

### HF-03 — One raw/pretokenized HF dataset path (P0, in progress)

**User outcome:** supply a Hub dataset or HF dataset object and tell TrainLM
whether it contains text or tokens.

**Progress on `feat/20-hf-dataset-sources`:** the scratch pretraining helper
accepts a dataset config name, branch or tag revision, and simple HF split
slices. Named refs resolve to one commit before the stream is opened. The
resolved commit and config enter the preparation fingerprint and run
configuration, and generated resume settings pin to that commit. When train
and validation use one underlying split, TrainLM accepts them only as
non-overlapping slices. Raw local text files and text-column tokenization
remain supported. The pretrained `run_finetuning` helper also accepts
prepared HF `Dataset` splits with `data_format="tokens"`: it validates aligned
token columns, checks known model embedding bounds, preserves unshifted labels
and masks, and pads ragged batches. Tokenized dataset input to random-init/CPT
pretraining, arbitrary local data files, and unified source/resume fingerprints
remain outstanding. This branch has CPU evidence only; it does not establish
live-Hub or TPU execution.

Runnable example: [examples/pretrain_from_scratch.py](../examples/pretrain_from_scratch.py).
Its 10,000-token budget is only a workflow smoke test, not a useful training
budget or quality claim.

For example, this branch supports choosing a named dataset revision and
holding out the final 10% of one split without requiring users to find a
commit SHA:

```python
from trainlm import TrainLMPretrainingConfig, run_pretraining

config = TrainLMPretrainingConfig(
    model_name_or_path="openai-community/gpt2",
    model_initialization="pretrained",
    dataset_id="roneneldan/TinyStories",
    dataset_revision="main",
    train_split="train[:90%]",
    validation_split="train[90%:]",
    text_field="text",
    output_dir="trainlm-cpt",
)
result = run_pretraining(config, show_progress=True)
print(result.summary())
```

The report records the requested ref and resolved SHA. A later resume uses the
resolved SHA so a moved branch cannot silently change the corpus.

For an already-tokenized Hub dataset, the pretrained fine-tuning API accepts
the `Dataset` splits directly. Use the same tokenizer that produced the token
IDs; TrainLM does not decode or retokenize them:

```python
from datasets import load_dataset
from trainlm import (
    TrainLMFineTuningConfig,
    TrainLMTrainingArguments,
    run_finetuning,
)

MODEL_ID = "your-org/your-causal-lm"
MODEL_REVISION = "<model-commit-sha>"
tokenized = load_dataset("your-org/your-tokenized-dataset", revision="<commit-sha>")

config = TrainLMFineTuningConfig(
    model_name_or_path=MODEL_ID,
    model_revision=MODEL_REVISION,
    dataset_id=None,
    dataset_revision=None,
    data_format="tokens",
    args=TrainLMTrainingArguments(output_dir="trainlm-finetuned", max_steps=100),
)
result = run_finetuning(
    config,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)
print(result.summary())
```

Replace the example IDs, commit, and split names with values from your run.
Dataset token IDs and tokenizer must be semantically compatible; TrainLM can
check dimensions and known ID bounds, but cannot prove what each ID means.

- Support dataset ID/config/subset/revision, named or sliced splits, local data
  files and selected `DatasetDict` splits through standard `datasets` loaders.
  Select train/eval splits explicitly; never silently reuse training rows for
  validation. Record source, transform, tokenizer and split fingerprints.
- Raw mode accepts a text field or formatter plus an HF tokenizer. Specify
  packing versus truncation, EOS/document boundaries and discarded-token
  policy. Do not silently discard most of a long document for CPT.
- Tokens mode accepts `input_ids` and valid optional `attention_mask`/`labels`,
  checks ranges and shapes, preserves `-100` masks and real EOS labels, and
  never re-tokenizes. Record the tokenizer ID, revision, or a stable local
  vocabulary/added-token mapping identity that produced the token IDs. Equal
  vocabulary sizes and in-range IDs do not prove matching token meanings;
  define an explicit caller-acknowledgment policy when provenance is unknown.
- Treat HF causal labels as unshifted labels aligned to `input_ids`; the
  training task shifts exactly once. Preserve and validate supported
  `loss_mask`, `attention_mask`, caller labels and model-input fields such as
  `position_ids` rather than silently dropping them. If `pad_token_id` equals
  `eos_token_id`, use the attention mask to ignore padding while retaining real
  EOS targets. Unmasked causal labels are derived only when absent.
- Reuse HF collators when they meet the contract; handle ragged sequences,
  padding and TPU shape buckets internally. Preserve custom collator control.
- Existing packing uses EOS-separated streams, not document-isolated
  attention; document that distinction. Cache identity must include tokenizer
  mapping/revision and packing policy, not just dataset name and vocabulary size.
- Acceptance: equivalent raw/tokenized examples produce equivalent supervised
  tokens, loss and updates; changed tokenizer/masks invalidate incompatible
  caches; token IDs with unknown or mismatched tokenizer provenance follow an
  explicit policy; labels are shifted once; `loss_mask` and padding behavior,
  including pad-equals-EOS, are covered. Empty, malformed and incompatible
  inputs fail before long training.

### HF-04 — Clear scratch/CPT lifecycle and budgets (P0)

**User outcome:** switch between new weights and an existing model while
keeping the same data, arguments and train/evaluate/save workflow.

- Consolidate the existing causal-LM helpers around shared components and
  expose CPT by name in documentation/API selection without another engine.
- Offer clear step/token/epoch budget rules, accumulation, precision,
  learning-rate schedule, eval/save cadence, best-model selection and logging.
  Reject conflicting budgets; keep the supervised-token definition explicit.
- Show model size, data/sequence settings, selected runtime and approximate
  work before a long run. Resource estimates must be labeled as estimates.
- Define exact resume versus new CPT run versus a deliberate budget extension.
  Keep best-weight exports distinct from the latest resumable checkpoint.
- Acceptance: four public journeys (random/pretrained × raw/tokens) share the
  lifecycle and error/report contracts. A pretrained start preserves weights;
  resume restores training state; evaluation remains held out. Defaults and
  existing public calls remain compatible. Native TPU evidence is a separate gate.

### HF-05 — Portable export and explicit HF publication (P0)

**User outcome:** save the model and tokenizer, optionally publish, then use
standard inference tools with no TrainLM dependency.

- Reuse HF `save_pretrained`/safetensors for the complete model, tokenizer,
  special-token and applicable generation configs. Record base model/config,
  tokenizer and dataset provenance, training settings and validation limits.
- Add an explicit Hub publication method using `huggingface_hub`. Let the user
  choose destination and privacy; new repos should default private. Inspect
  existing repo visibility and never change it implicitly. Return the repo URL
  and immutable uploaded revision; retain the local export if upload fails.
- Upload an allowlisted model export and model card. Exclude credentials,
  private corpus files, optimizer checkpoints and logs unless separately
  requested. Preserve relevant base-model/data license information; TrainLM's
  Apache-2.0 code license does not automatically license the resulting weights.
- Acceptance: local export reloads and produces matching deterministic output
  in a fresh Transformers process without TrainLM; mocked upload failures are
  recoverable; an explicitly authorized private Hub round trip validates the
  uploaded revision. Other inference engines are named only after verification.

### HF-06 — Larger corpora and durable resume (P1, before long jobs)

**User outcome:** train beyond memory-sized datasets and survive interruptions
without repeating or skipping training data silently.

- Unify streaming/IterableDataset, finite prepared data, shuffle/seed,
  packing/batching and token budgets. Define what happens on exhaustion and
  whether repetition is requested. Reuse HF/Arrow caches and existing packed IO.
- Check resume capability before training starts. Capture dataset/tokenizer/
  transformation identity and the cursor/RNG needed for supported loaders;
  do not promise exact resume for arbitrary streams or worker settings.
- Support durable run directories and a documented recovery flow containing
  optimizer, scheduler, RNG, data cursor and required preparation artifacts.
  The existing model ZIP alone cannot resume a run after VM replacement.
- Acceptance: interrupted/resumed updates match the documented reference;
  changed data/tokenizer is rejected before state mutation; retained-file and
  VM-replacement recovery are reported as different evidence.

### HF-07 — Three user recipes, one API (P0 alongside each feature)

**User outcome:** open Colab/Kaggle, choose their HF resources, and run a few
readable cells without copying library internals.

- Deliver recipes for scratch from an HF config/tokenizer, CPT from existing
  weights, and a pretokenized HF corpus. Each demonstrates evaluation,
  local save and standard Transformers reload; Hub push is clearly optional.
- Share library helpers for preparation, provenance, preflight and reports.
  Keep diagnostics in separate notebooks and detailed controls in guides.
- Document model/tokenizer/dataset ID, revision, subset/splits, text/tokens,
  training budget and output location where users select them. Explain
  random weights versus pretrained weights and checkpoint resume.
- Acceptance: examples execute against the public API with small local
  fixtures; current links and meaningful behavior are checked. Avoid brittle
  tests for incidental prose or compressing code into unreadable lines merely
  to satisfy a line-count limit. Mark planned examples as planned until runnable.

### HF-08 — Publish an evidence-based support matrix (P0 release gate)

**User outcome:** know which combinations are usable on their hardware.

- Track model architecture, initialization, tokenizer, data format, dtype,
  compile mode, runtime versions and tested device count for each result.
- CPU reference coverage verifies semantics. Use owner-run v5e-1 for native
  scratch/CPT, data preparation, export and resume; v5e-8 for collectives,
  DP/sharding and distributed checkpoint claims. Preserve reusable XLA code.
- Acceptance: complete the four journeys on the supported single-device path,
  including fresh-process HF reload and checkpoint recovery. Broader models,
  compile and scale stay explicitly unverified until measured. A 10K/1M-token
  completion proves a workflow, not convergence or useful generated text.

## First complete release slice and later work

The first useful HF-centered slice is a supported causal LM with an existing
HF tokenizer, raw or already tokenized data, random weights or CPT, one common
train/evaluate/save/resume lifecycle, and an HF-loadable export with optional
publication. HF-01–HF-05 plus their HF-07 recipes and HF-08 evidence define
that slice. HF-06 is required before recommending long streaming/preemptible
jobs. TPU queue delays should not block CPU-verifiable API work, but CPU
results must never be called TPU certification.

After that slice, consider corpus mixtures/weights and quality tools,
parameter-efficient training, richer monitoring integrations, larger-model
sharding, optimized kernels, additional architectures, and inference-engine
compatibility. These are separate priorities; the current plan does not claim
full Transformers Trainer parity or promise every Hub artifact will train.

Reuse upstream utilities where they fit: [Transformers loading/configuration](https://huggingface.co/docs/transformers/model_doc/auto),
[HF Datasets loading](https://huggingface.co/docs/datasets/loading),
[data collators](https://huggingface.co/docs/transformers/main_classes/data_collator),
and [Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload).
These are reference entry points; implementation must check behavior against
TrainLM's supported package profiles.
