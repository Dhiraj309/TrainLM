# TrainLM: Hugging Face user workflows and delivery backlog

**Preview an HF pretraining run.** The [scratch/CPT notebook](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) calls `config.training_plan(inspect_model=True)` to show the architecture, parameter count and equivalent FP32 parameter bytes before training. It uses the HF configuration and a PyTorch meta model. The normal plan stays offline. When training a new BPE tokenizer, the estimate assumes the requested vocabulary capacity. Activations, optimizer state and allocator overhead require separate memory measurements.

**Check pretokenized input.** Corpus producers can record `trainlm.tokenizer_vocab_sha256(tokenizer)` alongside their token rows. Pass that digest as `expected_tokenizer_vocab_sha256` in scratch, CPT or fine-tuning configs to reject a different token-to-ID mapping before training. The comparison includes added vocabulary entries. Its scope is vocabulary identity; formatting, token boundaries and label alignment are separate contracts. The check is optional for existing corpora.



The [HF workflow support matrix](HF_SUPPORT_MATRIX.md) links the scratch, CPT and tokenized corpus recipes to CPU and TPU evidence.

**Priority reset requested by the owner on 2026-11-17.** This is the active
product delivery plan. T0–T11 remains the architecture and accelerator evidence
plan. Story labels below distinguish merged work, the current implementation
branch, and work that remains planned.
PR #145 merged at `638c6643af0a6db19b491622805b74db19fcb09f`. See the [HF support matrix](HF_SUPPORT_MATRIX.md) for the current CPU and hardware results.
The baseline audit started from
`9591d8b9823c9a8757630d1b79e724e3590d7028` (PR #138); HF-02 then merged at
`c9ece9a2e9057260586636a428dfd7679c42e8fa` (PR #139).

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
| Start from pretrained weights for CPT | `TrainLMPretrainingConfig.continued_pretraining(model_id, ...)` selects the HF checkpoint and reuses the scratch/CPT lifecycle for raw text or tokenized HF Dataset rows | HF-02 is merged; model loading begins only when the run starts |
| Reuse an HF tokenizer | Scratch defaults to the model repository's tokenizer or accepts independent `tokenizer_name_or_path` and `tokenizer_revision`; `tokenizer_mode="train_bpe"` retains corpus training for random initialization | HF-02 fingerprints tokenizer mapping and resolved source; independently hosted token meaning remains the user's responsibility |
| Independently hosted tokenizer | `run_finetuning` accepts separate model/tokenizer IDs and revisions, resolves them independently, and checks known token-ID bounds and static context limits | Token meanings cannot be proven from Hub IDs or vocabulary sizes; the report marks them unverified and an explicit user policy remains |
| Raw HF datasets | Scratch streams pinned Hub text and accepts supplied raw-text or tokenized HF `Dataset` splits with fingerprinted cache identity; scratch/CPT and fine-tuning accept row formatters | Arbitrary structured local data file loading and unified provenance remain |
| Pretokenized HF datasets | Fine-tuning and scratch/CPT accept `input_ids` with optional aligned labels and masks; scratch/CPT preserves them in checksummed sidecars, derives labels only when absent, and shifts once | Verify tokenizer-to-dataset semantics and unify provenance and resume identity across workflows |
| Modern HF vocabularies | Packed writer and scratch config support uint32 IDs; scratch derives required capacity from the largest tokenizer ID, including sparse/added IDs | HF-02 removed the 65,536 ceiling; a CPU round-trip covers token ID 70,000 |
| Train/evaluate/save/resume | Public trainer and helper lifecycles exist; scratch has strong prepared-data identity checks | Share contracts across HF data paths; include tokenizer/data identity and declare which streams can resume exactly |
| Standard HF model export | `save_model` uses `save_pretrained` and saves the processing class when supplied | A CPU fresh-process check reloads model and tokenizer and generates from the export; accelerator export remains owner-validated |
| Publish to HF Hub | Completed pretraining and fine-tuning results expose `push_to_hub`; only model/tokenizer artifacts and a model card are published | Review license and provenance before public release; live Hub upload remains user-run |
| Inference | `examples/infer_local_export.py` and the scratch notebook load the export in a clean Python process using Transformers | Other inference engines require their own compatibility evidence; Hub-hosted inference remains a user-managed Transformers workflow |

Evidence paths:

- [Model acquisition policy](../src/trainlm/config/model.py) and
  [HF provider](../src/trainlm/model/huggingface.py): reuse these loaders.
- [Public trainer](../src/trainlm/api.py): `from_pretrained`, constructor,
  `save_model`, checkpoint identity and lifecycle.
- [Scratch preparation](../src/trainlm/pretraining.py): source fields,
  tokenizer fitting, config construction, packing and vocabulary validation.
- [Fine-tuning workflow](../src/trainlm/finetuning.py): `run_finetuning`,
  `_CausalLMCollator`, raw-text mapping and model/tokenizer revision coupling.
- [Hub publishing](../src/trainlm/hub.py): private-by-default repository
  creation, visibility checks for existing repos, model artifact allowlist,
  generated model card and immutable commit result; see the optional calls in
  [fine-tuning](../examples/finetune_text.py) and
  [pretraining](../examples/pretrain_from_scratch.py), and the
  [publisher contract tests](../tests/test_hub_publish.py).
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
supports `data_format="tokens"`; a single model-object trainer interface
remains future work. HF loading supplies `model`,
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
```

After a high-level `run_pretraining(config)` or `run_finetuning(config)` call,
publish explicitly:

```python
result = run_finetuning(config)  # or run_pretraining(config)
published = result.push_to_hub("my-account/my-model")  # private by default
print(published.repo_url, published.revision)
```

Authenticate first with `hf auth login` or `HF_TOKEN`. Set `private=False` only
when you intend to create or use a public repository. An existing repository's
visibility must match the requested setting and is never changed implicitly.
TrainLM uploads the selected model export and a basic model card; it excludes
run reports, corpus files, and resumable checkpoints. It preserves an existing
README. Review the model and data licenses before making a repository public;
TrainLM's Apache-2.0 software license does not grant a license to model weights
or training data. Dataset identifiers are not copied into the model card
automatically; add provenance to the README when you have confirmed it is safe
to disclose. A successful call returns the repository URL and full commit SHA
as `HubPublishResult`.

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

HF-01–HF-03 are merged through PR #140 at
`242f418f4bd89a814709db5801e4f393963d86e4`. HF-05's opt-in result publishing
is merged through PR #141; mocked CPU coverage passes, and live Hub round-trip
evidence remains outstanding. HF-04 lifecycle and epoch scheduling merged in PRs #142–143;
HF-06–HF-08 have partial implementation and evidence. IDs
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
non-overlapping slices. Local raw files support UTF-8 text, CSV, JSON/JSONL,
and Parquet; `local_data_format="auto"` selects the parser from the extension,
or you can set it explicitly. Select a structured source column with
`text_field`, or use the existing row formatter for multiple columns. Local
file hashes and the parser enter preparation-cache identity and the run report.
The pretrained `run_finetuning` helper also accepts
prepared HF `Dataset` splits with `data_format="tokens"`: it validates aligned
token columns, checks known model embedding bounds, preserves unshifted labels
and masks, and pads ragged batches. Tokenized scratch/CPT accepts optional
aligned labels and loss masks. It packs caller labels and masks in
checksummed sidecars, derives labels from `input_ids` only when labels are
absent, and applies one next-token shift. Special-token masks, position IDs,
and token-type IDs remain unsupported. TrainLM records the selected HF
tokenizer source and mapping, but cannot verify that the dataset token IDs were
produced by that tokenizer. Raw text datasets with multiple columns accept a
`formatting_func(row) -> str`; its stable `text_formatter_name` is included in
the preparation cache and checkpoint identity, and the caller supplies the
same formatter again on resume. Other local formats and unified
cross-workflow source/resume fingerprints remain outstanding.
Raw text HF `Dataset` splits now enter scratch/CPT through
`run_pretraining(..., train_dataset=..., eval_dataset=...)` or
`TrainLMTrainer.from_scratch`, using `dataset_source="provided"`. Their stable
HF fingerprints and row counts enter the prepared-data cache and run report;
resume requires the same splits again. This branch has CPU evidence only; it
does not establish live-Hub or TPU execution.

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

To reuse an HF text dataset already loaded in Python, pass its split objects
directly. They must expose the configured text column and a stable HF
fingerprint. For resume, load the same dataset revision and splits again.

```python
from datasets import load_dataset
from trainlm import TrainLMPretrainingConfig, run_pretraining

data = load_dataset("roneneldan/TinyStories", revision="main")
config = TrainLMPretrainingConfig(
    dataset_source="provided",
    output_dir="trainlm-cpt",
)
result = run_pretraining(
    config,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
)
print(result.summary())
```

If a raw dataset stores the text across several columns, provide one row
formatter and a stable name. TrainLM uses it for packed training/evaluation
data, and also for corpus-tokenizer training when using the default BPE path:

```python
def format_example(row):
    return f"Question: {row['question']}\nAnswer: {row['answer']}"

config = TrainLMPretrainingConfig(
    dataset_source="provided",
    text_formatter_name="question-answer-v1",
    output_dir="trainlm-cpt",
)
result = run_pretraining(
    config,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    formatting_func=format_example,
)
```

Pass the same formatter and name when resuming. Changing the name invalidates
the prepared-data cache rather than silently reusing data formatted another
way. The generated resume example asks for the formatter to be redefined and
passes it back into TrainLM. Treat the name like a formatter version and
change it whenever formatting behavior changes.

For local structured files, set the source column and let TrainLM choose the
parser from the matching train/validation extensions:

```python
from trainlm import TrainLMPretrainingConfig, run_pretraining

config = TrainLMPretrainingConfig(
    dataset_id=None,
    dataset_revision=None,
    train_file="data/train.csv",
    validation_file="data/validation.csv",
    text_field="content",
    output_dir="trainlm-cpt",
)
result = run_pretraining(config, show_progress=True)
```

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

Scratch/CPT token input is also available with `data_format="tokens"`. The
dataset must contain `input_ids` and may contain aligned `attention_mask`,
`labels`, and `loss_mask`. Labels use token IDs or `-100`; loss masks use zero
or one. TrainLM strips left/right attention padding, adds EOS when a row lacks
it, and applies the causal shift once. Caller labels and loss masks are
preserved through packed data. When labels are omitted, TrainLM uses
`input_ids`; when it appends EOS to labeled rows, it appends an EOS target
unless the previous target was ignored. An explicit loss mask's final value
is carried onto that appended EOS. Remove `special_tokens_mask`, positional
IDs, and token-type IDs because those fields are not yet supported here.
Choose the same model or tokenizer source used to create the IDs. TrainLM
records that source and its token mapping, but cannot prove the token meaning
matches the dataset's producer. Put the matching tokenizer source and immutable
revision in the config when it is independent of the model:

```python
from datasets import load_dataset
from trainlm import TrainLMPretrainingConfig, run_pretraining

tokenized = load_dataset("your-org/your-tokenized-corpus", revision="<data-sha>")
config = TrainLMPretrainingConfig(
    model_name_or_path="your-org/your-causal-lm",
    model_initialization="pretrained",
    dataset_source="provided",
    data_format="tokens",
    sequence_length=512,
    output_dir="trainlm-cpt",
)
result = run_pretraining(
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

**Progress on `feat/22-hf-scratch-cpt-lifecycle`:**
`TrainLMPretrainingConfig.training_plan()` provides a no-download pre-run
summary of the selected initialization, model/data settings, requested
accelerator, supervised-token target, and estimated optimizer-step count.
`TrainLMFineTuningConfig.training_plan()` summarizes step- and token-based
batches directly; for epoch budgets, pass an already-loaded finite dataset to
estimate batches and updates without loading another resource.
Both plans also show learning-rate settings, batch and accumulation values,
precision, logging, save/evaluation cadence, and best-model/early-stopping
settings. Scratch pretraining labels its derived default cadence as an
estimate; fine-tuning reports every configured stop limit. If several limits
are set, the first reached stops the run. Fine-tuning's token-to-step estimate
assumes one device and dense targets; masks, padding, or distributed world size
can change it. A plan requested with `accelerator="tpu"` marks step, token,
and finite packed-data epoch budgets implemented-but-unverified for the legacy
PyTorch/XLA coordinator. Epoch counts are resolved from each rank's finite
packed-data partition. The first epoch keeps the established order for resume
compatibility; later epochs deterministically reshuffle shards and batches from
the configured seed, and that order can be reconstructed after a checkpoint.
Epoch-based evaluation and checkpoint saving now run at those same finite
dataset boundaries. Token stopping uses exact global supervised-token counts
and may pass the target by the final microbatch. All TPU-coordinator paths
remain hardware-unverified.
For an explicitly named CPT entry point, use
`TrainLMPretrainingConfig.continued_pretraining(model_id, ...)`; it selects
pretrained initialization without downloading weights during config creation.
The default plan leaves parameter count unresolved. Use the optional architecture preview above to count parameters before fetching weights.
CPU tests validate the plan contracts; they do not certify TPU execution.

```python
print(config.training_plan())  # inspect estimated budget before run_pretraining(config)
```

Pretraining plans can preview the three run choices without touching files:

```python
# A new scratch run, or a new CPT run that loads the configured pretrained weights.
print(config.training_plan())
# Exact checkpoint resume with the same run configuration.
print(config.training_plan(resume_from_checkpoint="latest"))
# Continue the same run with a larger max_tokens budget.
print(config.training_plan(
    resume_from_checkpoint="latest", extend_training_budget=True
))
```

Pass the same lifecycle arguments to `run_pretraining()`. A new run needs a
fresh `output_dir`; exact resume restores the saved training state, while a
budget extension permits only a larger token target under the existing strict
compatibility checks. The plan describes the requested choice but does not
inspect the checkpoint or validate its saved configuration.

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

### Recover a prepared corpus on persistent storage

For long runs, set `output_dir` to a persistent filesystem before starting.
After a checkpoint is published and the writer has stopped, keep the whole
output directory, including the tokenizer, packed data, preparation manifest,
run config, report, and checkpoints. If its mount path changes, use the report
in the copied directory. Keep the original external model, tokenizer, local
source files and software versions available. Supply the original Dataset
objects and formatting function again if the first run used them.

```python
from trainlm import restore_pretraining_report, run_pretraining

saved = restore_pretraining_report("/mnt/restored/report.json",
                                  restored_output_dir="/mnt/restored")
result = run_pretraining(saved.config, resume_from_checkpoint=saved.checkpoint_path)
```

TrainLM restores optimizer, scheduler, RNG and packed cursor during training.
Exact resume keeps the saved budget. The CPU regression also exercises the
explicit budget extension after moving the directory. Physical TPU VM
replacement still needs an owner run.

### Save training curves as JSONL

Pass `report_to="jsonl"` to `TrainLMTrainingArguments` to append each configured training log and evaluation snapshot to `output_dir/metrics.jsonl`. The default is `"none"`. TensorBoard and W&B integrations are not available yet.

### Fine-tune from local CSV, JSONL, or Parquet

Set `train_file` and `validation_file` to paired files. TrainLM infers the shared format from their extensions or accepts `local_data_format="csv"`, `"json"`, or `"parquet"`. For Hub datasets, `dataset_revision` accepts a branch, tag, or commit SHA. TrainLM resolves named refs to a commit and records the resolved revision in the run summary.

### Reduce activation memory during fine-tuning

Set `gradient_checkpointing=True` in `TrainLMTrainingArguments` to enable the selected Hugging Face model's checkpointing support. TrainLM disables KV caching while checkpointing is active. Models without the checkpointing hook fail before training. The legacy XLA coordinator currently reports this option as unsupported.
