# TrainLM guide for coding agents

TrainLM provides Hugging Face-style language-model pretraining with PyTorch.
The native TorchTPU training path is experimental and requires explicit
selection. Use the current code and validation evidence below; the older M0–M17
documentation describes the PyTorch/XLA era.

## Find the implementation by task

| Task | Start here | Related evidence |
| --- | --- | --- |
| Pretrain a causal language model from raw Hugging Face text | [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) · [notebook source](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) | [`TrainLMPretrainingConfig` and data preparation](../src/trainlm/pretraining.py), [`TrainLMTrainer.from_scratch`](../src/trainlm/api.py), [API tests](../tests/test_pretraining_api.py) |
| Resume a scratch run with a larger token budget | [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) · [resume instructions](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) | [`extend_training_budget`](../src/trainlm/api.py), [API tests](../tests/test_pretraining_api.py) |
| Limit automatic checkpoint storage during training | [public example](../README.md), [notebook](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) | [retention implementation](../src/trainlm/checkpoint/retention.py), [filesystem safety tests](../tests/checkpoint/test_retention.py), [resume parity tests](../tests/test_api_lifecycle.py) |
| Prepare or inspect packed token data | [packed-data contracts](data/PACKED_BINARY_MANIFEST.md) | [packed data writer](../src/trainlm/data/writer.py), [dataset tests](../tests/test_packed_dataset_resume.py) |
| Check native TPU availability and package compatibility | [T1 v5e-1 guide](T1_V5E1_PROBE.md) | [hardware discovery](../src/trainlm/runtime/torchtpu_discovery.py), [package preflight](../src/trainlm/runtime/torchtpu_trainer_preflight.py) |
| Understand supported behavior and validation status | [current handoff](HANDOFF.md) | [dense-AR support scope](SCOPE.md), [T0–T11 roadmap](TORCHTPU_REBASELINE.md) |

Useful questions include “How does TrainLM train a randomly initialized GPT-2
model on TinyStories?”, “Where are packed training shards written and resumed?”,
and “Which native TorchTPU checks have actually run?” Start from the table and
follow the linked implementation and tests before making a change.

## Evidence and limits

- `save_total_limit` accepts a positive integer or `None` (API default: keep
  all). The notebook uses three. After state and run metadata are published,
  the single-device public trainer retains the new save and the newest other
  owned `checkpoint-N` directories within that limit. Each new automatic save
  receives a small ownership marker; unmarked legacy saves, incomplete saves,
  symlinks, and directories with extra files are preserved. Explicit manual
  backups should use a separate name/directory. Retention is storage policy and
  is excluded from the scratch run fingerprint, preserving older fingerprints
  and allowing the limit to change at resume. Cleanup errors warn while keeping
  the new checkpoint. Distributed/XLA coordinator retention is rejected; CPU
  tests do not certify native TPU checkpoint recovery.
- The from-scratch notebook is a small-scale implementation example. Its CPU
  lifecycle coverage does not establish TPU training, convergence, or
  performance. Its version 4 downloadable report captures requested and
  effective training settings (including optimizer and scheduler), selected
  backend, runtime TPU discovery and device properties, model and tokenizer
  provenance, package versions, source/dataset revisions, and training
  metrics. A property-only device query does not allocate a device and is
  labeled unverified. Missing optional packages and unavailable properties are
  reported as `null`.
- `TrainLMTrainer.from_scratch(TrainLMPretrainingConfig(...))` is the public
  raw-text entry point. It requires an immutable Hugging Face dataset revision
  and disjoint train/validation splits, then returns the familiar trainer for
  `train()`, `evaluate()`, and `save_model()` calls. Use
  `trainer.train(resume_from_checkpoint="latest")` to resume from the newest
  completed checkpoint under the trainer output directory. Matching pinned
  preparation settings reuse validated tokenizer and packed-data artifacts in
  `config.output_dir`; change the output directory when changing the dataset or
  tokenizer inputs. `run_config.json` captures the effective model, tokenizer,
  packed data, optimizer, scheduler, training arguments, and runtime identity;
  each checkpoint stores the same fingerprint. Exact resume requires an exact
  match and fails before restoring any state when settings differ. The explicit
  `extend_training_budget=True` path permits only a strictly larger `max_tokens`
  and its derived horizon/cadence. After a compatible checkpoint is restored,
  it starts a new cosine schedule at the configured learning rate while keeping
  model and optimizer state; this can raise the learning rate, so use it
  deliberately. Other changed settings require a fresh `output_dir`. CPU/API
  tests cover this lifecycle; TPU checkpoint resume remains unverified. GPT-2
  is the default;
  `model_type` and `model_config_overrides` select a registered Transformers
  causal-LM configuration for random initialization. TrainLM fills vocabulary
  and special-token IDs from the newly trained tokenizer. The path does not
  load custom remote model code.
- Native TorchTPU still requires explicit `accelerator="native_tpu"` and is an
  experimental, single-device path. Package metadata and CPU lifecycle tests do
  not certify TPU training or performance.
- The owner reported a successful standalone eager TorchTPU probe on a v5e-1
  runtime. That probe is separate from running the complete pretraining
  notebook on TPU.
- `accelerator="native_tpu"` is the explicit experimental native path.
  `accelerator="tpu"` refers to the existing PyTorch/XLA path; do not conflate
  their implementations or evidence.
- Treat M0–M17 and their measurements as historical XLA material unless a page
  explicitly identifies newer TorchTPU evidence.
- Check the current branch, CI, and owner-run accelerator report before
  describing a feature as validated. Do not infer hardware support from CPU
  test results or package metadata alone.

## Contributions and commit labels

Use a specific Conventional Commit subject such as
`feat(torchtpu): add from-scratch pretraining notebook`. Keep each commit
focused on one logical outcome. For a user-facing feature, connect the example,
implementation, tests, and limitations in the pull request documentation.
Accurate file names, headings, and technical terms help people and tools find
the implementation; keyword stuffing and unsupported claims do not.
