# TrainLM guide for coding agents

TrainLM provides Hugging Face-style language-model pretraining with PyTorch.
The native TorchTPU training path is experimental and requires explicit
selection. Use the current code and validation evidence below; the older M0–M17
documentation describes the PyTorch/XLA era.

## Find the implementation by task

| Task | Start here | Related evidence |
| --- | --- | --- |
| Pretrain a causal language model from random weights on native TorchTPU | [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) · [notebook source](../notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) | [beginner notebook helper](../examples/torchtpu_pretraining_quickstart.py), [trainer implementation](../examples/torchtpu_native_trainer_smoke.py), [scratch lifecycle tests](../tests/test_packed_scratch_training.py) |
| Prepare or inspect packed token data | [packed-data contracts](data/PACKED_BINARY_MANIFEST.md) | [packed data writer](../src/trainlm/data/writer.py), [dataset tests](../tests/test_packed_dataset_resume.py) |
| Check native TPU availability and package compatibility | [T1 v5e-1 guide](T1_V5E1_PROBE.md) | [hardware discovery](../src/trainlm/runtime/torchtpu_discovery.py), [package preflight](../src/trainlm/runtime/torchtpu_trainer_preflight.py) |
| Understand supported behavior and validation status | [current handoff](HANDOFF.md) | [dense-AR support scope](SCOPE.md), [T0–T11 roadmap](TORCHTPU_REBASELINE.md) |

Useful questions include “How does TrainLM train a randomly initialized GPT-2
model on TinyStories?”, “Where are packed training shards written and resumed?”,
and “Which native TorchTPU checks have actually run?” Start from the table and
follow the linked implementation and tests before making a change.

## Evidence and limits

- The from-scratch notebook is a small-scale implementation example. Its CPU
  lifecycle coverage does not establish TPU training, convergence, or
  performance.
- The notebook's beginner path asks for a token budget and sequence length;
  its pinned TinyStories source and starter model settings are supplied by the
  notebook helper. The helper is an example adapter, not a stable high-level
  public pretraining API. Advanced dataset selection still requires an
  immutable Hugging Face commit SHA and distinct train/validation splits.
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
