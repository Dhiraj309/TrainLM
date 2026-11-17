# TrainLM guide for coding agents

TrainLM provides Hugging Face-style language-model pretraining with PyTorch.
The native TorchTPU training path is experimental and requires explicit
selection. Use the current code and validation evidence below; the older M0–M17
documentation describes the PyTorch/XLA era.

## Find the implementation by task

| Task | Start here | Related evidence |
| --- | --- | --- |
| Fine-tune a pretrained causal LM | [short Colab notebook](../notebooks/TrainLM_Text_Fine_Tuning.ipynb) · [short Python example](../examples/finetune_text.py) · [full guide](tutorials/TEXT_FINETUNING.md) | [`run_finetuning`](../src/trainlm/finetuning.py), [API tests](../tests/test_finetuning_api.py), [notebook contract](../tests/contracts/test_current_colab_notebooks.py); use CPU or CUDA, native TPU fine-tuning is not validated |
| Configure fine-tuning evaluation, best-checkpoint selection, early stopping, or resume | [fine-tuning guide](tutorials/TEXT_FINETUNING.md) | [`TrainLMTrainingArguments`](../src/trainlm/api.py), [`restore_finetuning_config`](../src/trainlm/finetuning.py), [lifecycle tests](../tests/test_api_lifecycle.py) |
| Pretrain a causal LM from Hub text or local text files | [short Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) · [short Python example](../examples/pretrain_from_scratch.py) | [`run_pretraining`](../src/trainlm/pretraining_run.py), [`TrainLMPretrainingConfig`](../src/trainlm/pretraining.py), [CPU workflow tests](../tests/test_pretraining_run.py); defaults to random Hugging Face GPT-2; native TPU remains experimental |
| Change model dimensions, token budget, or resume a scratch run | [pretraining guide](tutorials/TPU_PACKED_BIN_PRETRAINING.md) · [README API reference](../README.md#start-here-pretrain-a-language-model-from-scratch) | [`restore_pretraining_report` and `bundle_pretraining_artifacts`](../src/trainlm/pretraining_run.py), [recovery tests](../tests/test_notebook_resume.py) |
| Use the lower-level trainer lifecycle or inspect checkpoints | [README API reference](../README.md) | [`TrainLMTrainer`](../src/trainlm/api.py), [checkpoint retention](../src/trainlm/checkpoint/retention.py), [lifecycle tests](../tests/test_api_lifecycle.py) |
| Check native TPU availability and package compatibility | [T1 v5e-1 guide](T1_V5E1_PROBE.md) | [hardware discovery](../src/trainlm/runtime/torchtpu_discovery.py), [package preflight](../src/trainlm/runtime/torchtpu_trainer_preflight.py) |
| Prepare or inspect packed token data | [packed-data contracts](data/PACKED_BINARY_MANIFEST.md) | [packed data writer](../src/trainlm/data/writer.py), [dataset tests](../tests/test_packed_dataset_resume.py) |
| Understand supported behavior and validation status | [current handoff](HANDOFF.md) | [dense-AR support scope](SCOPE.md), [T0–T11 roadmap](TORCHTPU_REBASELINE.md) |

Useful questions include “How does TrainLM train a randomly initialized GPT-2
model on TinyStories?”, “Where are packed training shards written and resumed?”,
and “Which native TorchTPU checks have actually run?” Start from the table and
follow the linked implementation and tests before making a change.

## Evidence and limits

- `save_total_limit` accepts a positive integer or `None` (API default: keep
  all). The quickstart notebooks leave retention at its API default. After state and run metadata are published,
  the single-device public trainer retains the new save and the newest other
  owned `checkpoint-N` directories within that limit. Each new automatic save
  receives a small ownership marker; unmarked legacy saves, incomplete saves,
  symlinks, and directories with extra files are preserved. Explicit manual
  backups should use a separate name/directory. Retention is storage policy and
  is excluded from the scratch run fingerprint, preserving older fingerprints
  and allowing the limit to change at resume. Cleanup errors warn while keeping
  the new checkpoint. Distributed/XLA coordinator retention is rejected; CPU
  tests do not certify native TPU checkpoint recovery. When periodic checkpoint
  saving is enabled, a successful run also saves its final step if the cadence
  did not already save it. Failed runs do not get this final save; `save_steps=None`
  keeps automatic saving disabled.
- `trainer.latest_checkpoint` returns the newest published checkpoint as a
  `Path`, or `None`; it uses the same lookup as
  `trainer.train(resume_from_checkpoint="latest")`.
- The from-scratch notebook and script are small workflow examples, not quality
  benchmarks. `run_pretraining` writes a detailed report; the artifact helper
  can bundle that report, a completed export, and a resume helper. Keep the run
  directory for checkpoint recovery. CPU tests do not establish TPU training,
  checkpoint recovery, convergence, or performance.
- `TrainLMTrainer.from_scratch(TrainLMPretrainingConfig(...))` is the public
  raw-text entry point. It accepts either an immutable Hugging Face dataset
  revision or paired local UTF-8 train/validation text files, then returns the
  familiar trainer for `train()`, `evaluate()`, and `save_model()` calls. Use
  `trainer.train(resume_from_checkpoint="latest")` to resume from the newest
  completed checkpoint under the trainer output directory. Matching pinned
  preparation settings reuse validated tokenizer and packed-data artifacts in
  `config.output_dir`; change the output directory when changing the dataset or
  tokenizer inputs. `run_config.json` captures the effective model, tokenizer,
  packed data, optimizer, scheduler, training arguments, and runtime identity;
  each checkpoint stores the same fingerprint. Exact resume requires an exact
  match and fails before restoring any state when settings differ.
  `lr_scheduler_type` supports `constant`, `linear`, `cosine`, or `wsd`;
  `warmup_steps=None` keeps the default automatic warmup (floor of 10% of the
  step budget, capped at 100). Set `warmup_ratio` (0–1, rounded up to a whole
  step) or an integer `warmup_steps` for an override; they cannot be combined.
  Set positive `save_steps` and `eval_steps` to control checkpoint and
  evaluation cadence independently; if omitted, both use the automatic interval
  of roughly 10% of the planned update budget. The run report records
  requested and effective values. `load_best_model_at_end=True` selects the
  lowest scratch `eval_loss`; `save_steps` must be a multiple of `eval_steps`,
  and `eval_steps` must reach at least one evaluation within the run. Add
  `early_stopping_patience` and optionally `early_stopping_threshold` to stop
  after validation loss plateaus. The selected step is available on
  `TrainLMPretrainingResult.best_model_selection`; early stopping can finish
  before the token target and is reported explicitly. Set AdamW `betas` and
  `eps` as usual; `max_grad_norm` defaults to `1.0`, and `0` disables gradient
  clipping. The run report records requested and effective values. TPU behavior
  remains experimental.
  `extend_training_budget=True` permits only a strictly larger `max_tokens`
  when effective schedule, warmup, and cadence still match the automatic
  cosine defaults. After a compatible checkpoint is restored,
  it starts a new cosine schedule at the configured learning rate while keeping
  model and optimizer state; this can raise the learning rate, so use it
  deliberately. Other changed settings require a fresh `output_dir`. CPU/API
  tests cover this lifecycle; the owner reports one successful same-runtime
  v5e-1 resume with budget extension. Fresh-process/post-reset resume remains
  unverified. GPT-2
  is the default;
  `model_type` and `model_config_overrides` select a registered Transformers
  causal-LM configuration for random initialization. TrainLM fills vocabulary
  and special-token IDs from the newly trained tokenizer. The path does not
  load custom remote model code.
- `gradient_checkpointing=True` enables the selected Hugging Face model's
  standard activation-checkpointing hook when available and disables KV cache
  during training. Export restores the original cache setting. It is off by
  default; CPU tests confirm training and export behavior, while TPU memory and
  speed effects remain unverified.
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
