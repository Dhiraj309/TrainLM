# Fine-tune a pretrained causal language model

[Open the copyable notebook in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/feat%2F01-pretrained-finetuning-api/notebooks/TrainLM_Text_Fine_Tuning.ipynb) · [View notebook source](../../notebooks/TrainLM_Text_Fine_Tuning.ipynb)

Use this workflow when you already have pretrained weights and want to adapt
them to a text domain. It is separate from [pretraining from scratch](../../README.md#start-here-pretrain-a-language-model-from-scratch),
which initializes random weights and learns a tokenizer.

`run_finetuning` owns dataset loading, tokenization, training, evaluation, and
local model export. Give it an immutable dataset revision and a distinct
validation split:

```python
from trainlm import (
    TrainLMFineTuningConfig,
    TrainLMTrainingArguments,
    run_finetuning,
)

config = TrainLMFineTuningConfig(
    model_name_or_path="Qwen/Qwen3-0.6B",
    dataset_id="roneneldan/TinyStories",
    dataset_revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    text_field="text",
    args=TrainLMTrainingArguments(
        output_dir="runs/qwen-tinystories",
        max_steps=1000,
        sequence_length=512,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
    ),
)

result = run_finetuning(config)
print(result.summary())
```

The model and tokenizer are saved under `runs/qwen-tinystories/model`. The
helper does not upload artifacts. `result.summary()` returns the requested and
resolved model revisions, pinned dataset revision, resolved tokenizer
revision, final step, loss, evaluation metrics, export path, and summary path.
The JSON report at `result.summary_path` also records the text field, split
names, and all `TrainLMTrainingArguments`. TrainLM pins model loading to the
tokenizer's resolved Hub commit when available. If both revisions are available
and differ, the run stops before training. Revision fields are `null` for local
sources or loaders that do not expose commit metadata. If a run fails, the same
file records the failed phase, error type, last known step, metrics, and
checkpoint path before the original exception is raised. The error message is
not copied into the report. Install the optional dataset dependency with
`pip install 'trainlm[finetuning]'`.

## Resume an interrupted run

Set `save_steps` to the checkpoint interval you can afford, and keep the full
`output_dir` between sessions. To continue from its newest checkpoint, load
the saved configuration with `restore_finetuning_config` and set `max_steps`
above the saved step. The helper restores the model, dataset, and other
training settings, and uses the summary file's directory as `output_dir`. The
saved checkpoint contains optimizer and trainer state; the exported `model`
directory alone is not enough to resume. On Colab or Kaggle, keep the summary
and checkpoints on persistent storage if a runtime reset should not discard
the run. In the notebook, a missing or incompatible summary produces recovery
instructions and skips training and bundling, so a previous model is not
presented as the resumed run.

```python
from trainlm import restore_finetuning_config, run_finetuning

config = restore_finetuning_config(
    "runs/qwen-tinystories/finetuning-summary.json",
    max_steps=2000,  # Must be greater than the saved step.
)
result = run_finetuning(config, resume_from_checkpoint="latest")
```

## Choose settings for your task

- Confirm the model is a causal language model and the selected dataset field
  contains plain text.
- Keep evaluation examples in a separate validation split. This helper does
  not create a split or infer one from training data.
- Set `sequence_length`, batch size, training budget, and evaluation/save
  cadence to fit the runtime and memory available to you. The example values
  are a starting point, not a convergence or performance guarantee.
- For chat or instruction fine-tuning, apply the model's chat template and
  task-specific label masking before using the generic trainer API. This
  convenience helper currently tokenizes a plain text column only.
- Native TorchTPU is experimental and has not been certified by this workflow.
  Set `accelerator="native_tpu"` explicitly to try that path; automatic TPU
  selection is not enabled. The legacy PyTorch/XLA coordinator is not supported
  by this one-call export path.

For custom preprocessing, already-tokenized datasets, or direct control of the
training lifecycle, use [`TrainLMTrainer.from_pretrained`](../../src/trainlm/api.py)
with `train()`, `evaluate()`, and `save_model()`.
