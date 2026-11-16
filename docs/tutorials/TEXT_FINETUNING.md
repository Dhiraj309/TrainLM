# Fine-tune a pretrained causal language model

[Open the copyable notebook in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/feat%2F01-pretrained-finetuning-api/notebooks/TrainLM_Text_Fine_Tuning.ipynb) · [View notebook source](../../notebooks/TrainLM_Text_Fine_Tuning.ipynb)

Use this workflow when you already have pretrained weights and want to adapt
them to a text domain. It is separate from [pretraining from scratch](../../README.md#start-here-pretrain-a-language-model-from-scratch),
which initializes random weights and learns a tokenizer.

`run_finetuning` owns dataset loading, tokenization, training, evaluation, and
local model export. When it loads a dataset from the Hub, give it an immutable
dataset revision and a distinct validation split:

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
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
    ),
)

result = run_finetuning(config)
print(result.summary())
```

## Train for a number of dataset epochs

Use `num_train_epochs` when you want the training budget tied to a finite
dataset instead of a fixed number of optimizer updates. Set `max_steps=None`
so the step budget does not conflict:

```python
epoch_args = TrainLMTrainingArguments(
    output_dir="runs/qwen-tinystories-epochs",
    max_steps=None,
    num_train_epochs=3,
    sequence_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
)
epoch_config = TrainLMFineTuningConfig(
    model_name_or_path=config.model_name_or_path,
    dataset_id=config.dataset_id,
    dataset_revision=config.dataset_revision,
    text_field=config.text_field,
    train_split=config.train_split,
    validation_split=config.validation_split,
    args=epoch_args,
)
result = run_finetuning(epoch_config)
```

TrainLM counts the batches after applying the per-device batch size, and ends
each epoch before starting the next. If an epoch ends partway through gradient
accumulation, TrainLM applies the accumulated update there. Fractional epoch
budgets round up to the next data batch. Epoch budgets currently require a
finite, map-style dataset and a single-device CPU, CUDA, or experimental native
TorchTPU runtime. Streaming datasets, distributed loaders, and the legacy XLA
coordinator should use `max_steps` instead.
`result.summary()` and the run report include the resolved batches-per-epoch,
batch target, and optimizer-step horizon.

## Schedule evaluation and checkpoints

Set `eval_strategy` and `save_strategy` to `"steps"` or `"epoch"`. Step
strategies use `eval_steps` or `save_steps`; epoch strategies run after the
last batch in each finite map-style dataset pass. At an epoch boundary, TrainLM
flushes any partial gradient-accumulation group before evaluating or saving.
Epoch schedules require a single-device runtime and are not available through
the legacy PyTorch/XLA coordinator. Omit the strategy to preserve the existing
behavior: a configured step interval schedules that action, while no interval
disables scheduled actions. `run_finetuning` still evaluates once at the end
when a validation split is configured. When saving is enabled, the last
successful training state is also checkpointed on normal completion.

## Use preloaded datasets and your own formatting

The same config can use dataset splits you loaded or filtered yourself. When
you pass both splits directly, `dataset_id` and `dataset_revision` are
optional. Include them if you want the run summary to record a source; for
local or transformed data, omit them instead of inventing a Hub identity. If
you load the splits from the Hub as below, keep the immutable ID and revision.
This example assumes a dataset whose rows contain `prompt` and `response`
fields; update the config and field names for your data. The formatter receives
one row and returns the text to tokenize:

```python
from datasets import load_dataset

train_dataset = load_dataset(
    config.dataset_id,
    revision=config.dataset_revision,
    split=config.train_split,
)
eval_dataset = load_dataset(
    config.dataset_id,
    revision=config.dataset_revision,
    split=config.validation_split,
)

def format_example(example):
    return f"### Prompt\n{example['prompt']}\n### Response\n{example['response']}"

result = run_finetuning(
    config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=format_example,
)
```

For local or transformed splits, create the config without any dataset
identity and pass the two splits directly:

```python
from datasets import Dataset

train_dataset = Dataset.from_dict({"text": ["Hello world.", "A second example."]})
eval_dataset = Dataset.from_dict({"text": ["A held-out example."]})
local_config = TrainLMFineTuningConfig(
    model_name_or_path="Qwen/Qwen3-0.6B",
    args=TrainLMTrainingArguments(output_dir="runs/local-finetuning"),
)
result = run_finetuning(
    local_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

Pass `tokenizer=` to reuse a tokenizer you loaded, or `data_collator=` to
control padding and labels. When a custom collator is set, the mapped batch
retains original dataset columns so the collator can build task-specific
labels. Caller-provided splits must expose Hugging Face Dataset's
`column_names` and `map` interface. On resume, pass the same splits, tokenizer,
formatter, and collator again; the summary records their source identity and
formatter name but cannot recreate Python objects or callback code.

The model and tokenizer are saved under `config.args.output_dir/model`. The
helper does not upload artifacts. `result.summary()` returns the requested and
resolved model revisions, pinned dataset revision, resolved tokenizer
revision, final step, loss, evaluation metrics, export path, and summary path.
The JSON report at `result.summary_path` also records the data source, text
field, split names, formatter name, and all `TrainLMTrainingArguments`. TrainLM
pins model loading to the tokenizer's resolved Hub commit when available. If
both revisions are available
and differ, the run stops before training. Revision fields are `null` for local
sources or loaders that do not expose commit metadata. If a run fails, the same
file records the failed phase, error type, last known step, metrics, and
checkpoint path before the original exception is raised. The error message is
not copied into the report. Install the optional dataset dependency with
`pip install 'trainlm[finetuning]'`.

## Resume an interrupted run

Set `save_strategy="steps"` with `save_steps` for step-based checkpoints, or
`save_strategy="epoch"` to save at epoch boundaries. Keep the full
`output_dir` between sessions. To continue from its newest checkpoint, load
the saved configuration with `restore_finetuning_config` and increase the same
training budget (`max_steps` or `num_train_epochs`). The helper restores the
model, dataset, and other training and scheduling settings, uses the summary
file's directory as `output_dir`, and checks local archive or native TPU
manifest metadata before returning the config. This catches missing or
malformed publication without unpickling local
checkpoint data; the trainer checks model/state compatibility during restore.
The saved checkpoint contains optimizer and trainer state; the exported
`model` directory alone is not enough to resume. On Colab or Kaggle, keep the
summary and checkpoints on persistent storage if a runtime reset should not
discard the run. In the notebook, a missing or incompatible summary or
checkpoint produces recovery instructions and skips training and bundling, so
a previous model is not presented as the resumed run.

```python
from trainlm import restore_finetuning_config, run_finetuning

config = restore_finetuning_config(
    "runs/qwen-tinystories/finetuning-summary.json",
    max_steps=2000,  # Must be greater than the saved step.
)
result = run_finetuning(config, resume_from_checkpoint="latest")
```

For an epoch-based run, supply `num_train_epochs` instead:

```python
config = restore_finetuning_config(
    "runs/qwen-tinystories-epochs/finetuning-summary.json",
    num_train_epochs=4,
)
result = run_finetuning(config, resume_from_checkpoint="latest")
```

## Choose settings for your task

- Confirm the model is a causal language model and the selected dataset field
  contains plain text, or provide `formatting_func` to build it from row fields.
- Keep evaluation examples in a separate validation split. This helper does
  not create a split or infer one from training data.
- Set `sequence_length`, batch size, training budget, and evaluation/save
  cadence to fit the runtime and memory available to you. The example values
  are a starting point, not a convergence or performance guarantee.
- For chat or instruction fine-tuning, use `formatting_func` to apply the
  model's chat template and a custom `data_collator` for task-specific label
  masking.
- Native TorchTPU is experimental and has not been certified by this workflow.
  Set `accelerator="native_tpu"` explicitly to try that path; automatic TPU
  selection is not enabled. The legacy PyTorch/XLA coordinator is not supported
  by this one-call export path.

For already-tokenized datasets, datasets without Hugging Face Dataset's
`map`/`column_names` interface, or direct control of the training lifecycle,
use [`TrainLMTrainer.from_pretrained`](../../src/trainlm/api.py) with `train()`,
`evaluate()`, and `save_model()`.
