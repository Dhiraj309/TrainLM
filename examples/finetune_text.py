"""Small pretrained-model fine-tuning example for CPU or CUDA."""

from trainlm import (
    TrainLMFineTuningConfig,
    TrainLMTrainingArguments,
    run_finetuning,
)


config = TrainLMFineTuningConfig(
    model_name_or_path="sshleifer/tiny-gpt2",
    dataset_id="roneneldan/TinyStories",
    dataset_revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    text_field="text",
    args=TrainLMTrainingArguments(
        output_dir="runs/tinystories-finetune",
        max_steps=3,  # Workflow smoke only; increase for real training.
        sequence_length=128,
        eval_strategy="steps",
        eval_steps=3,
        save_strategy="steps",
        save_steps=3,
        accelerator="auto",
    ),
)
result = run_finetuning(config)
print(result.summary())
