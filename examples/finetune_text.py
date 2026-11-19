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
    # Bound preprocessing as well as training for this workflow smoke test.
    train_split="train[:1000]",
    validation_split="validation[:100]",
    args=TrainLMTrainingArguments(
        output_dir="runs/tinystories-finetune",
        max_steps=3,  # Workflow smoke only; increase for real training.
        sequence_length=128,
        # run_finetuning performs one final evaluation automatically.
        eval_strategy="no",
        save_strategy="no",
        accelerator="auto",
    ),
)
result = run_finetuning(config)
print(result.summary())
# Optional after `hf auth login`: result.push_to_hub("you/tinystories-model")
