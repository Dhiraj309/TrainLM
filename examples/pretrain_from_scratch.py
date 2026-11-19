"""Short native TorchTPU example using a Hub architecture and raw HF text."""

from trainlm import TrainLMPretrainingConfig, run_pretraining


config = TrainLMPretrainingConfig(
    output_dir="runs/tinystories-gpt2",
    max_tokens=10_000,  # Workflow smoke only; increase for real training.
    sequence_length=128,
    model_name_or_path="openai-community/gpt2",
    model_initialization="random",  # HF config and tokenizer; no model weights.
    dataset_id="roneneldan/TinyStories",
    dataset_revision="main",  # TrainLM records the resolved immutable commit.
    train_split="train[:90%]",
    validation_split="train[90%:]",
    text_field="text",
    accelerator="native_tpu",
)
result = run_pretraining(config, show_progress=True)
print(result.summary())
print(
    f"architecture: {result.report['model']['model_class']} "
    f"({result.report['model']['model_type']})"
)
print(f"report: {result.report_path}")
print(f"model:  {result.export_path}")
# Optional after `hf auth login`: result.push_to_hub("you/tinystories-gpt2")
