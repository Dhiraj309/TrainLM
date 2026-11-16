"""Small native TorchTPU from-scratch pretraining example."""

from trainlm import TrainLMPretrainingConfig, run_pretraining


config = TrainLMPretrainingConfig(
    output_dir="runs/tinystories",
    max_tokens=10_000,  # Workflow smoke only; increase for real training.
    sequence_length=128,
    model_embedding_size=256,
    model_layers=4,
    model_attention_heads=4,
    accelerator="native_tpu",
)
result = run_pretraining(config, show_progress=True)
print(result.summary())
print(f"report: {result.report_path}")
print(f"model:  {result.export_path}")
