# TrainLM examples

## Start here

- [Pretrain from scratch](pretrain_from_scratch.py): one public API call that
  uses a Hugging Face GPT-2 config and tokenizer with random weights, streams
  raw TinyStories text, and takes a non-overlapping validation slice. The
  10,000-token setting is a workflow smoke test.
- [Fine-tune text](finetune_text.py): one public API call using a tiny model and
  three steps; increase the budget and choose a model for a real run.

## Runtime diagnostics

The `torchtpu_t1_*.py` files and `torchtpu_native_trainer_smoke.py` collect
runtime, package, lifecycle, or distributed evidence. They support the TPU
validation notebooks; they are diagnostic tools, not beginner training
examples. Start with the public API scripts above for ordinary use.

## CLI configuration

- [Generic dense causal-LM config](dense_ar_pretraining.yaml): current public
  trainer/CLI template. Replace the placeholder model and provide datasets.
- [SmolLM2 TPU lifecycle config](smollm2_135m_tpu_pretraining.yaml): retained
  as a historical PyTorch/XLA example (`accelerator="tpu"`); it is not the
  native TorchTPU from-scratch workflow.
