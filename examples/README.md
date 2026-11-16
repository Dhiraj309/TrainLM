# TrainLM examples

## Start here

- [Pretrain from scratch](pretrain_from_scratch.py): one public API call; the
  10,000-token setting is a workflow smoke test.
- [Fine-tune text](finetune_text.py): one public API call using a tiny model and
  three steps; increase the budget and choose a model for a real run.

## Runtime diagnostics

The `torchtpu_t1_*.py` files and `torchtpu_native_trainer_smoke.py` collect
runtime, package, lifecycle, or distributed evidence. They support the TPU
validation notebooks; they are diagnostic tools, not beginner training
examples. Start with the public API scripts above for ordinary use.

The YAML files show the more configurable CLI workflow.
