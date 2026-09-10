# Secure TPU pretraining from packed binary shards

This guide uses only TrainLM's public package surface. TrainLM validates every
manifest and payload before launching workers; users do not invoke worker
scripts, set PJRT variables, or parse stage logs.

## 1. Provide credentials as a secret

Create a read-only Hugging Face token and store it in the secret manager of the
TPU environment (for example, Kaggle Secrets or the Cloud TPU job secret
store). Expose it to the process as `HF_TOKEN`. Never paste a token into Python,
YAML, notebooks, shell history, logs, manifests, or checkpoint directories.
Do not print the environment variable. Revoke and rotate it if disclosure is
suspected.

## 2. Pin immutable revisions and splits

Use lowercase 40-character commit SHAs for both the model and packed dataset.
Keep training and evaluation manifests in separate, explicit shard lists.
TrainLM rejects mutable Hub model revisions before starting the TPU coordinator;
local model directories remain available for offline, immutable snapshots.

```python
import os

from trainlm import (
    HuggingFaceShardSourceConfig,
    HuggingFaceShardSpec,
    PackedBinDataset,
    TrainLMTrainer,
    TrainLMTrainingArguments,
)

if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Configure HF_TOKEN in the platform secret manager.")

model_revision = os.environ["MODEL_REVISION"]
data_revision = os.environ["DATA_REVISION"]

train_data = PackedBinDataset.from_hub(
    HuggingFaceShardSourceConfig(
        repo_id="org/packed-corpus",
        revision=data_revision,
        shards=(HuggingFaceShardSpec("train-000", "train/000.manifest.json"),),
    ),
    sequence_length=2048,
    split="train",
)
eval_data = PackedBinDataset.from_hub(
    HuggingFaceShardSourceConfig(
        repo_id="org/packed-corpus",
        revision=data_revision,
        shards=(HuggingFaceShardSpec("eval-000", "eval/000.manifest.json"),),
    ),
    sequence_length=2048,
    split="eval",
)

trainer = TrainLMTrainer.from_pretrained(
    "org/model",
    revision=model_revision,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=TrainLMTrainingArguments(
        output_dir="runs/secure-tpu",
        accelerator="tpu",
        bf16=True,
        max_steps=1000,
        save_steps=100,
        eval_steps=100,
    ),
)
print(trainer.explain(format="text"))
trainer.train()
```

`PackedBinDataset.from_directory(...)` is the offline alternative. Copy the
manifest, payload, and optional document index together; validation rejects
missing files, unsafe paths, size/hash mismatches, duplicate shard IDs, and
incompatible token layouts before coordinator launch.

## 3. Resume only committed checkpoints

Resume from the checkpoint directory published by TrainLM, not a temporary
file or an individual rank shard:

```python
trainer.train(resume_from_checkpoint="runs/secure-tpu/checkpoint-500")
```

The checkpoint manifest must be committed and its topology must match the TPU
job. Keep checkpoint storage private because optimizer state and training data
position can reveal information even when model weights are intended for
publication.

## 4. Export without claiming unsupported behavior

For local CPU/CUDA runs, `trainer.save_model("export/model")` writes a plain
Hugging Face checkpoint. The current TPU coordinator does **not** yet expose
canonical model export through `save_model()`; it raises `NotImplementedError`.
Do not rename rank-local training shards and present them as a Hugging Face
model. TPU release certification requires the plain-HF export gate and a clean
Transformers-only reload before publication.

## 5. Operational checklist

- Pin model and dataset commit SHAs; record them with the run artifacts.
- Grant the token read-only access and never serialize or log it.
- Use disjoint, explicit train/eval shard lists.
- Inspect `trainer.explain()` before launch and retain its output.
- Resume only a committed checkpoint with matching topology.
- Retain coordinator summaries, metrics, profiles, and HLO evidence privately.
- Publish only canonical exports that pass a clean-environment reload.
