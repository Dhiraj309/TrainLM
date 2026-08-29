# Canonical Hugging Face export

M7-F3 keeps the public checkpoint format independent from TrainLM internals.
`plan_canonical_hf_export(manifest)` validates a committed
`HFExportManifest` before a backend-specific writer invokes the normal
Transformers `save_pretrained` path.

The plan requires:

- `config.json` and model weights in safetensors format;
- the canonical `huggingface` parameter layout;
- every runtime transform to be reversed;
- tied-weight aliases to remain explicit; and
- no optimizer, scheduler, trainer, runtime, RNG, or data artifacts.

The planner performs no file I/O and does not modify the loaded HF model.  A
writer may shard weights and add the standard index, tokenizer, generation
config, and metadata only after the plan succeeds.  The committed export is
then reloadable by plain `AutoModelForCausalLM.from_pretrained` without
TrainLM.

## TPU acceptance procedure

For each certified model/layout:

1. synchronize the TPU and reverse all temporary runtime transforms;
2. build and validate the canonical export plan;
3. write a committed safetensors export atomically;
4. reload it with plain Transformers; and
5. compare deterministic logits and tied-weight aliases with the source model.

The lifecycle and distributed-state contracts remain separate: training-only
state belongs in the internal resume checkpoint, never in the HF export.
