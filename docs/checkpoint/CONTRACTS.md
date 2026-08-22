# Checkpoint and Hugging Face export contracts

TrainLM has two deliberately separate persistence formats. They serve different
purposes and must never be substituted for one another.

| Format | Purpose | Portable to plain Transformers? |
|---|---|---|
| `ResumeManifest` | Exact continuation after shutdown or preemption | No |
| `HFExportManifest` | Evaluation, sharing, and `from_pretrained` loading | Yes |

Both use schema version 1 and content-address every artifact with a SHA-256
digest and byte size. Paths are safe relative POSIX paths; absolute paths,
parent traversal, and duplicate paths are rejected.

## Exact internal resume

A committed resume checkpoint contains all of the following:

- model parameters and buffers;
- optimizer slots keyed by canonical parameter name;
- scheduler state;
- trainer progress, including optimizer/micro steps, tokens, and samples;
- backend/runtime state;
- RNG state;
- data-pipeline state and an exact cursor for every replica and worker;
- backend, precision, world size, logical mesh, and canonical/sharded layout;
- capability fingerprint, execution-plan ID, runtime parameter layout, and
  every applied reversible transformation.

Each component has a `StateDescriptor` specifying implementation, independent
format version, layout, keying scheme, and its artifact IDs. This avoids
guessing how an optimizer or sharded state file should be interpreted.

An internal resume manifest is exact only for a compatible framework/backend
and topology. Later implementations may explicitly reshard canonical states,
but no loader may infer or silently reinterpret an incompatible layout.

## Hugging Face export

A committed export requires `config.json` and safetensors model weights in the
canonical Hugging Face parameter layout. It records architecture/config class,
Transformers version, dtype, tied-weight aliases, source checkpoint, and all
reversed runtime transforms. No transform may remain active.

Optimizer, scheduler, trainer, runtime, RNG, and data state are forbidden from
the export contract. Optional standard artifacts may include the safetensors
index, generation config, tokenizer files, and model-card metadata. The extra
TrainLM manifest does not change ordinary `AutoModelForCausalLM.from_pretrained`
behavior.

## Atomic publication and incomplete saves

Writers must follow this sequence:

1. create a unique transaction directory using `.incomplete`;
2. write artifact files and compute their digests;
3. write a `staging` manifest;
4. synchronize asynchronous writes and participating workers;
5. verify required roles, state descriptors, shards, cursors, transforms,
   digests, and sizes;
6. publish with one atomic directory rename or a final `COMPLETED` marker; and
7. expose a manifest whose commit status is `complete`.

`staging` and `failed` transactions are serializable for diagnosis but are
never resumable/loadable. Readers must ignore them. Cleanup or recovery may be
implemented separately, but automatic guessing from partial files is forbidden.

The runtime `before_checkpoint` and `after_checkpoint` hooks coordinate device
work; they do not weaken these publication rules.

## Versioned schemas

- `schemas/checkpoint/resume_manifest_v1.schema.json`
- `schemas/checkpoint/hf_export_manifest_v1.schema.json`

This milestone defines contracts only. Distributed writing, asynchronous
lifecycle management, resume loading, resharding, and `save_pretrained` export
are implemented and tested on TPU in M7.

