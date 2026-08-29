# Generic TPU round-trip contract

M6-F4 verifies that a generic Hugging Face causal model remains usable after
an internal checkpoint resume and after canonical Hugging Face export.  The
model implementation is never replaced by a family-specific copy.

`evaluate_round_trip` is an evidence-boundary helper.  It compares:

- reference and resumed state dictionaries;
- the reference and resumed next-update outputs; and
- reference logits with logits from the reloaded canonical export.

It records missing/unexpected state keys and maximum absolute errors, then
reports whether every comparison is within the configured tolerance.  It does
not perform file I/O or run in the training hot path.  Callers must synchronize
the TPU before collecting comparison tensors.

## TPU acceptance procedure

For every M6 representative family and layout cluster:

1. Run a short generic training segment and save a committed internal resume
   checkpoint.
2. Continue once without interruption and record the next-update state/output.
3. Reload the resume checkpoint on the same topology and compare the next
   update with the uninterrupted reference using the helper.
4. Export the model to canonical safetensors, reload it through plain
   `AutoModelForCausalLM`, and compare deterministic logits.
5. Record model revision, TrainLM/runtime versions, topology, tolerance, and
   manifest IDs with the evidence.

The gate requires equivalent next-update state, clean HF reload, and no
remaining runtime transforms.  Sharded writing, asynchronous lifecycle, and
production telemetry are implemented in M7; this contract only defines the
comparison and acceptance boundary.
