# Attention-layout compatibility

TrainLM keeps each Hugging Face attention implementation intact.  The generic
path can describe the layout needed by the TPU conformance matrix through
`detect_attention_layout(config, model=None)`:

- `mha`: query and key/value head counts are equal;
- `gqa`: fewer key/value heads than query heads, with an even grouping;
- `mqa`: one key/value head (or an explicit multi-query flag); and
- `unknown`: incomplete or incompatible evidence.

The result also reports `packed`, `separate`, or `unknown` QKV projections.
Explicit configuration fields are preferred.  Module structure is inspected
only as a fallback for projection packing, and no model-family name is used.
Unknown or contradictory layouts remain compatible-mode fallbacks; they are
never silently rewritten into an optimized kernel layout.

## TPU conformance procedure

The M6-F2 matrix covers tiny official configurations for MHA, GQA, and MQA,
including both packed and separate QKV projections:

1. Load each model with the normal `AutoModelForCausalLM` provider.
2. Record the detected head and projection layout and verify the model's
   forward signature receives the unchanged inputs.
3. Run five finite updates at a fixed shape and accumulation structure.
4. Confirm stable post-warmup graphs, finite loss/gradients, and no unexpected
   CPU fallback.
5. Save model revision, runtime versions, hardware geometry, and graph/fallback
   evidence.

This is a semantic compatibility gate.  Optimized attention kernels and
throughput claims remain disabled until the capability planner and later M10
performance gates provide hardware evidence.
