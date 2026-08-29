# Positional-semantics compatibility

TrainLM keeps the Hugging Face model implementation unchanged.  The generic
model path can describe the positional semantics exposed by a model so that
the TPU conformance matrix can select the right inputs and acceptance checks.

`detect_position_semantics(config, model=None)` returns one of:

- `learned` for learned/absolute position embeddings;
- `rope` for rotary position embeddings;
- `alibi` for ALiBi-style attention bias; or
- `unknown` when the configuration is ambiguous.

Detection is deliberately conservative.  Explicit configuration fields and,
only when needed, module class names are inspected.  A field such as
`max_position_embeddings` is not enough to infer a position scheme, and
contradictory indicators return `unknown` rather than guessing from a model
family name.  The detector is descriptive in M6; it does not mutate the model
or claim that a TPU kernel is available.  Capability-driven replacements are
planned for M8 onward.

## TPU conformance procedure

For each representative learned-position, RoPE, and ALiBi model:

1. Load the model through the normal `AutoModelForCausalLM` provider.
2. Record the detector result and the model's position/attention inputs.
3. Run five finite updates with fixed sequence length, mask shape, position IDs,
   and accumulation structure.
4. Confirm that the post-warmup graph count is stable, no unexpected CPU
   fallback occurs, and loss/gradients remain finite.
5. Save the evidence with the exact model revision, runtime versions, and
   hardware geometry.

This matrix proves semantic compatibility before any positional or attention
optimization is enabled.  Throughput and MFU claims remain gated by the M5
baseline and later optimization milestones.
