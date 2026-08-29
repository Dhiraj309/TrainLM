# Transformer-block layout compatibility

TrainLM does not copy or patch Hugging Face block implementations.  It can
describe the structural variants that the generic TPU conformance matrix must
exercise through `detect_block_layout(config, model=None)`:

- normalization: `layernorm`, `rmsnorm`, or `unknown`;
- MLP/activation: `gelu`, `swiglu`, `geglu`, or `unknown`; and
- residual path: `serial`, `parallel`, or `unknown`.

Explicit configuration fields are preferred.  Module class names are an
optional fallback when a configuration omits a layout.  Conflicting fields
remain `unknown`; TrainLM never infers an optimization from a model-family
name or silently changes the block's mathematical semantics.

## TPU conformance procedure

The M6-F3 matrix uses tiny representative configurations for LayerNorm and
RMSNorm, GELU and gated MLPs, and serial and parallel residual blocks:

1. Load each unchanged model through the Hugging Face provider.
2. Record the detected block layout and verify forward inputs/outputs remain
   model-native.
3. Run five finite updates with fixed shapes and accumulation structure.
4. Confirm stable post-warmup graphs, finite loss/gradients, and no unexpected
   CPU fallback.
5. Save model revision, runtime versions, hardware geometry, and graph/fallback
   evidence.

This establishes semantic compatibility before M8 capability transformations
or TPU-specific kernels are enabled.  Throughput and MFU remain governed by
the later optimization and parity gates.
