# TrainLM

TrainLM is a Hugging Face-native framework for efficient language-model
pretraining, with Google TPU execution provided through replaceable runtime and
kernel backends.

The project is implementing its dense autoregressive V1 foundation. Before
relying on a model-support or performance claim, read the
[dense-AR support contract](docs/SCOPE.md).

Environment support and exact CPU/TPU profiles are defined by the
[dependency compatibility policy](docs/DEPENDENCIES.md).

Model architecture and training-policy boundaries are defined by the
[configuration ownership contract](docs/configuration/OWNERSHIP.md).

Device-specific execution is isolated behind the
[execution backend protocol](docs/runtime/BACKEND_PROTOCOL.md).
The optional PyTorch/XLA implementation follows the
[XLA runtime contract](docs/runtime/PYTORCH_XLA.md).
Its logical data-parallel mesh follows the
[XLA SPMD mesh contract](docs/runtime/XLA_SPMD.md).
Persistent compilation caching and fixed-shape enforcement follow the
[XLA cache and shape-guard contract](docs/runtime/XLA_CACHE_GUARD.md).
Accumulation choices and evidence-gated fallbacks follow the
[XLA accumulation strategy contract](docs/runtime/XLA_ACCUMULATION.md).
Compile counters, fallback reports, HLO attachments, and XProf metadata follow
the [XLA diagnostics contract](docs/runtime/XLA_DIAGNOSTICS.md).

Next-token shifting, masking, normalization, and accounting are fixed by the
[causal language-model task contract](docs/tasks/CAUSAL_LM.md).

Optimization discovery and provider selection exchange immutable
[capability and execution-plan schemas](docs/optimization/SCHEMAS.md).

Exact resume state and plain Transformers output follow separate
[checkpoint and Hugging Face export contracts](docs/checkpoint/CONTRACTS.md).
Generic TPU save/resume/export comparisons follow the
[round-trip contract](docs/checkpoint/TPU_ROUND_TRIP.md).
Distributed checkpoint state ownership follows the
[distributed resume contract](docs/checkpoint/DISTRIBUTED_RESUME.md).
Asynchronous publication follows the
[checkpoint lifecycle contract](docs/checkpoint/ASYNC_LIFECYCLE.md).
Canonical model publication follows the
[Hugging Face export contract](docs/checkpoint/CANONICAL_HF_EXPORT.md).

Generic dense causal models are acquired through the
[Hugging Face causal-model provider](docs/models/HUGGINGFACE_PROVIDER.md).
Model batches follow the model's own
[forward-signature dispatch contract](docs/models/BATCH_DISPATCH.md).
Trainer lifecycle, stop, evaluation, and checkpoint hook ordering follow the
[backend-neutral lifecycle contract](docs/training/LIFECYCLE.md).
Microbatch reduction and token accounting follow the
[token-normalized accumulation contract](docs/training/ACCUMULATION.md).
Optimizer construction and moment precision follow the
[optimizer state policy](docs/optimization/OPTIMIZER_POLICY.md).
Token-indexed warmup/stable/decay scheduling follows the
[WSD scheduler contract](docs/training/TOKEN_SCHEDULER.md).
Streaming, token-weighted evaluation follows the
[streaming evaluation contract](docs/evaluation/STREAMING_EVALUATION.md).
Host callbacks consume immutable, materialized metrics through the
[sync-safe callback contract](docs/training/CALLBACKS.md).
Generic dense-AR trainability is checked across representative HF families by
the [multi-family overfit matrix](docs/training/MULTI_FAMILY_OVERFIT.md).
Updated models must pass the
[plain Hugging Face round-trip contract](docs/models/PLAIN_HF_ROUNDTRIP.md).
Representative non-Llama families are exercised by the
[dense-AR CPU conformance matrix](docs/models/DENSE_AR_CPU_MATRIX.md).
Every loaded model has a stable
[generic compatibility explanation](docs/models/COMPATIBILITY_EXPLANATION.md)
with explicit capabilities, providers, and fallbacks.
Positional encoding detection and the TPU conformance procedure are defined in
the [positional-semantics contract](docs/models/POSITION_SEMANTICS.md).
Attention head and QKV projection coverage follows the
[attention-layout contract](docs/models/ATTENTION_LAYOUTS.md).
Normalization, MLP, and residual coverage follows the
[transformer-block contract](docs/models/BLOCK_LAYOUTS.md).

Packed `.bin` inputs are accepted only through the versioned
[binary shard manifest and integrity contract](docs/data/PACKED_BINARY_MANIFEST.md).
Remote shards use the
[revision-pinned Hugging Face dataset source](docs/data/HUGGINGFACE_SHARD_SOURCE.md).
Validated payloads are exposed through the
[contiguous fixed-shape batch reader](docs/data/CONTIGUOUS_BATCH_READER.md).
Shard order and rank ownership follow the
[deterministic packed-data partition contract](docs/data/DETERMINISTIC_PARTITIONING.md).
Packed reads can overlap training through the bounded, backend-aware
[asynchronous prefetch contract](docs/data/ASYNC_PREFETCH.md).
Exact next-batch restart state follows the
[resumable cursor contract](docs/data/RESUMABLE_CURSOR.md).

TrainLM distinguishes models that are **Compatible**, **Optimized**, and
hardware **Certified**. Generic execution is never presented as TPU performance
certification.
