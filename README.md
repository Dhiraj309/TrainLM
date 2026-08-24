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

Next-token shifting, masking, normalization, and accounting are fixed by the
[causal language-model task contract](docs/tasks/CAUSAL_LM.md).

Optimization discovery and provider selection exchange immutable
[capability and execution-plan schemas](docs/optimization/SCHEMAS.md).

Exact resume state and plain Transformers output follow separate
[checkpoint and Hugging Face export contracts](docs/checkpoint/CONTRACTS.md).

Generic dense causal models are acquired through the
[Hugging Face causal-model provider](docs/models/HUGGINGFACE_PROVIDER.md).
Model batches follow the model's own
[forward-signature dispatch contract](docs/models/BATCH_DISPATCH.md).
Updated models must pass the
[plain Hugging Face round-trip contract](docs/models/PLAIN_HF_ROUNDTRIP.md).
Representative non-Llama families are exercised by the
[dense-AR CPU conformance matrix](docs/models/DENSE_AR_CPU_MATRIX.md).
Every loaded model has a stable
[generic compatibility explanation](docs/models/COMPATIBILITY_EXPLANATION.md)
with explicit capabilities, providers, and fallbacks.

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

TrainLM distinguishes models that are **Compatible**, **Optimized**, and
hardware **Certified**. Generic execution is never presented as TPU performance
certification.
