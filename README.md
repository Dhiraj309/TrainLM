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

TrainLM distinguishes models that are **Compatible**, **Optimized**, and
hardware **Certified**. Generic execution is never presented as TPU performance
certification.
