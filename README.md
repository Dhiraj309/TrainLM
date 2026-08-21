# TrainLM

TrainLM is a Hugging Face-native framework for efficient language-model
pretraining, with Google TPU execution provided through replaceable runtime and
kernel backends.

The project is currently implementing its dense autoregressive V1 foundation.
Before relying on a model-support or performance claim, read:

- [Dense-AR support contract](docs/SCOPE.md)
- [Implementation roadmap](docs/ROADMAP.md)

TrainLM distinguishes models that are **Compatible**, **Optimized**, and
hardware **Certified**. Generic execution is never presented as TPU performance
certification.
