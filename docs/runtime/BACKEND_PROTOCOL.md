# Execution backend protocol

The training engine depends on `ExecutionBackend`, a structural Python
protocol. It does not import PyTorch/XLA, CUDA extensions, JAX, or future
TorchTPU packages. A backend can therefore be installed and selected without
changing the trainer or Hugging Face model implementation.

## Ownership boundary

| Concern | Backend responsibility |
|---|---|
| Identity | backend name, device, precision, ranks, world size |
| Lifecycle | initialize/finalize, training begin/end, step begin/end |
| Preparation | model, optimizer, dataloader, and batch preparation |
| Precision | backend-native autocast context |
| Compilation | compile a model or return it unchanged |
| Distribution | create a logical mesh and shard model/optimizer state |
| Execution | backward, gradient clipping, optimizer step, zeroing gradients |
| Coordination | device synchronization and named worker barriers |
| Checkpointing | before/after coordination plus backend-owned runtime state |
| Diagnostics | portable identity fields plus backend-specific scalar facts |

`LogicalMesh` contains only named integer axes. An XLA implementation may turn
it into an XLA mesh, while another backend may map it to DTensor or its own
topology object. Those native objects never enter trainer-facing packages.

Checkpoint hooks coordinate workers but intentionally do not specify file
formats. Internal resume and Hugging Face export formats are defined separately
in M1-F5.

## Portable baseline and compatibility

`TorchRuntime` is the single-process CPU/CUDA baseline. The historical
`Runtime` import remains an alias so existing applications continue to work:

```python
from trainlm.runtime import ExecutionBackend, Runtime, TorchRuntime

runtime: ExecutionBackend = TorchRuntime(
    device="cpu",
    precision="fp32",
    compile_enabled=False,
)

assert isinstance(Runtime(), TorchRuntime)
```

Backends may mutate or wrap an optimizer during `prepare_optimizer`, but must
preserve its association with an already-created scheduler. A future assembly
layer can prepare model and optimizer before constructing the scheduler; until
then, replacement rather than in-place preparation is unsupported.

## Backend-package rule

Imports such as `torch_xla` are allowed only in their concrete backend package.
They are forbidden in `trainlm.training` and the backend protocol module. This
is enforced by a source-level contract test, allowing the core package and its
CPU tests to remain installable without TPU dependencies.

