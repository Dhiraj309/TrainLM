# Optional PyTorch/XLA backend

`XlaRuntime` implements the same `ExecutionBackend` contract as the portable
`TorchRuntime`. Importing `trainlm.runtime` does not import `torch_xla`; the
optional package is loaded only when `XlaRuntime` is constructed. CPU/CUDA
installations therefore remain usable without TPU dependencies.

Install the pinned TPU profile before constructing it:

```bash
python -m pip install -e ".[tpu-xla]" -c constraints/tpu-xla-2.9.txt
```

Use it with the existing trainer without changing the Hugging Face model or
task:

```python
from trainlm.runtime import XlaRuntime

runtime = XlaRuntime(precision="bf16")
trainer = Trainer(
    config=config,
    model=model,
    runtime=runtime,
    optimizer=optimizer,
    scheduler=scheduler,
    task=CausalLMTask(),
    train_dataloader=train_loader,
)
```

The backend delegates optimizer updates to `xm.optimizer_step`, flushes lazy
graphs at explicit step/end boundaries, and uses XLA rendezvous for named
barriers. Model-only compilation remains disabled by design. For an explicit
device-step callable, opt into the PyTorch/XLA training compiler:

```python
runtime = XlaRuntime(precision="bf16", compile_training=True)
compiled_step = runtime.compile_training_step(device_step)
```

The callable should contain only forward, loss, backward, reduction, gradient
clipping, and optimizer-update operations. Data loading, host token accounting,
logging, callbacks, and checkpointing stay outside it. PyTorch/XLA currently
documents `torch_xla.compile` as the recommended training-step boundary;
TrainLM keeps this hook explicit until the M5 accumulation spike selects and
validates one complete trainer integration. Runtime state records backend,
compiler mode, and topology for future checkpoint contracts.
