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
barriers. M5-F1 deliberately does not claim SPMD sharding, graph caching,
compiled operation fusion, or throughput certification; M5-F2 through M5-F7
own those concerns. Runtime state records backend identity and topology for
future checkpoint contracts.
