# XLA SPMD data-parallel mesh

`XlaRuntime.create_mesh(LogicalMesh(...))` validates that the product of the
requested axes equals the XLA world size, then constructs a native
`torch_xla.distributed.spmd.Mesh`. The returned `XlaMesh` keeps the portable
logical mesh and native object together; native objects do not enter trainer
or task APIs.

M5-F2 uses a data-parallel convention:

- model parameters are replicated with an empty `PartitionSpec`;
- every non-scalar batch tensor is sharded on its leading `data` dimension;
- `xm.optimizer_step` performs XLA-native cross-replica gradient reduction.

```python
mesh = runtime.create_mesh(LogicalMesh({"data": 8}))
model = runtime.shard_model(model, mesh)
optimizer = runtime.shard_optimizer(optimizer, mesh)
```

This stage does not introduce tensor, sequence, pipeline, or FSDP axes. It
also does not change model implementations or the generic task contract.
Those layouts are validated in M6 after the single-axis DP path is stable.
Static batch/sequence enforcement, graph caching, and compiled operation
boundaries remain M5-F3 and M5-F4 responsibilities.
