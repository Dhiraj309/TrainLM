# Canonical state conversion for packed parameters

M8-F5 adds explicit `StateLayout`, `StateTensor`, and `TensorPack` declarations
to the optimization layer. The checkpoint package remains framework independent.
No model-family inference, kernel implementation, provider registration, or
public trainer integration is introduced by this codec.

## Mapping contract

`StateLayout.from_module()` captures registered parameter/buffer names, shapes,
dtypes, gradient flags, persistence, and tensor aliases. `TensorPack` declares an
ordered list of canonical source names, a packed target name, and a concatenation
dimension. The inverse splits using the declared source sizes, including unequal
sizes. A single-source pack provides an explicit rename. Mappings reject missing
or overlapping sources, target collisions, incompatible shapes/dtypes/grad flags,
invalid dimensions, and alias cycles. JSON metadata is versioned and has a stable
content fingerprint.

```python
from trainlm.optimization import StateLayout, TensorPack

layout = StateLayout.from_module(block, packs=(
    TensorPack(("q_proj.weight", "k_proj.weight", "v_proj.weight"), "qkv_weight", 0),
))
packed = layout.convert(block.state_dict())
canonical = layout.convert(packed, to_canonical=True)
```

State keys, shapes, dtypes, dense materialization, and alias values are validated
before conversion. Outputs are detached copies; alias keys are reattached to
their canonical owner's output tensor. Unchanged embedding/head ties remain
shared. Packing a source that belongs to a tied group is rejected, including
ties outside the target module: preserving such ties would require an explicit
view/gradient/optimizer implementation. Only floating parameters may be packed;
buffers remain in canonical layout.

## Application and undo

A layout-changing `ModelTransformation` must have
`parameter_layout_change=True` and a registered handler whose `layout` is a
`StateLayout` or a factory receiving an isolated source-module copy. Application
validates source metadata, the candidate's transformed metadata, and exact
inverse-mapped tensor values before commit. The existing inverse handler must
still reconstruct canonical modules and values. Unmapped tensors retain their
original objects; only declared packed parameters acquire new ownership.

Apply at initialization with cleared gradients and before constructing an
optimizer. Layout changes must not silently discard pending gradients. Shared
module aliases, hooks, modes, and transaction rollback retain the M8-F4 guards.
Decide device placement before taking a handle that will be used for later
state I/O: moving a model can replace buffers/parameters and is treated as stale
ownership. All GPU/TPU execution still needs hardware validation.

`applied.revert(optimizer)` splits current packed values and gradients into the
retained canonical modules, converts named optimizer slots, and redirects the
supplied optimizer to canonical parameters. Current mode, unchanged aliases,
moment dtypes, and optimizer-group hyperparameters are retained. All inverse
checks complete before mutation; failed conversion/commit restores the live
packed graph and optimizer. If an optimizer already exists, pass it to undo or
rebuild it afterwards; external optimizer references cannot be discovered.

## Optimizer state

`named_optimizer_state()` keys slots by canonical owner names, never positional
parameter IDs. `convert_optimizer_state()` packs/splits standard AdamW moments,
AMSGrad maxima, and SGD momentum. First- and second-moment precision are preserved
independently of parameter dtype. Packed sources must belong to one complete
optimizer group and have matching slot sets, initialized-state coverage, scalar
step counts, and per-slot dtypes/devices. Incompatible states are rejected rather
than reset or cast. Unknown slots, fused/capturable/differentiable optimizer
state, and parameter membership changes require a separate audited implementation.

## State I/O and plain HF export

The application handle exposes:

- `canonical_state_dict()` / `load_canonical_state_dict()` for canonical model
  weights without changing the active transformed graph;
- `training_state_dict(optimizer, scheduler=None)` and
  `load_training_state_dict(payload, optimizer, scheduler=None)` for model,
  named optimizer, optional scheduler, nonpersistent runtime buffers, and Torch
  CPU/already-initialized CUDA RNG state;
- `canonical_model()` for a separate canonical module graph with current values;
- `export_huggingface(path)` for that graph's standard `save_pretrained()` export
  using safetensors, without editing the live graph or leaking packed names.

```python
payload = applied.training_state_dict(optimizer, scheduler)
torch.save(payload, checkpoint_path)
# Construct the same canonical model, apply the same plan/mappings, and create
# its optimizer/scheduler before restoring into that prepared transformed model.
resumed.load_training_state_dict(
    torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    resumed_optimizer, resumed_scheduler,
)
applied.export_huggingface(export_path)
```

Training-state restore checks schema, plan/layout identity, optimizer/scheduler
classes, compact moment policy, parameter-group membership, tensor metadata,
aliases, and slot compatibility before copying data. Late failures restore model
tensors, optimizer references, scheduler, and RNG. Call at optimizer boundaries;
gradients are not checkpointed. The enclosing engine still owns trainer progress,
runtime/distributed state, data cursors, and other RNG sources. This local codec
is not the M7 distributed `ResumeManifest` or a new public trainer checkpoint
format. [M8-F6 public integration](PUBLIC.md) now wires it into local facade and
per-rank worker checkpoints and exports. Canonical model state
can be imported independently, while exact optimizer continuation requires the
same audited architecture, plan, mapping, optimizer, and scheduler configuration.

CPU fixtures exercise real QKV weight/bias packing, unequal splits, aliases,
standard/compact AdamW continuation with dropout and runtime buffers, rollback,
trained undo with an existing optimizer, and plain Transformers model export/
reload after a mapped projection is trained. These checks are implementation
evidence, not GPU/TPU kernel certification.

The full CPU regression command
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q`
executed 432 tests: 426 passed and 6 failed. All 22 new layout tests passed.
The failures match onboarding: five Phi padding-configuration errors with
Transformers 5.0.0 and the existing TrainLM save/reload weight mismatch.
