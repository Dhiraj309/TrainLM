# Optimizer and state-dtype policy

`OptimizerFactory` is the single construction boundary for TrainLM optimizers.
It currently creates decoupled AdamW through a backend-neutral subclass that
keeps first (`mu`) and second (`nu`) moments in independently selected dtypes.

```python
from trainlm.config import OptimizerConfig
from trainlm.optimization import OptimizerFactory

optimizer = OptimizerFactory.create(
    model.parameters(),
    OptimizerConfig(
        learning_rate=2e-4,
        weight_decay=0.1,
        parameter_dtype="float32",
        mu_dtype="bfloat16",
        nu_dtype="float32",
        fused=False,
    ),
)
```

The factory validates parameter dtype, learning-rate/beta/epsilon ranges,
non-negative decoupled weight decay, and supported moment dtypes before
construction. It does not silently cast model parameters; model and optimizer
assembly must agree on parameter dtype first.

Moment tensors are initialized and updated by PyTorch AdamW, then normalized to
the selected policy after each step. The policy is exposed on the optimizer so
checkpoint writers can record it. Backend preparation happens after factory
construction, preserving the existing `prepare_optimizer` ownership boundary.

`fused` remains an explicit provider choice. A backend may reject or replace a
fused implementation during `prepare_optimizer`; the generic factory does not
claim that a fused kernel exists on every device. Mixed moment dtypes should be
benchmarked and certified per backend before becoming a production default.

Legacy direct `torch.optim.AdamW` construction remains possible for callers,
but it does not provide TrainLM's explicit dtype-policy metadata and is not the
certified path.
