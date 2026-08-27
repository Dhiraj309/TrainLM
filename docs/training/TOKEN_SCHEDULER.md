# Token-based WSD scheduler

`TokenWSD` is TrainLM's warmup-stable-decay learning-rate schedule. It is
indexed by cumulative supervised tokens consumed by the trainer, not by host
loop iterations. This keeps the schedule stable when gradient accumulation,
masking, world size, or batch geometry changes.

```python
from trainlm.config import SchedulerConfig
from trainlm.training import create_scheduler

scheduler = create_scheduler(
    optimizer,
    SchedulerConfig(
        name="wsd",
        horizon_tokens=20_000_000_000,
        warmup_fraction=0.01,
        stable_fraction=0.95,
        min_lr_ratio=0.05,
    ),
)
```

The fractions cover the horizon: warmup occupies the first fraction, stable
learning rate the next fraction, and the remaining interval linearly decays to
`min_lr_ratio`. Tokens at or beyond the horizon stay at the minimum ratio.
Zero-length warmup or decay regions are valid. Fraction boundaries are mapped
to integer token counts deterministically.

After each optimizer update, `Trainer` advances a WSD scheduler with the
state's cumulative token count. Standard PyTorch schedulers retain their
existing `step()` behavior. A WSD scheduler rejects decreasing token values,
and its state dict includes the token position for exact checkpoint resume.

The scheduler does not read model outputs or synchronize devices. The trainer
performs the one sparse scalar read needed for progress reporting after its
backend synchronization; future backends may replace that materialization
boundary while preserving the token-indexed contract.
