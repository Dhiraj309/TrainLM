# Token-normalized gradient accumulation

TrainLM treats `TrainerConfig.gradient_accumulation_steps` as the number of
microbatches in one optimizer update. For exact task results, each microbatch
contributes:

```text
loss_numerator += loss * supervised_tokens
gradient       += d(loss * supervised_tokens)
```

After the final microbatch, TrainLM scales the accumulated gradients by
`1 / total_supervised_tokens`, then clips, steps the optimizer, and advances
the scheduler. The reported update loss uses the same weighted numerator and
token denominator. This is equivalent to a single concatenated batch even
when masks or ignored-token counts differ between microbatches.

`TrainerState.micro_step` counts completed task microbatches, while `step`
counts optimizer updates. `tokens_seen` and `samples_seen` advance once per
completed optimizer update using the exact task counts.

Legacy model-owned losses report `exact=False` and no supervised-token count.
They remain compatible for one-microbatch updates, but accumulation greater
than one is rejected instead of silently averaging incompatible loss scales.

Gradient scaling is an `ExecutionBackend` operation so XLA and future TorchTPU
backends can keep the normalization in their compiled device graph.
