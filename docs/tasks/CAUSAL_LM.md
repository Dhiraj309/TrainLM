# Causal language-model task contract

`CausalLMTask` owns dense autoregressive training semantics independently of
the selected Hugging Face model family and execution backend. The trainer sees
only a `LanguageModelTask` and a `TaskResult`.

## Canonical semantics

For an input sequence `[t0, t1, ..., tn]`, logits at positions
`[0, ..., n-1]` predict labels `[t1, ..., tn]`. The task:

1. accepts `input_ids` and optional same-shaped `labels`;
2. uses `input_ids` as labels when labels are absent;
3. shifts targets exactly once;
4. combines `attention_mask`, optional `loss_mask`, and `ignore_index`;
5. removes task-only fields before model dispatch;
6. extracts logits from HF attribute, mapping, or tuple outputs;
7. computes summed cross-entropy and applies the configured normalization;
8. optionally adds z-loss; and
9. returns exact input, target, supervised, ignored, and sequence counts.

The default normalization is `supervised_tokens`. This makes loss scale stable
when padding or masked targets vary between batches. `batch` normalization is
available only when explicitly requested.

Token accounting happens before `ExecutionBackend.prepare_batch`, keeping the
normal host-input path free of accelerator scalar reads. Backends or input
pipelines that yield device-resident batches must eventually provide equivalent
host metadata to avoid a synchronization; that integration belongs to the data
pipeline milestones.

## Trainer boundary

The trainer dispatches `training_step` or `evaluation_step`, backpropagates the
returned scalar loss, and updates progress from returned token counts. It does
not inspect `input_ids`, labels, masks, logits, or model-output classes.
Evaluation aggregation is also task-owned, so variable-length or differently
masked batches are weighted by their actual normalization units rather than
being averaged per batch.

The trainer never constructs a concrete task. The application/assembly layer
must pass one explicitly, keeping future task types independent of trainer
control flow.

```python
from trainlm.tasks import CausalLMTask

task = CausalLMTask(
    ignore_index=-100,
    normalization="supervised_tokens",
    z_loss=1e-4,
)
```

The initial implementation uses portable PyTorch cross-entropy with FP32
logits. It is the correctness reference, not the final throughput path.
Chunked logits, fused linear-cross-entropy, Pallas, and backend-native kernels
will implement the same task contract in later milestones.

## Legacy model-owned loss

The former callable `loss_fn(model, batch, runtime)` remains available through
`LossTaskAdapter`. Because arbitrary model-owned loss does not expose reliable
shift or mask semantics, the adapter marks token counts as inexact and reports
zeros. It must not be used for parity benchmarks or token-based stopping.

Selecting `loss.implementation: model` therefore requires an explicit legacy
`loss_fn`. The default `loss.implementation: causal_lm` uses canonical,
task-owned semantics.
