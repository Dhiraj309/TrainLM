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
5. keeps full masked labels for compatible HF-native causal loss and shifted
   labels for TrainLM cross-entropy;
6. extracts scalar loss and logits from HF attribute, mapping, or tuple outputs;
7. uses compatible model loss or computes summed cross-entropy with the
   configured normalization;
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
    loss_implementation="auto",
)
```

`auto` uses model-provided loss only when `forward` explicitly accepts labels,
the output exposes a scalar loss, `ignore_index` is `-100`, normalization is by
supervised tokens, and z-loss is disabled. Otherwise it uses portable PyTorch
cross-entropy with FP32 logits. `model` makes native loss mandatory;
`causal_lm` always selects the TrainLM correctness reference.

Every `TaskResult` records `loss_source` as `model`,
`trainlm_cross_entropy`, or `legacy_model`.

The portable cross-entropy is the correctness reference, not the final
throughput path.
Chunked logits, fused linear-cross-entropy, Pallas, and backend-native kernels
will implement the same task contract in later milestones.

## Legacy model-owned loss

The former callable `loss_fn(model, batch, runtime)` remains available through
`LossTaskAdapter`. Because arbitrary model-owned loss does not expose reliable
shift or mask semantics, the adapter marks token counts as inexact and reports
zeros. It must not be used for parity benchmarks or token-based stopping.

The default `loss.implementation: auto` belongs to `CausalLMTask`. The legacy
adapter remains only for applications still passing the former `loss_fn`.
