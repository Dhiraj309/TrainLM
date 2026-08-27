# Streaming evaluation

`TrainLMTrainer.evaluate()` consumes evaluation batches one at a time. Tasks
that implement `aggregate_evaluation_stream(results)` receive an iterator and
must reduce it to scalar metrics without retaining model outputs or task
results. Tasks that only implement the existing sequence-based
`aggregate_evaluation(results)` remain compatible through the fallback
compatibility path.

`CausalLMTask` uses the configured normalization unit (`supervised_tokens` by
default, or `sequences` for batch normalization). It accumulates weighted loss
in a detached scalar tensor and materializes it once at the end, then reports:

- `eval_loss`: token- or sequence-weighted mean loss;
- `eval_perplexity`: `exp(eval_loss)`, with overflow reported as `inf`.

The evaluator temporarily switches the model to evaluation mode, executes
under `torch.no_grad()`, and restores the prior training mode and trainer
phase. Evaluation does not mutate training counters, optimizer state,
scheduler state, or checkpoint state.

Custom tasks can opt in as follows:

```python
from collections.abc import Iterable

from trainlm.tasks import TaskResult


def aggregate_evaluation_stream(
    self,
    results: Iterable[TaskResult],
) -> dict[str, float]:
    total_loss = None
    total_tokens = 0
    for result in results:
        tokens = result.tokens.supervised_tokens
        contribution = result.loss.detach() * tokens
        total_loss = (
            contribution if total_loss is None else total_loss + contribution
        )
        total_tokens += tokens
    if total_tokens == 0 or total_loss is None:
        raise ValueError("Evaluation contains no supervised tokens.")
    return {"eval_loss": (total_loss / total_tokens).item()}
```

For TPU execution, keep the reduction boundary explicit and return only
materialized scalar metrics to callbacks; do not call `.item()` inside the
compiled model or training step.
