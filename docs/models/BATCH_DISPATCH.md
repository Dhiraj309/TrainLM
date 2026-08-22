# Forward-aware batch dispatch

TrainLM does not maintain a per-family list of legal model inputs. Instead,
`ForwardBatchDispatcher` inspects the loaded model's bound `forward` signature
once and the causal task reuses that immutable contract for every step.

The dispatcher:

- preserves declared inputs such as `attention_mask`, `position_ids`,
  `token_type_ids`, `cache_position`, `past_key_values`, and family extensions;
- does not treat `**kwargs` as permission to leak arbitrary dataset metadata;
- supports explicitly named passthrough fields only when `**kwargs` exists;
- removes dataset metadata and unsupported optional fields while reporting
  their names in `BatchDispatch.dropped_fields`;
- fails before model execution when a required keyword input is absent; and
- rejects required positional-only inputs because TrainLM batches are mappings.

```python
from trainlm.model import ForwardBatchDispatcher

dispatcher = ForwardBatchDispatcher.from_model(model)
dispatch = dispatcher.dispatch(batch)
outputs = model(**dispatch.inputs)
```

For a remote model that intentionally consumes a field only through
`**kwargs`, opt in by name:

```python
dispatcher = ForwardBatchDispatcher.from_model(
    model,
    passthrough_fields=("family_extension",),
)
```

`CausalLMTask` owns labels and loss masks, so those task-only values are removed
before model dispatch. The trainer itself never interprets model input names.
This story filters calls only; generic HF output and model-loss behavior are
defined by M2-F3.
