# Multi-family overfit validation

The dense-AR training contract is validated with the same generic path for
each representative Hugging Face family in
`tests/model/dense_ar_fixtures.py`. The matrix loads each model through
`load_huggingface_causal_lm`, constructs the ordinary `CausalLMTask`, and
trains it with `Trainer`; no model-type conditionals or family adapters are
used by the test.

Every case uses a deterministic repeated-token batch and checks:

- all recorded losses are finite;
- every completed update has finite gradients;
- the minimum post-warm-up loss falls substantially below the initial loss;
- the updated model saves and reloads through plain Transformers
  `save_pretrained`/`from_pretrained`.

Resume behavior is owned by the lifecycle/checkpoint-hook contract and is
covered independently in `tests/training/test_lifecycle.py`. TPU execution,
throughput, and memory certification remain M5+ target-hardware gates; this
matrix only proves backend-neutral semantic trainability.
