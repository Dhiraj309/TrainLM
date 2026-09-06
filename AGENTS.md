# TrainLM agent handoff

Before changing this repository, read [`docs/IMPLEMENTATION_CONTEXT.md`](docs/IMPLEMENTATION_CONTEXT.md)
and the relevant sections of [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Working rules

- TrainLM's product surface is HF-like: users provide model, datasets,
  tokenizer/processing class, and familiar training arguments. Worker launch,
  PJRT/rank setup, preflight, manifests, cache, and log parsing stay private.
- Preserve Hugging Face model implementations as the semantic source of truth.
  Add optimization through capability inspection, explicit adapters, and
  reversible plans; never rely on model-family name guesses.
- Current scope is dense decoder-only autoregressive models. MoE and DLLM are
  future extensions.
- Do not run TPU training locally. TPU validation belongs to the owner's
  Kaggle/Cloud TPU environment.
- Do not run git commands. The repository owner handles fetch, rebase, commit,
  push, and merge manually.
- Use one feature/story per commit. In every implementation turn, update both
  `docs/ROADMAP.md` and `docs/IMPLEMENTATION_CONTEXT.md`, and report the
  suggested commit message.

## Current continuation point

PR4 (`milestone/m8-m9-optimization-core`) is active. M8-F0's first public
facade slice exists in `src/trainlm/api.py` and delegates to the existing
backend-neutral engine on CPU/CUDA. The next story is to hide TPU coordinator
and worker orchestration behind `TrainLMTrainer.train()` while keeping the
same concise user code. Do not mark M8-F0 complete until that public path
reaches the TPU coordinator without subprocess code or stage-log parsing in
the user's workflow.

