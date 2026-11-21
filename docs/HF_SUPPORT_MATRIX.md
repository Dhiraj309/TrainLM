# Hugging Face pretraining workflows

The [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) exposes the model, dataset, splits, initialization and input format in its training cell. Use a fresh output directory per run. All three recipes evaluate and save a Transformers model and tokenizer.

| Workflow | Evidence | Outstanding check |
| --- | --- | --- |
| Random weights, tokenizer trained on raw text | Owner ran an 8.3M GPT-2 on one Colab v5e device, fp32 eager, torch 2.14.1+cpu, torch-tpu 0.2.0, libtpu 0.0.50. A kernel restart with retained files resumed step 79 through step 95. | Replacement TPU VM |
| Random weights from HF config, existing HF tokenizer | CPU model and tokenizer loading, packing, checkpoint and export contracts | Owner run with chosen HF assets on v5e-1 |
| Random initialization or CPT from a GPT-2 source, streamed text or token rows | Four CPU runs with tiny local GPT-2 assets, fp32 eager. All four evaluated, exported and generated through Transformers | Live Hub data and native v5e-1 CPT |
| Prepared packed epochs and resume | CPU fresh and resumed runs reached six updates over two epochs. The v5e-1 notebook exercises combined and independent epoch schedules. | Each TPU backend |
| Copy a complete run directory and continue with a larger budget | CPU directory move, restore and continued training | Recovery on a replacement VM |
| Publish a reviewed export | Upload allowlist, visibility and commit handling covered with a mocked Hub | Authenticated upload and fresh Transformers reload at the returned Hub revision |

Keep the model revision, tokenizer mapping, dataset revision and report with each hardware result. Compile, sharding, distributed checkpoints and scaling require separate runs. See the [runtime evidence](TORCHTPU_REBASELINE.md), [CPU workflow tests](../tests/test_pretraining_run.py) and [packed epoch test](../tests/test_packed_scratch_training.py).
