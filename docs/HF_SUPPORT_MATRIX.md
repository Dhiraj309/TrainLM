# Hugging Face pretraining workflows

The [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) exposes the model, dataset, splits, initialization and input format in its training cell. Use a fresh output directory per run. All three recipes evaluate and save a Transformers model and tokenizer.

| Workflow | Evidence | Outstanding check |
| --- | --- | --- |
| Random weights, tokenizer trained on raw text | Owner ran an 8.3M GPT-2 on one Colab v5e device, fp32 eager, torch 2.14.1+cpu, torch-tpu 0.2.0, libtpu 0.0.50. A kernel restart with retained files resumed step 79 through step 95. | Replacement TPU VM |
| Random weights from HF config, existing HF tokenizer | CPU model and tokenizer loading, packing, checkpoint and export contracts | Owner run with chosen HF assets on v5e-1 |
| Random initialization or CPT from a GPT-2 source, streamed text or token rows | Four fp32 eager CPU runs on public Hub assets. All four evaluated, checkpointed, exported and generated in fresh Transformers processes | Repeat the four cases on native v5e-1 |
| Prepared packed epochs and resume | CPU fresh and resumed runs reached six updates over two epochs. The v5e-1 notebook exercises combined and independent epoch schedules. | Each TPU backend |
| Copy a complete run directory and continue with a larger budget | CPU directory move, restore and continued training | Recovery on a replacement VM |
| Publish a reviewed export | Upload allowlist, visibility and commit handling covered with a mocked Hub | Authenticated upload and fresh Transformers reload at the returned Hub revision |

Keep the model revision, tokenizer mapping, dataset revision and report with each hardware result. Compile, sharding, distributed checkpoints and scaling require separate runs. See the [runtime evidence](TORCHTPU_REBASELINE.md), [CPU workflow tests](../tests/test_pretraining_run.py) and [packed epoch test](../tests/test_packed_scratch_training.py).


### Reproduce the four HF workflows

Run the [validation notebook on Colab v5e-1](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_TPU_Epoch_Eval_Save_V5e1.ipynb) or use the [same runner](../examples/validate_hf_pretraining.py) on CPU:

```sh
python examples/validate_hf_pretraining.py --accelerator cpu --output-dir hf-checks
```

Each 30-token run uses one review from the training split and a different review from the test split. Raw runs stream text from the Hub. Token runs read JSONL encoded with the same model tokenizer. The public CPU run used `hf-internal-testing/tiny-random-gpt2@71034c5` and `lhoestq/demo1@87ecf16`, PyTorch 2.9.0+cpu and Transformers 5.15.0. Evaluation loss was 6.9281 for both random-weight runs and 6.9024 for both pretrained runs. Each export generated one new token in a separate CPU process. The output directory includes individual run reports and a combined `matrix.json`.


The same CPU matrix also passed with `--model hf-internal-testing/tiny-random-LlamaForCausalLM` at revision `9fb1912`. Both raw and pretokenized inputs yielded eval loss 10.4047 from random weights and 10.3720 from pretrained weights. Transformers warned that this test checkpoint declares `pad_token_id=-1`. All four exports generated in separate CPU processes.

A separate CPU process reopened a copy of the complete GPT-2 CPT run under a new path and continued from the first update. At step two, all 65 exported state tensors matched the uninterrupted reference exactly. This covers a copied filesystem on CPU. Physical TPU VM replacement remains outstanding.
