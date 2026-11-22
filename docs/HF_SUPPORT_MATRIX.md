# Hugging Face pretraining workflows

The [Colab notebook](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) exposes the model, dataset, splits, initialization and input format in its training cell. Use a fresh output directory per run. All three recipes evaluate and save a Transformers model and tokenizer.

The entries below separate software/CPU evidence from owner-run hardware
evidence. “Implemented” means a code path exists; it does not imply every
model/data/runtime combination is supported.

| Journey | Model/init | Input and tokenizer | Device/runtime evidence | Status and next evidence |
| --- | --- | --- | --- | --- |
| Four-case pretraining matrix | `hf-internal-testing/tiny-random-gpt2`, random and pretrained | Hub raw review text and pretokenized JSONL; GPT-2 tokenizer; vocabulary digest checked for token rows | CPU, PyTorch 2.9.0+cpu, Transformers 5.15.0; fp32, eager, one process | **CPU passed:** finite evaluation, checkpoint, export and fresh-process Transformers generation in all four cases. Repeat on native v5e-1. |
| Llama architecture cross-check | `hf-internal-testing/tiny-random-LlamaForCausalLM`, random and pretrained | Same raw and tokenized review fixture; same tokenizer | CPU; fp32, eager, one process | **CPU passed:** all four cases evaluate, checkpoint, export and generate. The fixture reports `pad_token_id=-1`; broader tokenizer behavior is not established. |
| Owner scratch run | GPT-2 config, random initialization; 8.3M parameters | TinyStories raw text; HF GPT-2 tokenizer | Colab attached runtime confirmed by owner as v5e-1; PyTorch 2.14.1+cpu, TorchTPU 0.2.0, libtpu 0.0.50; fp32, eager, one visible TPU device | **Owner reported passed:** 79-step fresh run and same-device-session-restart resume through step 95. Replacement-VM recovery remains unverified. |
| HF config plus existing tokenizer | Small HF architecture and tokenizer | Hub text or supplied pretokenized Dataset | CPU contracts exist; TPU not yet run | Run the owner-selected assets on v5e-1 and record exact IDs/revisions, dtype, compile mode, package versions and device count. |
| Finite packed epochs and resume | Tiny local model fixture | Prepared packed train/evaluation shards | CPU fresh/resume exercised; v5e-1 notebook has the schedules but no recorded owner result | Run each enabled native TPU lifecycle case and record checkpoint recovery separately. |
| Portable recovery archive | Same run as its source report | Optional archive includes prepared shards, tokenizer, manifests and checkpoints | API path implemented; no recovery archive has been owner-tested after VM replacement | Test download, extraction, `restore_pretraining_report(..., restored_output_dir=...)`, and continuation on a replacement v5e-1 VM. The archive may contain private prepared corpus data. |
| Stateful iterable checkpoint/resume | User-supplied model | Pre-tokenized `torch.utils.data.IterableDataset`; zero workers; explicit identity, stable HF fingerprint, or a source-validating `validate_state_dict()` hook | Public trainer gate and identity comparison added; no CPU or accelerator run for this change | Test a pinned streaming HF Dataset through checkpoint/resume and require exact next-batch/model parity before claiming support. |
| High-level streamed fine-tuning | Pretrained causal LM and its HF tokenizer | `TrainLMFineTuningConfig(streaming=True)` loads pinned Hub raw-text splits; finite `max_steps`; bounded evaluation; zero workers | CPU passed: mocked Hub loader/pin/restore contract and exact stateful iterable resume parity; no live Hub download or TPU run | Owner-run v5e-1 test remains. Tokenized streams and epoch schedules remain unsupported. |
| Publish a reviewed export | Export from any completed supported run | Model/tokenizer export only; private by default | Upload allowlist, visibility and receipt covered by mocked CPU tests | Authenticate, upload to a private test repository, reload the returned immutable revision in a fresh Transformers process, then remove the test repo manually if desired. |

Keep the model revision, tokenizer mapping, dataset revision and report with each hardware result. Compile, sharding, distributed checkpoints and scaling require separate runs. See the [runtime evidence](TORCHTPU_REBASELINE.md), [CPU workflow tests](../tests/test_pretraining_run.py) and [packed epoch test](../tests/test_packed_scratch_training.py).


### Reproduce the four HF workflows

Run the [validation notebook on Colab v5e-1](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_TPU_Epoch_Eval_Save_V5e1.ipynb) or use the [same runner](../examples/validate_hf_pretraining.py) on CPU:

```sh
python examples/validate_hf_pretraining.py --accelerator cpu --output-dir hf-checks
```

Each 30-token run uses one review from the training split and a different review from the test split. Raw runs stream text from the Hub. Token runs read JSONL encoded with the same model tokenizer. The public CPU run used `hf-internal-testing/tiny-random-gpt2@71034c5` and `lhoestq/demo1@87ecf16`, PyTorch 2.9.0+cpu and Transformers 5.15.0. Evaluation loss was 6.9281 for both random-weight runs and 6.9024 for both pretrained runs. Each export generated one new token in a separate CPU process. The output directory includes individual run reports and a combined `matrix.json`.


The same CPU matrix also passed with `--model hf-internal-testing/tiny-random-LlamaForCausalLM` at revision `9fb1912`. Both raw and pretokenized inputs yielded eval loss 10.4047 from random weights and 10.3720 from pretrained weights. Transformers warned that this test checkpoint declares `pad_token_id=-1`. All four exports generated in separate CPU processes.

A separate CPU process reopened a copy of the complete GPT-2 CPT run under a new path and continued from the first update. At step two, all 65 exported state tensors matched the uninterrupted reference exactly. This covers a copied filesystem on CPU. Physical TPU VM replacement remains outstanding.
