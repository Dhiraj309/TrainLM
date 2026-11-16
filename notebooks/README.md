# TrainLM notebooks

## User training quickstarts

- [Pretrain from scratch](TrainLM_Native_TorchTPU_From_Scratch.ipynb) ([open in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) · [open in Kaggle](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Dhiraj309/TrainLM/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)) uses
  Hugging Face GPT-2 with randomly initialized weights. The training cell sets
  `model_type="gpt2"`, dimensions, context length, and the experimental
  `native_tpu` backend. It does not load a pretrained model checkpoint.
- [Fine-tune text](TrainLM_Text_Fine_Tuning.ipynb) starts from a pretrained
  causal-LM checkpoint and targets CPU or CUDA.

## Runtime and TPU validation

- [v5e-1 single-device probe](TorchTPU_T1_v5e1_Probe.ipynb) checks the
  installed environment, native device discovery, eager training, and optional
  compile/lifecycle behavior.
- [v5e-8 distributed probe](TorchTPU_T1_v5e8_Distributed_Probe.ipynb) checks
  single-host collective startup. It is intended for a suitable v5e-8 runtime.
- [Native single-device trainer smoke](TrainLM_Native_TorchTPU_Single_Device_Training.ipynb)
  checks TrainLM's experimental trainer using a pretrained DistilGPT-2
  checkpoint. It complements the standalone probe and the scratch workflow.

These notebooks collect evidence for their stated scope; CPU runs and
standalone probes do not certify end-to-end TPU training or performance.

## Historical notebook

[TrainLM TPU validation](TrainLM_TPU_Validation.ipynb) records an earlier
PyTorch/XLA workflow. It remains for historical reference and does not validate
the current native TorchTPU path.
