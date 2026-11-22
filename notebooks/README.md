# TrainLM notebooks

## User training quickstarts

- [Pretrain from scratch](TrainLM_Native_TorchTPU_From_Scratch.ipynb) ([open in Colab](https://colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb) · [open in Kaggle](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Dhiraj309/TrainLM/main/notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb)) uses
  an HF model config and tokenizer, with random weights, continued pretraining,
  raw Hub text, or tokenized HF Dataset splits selected in the public API cell.
  Token input is pinned to the resolved Dataset revision and can be checked
  against a tokenizer vocabulary digest. Resume reloads the same splits. Its
  optional recovery archive includes checkpoints and prepared data and may be
  large or private. The notebook targets experimental single-device native
  TorchTPU and keeps the default smoke budget small.
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
- [v5e-1 epoch evaluation and checkpoint validation](TrainLM_TPU_Epoch_Eval_Save_V5e1.ipynb)
  runs the focused CPU contracts and, on native TorchTPU, all four HF pretraining workflows with fresh-process Transformers reload. It then runs a tiny single-device
  epoch evaluation/checkpoint, and checkpoint-resume check. It prefers native
  TorchTPU when available and can select legacy PyTorch/XLA explicitly. The
  report labels which backend ran; a native run does not validate the XLA path.

These notebooks collect evidence for their stated scope; CPU runs and
standalone probes do not certify end-to-end TPU training or performance.

## Historical notebook

[TrainLM TPU validation](TrainLM_TPU_Validation.ipynb) records an earlier
PyTorch/XLA workflow. It remains for historical reference and does not validate
the current native TorchTPU path.
