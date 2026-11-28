---
license: apache-2.0
library_name: transformers
datasets:
  - Dhiraj45/laughlm-135m-shards-v2
tags:
  - tpu
  - torchtpu
  - pallas
  - looped-transformer
  - pretraining
  - trainlm
---

# `LaughLM-Loop-135M` (`131.07M`-Token Stage-1 Pilot on Cloud `TPU v5e-1`)

Early **Stage-1 pretraining pilot** (`131.07M` tokens = `0.37%` of the full 35B target) of **`LaughLM-Loop-135M`** trained end-to-end from scratch on a single **Google Cloud TPU `v5e-1`** (`16.0 GiB` HBM, `197 TFLOP/s` BF16 peak) using native [`TorchTPU`](https://github.com/Dhiraj309/TrainLM) (`torch-tpu==0.2.0`, zero `torch_xla`), custom VMEM Pallas kernels, and `adamw_bf16` + `chunked_linear` cross-entropy.

- **Codebase & Training Pipeline:** [`Dhiraj309/TrainLM`](https://github.com/Dhiraj309/TrainLM)
- **Pre-Tokenized Dataset:** [`Dhiraj45/laughlm-135m-shards-v2`](https://huggingface.co/datasets/Dhiraj45/laughlm-135m-shards-v2)

---

## Architecture Overview

`LaughLM-Loop-135M` replaces deep-narrow layers with a **Looped-Depth Transformer** (`Prelude + Shared Recurrent Core + Coda`) specifically aligned (`128`-multiple dimensions) to saturate Cloud TPU MXUs and hit unpadded `pallas_splash_attention`:

| Component | Configuration |
| :--- | :--- |
| **Structure** | `2` Prelude layers + `4` Shared Core layers (looped $R=4$ times) + `2` Coda layers |
| **Effective Active Depth** | **`20` layers** (`2 + 4 * 4 + 2`) at $R=4$ (`truncated_bptt_steps=4`) |
| **Hidden / FFN Geometry** | `hidden_size=896`, `intermediate_size=2432`, `vocab_size=49152` (tied embeddings) |
| **Attention** | `num_attention_heads=7`, `num_key_value_heads=1` (MQA), **`head_dim=128`** (unpadded Splash Attention) |
| **Stored Parameters** | `~111M` stored (`~135M`-class capacity, `~167M` non-embedding / `211M` total active params at $R=4$) |
| **Recurrent Boundary Kernel** | Custom VMEM-fused Pallas kernel `fused_recurrent_inject_rmsnorm`: $\text{RMSNorm}(h_t + \sigma(\text{gate}) \cdot h_0)$ |

---

## `TPU v5e-1` Micro-Benchmark MFU Leaderboard (`seq=2048`, `torch-tpu==0.2.0`)

| Model | Mode | Batch Size | Throughput | Non-Logit MFU | Total MFU | Host Wait |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `SmolLM2-135M` | Eager | 8 | `28,881.3 tok/s` | `15.57%` | `18.06%` | `0.0%` |
| `SmolLM2-135M` | Compiled | 8 | `54,645.4 tok/s` | `29.46%` | `34.17%` | `0.0%` |
| `SmolLM2-360M` | Compiled | 4 | `23,549.2 tok/s` | `31.59%` | `34.98%` | `0.0%` |
| `Qwen2.5-0.5B` | Compiled | 2 | `24,782.6 tok/s` | `33.66%` | `43.94%` | `0.0%` |
| `Qwen3-0.6B` | Compiled | 2 | `11,560.0 tok/s` | `19.64%` | `25.12%` | `0.0%` |
| `LaughLM-Loop-135M` | Compiled (`Pallas Off`, `eff_L=20`) | 4 | `62,873.3 tok/s` | `46.13%` | `54.56%` | `0.0%` |
| **`LaughLM-Loop-135M`** | **Compiled (`Pallas On`, `eff_L=20`)** | **4** | **`62,947.8 tok/s`** | **`46.18%`** | **`54.63%`** | **`0.0%`** |
| `LaughLM-Loop-135M` | Compiled (`Pallas On`, `eff_L=20`) | 8 | `61,004.9 tok/s` | `44.76%` | `52.94%` | `0.0%` |

---

## Stage-1 Pilot Run Summary (`131.07M` Tokens on `TPU v5e-1`)

- **Command / Config:** `stage1_base_30b.py --loop-preset pilot-1b --arch laughlm_loop` (`1,000` optimizer steps, `micro_batch_size=4`, `grad_accum=16`, `global_batch=64`, `seq_len=2048` = `131,072` tokens/step; `131,072,000` total tokens on `stream_stage1_w00_shard_00000.bin`).
- **Sustained Training Throughput:** **`73,240.1 tok/s`** | **`53.73%` non-logit MFU** (**`~63.6%` total MFU**) | **`0.0%` host wait**.
- **XLA Compilation Cache:** `99.9%` hit rate (`17,145 / 17,157` cache requests, `12` total compilations).
- **Convergence:** Cross-entropy loss dropped from `10.9748` (`58,384` PPL at step 5) to **`4.2332` (`68.93` PPL at step 1,000; lowest 5-step window `3.9422` / `51.53` PPL at step 885)**.

---

## Held-Out Test-Time Loop Scaling (`stream_stage1_w00_shard_00029.bin`, `131,072` Tokens)

Evaluated strictly on held-out shard `00029` (zero training overlap):

| Recurrent Loops ($R$) | Effective Layers | Held-Out Loss | Perplexity ($\text{PPL}$) | $\Delta$ PPL vs. $R=1$ | Eager Eval Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$R = 1$** *(warmup + eval)* | 8 | `4.8129` | `123.09` | Baseline | `2,414.0 tok/s` *(includes initial XLA trace)* |
| **$R = 2$** | 12 | `4.7035` | `110.33` | `-10.4%` | `44,089.2 tok/s` |
| **$R = 4$** *(trained depth)* | **20** | **`4.2198`** | **`68.02`** | **`-44.7%`** | **`30,042.5 tok/s`** |
| **$R = 6$** *(extrapolated)* | 28 | `4.2892` | `72.91` | `-40.8%` | `22,368.5 tok/s` |
| **$R = 8$** *(extrapolated)* | 36 | `4.2511` | `70.18` | `-43.0%` | `16,652.9 tok/s` |

---

## Qualitative Pilot Generations (`R=4, eff_L=20`, `0.37%` of 35B Target)

Representative greedy/top-$k$ completions from the `131.07M`-token checkpoint (`TTFT ~69–89 ms`, `8.0–9.9 tok/s` eager CPU decode):

- **Narrative (`Once upon a time, in a quiet village by the sea`):**
  > `Once upon a time, in a quiet village by the sea, I can see you as an ocean turtle. We visited a 31-mile ship that was built in the 1980s, and the 2024-mile ship that was built in the 1970s. We were also known to have a very large colony` (`TTFT=76.2 ms`, `8.85 tok/s`)
- **Dialogue / Explanatory (`The key difference between a star and a planet is`):**
  > `The key difference between a star and a planet is that the universe is not an absolute element of the mind. “We are a complex human and are at the center of the earth,” said Dr. Gaudimrabran, the world's first leader and co-author. “I am able to prove myself as true in this world.”` (`TTFT=88.8 ms`, `8.00 tok/s`)
- **Academic (`In 1969, engineers at NASA`):**
  > `In 1969, engineers at NASA in the National Academy of Engineering (2013). By the University of New York (2014), with its own technical language, it is possible to find out more about the science of the study. The researchers have demonstrated that researchers in the United States have been able to produce the data on the` (`TTFT=80.5 ms`, `9.40 tok/s`)

---

## Loading & Inference (`TrainLM`)

Checkpoints in this repository are organized by step directory (`checkpoint-250/`, ..., `final_model/`), so pass `subfolder="final_model"` when loading from the Hub:

```python
import torch
from transformers import AutoTokenizer
from trainlm.model import LaughLMLoopForCausalLM

repo_id = "Dhiraj45/laughlm-loop-135m-pilot"
tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="final_model")
model = LaughLMLoopForCausalLM.from_pretrained(
    repo_id,
    subfolder="final_model",
    torch_dtype=torch.bfloat16,
)
model.config.use_pallas_recurrent_norm = False  # set True on compiled TorchTPU
model.eval()

input_ids = tokenizer.encode("Once upon a time,", return_tensors="pt")
with torch.no_grad():
    logits = model(input_ids=input_ids, num_loops=4).logits
print(logits.shape)  # (1, seq_len, 49152)
```
