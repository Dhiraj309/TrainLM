TrainLM

TrainLM is a minimal, high-performance LLM pretraining framework built with JAX and Flax NNX.

The project focuses on training modern decoder-only language models efficiently on TPU v5e and other JAX-supported accelerators while keeping the codebase small, maintainable, and fully interoperable with the Hugging Face ecosystem.

TrainLM is intentionally opinionated.

Rather than supporting every transformer architecture and training strategy, it focuses on one optimized Llama-style architecture and one clean training stack.

---

Goals

- Train strong small-to-medium language models

- Achieve excellent TPU v5e throughput

- Keep the implementation minimal

- Use modern JAX and Flax NNX APIs

- Reuse existing ecosystem components whenever possible

- Export directly to Hugging Face format

- Maintain compatibility with:

  - Transformers
  - PEFT
  - TRL
  - QLoRA
  - Unsloth
  - vLLM
  - TGI

---

Design Philosophy

TrainLM prioritizes:

1. Simplicity
2. Correctness
3. Throughput
4. Maintainability

over:

- Feature count
- Architectural flexibility
- Framework complexity
- Research abstractions

If a feature is not required to train a modern Llama-style language model, it is generally out of scope.

---

Supported Architecture

TrainLM supports exactly one model family:

- Decoder-only Transformer
- Pre-Norm
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU
- Causal Attention
- Optional tied embeddings

Configuration controls model variants.

Separate GPT, Gemma, Phi, Qwen, TinyLlama, or SmolLM implementations are not planned.

---

Technology Stack

Core libraries:

- JAX
- Flax NNX
- Optax
- Orbax Checkpointing
- Hugging Face Datasets
- Transformers
- Tokenizers

Current target versions:

- Python 3.12+
- JAX 0.7.x
- Flax 0.11.x
- Optax 0.2.x
- Orbax 0.12.x

---

Project Status

Current phase:

- Project initialization
- HF-compatible LlamaConfig
- Rotary Position Embeddings (RoPE)

In progress:

- Attention
- SwiGLU MLP
- Decoder blocks
- Llama model

Planned:

- Training loop
- Dataset pipeline
- Checkpointing
- Distributed training
- HF export
- Export parity validation

TrainLM is not yet ready for production training.

---

Roadmap

Phase 1 — Model Foundation

- HF-compatible configuration
- RoPE
- GQA attention
- SwiGLU MLP
- Decoder block
- LlamaForCausalLM

Phase 2 — Training

- AdamW
- Warmup + cosine schedule
- Train state
- Train step
- Tiny overfit validation

Phase 3 — Data

- Hugging Face Datasets
- Streaming support
- Continuous token packing

Phase 4 — Checkpointing

- Orbax checkpointing
- Save/restore:
  - model
  - optimizer
  - step
  - RNG

Phase 5 — Distributed Training

- Replicated training
- NamedSharding
- GSPMD
- Parameter sharding

Phase 6 — Hugging Face Export

- config.json
- generation_config.json
- model weights
- tokenizer files

Phase 7 — Validation

- HF export parity
- Distributed parity
- Throughput benchmarking

---

Planned Project Structure

src/trainlm/
├── model/
│   ├── config.py
│   ├── rope.py
│   ├── attention.py
│   ├── mlp.py
│   ├── decoder.py
│   ├── model.py
│   └── loss.py
│
├── training/
│   ├── state.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── train_step.py
│   └── trainer.py
│
├── data/
│   ├── tokenizer.py
│   ├── dataset.py
│   └── packing.py
│
├── checkpointing/
│   └── manager.py
│
├── export/
│   ├── hf_config.py
│   ├── hf_weights.py
│   └── export_hf.py
│
└── utils/
    ├── dtype.py
    └── mesh.py

---

Configuration Philosophy

TrainLM follows Hugging Face naming wherever possible.

Examples:

- vocab_size
- hidden_size
- intermediate_size
- num_hidden_layers
- num_attention_heads
- num_key_value_heads
- max_position_embeddings
- rope_theta
- rms_norm_eps
- tie_word_embeddings

This simplifies export and interoperability.

---

Distributed Training Philosophy

Development proceeds in stages:

1. Single-device correctness
2. Replicated multi-device training
3. Parameter sharding

Correctness always comes before sharding complexity.

TrainLM does not use pmap for new implementations.

Distributed training is built on:

- Mesh
- NamedSharding
- PartitionSpec
- GSPMD

---

Hugging Face Compatibility

A primary goal of TrainLM is seamless interoperability.

Exports should load directly into:

from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(...)

Export is considered complete only after validating logits parity between:

- TrainLM
- Hugging Face Transformers

on identical inputs.

---

Testing Philosophy

Every major feature should include tests.

Required categories:

- Configuration
- Model initialization
- Forward pass
- Attention correctness
- Loss correctness
- Optimizer updates
- Checkpoint save/restore
- Distributed parity
- HF export parity

Small deterministic tests are preferred.

---

Non-Goals

TrainLM is not intended to be:

- A research framework
- A model zoo
- A post-training framework
- An RLHF framework
- A serving framework

The project focuses exclusively on efficient language model pretraining.

---

License

Apache 2.0
