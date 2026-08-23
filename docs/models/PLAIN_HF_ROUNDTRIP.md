# Plain Hugging Face round-trip certification

TrainLM's compatibility promise requires a trained dense causal model to leave
the framework through the ordinary Transformers contract. A successful export
must reload with `AutoModelForCausalLM.from_pretrained` without importing a
TrainLM architecture or requiring a TrainLM checkpoint reader.

M2-F4 certifies the following flow on CPU for every V1 representative family,
with both tied and untied language-model heads:

1. create a tiny official HF causal model from its native configuration;
2. run one optimizer update through `CausalLMTask`;
3. save `config.json` and the mandatory Transformers v5 safetensors using
   `save_pretrained`;
4. reload locally using only `AutoModelForCausalLM`;
5. compare every state-dict key, shape, dtype, and tensor value;
6. compare deterministic evaluation logits; and
7. verify tied parameters remain aliases and untied parameters remain distinct.

The matrix covers GPT-2, OPT, GPT-NeoX, BLOOM, Falcon, Phi, Llama, Mistral,
Qwen2, and Gemma. It deliberately uses no TrainLM model class. The test export
is a plain HF directory rather than the internal exact-resume format defined in
the checkpoint contract.

This story proves serialization compatibility after an update. M2-F5 extends
the shared tiny-model fixtures into the full construct, forward, loss,
backward, update, export, and overfit conformance matrix.
