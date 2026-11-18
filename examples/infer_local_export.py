"""Load a TrainLM export with ordinary Transformers in a new Python process."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "runs/tinystories-gpt2/baseline-export"
PROMPT = "Once upon a time"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=40, do_sample=False)

print(tokenizer.decode(output[0], skip_special_tokens=True))
