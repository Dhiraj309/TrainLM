"""Generate text from a TrainLM export using ordinary Transformers on CPU."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", default="runs/tinystories-gpt2/baseline-export")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    if output.shape[-1] <= inputs.input_ids.shape[-1]:
        raise RuntimeError("Generation returned no new token.")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
