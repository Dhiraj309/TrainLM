from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class TransformersAdapter:
    """
    Hugging Face chat-based adapter using apply_chat_template.

    This adapter is intentionally thin:
    - Accepts structured messages
    - Returns raw model output
    - Does NOT implement tool calling, parsing, or loops
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        device: str = "cpu",  # change to "cuda" if GPU available
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()  # IMPORTANT: disable dropout for stable inference

        # Guard: ensure model supports chat template
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError(
                f"Model '{model_name}' does not support chat templates"
            )

    def generate(self, messages: list[dict]) -> str:
        """
        Generate response from structured chat messages.

        Expected message format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
        """

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only newly generated tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()
