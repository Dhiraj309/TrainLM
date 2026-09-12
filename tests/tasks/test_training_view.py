from dataclasses import replace

import torch

from trainlm.optimization import ComponentCapability, inspect_dense_causal_lm
from trainlm.tasks import LinearCausalLMTrainingView


class FixtureCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)
        self.body = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 11)
        self.full_forward_calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids):
        self.full_forward_calls += 1
        return self.head(self.body(self.embedding(input_ids)))


def hidden_provider(model, inputs):
    return model.body(model.embedding(inputs["input_ids"]))


def test_training_view_bypasses_full_logits_and_preserves_model_state():
    model = FixtureCausalLM()
    before = tuple(model.state_dict())
    view = LinearCausalLMTrainingView(
        model=model,
        capabilities=inspect_dense_causal_lm(model),
        hidden_state_provider=hidden_provider,
        provider_id="fixture.body",
    )
    labels = torch.randint(0, 11, (2, 5))

    loss, _ = view.loss({"input_ids": labels}, labels, chunk_size=3)
    loss.backward()

    assert model.full_forward_calls == 0
    assert tuple(model.state_dict()) == before
    assert model.embedding.weight.grad is not None
    assert model.body.weight.grad is not None
    assert model.head.weight.grad is not None


def test_training_view_requires_inspected_linear_head():
    model = FixtureCausalLM()
    capabilities = replace(
        inspect_dense_causal_lm(model),
        lm_head=ComponentCapability.unknown("No output-head evidence."),
    )

    try:
        LinearCausalLMTrainingView(
            model=model,
            capabilities=capabilities,
            hidden_state_provider=hidden_provider,
            provider_id="fixture.body",
        )
    except ValueError as exc:
        assert "inspected linear LM head" in str(exc)
    else:
        raise AssertionError("Expected unknown LM head to block optimized view.")


def test_training_view_rejects_hidden_width_mismatch():
    model = FixtureCausalLM()
    view = LinearCausalLMTrainingView(
        model=model,
        capabilities=inspect_dense_causal_lm(model),
        hidden_state_provider=lambda model, inputs: torch.zeros(2, 5, 3),
        provider_id="fixture.invalid",
    )

    try:
        view.hidden_states({"input_ids": torch.ones(2, 5, dtype=torch.long)})
    except ValueError as exc:
        assert "projection width" in str(exc)
    else:
        raise AssertionError("Expected hidden width mismatch to fail.")
