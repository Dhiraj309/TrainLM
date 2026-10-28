from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from trainlm.optimization import inspect_dense_causal_lm
from trainlm.tasks import LinearCausalLMTrainingView, extract_hidden_state


class LayoutModel(torch.nn.Module):
    def __init__(self, *, tied: bool, bias: bool, output_style: str):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 6)
        self.body = torch.nn.Linear(6, 6)
        self.head = torch.nn.Linear(6, 13, bias=bias)
        if tied:
            self.head.weight = self.embedding.weight
        self.output_style = output_style

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def backbone(self, input_ids):
        hidden = self.body(self.embedding(input_ids))
        if self.output_style == "mapping":
            return {"last_hidden_state": hidden}
        if self.output_style == "tuple":
            return (hidden,)
        return SimpleNamespace(last_hidden_state=hidden)

    def forward(self, input_ids):
        return self.head(extract_hidden_state(self.backbone(input_ids)))


def _provider(model, inputs):
    return extract_hidden_state(model.backbone(inputs["input_ids"]))


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("output_style", ["mapping", "tuple", "object"])
def test_dense_head_layout_loss_gradient_alias_and_export_parity(
    tied, bias, output_style
):
    torch.manual_seed(23)
    model = LayoutModel(tied=tied, bias=bias, output_style=output_style)
    input_ids = torch.randint(0, 13, (2, 5))
    labels = input_ids.clone()
    labels[0, 3] = -100
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    alias_before = model.embedding.weight is model.head.weight

    logits = model(input_ids)
    reference = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, 13),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    reference.backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)

    view = LinearCausalLMTrainingView(
        model=model,
        capabilities=inspect_dense_causal_lm(model),
        hidden_state_provider=_provider,
        provider_id=f"fixture.{output_style}",
    )
    actual, _ = view.loss(
        {"input_ids": input_ids}, labels, chunk_size=3,
        rematerialization="per_chunk",
    )
    actual.backward()

    torch.testing.assert_close(actual, reference)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_gradients[name])
    assert (model.embedding.weight is model.head.weight) is alias_before
    assert tuple(model.state_dict()) == tuple(before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_hidden_output_adapter_rejects_ambiguous_or_missing_values():
    with pytest.raises(KeyError, match="does not contain"):
        extract_hidden_state({"hidden_states": torch.zeros(1, 2, 3)})
    with pytest.raises(TypeError, match="must be a tensor"):
        extract_hidden_state({"last_hidden_state": "not-a-tensor"})
    with pytest.raises(IndexError, match="out of range"):
        extract_hidden_state((), tuple_index=0)
