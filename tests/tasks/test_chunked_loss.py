import pytest
import torch
import torch.nn.functional as F

from trainlm.tasks import chunked_linear_causal_cross_entropy


def _reference(hidden, weight, labels, bias, mask, z_loss):
    logits = F.linear(hidden[..., :-1, :].float(), weight.float(), bias.float())
    targets = torch.where(mask[..., 1:], labels[..., 1:], -100)
    denominator = targets.ne(-100).sum()
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=-100, reduction="sum",
    ) / denominator
    log_z = torch.logsumexp(logits, dim=-1)
    z_value = torch.where(targets.ne(-100), log_z.square(), 0.0).sum() / denominator
    return ce + z_loss * z_value, z_value


def test_chunked_loss_and_gradients_match_full_logits_reference():
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    weight = torch.randn(9, 4, requires_grad=True)
    bias = torch.randn(9, requires_grad=True)
    labels = torch.randint(0, 9, (2, 5))
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.bool)
    expected, expected_z = _reference(hidden, weight, labels, bias, mask, 0.05)
    expected.backward()
    expected_gradients = tuple(value.grad.clone() for value in (hidden, weight, bias))

    for value in (hidden, weight, bias):
        value.grad = None
    actual, actual_z = chunked_linear_causal_cross_entropy(
        hidden, weight, labels, bias=bias, loss_mask=mask, chunk_size=3, z_loss=0.05
    )
    actual.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_z, expected_z)
    for value, expected_gradient in zip((hidden, weight, bias), expected_gradients):
        torch.testing.assert_close(value.grad, expected_gradient)


def test_chunking_never_projects_more_than_requested(monkeypatch):
    sizes = []
    original = F.linear

    def recording_linear(input, weight, bias=None):
        sizes.append(input.shape[0])
        return original(input, weight, bias)

    monkeypatch.setattr(F, "linear", recording_linear)
    chunked_linear_causal_cross_entropy(
        torch.randn(2, 6, 4), torch.randn(7, 4), torch.randint(0, 7, (2, 6)),
        chunk_size=3,
    )
    assert sizes == [3, 3, 3, 1]


def test_per_chunk_rematerialization_matches_loss_and_gradients():
    torch.manual_seed(11)
    source = (
        torch.randn(2, 5, 4),
        torch.randn(7, 4),
        torch.randn(7),
    )
    labels = torch.randint(0, 7, (2, 5))

    def run(policy):
        hidden, weight, bias = (
            value.detach().clone().requires_grad_(True) for value in source
        )
        loss, z_value = chunked_linear_causal_cross_entropy(
            hidden,
            weight,
            labels,
            bias=bias,
            chunk_size=3,
            z_loss=0.02,
            rematerialization=policy,
        )
        loss.backward()
        return loss.detach(), z_value.detach(), tuple(
            value.grad for value in (hidden, weight, bias)
        )

    expected = run("disabled")
    actual = run("per_chunk")
    for actual_value, expected_value in zip(actual, expected):
        if isinstance(actual_value, tuple):
            for actual_gradient, expected_gradient in zip(actual_value, expected_value):
                torch.testing.assert_close(actual_gradient, expected_gradient)
        else:
            torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"z_loss": -1.0}, "z_loss"),
        ({"rematerialization": "layer"}, "rematerialization"),
        ({"loss_mask": torch.zeros(2, 5, dtype=torch.bool)}, "no supervised"),
    ],
)
def test_chunked_loss_rejects_invalid_or_empty_requests(kwargs, message):
    with pytest.raises(ValueError, match=message):
        chunked_linear_causal_cross_entropy(
            torch.randn(2, 5, 4),
            torch.randn(7, 4),
            torch.randint(0, 7, (2, 5)),
            **kwargs,
        )
