"""Family-neutral dense causal-LM structural inspection."""

from types import SimpleNamespace

import torch

from trainlm.optimization import inspect_dense_causal_lm


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)


class StructuralModel(torch.nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="fixture",
            architectures=["StructuralModel"],
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_theta=10000.0,
            hidden_act="silu",
            intermediate_size=16,
        )
        self.embed = torch.nn.Embedding(32, 8)
        self.attention = Attention()
        self.norm = torch.nn.LayerNorm(8)
        self.head = torch.nn.Linear(8, 32, bias=False)
        self.head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        return self.head(self.norm(self.embed(input_ids)))


def test_inspector_uses_structural_evidence_without_family_guesses():
    report = inspect_dense_causal_lm(StructuralModel())

    assert report.attention.kind == "gqa"
    assert report.position.kind == "rope"
    assert report.normalization.kind == "layer_norm"
    assert report.projections.kind == "separate_qkv"
    assert report.embedding.kind == "token_embedding"
    assert report.lm_head.kind == "linear"
    assert {fact.name: fact.value for fact in report.lm_head.facts}["tied"] is True
    assert report.residual.status == "unknown"


def test_inspector_reports_missing_semantics_as_unknown():
    report = inspect_dense_causal_lm(torch.nn.Linear(4, 4))

    assert report.attention.status == "unknown"
    assert report.position.status == "unknown"
    assert report.mlp.status == "unknown"
    assert report.projections.status == "unknown"


def test_inspector_requires_torch_module():
    try:
        inspect_dense_causal_lm(object())
    except TypeError as exc:
        assert "torch.nn.Module" in str(exc)
    else:
        raise AssertionError("Expected non-module inspection to fail.")
