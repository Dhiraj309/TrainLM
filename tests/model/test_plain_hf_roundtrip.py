"""Plain Transformers round-trip after one TrainLM task update."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from trainlm.model import normalize_causal_lm_output
from trainlm.runtime import Runtime
from trainlm.tasks import CausalLMTask

from .dense_ar_fixtures import DENSE_AR_FIXTURES, DenseARFixture


ROUNDTRIP_CASES = tuple(
    pytest.param(fixture, tied, id=f"{fixture.name}-{'tied' if tied else 'untied'}")
    for fixture in DENSE_AR_FIXTURES
    for tied in (True, False)
)


@pytest.mark.parametrize(("fixture", "tied"), ROUNDTRIP_CASES)
def test_plain_hf_roundtrip_preserves_updated_state_outputs_and_ties(
    fixture: DenseARFixture,
    tied: bool,
    tmp_path,
):
    model = AutoModelForCausalLM.from_config(fixture.config_factory(tied))
    assert not type(model).__module__.startswith("trainlm.")

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = CausalLMTask(loss_implementation="causal_lm").training_step(
        model,
        {"input_ids": input_ids, "attention_mask": attention_mask},
        Runtime(),
    )
    result.loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        expected_logits = normalize_causal_lm_output(
            model(input_ids=input_ids, attention_mask=attention_mask)
        ).logits
    expected_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    export_dir = tmp_path / f"{fixture.name}-export"
    # Transformers v5 always writes model weights as safetensors.
    model.save_pretrained(export_dir)

    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.safetensors").is_file()
    assert not (export_dir / "trainlm_export.json").exists()

    reloaded = AutoModelForCausalLM.from_pretrained(
        export_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    assert not type(reloaded).__module__.startswith("trainlm.")
    reloaded.eval()

    reloaded_state = reloaded.state_dict()
    assert tuple(reloaded_state) == tuple(expected_state)
    for name, expected in expected_state.items():
        actual = reloaded_state[name]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)

    with torch.no_grad():
        actual_logits = normalize_causal_lm_output(
            reloaded(input_ids=input_ids, attention_mask=attention_mask)
        ).logits
    assert torch.allclose(actual_logits, expected_logits, rtol=0.0, atol=1e-6)

    input_weight = reloaded.get_input_embeddings().weight
    output_weight = reloaded.get_output_embeddings().weight
    assert (input_weight is output_weight) is tied
    assert reloaded.config.tie_word_embeddings is tied
