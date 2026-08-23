"""Generic CPU conformance matrix for representative dense causal LMs."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from trainlm.model import (
    ForwardBatchDispatcher,
    load_huggingface_causal_lm,
    normalize_causal_lm_output,
)
from trainlm.runtime import Runtime
from trainlm.tasks import CausalLMTask

from .dense_ar_fixtures import DENSE_AR_FIXTURES, DenseARFixture


@pytest.mark.parametrize("fixture", DENSE_AR_FIXTURES, ids=lambda item: item.name)
def test_dense_ar_family_conforms_to_generic_training_path(
    fixture: DenseARFixture,
    tmp_path,
):
    torch.manual_seed(0)
    loaded = load_huggingface_causal_lm(fixture.source(tied=True))
    model = loaded.model

    assert loaded.metadata.model_type == fixture.model_type
    assert loaded.config.model_type == fixture.model_type
    assert isinstance(loaded.config, fixture.config_class)
    assert type(model).__module__.startswith("transformers.")
    assert all(
        not type(module).__module__.startswith("trainlm.")
        for module in model.modules()
    )

    input_ids = torch.tensor(
        [[1, 2, 3, 4], [4, 3, 2, 1]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "document_ids": torch.tensor([10, 11]),
        "shard_offsets": torch.tensor([100, 200]),
    }

    dispatch = ForwardBatchDispatcher.from_model(model).dispatch(batch)
    assert dispatch.forwarded_fields == ("input_ids", "attention_mask")
    assert dispatch.dropped_fields == ("document_ids", "shard_offsets")
    with torch.no_grad():
        logits = normalize_causal_lm_output(model(**dispatch.inputs)).logits
    assert logits.shape == (2, 4, 32)

    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)
    task = CausalLMTask(loss_implementation="auto")
    losses = []

    model.train()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        result = task.training_step(model, batch, Runtime())
        assert torch.isfinite(result.loss)
        result.loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()
        losses.append(result.loss.detach().item())

    assert any(
        not torch.equal(parameter.detach(), initial_parameters[name])
        for name, parameter in model.named_parameters()
    )
    assert min(losses[1:]) < losses[0] * 0.8

    model.eval()
    with torch.no_grad():
        expected_logits = normalize_causal_lm_output(
            model(input_ids=input_ids, attention_mask=attention_mask)
        ).logits

    export_dir = tmp_path / fixture.name
    model.save_pretrained(export_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(
        export_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    assert type(reloaded).__module__.startswith("transformers.")
    reloaded.eval()
    with torch.no_grad():
        actual_logits = normalize_causal_lm_output(
            reloaded(input_ids=input_ids, attention_mask=attention_mask)
        ).logits
    assert torch.allclose(actual_logits, expected_logits, rtol=0.0, atol=1e-6)
