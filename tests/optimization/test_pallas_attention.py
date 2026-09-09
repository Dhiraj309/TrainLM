import pytest

from trainlm.optimization import PallasAttentionRuntime, pallas_mha_provider


def runtime(kernel, **values):
    defaults = dict(
        torch_xla_version="2.8.0",
        tested_torch_xla_versions=("2.8.0",),
        kernel=kernel,
        backward_verified=True,
    )
    return PallasAttentionRuntime(**{**defaults, **values})


def test_guarded_mha_provider_calls_explicit_kernel_adapter():
    calls = []

    def kernel(query, key, value, *, causal, scale):
        calls.append((query, key, value, causal, scale))
        return "output"

    provider = pallas_mha_provider(runtime(kernel))
    output, weights = provider.attention_forward(
        object(), "q", "k", "v", scaling=0.125
    )

    assert (output, weights) == ("output", None)
    assert calls == [("q", "k", "v", True, 0.125)]
    assert provider.layouts == ("mha",)
    assert provider.mask_layouts == ("causal",)


def test_untested_xla_version_is_rejected_before_provider_creation():
    with pytest.raises(RuntimeError, match="untested torch_xla version"):
        pallas_mha_provider(
            runtime(lambda *args, **kwargs: None, torch_xla_version="2.9.0")
        )


def test_missing_backward_evidence_is_rejected():
    with pytest.raises(RuntimeError, match="backward correctness evidence"):
        pallas_mha_provider(
            runtime(lambda *args, **kwargs: None, backward_verified=False)
        )


def test_unsupported_mask_and_dropout_are_explicit_errors():
    provider = pallas_mha_provider(runtime(lambda *args, **kwargs: None))
    with pytest.raises(ValueError, match="registered causal mask"):
        provider.attention_forward(object(), "q", "k", "v", attention_mask="dense")
    with pytest.raises(ValueError, match="dropout"):
        provider.attention_forward(object(), "q", "k", "v", dropout=0.1)


def test_registered_mask_does_not_materialize_dense_tensor():
    provider = pallas_mha_provider(runtime(lambda *args, **kwargs: None))
    assert provider.mask_factory(batch_size=2, sequence_length=4096) is None
