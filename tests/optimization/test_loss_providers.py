from trainlm.optimization import (
    OptimizationPlanner,
    causal_loss_provider_specs,
    causal_loss_request,
)

from .test_capabilities import capabilities


def test_loss_catalog_uses_portable_fallback_without_runtime_evidence():
    plan = OptimizationPlanner(causal_loss_provider_specs()).plan(
        capabilities(),
        backend="xla",
        precision="bf16",
        policy="auto",
        requests=(causal_loss_request(),),
    )

    decision = plan.decisions[0]
    assert decision.status == "fallback"
    assert decision.selected_provider == "trainlm.chunked_linear_ce"
    assert any("pallas_linear_ce_backward" in item for item in decision.evidence)


def test_loss_catalog_selects_pallas_only_with_forward_and_backward_evidence():
    plan = OptimizationPlanner(causal_loss_provider_specs()).plan(
        capabilities(),
        backend="xla",
        precision="bf16",
        policy="auto",
        requests=(causal_loss_request(),),
        runtime_features=(
            "torch_xla",
            "pallas_linear_ce",
            "pallas_linear_ce_backward",
        ),
    )

    assert plan.decisions[0].status == "selected"
    assert plan.decisions[0].selected_provider == "trainlm.pallas_linear_ce"


def test_loss_catalog_is_declarative_and_does_not_import_optional_providers():
    specs = causal_loss_provider_specs()

    assert [item.provider_id for item in specs] == [
        "trainlm.pallas_linear_ce",
        "trainlm.tokamax_linear_ce",
        "trainlm.chunked_linear_ce",
    ]
