import json
from dataclasses import replace

import pytest

from trainlm.optimization import ComponentCapability, OptimizationExplanation

from .test_capabilities import capabilities
from .test_execution_plan import plan


def test_explanation_has_stable_human_and_json_views():
    report = OptimizationExplanation(
        backend="pytorch-xla",
        precision="bf16",
        selected_path="tpu_coordinator",
        certification="compatible",
        capabilities=capabilities(),
        execution_plan=plan(),
        graph_evidence=("hlo:artifact.txt",),
        limitations=("Target-hardware certification is pending.",),
    )

    payload = json.loads(report.to_json())
    assert payload["schema_version"] == 1
    assert payload["execution_plan"]["decisions"][0]["status"] == "fallback"
    assert payload["execution_plan"]["transformations"][0]["transform_id"] == "pack-qkv"
    assert "Backend: pytorch-xla (bf16)" in report.to_text()
    assert "selected=torch_sdpa" in report.to_text()


def test_strict_report_rejects_unknown_capabilities_before_launch():
    inspected = replace(
        capabilities(), residual=ComponentCapability.unknown("No graph evidence.")
    )
    report = OptimizationExplanation(
        backend="xla", precision="bf16", selected_path="tpu", capabilities=inspected
    )

    with pytest.raises(RuntimeError, match="Unproven model capabilities"):
        report.require_supported()


def test_invalid_output_metadata_is_rejected():
    with pytest.raises(ValueError, match="graph_evidence"):
        OptimizationExplanation(
            backend="cpu", precision="fp32", selected_path="local", graph_evidence=("",)
        )
