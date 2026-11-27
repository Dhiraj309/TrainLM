import ast
import json
import re
from pathlib import Path


NOTEBOOK = Path("notebooks/TrainLM_v5e1_MFU_Benchmark.ipynb")


def test_v5e1_mfu_benchmark_notebook_is_runnable_and_focused():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(
        re.fullmatch(r"[a-zA-Z0-9-_]+", cell.get("id", ""))
        for cell in notebook["cells"]
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 8
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")

    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for required in (
        "TrainLM_v5e1_MFU_Benchmark.ipynb",
        "--no-deps",
        "MODELS_TO_RUN",
        "HuggingFaceTB/SmolLM2-135M",
        "HuggingFaceTB/SmolLM2-360M",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen3-0.6B",
        "RUN_EAGER",
        "RUN_COMPILED",
        "RUN_PROFILER",
        "USE_SYNTHETIC_DATA",
        "chunked_linear",
        "torch.profiler.profile",
        "trainlm_v5e1_mfu_benchmark.json",
        "--force-reinstall",
        "feat/laughlm-loop-pallas",
        "remote_revision",
        "laughlm-loop-135m-pallas-on",
        "laughlm-loop-135m-pallas-on-bs8",
        "is_pallas_recurrent_norm_available()",
        "is_pallas_splash_available()",
    ):
        assert required in source, f"Missing {required!r} in {NOTEBOOK}"
    config_cell = next(c for c in notebook["cells"] if c.get("id") == "config")
    config_src = "".join(config_cell["source"])
    assert 'MODELS_TO_RUN = (\n    "laughlm-loop-135m-pallas-on",\n    "laughlm-loop-135m-pallas-on-bs8",\n)' in config_src
    assert "RUN_EAGER = False" in config_src
    assert "RUN_COMPILED = True" in config_src
    assert "RUN_PROFILER = False" in config_src
    assert "LinearCausalLMTrainingView" not in source
    assert 'accelerator="auto"' not in source
    assert "native_tpu" in source

