"""The trainer smoke launcher must remain usable before Torch is installed."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/torchtpu_native_trainer_smoke.py"
NOTEBOOK = ROOT / "notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb"


class NativeTrainerSmokeProcessTests(unittest.TestCase):
    def test_help_does_not_initialize_torch(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--resume", result.stdout)
        self.assertIn("--compile", result.stdout)
        self.assertIn("--from-scratch", result.stdout)
        self.assertIn("--train-examples", result.stdout)

    def test_incompatible_modes_fail_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
             "--compile", "--resume"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Run checkpoint/resume after the eager baseline", result.stderr)
        self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_invalid_scratch_step_count_fails_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
             "--from-scratch", "--max-steps", "0"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be positive", result.stderr)
        self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_notebook_keeps_device_import_inside_bounded_child(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = ["".join(cell["source"]) for cell in notebook["cells"]
                 if cell["cell_type"] == "code"]
        source = "\n".join(cells)
        self.assertNotIn("import torch_tpu", source)
        self.assertNotIn("TrainLMTrainer(", source)
        self.assertIn("run_probe_process", source)
        self.assertIn("torchtpu_native_trainer_smoke.py", source)


if __name__ == "__main__":
    unittest.main()
