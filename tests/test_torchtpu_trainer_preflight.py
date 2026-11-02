"""Notebook training preflight classifies metadata before any Torch import."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_torchtpu_install_preflight import FakeMetadata, INSPECT, ROOT, pip_ok
from tests.test_torchtpu_trainer_evidence import SHA, successful_process


ASSESS = runpy.run_path(str(
    ROOT / "src/trainlm/runtime/torchtpu_trainer_preflight.py"
))["assess_trainer_profile"]


class TrainerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.versions = {
            "torch": "2.13.0", "torch_tpu": "0.2.0", "libtpu": "0.0.50",
            "transformers": "5.15.0", "huggingface-hub": "1.16.4",
            "PyYAML": "6.0.3",
        }

    def profile(self, versions=None, *, upstream=None, pip_runner=pip_ok):
        return INSPECT(
            ROOT, provider=FakeMetadata(versions or self.versions,
                                        upstream or ["torch>=2.11,<=2.15", "libtpu>=0.0.50"]),
            pip_runner=pip_runner,
        )

    def test_compatible_profile_proceeds_without_certification(self):
        result = ASSESS(self.profile())
        self.assertEqual(result["status"], "metadata_compatible")
        self.assertTrue(result["proceed"])
        self.assertIn("metadata_only", result["limitations"])

    def test_owner_observed_colab_mismatch_is_explicit_experiment(self):
        def ipython_conflict(*args, **kwargs):
            return SimpleNamespace(
                returncode=1,
                stdout="ipython 7.34.0 requires jedi, which is not installed.\n",
                stderr="",
            )

        result = ASSESS(self.profile(
            {**self.versions, "torch": "2.14.1+cpu"},
            pip_runner=ipython_conflict,
        ))
        self.assertEqual(result["status"], "experimental_mismatch")
        self.assertTrue(result["proceed"])
        self.assertTrue(result["source_torch_mismatch"])
        self.assertEqual(result["unrelated_pip_conflict_count"], 1)

    def test_missing_or_upstream_conflicting_tpu_stack_blocks(self):
        missing = {key: value for key, value in self.versions.items()
                   if key != "torch_tpu"}
        self.assertEqual(ASSESS(self.profile(missing))["status"], "blocked")
        upstream = self.profile(upstream=["torch<2.13", "libtpu>=0.0.50"])
        self.assertEqual(ASSESS(upstream)["status"], "blocked")

    def test_scratch_mode_requires_optional_datasets_distribution(self):
        missing = ASSESS(self.profile(), require_datasets=True)
        self.assertEqual(missing["status"], "blocked")
        supplied = self.profile({**self.versions, "datasets": "3.3.2"})
        ready = ASSESS(supplied, require_datasets=True)
        self.assertTrue(ready["proceed"])
        self.assertEqual(ready["installed_versions"]["datasets"], "3.3.2")

    def test_training_stack_and_unparsed_pip_conflicts_block(self):
        def training_conflict(*args, **kwargs):
            return SimpleNamespace(
                returncode=1,
                stdout="transformers 5.15.0 requires torch<2.13, but you have torch 2.13.0.\n",
                stderr="",
            )

        def unparsed_conflict(*args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="unknown conflict\n", stderr="")

        self.assertIn("training_stack_pip_conflict",
                      ASSESS(self.profile(pip_runner=training_conflict))["reasons"])
        self.assertIn("unclassified_pip_conflict",
                      ASSESS(self.profile(pip_runner=unparsed_conflict))["reasons"])

    def test_notebook_runs_gate_before_trainer_child(self):
        path = ROOT / "notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb"
        notebook = json.loads(path.read_text())
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code")
        self.assertLess(source.index("assess_profile(inspect_profile(repo_root))"),
                        source.index("result = run_process("))
        self.assertIn("if not preflight[\"proceed\"]:", source)
        self.assertIn("report[\"preflight\"] = preflight", source)

    def test_notebook_block_writes_report_without_launching_trainer(self):
        report, launches = self.run_notebook_cell({
            "status": "blocked", "proceed": False,
        }, expect_failure=True)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["checks"]["train"], "not_run")
        self.assertEqual(launches, [])

    def test_notebook_allowed_run_keeps_preflight_in_report(self):
        report, launches = self.run_notebook_cell({
            "status": "experimental_mismatch", "proceed": True,
        })
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["preflight"]["status"], "experimental_mismatch")
        self.assertEqual(len(launches), 1)

    def run_notebook_cell(self, preflight, *, expect_failure=False):
        path = ROOT / "notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb"
        notebook = json.loads(path.read_text())
        cell = next("".join(item["source"]) for item in notebook["cells"]
                    if item["cell_type"] == "code"
                    and "result = run_process(" in "".join(item["source"]))
        real_run_path = runpy.run_path
        launches = []

        def fake_run_path(path):
            if path.endswith("torchtpu_t1_install_preflight.py"):
                return {"inspect_profile": lambda root: {}}
            if path.endswith("torchtpu_trainer_preflight.py"):
                return {"assess_trainer_profile": lambda profile: preflight}
            if path.endswith("torchtpu_probe_process.py"):
                return {"run_probe_process": lambda *args, **kwargs: (
                    launches.append(args) or successful_process())}
            return real_run_path(path)

        with tempfile.TemporaryDirectory() as directory:
            globals_for_cell = {"Path": Path, "tempfile": tempfile,
                                "sys": sys, "repo_root": ROOT,
                                "source_revision": SHA}
            with (mock.patch("runpy.run_path", side_effect=fake_run_path),
                  mock.patch.object(tempfile, "mkdtemp", return_value=directory),
                  redirect_stdout(io.StringIO())):
                if expect_failure:
                    with self.assertRaisesRegex(RuntimeError, "preflight blocked"):
                        exec(cell, globals_for_cell)
                else:
                    exec(cell, globals_for_cell)
            report = json.loads((Path(directory) / "trainlm_native_trainer_report.json").read_text())
        return report, launches


if __name__ == "__main__":
    unittest.main()
