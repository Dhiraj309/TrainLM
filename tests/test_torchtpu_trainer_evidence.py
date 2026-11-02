"""Trainer evidence must remain bounded and fail closed without Torch installed."""

import json
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
build_report = runpy.run_path(str(ROOT / "src/trainlm/runtime/torchtpu_trainer_evidence.py"))[
    "build_trainer_report"
]
SHA = "a" * 40


def successful_process(*, compile_requested=False, resume_requested=False):
    checks = {
        "train": "passed", "export": "passed",
        "compile": "passed" if compile_requested else "not_run",
        "resume": "passed" if resume_requested else "not_run",
        "exact_weight_match": "passed" if resume_requested else "not_run",
    }
    summary = {
        "schema_version": 1,
        "device": {
            "schema_version": 1, "status": "ready", "device_type": "tpu",
            "device_count_visible": 1, "allocation": "passed",
            "device_name": "TPU v5e", "torch_version": "2.14.1+cpu",
            "torch_tpu_version": "0.2.0", "libtpu_version": "0.0.50",
        },
        "packages": {"torch": "2.14.1+cpu", "torch_tpu": "0.2.0",
                     "libtpu": "0.0.50", "transformers": "5.20.2",
                     "tokenizers": "0.22.1"},
        "checks": checks,
    }
    markers = ["train=PASS", "export=PASS"]
    if compile_requested:
        markers.append("compile=PASS")
    if resume_requested:
        markers.extend(["resume=PASS", "exact_weight_match=PASS"])
    markers.extend(["trainer_summary=" + json.dumps(summary),
                    "result=trainer_smoke_passed"])
    return {"status": "completed", "exit_code": 0, "error_type": None,
            "fatal_markers": [], "cleanup_completed": None,
            "stdout": "\n".join(markers), "stderr": ""}


class TrainerEvidenceTests(unittest.TestCase):
    def report(self, process, **modes):
        return build_report(process, source_revision=SHA,
                            compile_requested=modes.get("compile_requested", False),
                            resume_requested=modes.get("resume_requested", False))

    def test_eager_compile_and_resume_reports(self):
        for modes in ({}, {"compile_requested": True}, {"resume_requested": True}):
            with self.subTest(modes=modes):
                process = successful_process(**modes)
                report = self.report(process, **modes)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["device"]["device_name"], "TPU v5e")
                self.assertEqual(report["packages"]["torch"], "2.14.1+cpu")
                self.assertEqual(report["packages"]["tokenizers"], "0.22.1")
                self.assertNotIn("stdout", report)
                self.assertNotIn("stderr", report)

    def test_missing_duplicate_or_invalid_summary_fails(self):
        process = successful_process()
        summary_line = next(line for line in process["stdout"].splitlines()
                            if line.startswith("trainer_summary="))
        for stdout in (process["stdout"].replace(summary_line, ""),
                       process["stdout"] + "\n" + summary_line,
                       process["stdout"].replace(summary_line, "trainer_summary={")):
            with self.subTest(stdout=stdout[-20:]):
                result = self.report({**process, "stdout": stdout})
                self.assertEqual(result["status"], "failed")

    def test_missing_phase_marker_or_mismatched_mode_fails(self):
        process = successful_process()
        self.assertEqual(self.report({**process, "stdout": process["stdout"].replace(
            "export=PASS", "")})["status"], "failed")
        self.assertEqual(self.report(process, compile_requested=True)["status"], "failed")

    def test_timeout_and_fatal_exit_never_pass_or_leak_logs(self):
        process = successful_process()
        process["stderr"] = "secret-token=do-not-share"
        process["stdout"] += "\nsecret-token=do-not-share"
        for overrides in ({"status": "timeout", "cleanup_completed": True},
                          {"fatal_markers": ["python_fatal"]},
                          {"exit_code": 1}):
            with self.subTest(overrides=overrides):
                report = self.report({**process, **overrides})
                self.assertEqual(report["status"], "failed")
                self.assertNotIn("do-not-share", json.dumps(report))

    def test_invalid_device_or_package_text_fails_closed(self):
        process = successful_process()
        lines = process["stdout"].splitlines()
        index = next(i for i, line in enumerate(lines)
                     if line.startswith("trainer_summary="))
        summary = json.loads(lines[index].split("=", 1)[1])
        summary["device"]["device_count_visible"] = 8
        summary["packages"]["torch"] = "secret=/tmp/private"
        lines[index] = "trainer_summary=" + json.dumps(summary)
        report = self.report({**process, "stdout": "\n".join(lines)})
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["device"])
        self.assertNotIn("private", json.dumps(report))

    def test_scratch_model_dataset_and_metrics_are_sanitized(self):
        process = successful_process()
        lines = process["stdout"].splitlines()
        index = next(i for i, line in enumerate(lines)
                     if line.startswith("trainer_summary="))
        summary = json.loads(lines[index].split("=", 1)[1])
        summary["model"] = {
            "id": "distilbert/distilgpt2", "revision": "b" * 40,
            "initialization": "random_from_config", "parameter_count": 82000000,
        }
        summary["dataset"] = {
            "id": "roneneldan/TinyStories", "revision": "c" * 40,
            "train_examples": 10000, "validation_examples": 1000,
            "train_tokens": 1200000, "validation_tokens": 120000,
            "train_documents": 10000, "validation_documents": 1000,
            "train_shard_sha256": "d" * 64,
            "validation_shard_sha256": "e" * 64,
        }
        summary["metrics"] = {"final_train_loss": 4.2, "eval_loss": 3.8,
                              "eval_perplexity": 44.7, "secret": "omit"}
        lines[index] = "trainer_summary=" + json.dumps(summary)
        report = self.report({**process, "stdout": "\n".join(lines)})
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["model"]["initialization"], "random_from_config")
        self.assertEqual(report["dataset"]["validation_examples"], 1000)
        self.assertEqual(report["dataset"]["train_tokens"], 1200000)
        self.assertEqual(report["dataset"]["train_shard_sha256"], "d" * 64)
        self.assertEqual(report["metrics"]["eval_loss"], 3.8)
        self.assertNotIn("secret", json.dumps(report))

    def test_explicit_scratch_config_and_tokenizer_provenance_are_reported(self):
        process = successful_process()
        lines = process["stdout"].splitlines()
        index = next(i for i, line in enumerate(lines)
                     if line.startswith("trainer_summary="))
        summary = json.loads(lines[index].split("=", 1)[1])
        summary["model"] = {
            "id": "gpt2", "revision": None,
            "initialization": "random_from_config", "model_type": "gpt2",
            "config_source": "explicit_gpt2_config",
            "config": {
                "vocab_size": 16000, "n_positions": 128,
                "n_embd": 768, "n_layer": 6, "n_head": 12,
            },
            "parameter_count": 55000000,
        }
        summary["dataset"] = {
            "id": "roneneldan/TinyStories", "revision": "c" * 40,
            "train_examples": 10000, "validation_examples": 1000,
            "tokenizer": {
                "kind": "byte_level_bpe_from_training_split",
                "training_documents": 10000, "vocab_size": 16000,
            },
        }
        lines[index] = "trainer_summary=" + json.dumps(summary)
        report = self.report({**process, "stdout": "\n".join(lines)})

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["model"]["revision"], None)
        self.assertEqual(report["model"]["config"]["n_layer"], 6)
        self.assertEqual(
            report["dataset"]["tokenizer"]["kind"],
            "byte_level_bpe_from_training_split",
        )


if __name__ == "__main__":
    unittest.main()
