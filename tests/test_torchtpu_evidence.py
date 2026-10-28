"""T1 exports preserve scope, reject false success, and omit raw logs."""

import json
from pathlib import Path
import runpy
import unittest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1]
                          / "src/trainlm/runtime/torchtpu_evidence.py"))
BUILD = MODULE["build_evidence"]


def discovery(name="TPU v5e"):
    return {"schema_version": 1, "status": "ready", "device_type": "tpu",
            "allocation": "passed", "device_count_visible": 1, "device_name": name,
            "torch_version": "2.14.1+cpu", "torch_tpu_version": "0.2.0",
            "libtpu_version": "0.0.50"}


def process(stdout="", **overrides):
    return dict({"status": "completed", "exit_code": 0, "stdout": stdout,
                 "stderr": "", "fatal_markers": [], "error_type": None,
                 "cleanup_completed": None}, **overrides)


def smoke(device="tpu", compiled=False):
    lines = list(MODULE["_SMOKE_MARKERS"])
    lines += [f"device={device}"]
    lines += ["compile=torch.compile enabled", "compile=PASS"] if compiled else ["compile=eager"]
    if device == "tpu":
        lines += ["tpu_report=" + json.dumps(discovery())]
    return process("\n".join(lines))


def build(**kwargs):
    return BUILD(source_revision="a" * 40, notebook_host="Colab", **kwargs)


class EvidenceTests(unittest.TestCase):
    def test_package_incompatibility_is_preserved_beside_native_smoke_pass(self):
        profile = {"schema_version": 1, "status": "incompatible",
                   "issues": ["requirement_conflict"],
                   "installed_versions": {"torch": "2.14.1+cpu"},
                   "source_requirements": [{"package": "torch", "required": "<2.14,>=2.9",
                                             "status": "mismatch"}],
                   "torch_tpu_requirements": []}
        report = build(package_process=process(json.dumps(profile),
                                               status="failed", exit_code=1),
                       eager_process=smoke())
        self.assertEqual(report["checks"]["package_profile"]["status"], "incompatible")
        self.assertEqual(report["checks"]["eager"]["status"], "passed")
        self.assertEqual(report["checks"]["eager"]["scope"], "native_tpu")
        self.assertEqual(report["checks"]["compile"]["status"], "not_run")
        self.assertEqual(report["native_training"], "not_integrated")
        self.assertEqual(report["review"], "owner_review_required")

    def test_cpu_smoke_and_skip_remain_separate_from_native_evidence(self):
        checks = build(eager_process=smoke("cpu"))["checks"]
        self.assertEqual(checks["eager"]["status"], "passed")
        self.assertEqual(checks["eager"]["scope"], "non_tpu")
        self.assertEqual(checks["eager"]["device"], "cpu")
        self.assertEqual(checks["distributed"], {"status": "not_run"})

    def test_compile_requires_explicit_compile_markers(self):
        self.assertEqual(build(compile_process=smoke())["checks"]["compile"]["status"], "failed")
        self.assertEqual(build(compile_process=smoke(compiled=True))[
            "checks"]["compile"]["status"], "passed")

    def test_duplicate_discovery_cannot_be_mixed_into_one_run(self):
        result = smoke()
        result["stdout"] += "\ntpu_report=" + json.dumps(discovery("TPU v6e"))
        self.assertEqual(build(eager_process=result)["checks"]["eager"]["status"], "failed")

    def test_no_raw_output_addresses_or_unrecognized_fields_in_export(self):
        result = smoke()
        result["stderr"] = "private-coordinator.example:8471 token=do_not_export"
        result["secret"] = "do_not_export"
        report = build(eager_process=result)
        encoded = json.dumps(report)
        self.assertNotIn("do_not_export", encoded)
        self.assertNotIn("private-coordinator", encoded)
        self.assertNotIn("stderr", encoded)

    def test_fatal_teardown_defeats_even_incorrectly_marked_success(self):
        result = smoke()
        result["stderr"] = "RAW: Dumping core"
        check = build(eager_process=result)["checks"]["eager"]
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["fatal_markers"], ["core_dump"])

    def test_distributed_reconciles_ranks_and_exit_status(self):
        summary = {"status": "passed", "expected_world_size": 2, "reports_received": 2,
                   "ranks": [0, 1], "local_ranks": [0, 1], "worker_exit_code": 0}
        def check():
            return build(distributed_process=process(
                "collective_summary=" + json.dumps(summary)))['checks']['distributed']
        self.assertEqual(check()["status"], "passed")
        summary["ranks"] = [0, 0]
        self.assertEqual(check()["status"], "failed")
        summary["ranks"] = [False, True]
        self.assertEqual(check()["status"], "failed")
        summary["ranks"] = [0, 1]
        summary["worker_exit_code"] = 1
        self.assertEqual(check()["status"], "failed")

    def test_lifecycle_rechecks_consistency_and_omits_artifact_paths(self):
        attempts = [{"attempt": i, "status": "passed", "exit_code": 0,
                     "report": discovery()} for i in (1, 2)]
        report = {"schema_version": 1, "status": "passed", "requested_attempts": 2,
                  "completed_attempts": 2, "observations_consistent": True,
                  "attempts": attempts, "artifact_dir": "/private/owner/path"}
        check = build(lifecycle_report=report)["checks"]["lifecycle"]
        self.assertEqual(check["status"], "passed")
        self.assertNotIn("/private", json.dumps(check))
        attempts[1]["report"]["device_name"] = "TPU v6e"
        self.assertEqual(build(lifecycle_report=report)["checks"]["lifecycle"]["status"], "failed")

    def test_malformed_profile_and_missing_stdout_cannot_pass(self):
        self.assertEqual(build(package_process=process())["checks"]["package_profile"]["status"], "failed")
        report = {"schema_version": 1, "status": "metadata_compatible",
                  "installed_versions": [], "issues": []}
        self.assertEqual(build(package_process=process(json.dumps(report)))[
            "checks"]["package_profile"]["status"], "invalid_report")

    def test_export_requires_full_source_revision(self):
        with self.assertRaises(ValueError):
            BUILD(source_revision="main", notebook_host="Colab")


if __name__ == "__main__":
    unittest.main()
