"""Read-only TorchTPU install-profile classification without installed torch."""

from importlib import metadata
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

from packaging.utils import canonicalize_name


INSPECT = runpy.run_path(
    str(Path(__file__).resolve().parents[1]
        / "examples/torchtpu_t1_install_preflight.py")
)["inspect_profile"]
ROOT = Path(__file__).resolve().parents[1]


class FakeMetadata:
    def __init__(self, versions, upstream_requires=()):
        self.versions = {canonicalize_name(key): value
                         for key, value in versions.items()}
        self.upstream_requires = upstream_requires

    def version(self, name):
        key = canonicalize_name(name)
        if key not in self.versions:
            raise metadata.PackageNotFoundError(name)
        return self.versions[key]

    def distribution(self, name):
        self.version(name)
        return SimpleNamespace(
            requires=list(self.upstream_requires),
            metadata={"Requires-Python": ">=3.10"},
        )


def pip_ok(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout="No broken requirements found.\n",
                           stderr="")


class InstallPreflightTests(unittest.TestCase):
    def setUp(self):
        self.versions = {
            "trainlm": "0.1.0", "torch": "2.13.0", "torch_tpu": "0.2.0",
            "libtpu": "0.0.50", "transformers": "5.15.0",
            "huggingface-hub": "1.16.4", "PyYAML": "6.0.3",
        }

    def test_metadata_compatible_does_not_claim_hardware_support(self):
        provider = FakeMetadata(self.versions, ["torch>=2.9,<2.14", "libtpu==0.0.50"])
        report = INSPECT(ROOT, provider=provider, pip_runner=pip_ok)
        self.assertEqual(report["status"], "metadata_compatible")
        self.assertEqual(report["scope"], "installed_metadata_only")
        self.assertEqual(report["pip_check"]["status"], "passed")

    def test_owner_observed_torch_version_matches_source_and_upstream_ranges(self):
        versions = {**self.versions, "torch": "2.14.1+cpu"}
        provider = FakeMetadata(versions, ["torch==2.14.1", "libtpu==0.0.50"])
        report = INSPECT(ROOT, provider=provider, pip_runner=pip_ok)
        self.assertEqual(report["status"], "metadata_compatible")
        torch_check = next(item for item in report["source_requirements"]
                           if item["package"] == "torch")
        self.assertEqual(torch_check["required"], "<2.15,>=2.9")
        self.assertEqual(torch_check["status"], "matched")

    def test_unrelated_ipython_conflict_is_visible_but_not_blocking(self):
        def ipython_conflict(*args, **kwargs):
            return SimpleNamespace(
                returncode=1,
                stdout="ipython 7.34.0 requires jedi, which is not installed.\n",
                stderr="",
            )

        versions = {**self.versions, "torch": "2.14.1+cpu"}
        report = INSPECT(
            ROOT,
            provider=FakeMetadata(versions, ["torch==2.14.1", "libtpu==0.0.50"]),
            pip_runner=ipython_conflict,
        )
        self.assertEqual(report["status"], "metadata_compatible")
        self.assertEqual(report["pip_check"]["status"], "conflicts")
        self.assertFalse(report["pip_check"]["blocking"])
        self.assertEqual(report["pip_check"]["affected_packages"], ["ipython"])
        self.assertNotIn("pip_conflicts", report["issues"])

    def test_training_stack_pip_conflict_is_blocking(self):
        def torch_conflict(*args, **kwargs):
            return SimpleNamespace(
                returncode=1,
                stdout="transformers 5.15.0 requires torch<2.13, but you have torch 2.13.0.\n",
                stderr="",
            )

        report = INSPECT(ROOT, provider=FakeMetadata(self.versions), pip_runner=torch_conflict)
        self.assertEqual(report["status"], "incompatible")
        self.assertTrue(report["pip_check"]["blocking"])
        self.assertIn("pip_conflicts", report["issues"])

    def test_missing_tpu_packages_are_incomplete(self):
        versions = {key: value for key, value in self.versions.items()
                    if key not in {"torch_tpu", "libtpu"}}
        report = INSPECT(ROOT, provider=FakeMetadata(versions), pip_runner=pip_ok)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("missing_distributions", report["issues"])

    def test_pip_conflict_is_counted_without_echoing_output(self):
        provider = FakeMetadata(self.versions, ["torch>=2.9,<2.14"])

        def conflict(*args, **kwargs):
            return SimpleNamespace(returncode=1,
                                   stdout="secret=do_not_echo has broken requirement\n",
                                   stderr="")

        report = INSPECT(ROOT, provider=provider, pip_runner=conflict)
        self.assertEqual(report["status"], "incompatible")
        self.assertEqual(report["pip_check"]["problem_count"], 1)
        self.assertEqual(report["pip_check"]["affected_packages"], [])
        self.assertNotIn("do_not_echo", str(report))


if __name__ == "__main__":
    unittest.main()
