"""Contracts for runtime discovery without requiring a TPU or torch install."""

import runpy
from pathlib import Path
from types import SimpleNamespace
import unittest


CONTRACT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "src/trainlm/runtime/torchtpu_discovery.py")
)
DISCOVER = CONTRACT["discover_tpu"]
SELECT_AUTO = CONTRACT["select_auto_backend"]


class DiscoveryTests(unittest.TestCase):
    def test_no_accelerator_api(self):
        report = DISCOVER(SimpleNamespace(__version__="mock"))
        self.assertEqual(report["status"], "api_unavailable")
        self.assertEqual(report["allocation"], "not_run")

    def test_cpu_is_unavailable_without_allocation(self):
        accelerator = SimpleNamespace(
            current_accelerator=lambda: SimpleNamespace(type="cpu"),
            is_available=lambda: True,
            device_count=lambda: 1,
        )
        report = DISCOVER(SimpleNamespace(accelerator=accelerator))
        self.assertEqual(report["status"], "unavailable")
        self.assertIsNone(report["device_count_visible"])

    def test_tpu_properties_and_visible_count(self):
        accelerator = SimpleNamespace(
            current_accelerator=lambda: SimpleNamespace(type="tpu"),
            is_available=lambda: True,
            device_count=lambda: 1,
        )
        props = SimpleNamespace(
            name="TPU v5e", topology="1x1", hbm_bytes_per_device=16,
            hbm_bytes_per_chip=16, devices_per_chip=1,
        )
        torch = SimpleNamespace(
            accelerator=accelerator,
            device=lambda name: name,
            empty=lambda *args, **kwargs: SimpleNamespace(cpu=lambda: None),
            tpu=SimpleNamespace(get_device_properties=lambda: props),
        )
        report = DISCOVER(torch)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["device_name"], "TPU v5e")
        self.assertEqual(report["device_count_visible"], 1)
        self.assertEqual(report["allocation"], "passed")

    def test_busy_tpu_is_failure_without_cpu_fallback(self):
        accelerator = SimpleNamespace(
            current_accelerator=lambda: SimpleNamespace(type="tpu"),
            is_available=lambda: True,
            device_count=lambda: 1,
        )
        def fail(*args, **kwargs):
            raise RuntimeError("secret=should_not_appear")
        report = DISCOVER(SimpleNamespace(
            accelerator=accelerator, device=lambda name: name, empty=fail,
        ))
        self.assertEqual(report["status"], "allocation_failed")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertNotIn("should_not_appear", str(report))
        self.assertEqual(report["allocation"], "failed")

    def test_parent_preflight_skips_allocation_and_device_properties(self):
        accelerator = SimpleNamespace(
            current_accelerator=lambda: SimpleNamespace(type="tpu"),
            is_available=lambda: True,
            device_count=lambda: 1,
        )
        def must_not_call(*args, **kwargs):
            raise AssertionError("parent must not allocate or inspect properties")
        torch = SimpleNamespace(
            accelerator=accelerator,
            empty=must_not_call,
            tpu=SimpleNamespace(get_device_properties=must_not_call),
        )
        report = DISCOVER(torch, verify_allocation=False, include_properties=False)
        self.assertEqual(report["status"], "available_unverified")
        self.assertEqual(report["allocation"], "not_run")
        self.assertIsNone(report["device_name"])

    def test_auto_routing_preserves_cpu_cuda_and_blocks_native_tpu(self):
        absent = {"status": "unavailable", "device_type": "cpu"}
        self.assertEqual(SELECT_AUTO(absent, cuda_available=False), "cpu")
        self.assertEqual(SELECT_AUTO(absent, cuda_available=True), "cuda")
        with self.assertRaisesRegex(RuntimeError, "Native TorchTPU was detected"):
            SELECT_AUTO(
                {"status": "available_unverified", "device_type": "tpu"},
                cuda_available=False,
            )
        with self.assertRaisesRegex(RuntimeError, "accelerator discovery failed"):
            SELECT_AUTO(
                {"status": "query_failed", "torch_tpu_version": "0.2.0"},
                cuda_available=False,
            )


if __name__ == "__main__":
    unittest.main()
