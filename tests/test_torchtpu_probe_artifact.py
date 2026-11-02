"""Native probe artifact stays bounded and is produced by its device process."""

import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SELECT_DEVICE = runpy.run_path(
    str(Path(__file__).resolve().parents[1]
        / "examples/torchtpu_t1_v5e1_probe.py")
)["_select_device"]


class ProbeArtifactTests(unittest.TestCase):
    def test_device_owning_probe_writes_normalized_report(self):
        accelerator = SimpleNamespace(
            current_accelerator=lambda: SimpleNamespace(type="tpu"),
            is_available=lambda: True,
            device_count=lambda: 1,
        )
        torch = SimpleNamespace(
            __version__="2.14.1+cpu",
            accelerator=accelerator,
            device=lambda kind: kind,
            empty=lambda *args, **kwargs: SimpleNamespace(cpu=lambda: None),
            tpu=SimpleNamespace(get_device_properties=lambda: SimpleNamespace(
                name="TPU v5e", topology=[1, 1, 1],
            )),
        )
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "probe" / "report.json"
            args = SimpleNamespace(device="auto", allow_cpu=False,
                                   report_json=destination)
            SELECT_DEVICE(torch, args)
            report = json.loads(destination.read_text())
        self.assertEqual(args.device, "tpu")
        self.assertEqual(report["source"], "torchtpu_discovery")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["device_name"], "TPU v5e")
        self.assertEqual(report["topology"], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
