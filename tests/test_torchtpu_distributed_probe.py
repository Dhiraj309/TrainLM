"""Pure launch/worker contracts; hardware execution remains owner-run."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
import unittest


CONTRACT = runpy.run_path(
    str(Path(__file__).resolve().parents[1]
        / "src/trainlm/runtime/torchtpu_distributed_probe.py")
)
INSPECT = CONTRACT["inspect_launch_environment"]
PREPARE = CONTRACT["prepare_launch_environment"]
SUMMARIZE = CONTRACT["summarize_workers"]


class LaunchContractTests(unittest.TestCase):
    def test_complete_single_host_topology_redacts_addresses(self):
        secret = "private-coordinator.example:8471"
        env = {
            "WORLD_SIZE": "2",
            "TORCH_TPU_TOPOLOGY": "2,1,1",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"{secret},host2:8472",
            "NODE_RANK": "0",
        }
        report = INSPECT(env)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["topology_shape"], [2, 1, 1])
        self.assertEqual(report["coordinator_address_count"], 2)
        self.assertNotIn(secret, json.dumps(report))

    def test_missing_and_inconsistent_topology_fail_closed(self):
        self.assertEqual(INSPECT({})["status"], "incomplete")
        env = {
            "WORLD_SIZE": "8",
            "TORCH_TPU_TOPOLOGY": "2,2,1",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": "a,b,c,d",
        }
        report = INSPECT(env)
        self.assertEqual(report["status"], "invalid")
        self.assertIn("topology_world_mismatch", report["issues"])
        self.assertIn("address_world_mismatch", report["issues"])
        env["NODE_RANK"] = "1"
        self.assertIn("multi_host_unsupported", INSPECT(env)["issues"])

    def test_helper_is_called_before_inspection_and_failures_are_sanitized(self):
        env = {}

        def helper():
            env.update({
                "WORLD_SIZE": "2",
                "TORCH_TPU_TOPOLOGY": "2,1,1",
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": "host1:1,host2:2",
            })

        torch = SimpleNamespace(tpu=SimpleNamespace(
            distributed=SimpleNamespace(set_tpu_launch_env=helper)))
        self.assertEqual(PREPARE(torch, env)["status"], "ready")
        self.assertEqual(PREPARE(SimpleNamespace(), {})["status"], "api_unavailable")

        def fail():
            raise RuntimeError("secret=do_not_print")

        torch.tpu.distributed.set_tpu_launch_env = fail
        failure = PREPARE(torch, env)
        self.assertEqual(failure["status"], "preparation_failed")
        self.assertNotIn("do_not_print", str(failure))

    def test_worker_reconciliation_requires_every_unique_rank_and_collective(self):
        records = [
            {"rank": 0, "local_rank": 0, "world_size": 2, "collective": "passed"},
            {"rank": 1, "local_rank": 1, "world_size": 2, "collective": "passed"},
        ]
        self.assertEqual(SUMMARIZE(records, 2)["status"], "passed")
        self.assertEqual(SUMMARIZE(records[:1] * 2, 2)["status"], "failed")
        records[1]["local_rank"] = 0
        self.assertEqual(SUMMARIZE(records, 2)["status"], "failed")
        records[1]["local_rank"] = 1
        records[1]["collective"] = "failed"
        self.assertEqual(SUMMARIZE(records, 2)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
