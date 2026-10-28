"""Bounded native probe process lifecycle without running torch or a TPU."""

import json
from pathlib import Path
import runpy
import signal
import subprocess
from tempfile import TemporaryDirectory
import unittest


MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1]
        / "examples/torchtpu_t1_lifecycle_probe.py")
)
RUN = MODULE["run_lifecycle"]
RUN_ONCE = MODULE["run_once"]
MARKERS = MODULE["PASS_MARKERS"]


def ready_report(name="TPU v5e"):
    return {
        "schema_version": 1, "status": "ready", "device_type": "tpu",
        "allocation": "passed", "device_count_visible": 1,
        "device_name": name, "torch_version": "2.14.1+cpu",
    }


class FinishedProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = 0
        self.timeouts = []

    def communicate(self, timeout):
        self.timeouts.append(timeout)
        return "\n".join(MARKERS) + "\n", ""


class LifecycleTests(unittest.TestCase):
    def test_two_fresh_children_and_consistent_device_reports(self):
        commands = []
        processes = []

        def launch(command, **kwargs):
            commands.append((command, kwargs))
            destination = Path(command[command.index("--report-json") + 1])
            self.assertFalse(destination.exists())
            destination.write_text(json.dumps(ready_report()))
            process = FinishedProcess(100 + len(commands))
            processes.append(process)
            return process

        with TemporaryDirectory() as directory:
            report = RUN(Path(directory), repeats=2, timeout=12,
                         process_factory=launch,
                         kill_group=lambda *args: self.fail("unexpected kill"))
            self.assertTrue(Path(report["artifact_dir"]).is_dir())
            saved = json.loads((Path(report["artifact_dir"]) / "summary.json").read_text())
            self.assertEqual(saved, report)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["observations_consistent"])
        self.assertEqual(report["completed_attempts"], 2)
        self.assertEqual([process.timeouts for process in processes], [[12], [12]])
        self.assertTrue(all(args["start_new_session"] for _, args in commands))
        self.assertTrue(all("--device" in command and "tpu" in command
                            for command, _ in commands))

    def test_timeout_terminates_process_group_and_stops_retries(self):
        launches = []
        signals = []

        class HungProcess:
            pid = 222
            returncode = None
            calls = 0

            def communicate(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("probe", timeout)
                return "", ""

        def launch(command, **kwargs):
            launches.append(command)
            return HungProcess()

        with TemporaryDirectory() as directory:
            report = RUN(Path(directory), repeats=2, timeout=1,
                         process_factory=launch,
                         kill_group=lambda pid, sig: signals.append((pid, sig)))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_attempts"], 1)
        self.assertEqual(report["attempts"][0]["status"], "timeout")
        self.assertEqual(signals, [(222, signal.SIGTERM), (222, signal.SIGKILL)])
        self.assertEqual(len(launches), 1)

    def test_changed_hardware_fails_repeat_consistency(self):
        launches = []

        def launch(command, **kwargs):
            launches.append(command)
            destination = Path(command[command.index("--report-json") + 1])
            destination.write_text(json.dumps(ready_report(
                "TPU v5e" if len(launches) == 1 else "TPU v6e"
            )))
            return FinishedProcess(300 + len(launches))

        with TemporaryDirectory() as directory:
            report = RUN(Path(directory), repeats=2, timeout=5,
                         process_factory=launch,
                         kill_group=lambda *args: None)
        self.assertEqual(report["status"], "failed")
        self.assertEqual([a["status"] for a in report["attempts"]],
                         ["passed", "passed"])
        self.assertFalse(report["observations_consistent"])

    def test_interrupt_terminates_child_and_saves_summary(self):
        signals = []

        class InterruptedProcess:
            pid = 444
            returncode = None
            calls = 0

            def communicate(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt
                return "", ""

        with TemporaryDirectory() as directory:
            report = RUN(Path(directory), repeats=2, timeout=5,
                         process_factory=lambda *args, **kwargs: InterruptedProcess(),
                         kill_group=lambda pid, sig: signals.append((pid, sig)))
            saved = json.loads((Path(report["artifact_dir"]) / "summary.json").read_text())
        self.assertEqual(report, saved)
        self.assertEqual(report["attempts"][0]["status"], "interrupted")
        self.assertEqual(signals, [(444, signal.SIGTERM), (444, signal.SIGKILL)])

    def test_fatal_teardown_fails_even_with_success_markers_and_zero_exit(self):
        class FatalProcess(FinishedProcess):
            def communicate(self, timeout):
                stdout, _ = super().communicate(timeout)
                return stdout, "RAW: Dumping core secret=do_not_export\n"

        def launch(command, **kwargs):
            destination = Path(command[command.index("--report-json") + 1])
            destination.write_text(json.dumps(ready_report()))
            return FatalProcess(777)

        with TemporaryDirectory() as directory:
            report = RUN(Path(directory), process_factory=launch,
                         kill_group=lambda *args: None)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_attempts"], 1)
        self.assertEqual(report["attempts"][0]["fatal_markers"], ["core_dump"])
        self.assertNotIn("do_not_export", json.dumps(report))

    def test_missing_success_markers_fail_without_echoing_worker_error(self):
        class IncompleteProcess:
            pid = 555
            returncode = 0

            def communicate(self, timeout):
                return MARKERS[0] + "\n", "RuntimeError: secret=do_not_echo\n"

        def launch(command, **kwargs):
            destination = Path(command[command.index("--report-json") + 1])
            destination.write_text(json.dumps(ready_report()))
            return IncompleteProcess()

        with TemporaryDirectory() as directory:
            attempt = RUN_ONCE(1, Path(directory), 5, process_factory=launch,
                               kill_group=lambda *args: None)
        self.assertEqual(attempt["status"], "failed")
        self.assertIn("forward", attempt["missing_markers"])
        self.assertEqual(attempt["error_type"], "RuntimeError")
        self.assertNotIn("do_not_echo", str(attempt))


if __name__ == "__main__":
    unittest.main()
