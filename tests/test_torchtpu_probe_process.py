"""Exercise probe failure handling and real Linux child-group cleanup."""

import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest


CONTRACT = runpy.run_path(str(Path(__file__).resolve().parents[1]
                            / "src/trainlm/runtime/torchtpu_probe_process.py"))
RUN = CONTRACT["run_probe_process"]


class ProbeProcessTests(unittest.TestCase):
    def test_zero_exit_with_fatal_teardown_is_failed(self):
        for marker in ("RAW: Dumping core", "exit() hanging: exiting process",
                       "Fatal Python error: abort"):
            with self.subTest(marker=marker):
                result = RUN([sys.executable, "-c", "import sys; print(sys.argv[1])",
                              marker], cwd=Path.cwd(), timeout=5)
                self.assertEqual(result["exit_code"], 0)
                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["fatal_markers"])

    def test_normal_runtime_warning_is_not_a_failure(self):
        result = RUN([sys.executable, "-c",
                      "import sys; print('WARNING: cache disabled', file=sys.stderr)"],
                     cwd=Path.cwd(), timeout=5)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["fatal_markers"], [])

    def test_interrupt_kills_descendants_after_leader_exits(self):
        calls = []

        class Interrupted:
            pid = 100
            returncode = None

            def communicate(self, timeout):
                if not calls:
                    raise KeyboardInterrupt
                self.returncode = 0
                return "", ""

        result = RUN(["unused"], cwd=Path.cwd(), timeout=1,
                     process_factory=lambda *a, **kw: Interrupted(),
                     kill_group=lambda pid, sig: calls.append((pid, sig)))
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(result["cleanup_completed"])
        self.assertEqual(calls, [(100, signal.SIGTERM), (100, signal.SIGKILL)])

    def test_timeout_retains_partial_output_for_local_notebook_diagnosis(self):
        class TimedOut:
            pid = 111
            returncode = None
            calls = 0

            def communicate(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("probe", timeout,
                                                    output=b"device=tpu\n", stderr=b"initializing\n")
                self.returncode = -15
                return "", ""

        result = RUN(["unused"], cwd=Path.cwd(), timeout=1,
                     process_factory=lambda *a, **kw: TimedOut(),
                     kill_group=lambda *args: None)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["stdout"], "device=tpu\n")
        self.assertEqual(result["stderr"], "initializing\n")

    @unittest.skipUnless(sys.platform == "linux", "Linux process-group contract")
    def test_real_timeout_kills_grandchild_that_ignores_sigterm(self):
        processes = []

        def launch(*args, **kwargs):
            process = subprocess.Popen(*args, **kwargs)
            processes.append(process)
            return process

        code = """
import os, signal, sys, time
from pathlib import Path
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(sys.argv[1]).write_text(str(os.getpid()))
    os.close(1)
    os.close(2)
time.sleep(60)
"""
        with TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            try:
                result = RUN([sys.executable, "-c", code, str(pid_file)],
                             cwd=Path.cwd(), timeout=1, process_factory=launch)
                self.assertEqual(result["status"], "timeout")
                self.assertTrue(result["cleanup_completed"])
                child_stat = Path(f"/proc/{int(pid_file.read_text())}/stat")
                deadline = time.monotonic() + 2
                while child_stat.exists() and time.monotonic() < deadline:
                    if child_stat.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.01)
                if child_stat.exists():
                    self.assertEqual(child_stat.read_text().split()[2], "Z")
            finally:
                for process in processes:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
