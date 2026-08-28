import conftest_paths  # noqa: F401
import os
import stat
import tempfile
import time
import unittest

from autotune_core.bench_runner import BenchRunner, CancellationToken
from autotune_core.models import BenchBinaryIdentity, BenchmarkTarget, ExecutionEnvironment
from autotune_core.results import BenchmarkCase, BenchmarkWorkload
from autotune_core.run_store import RunStore


def _script(body):
    path = os.path.join(tempfile.mkdtemp(), "fake-bench.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env python3\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def _case(binary):
    env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", binary, "b", "f", "v", "artifact"), "now")
    return BenchmarkCase("case", "stage", "candidate", "cpu", {}, BenchmarkWorkload("pp", 4, 0, 0), env)


class TestBenchRunner(unittest.TestCase):
    def _run(self, script, timeout=2, cancellation=None, max_parse_bytes=1024 * 1024):
        root = tempfile.mkdtemp()
        store = RunStore(root, instance_id="runner", pid=os.getpid())
        store.create_run("run", {}, {})
        runner = BenchRunner(store, poll_interval=0.01, terminate_grace_seconds=0.05, max_parse_bytes=max_parse_bytes)
        return runner.run_case("run", BenchmarkTarget("/model.gguf", "fingerprint"), _case(script), 2, timeout, cancellation), root

    def test_repetitions_use_one_process_and_expand_samples(self):
        script = _script("import json\nprint(json.dumps({'n_prompt':4,'n_gen':0,'n_depth':0,'samples_ts':[1,2],'samples_ns':[10,20]}))\n")
        (status, measurements, artifact), _ = self._run(script)
        self.assertEqual(status, "completed")
        self.assertEqual(len(measurements), 2)
        self.assertIn("--repetitions", artifact["argv"])

    def test_timeout_and_cancellation_terminate_only_owned_process_group(self):
        script = _script("import time\ntime.sleep(10)\n")
        (status, _, _), root = self._run(script, timeout=0.03)
        self.assertEqual(status, "failed")
        self.assertEqual(RunStore(root).load_manifest("run")["status"], "running")
        token = CancellationToken(); token.cancel()
        (status, _, _), root = self._run(script, cancellation=token)
        self.assertEqual(status, "cancelled")
        self.assertEqual(RunStore(root).load_manifest("run")["status"], "running")

    def test_cancellation_signals_only_the_spawned_process_group(self):
        script = _script("import time\ntime.sleep(10)\n")
        root = tempfile.mkdtemp()
        store = RunStore(root, instance_id="runner", pid=os.getpid())
        store.create_run("run", {}, {})
        runner = BenchRunner(store, poll_interval=0.01, terminate_grace_seconds=0.05)
        seen, original = [], runner._terminate_owned_group
        def owned_only(process):
            seen.append(process.pid)
            original(process)
        runner._terminate_owned_group = owned_only
        token = CancellationToken(); token.cancel()
        runner.run_case("run", BenchmarkTarget("/m.gguf", "f"), _case(script), 1, 2, token)
        self.assertEqual(len(seen), 1)
        self.assertNotEqual(seen[0], os.getpid())

    def test_large_outputs_do_not_deadlock_pipes(self):
        script = _script("import sys\nsys.stdout.write('x'*2000000); sys.stderr.write('y'*2000000)\n")
        started = time.monotonic()
        (status, _, artifact), root = self._run(script, timeout=2, max_parse_bytes=1024)
        self.assertEqual(status, "failed")
        self.assertLess(time.monotonic() - started, 2)
        self.assertGreater(os.path.getsize(os.path.join(root, 'runs', 'run', 'invocations', artifact['stderr_path'])), 1000000)

    def test_missing_or_non_executable_binary_becomes_failed_artifact(self):
        for binary in ("/no/such/llama-bench", _script("print('x')")):
            if binary != "/no/such/llama-bench":
                os.chmod(binary, stat.S_IRUSR | stat.S_IWUSR)
            (status, _, artifact), root = self._run(binary)
            self.assertEqual(status, "failed")
            self.assertTrue(artifact["error"])
            self.assertTrue(os.path.exists(os.path.join(root, "runs", "run", "invocations", artifact["invocation_id"] + ".json")))
            self.assertEqual(RunStore(root).load_manifest("run")["status"], "running")

    def test_termination_race_is_safe(self):
        class Gone:
            pid = 999999
            def poll(self): return None
            def kill(self): raise ProcessLookupError()
            def terminate(self): raise ProcessLookupError()
        runner = BenchRunner(RunStore(tempfile.mkdtemp()), clock=time.monotonic)
        original = os.killpg
        try:
            os.killpg = lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError())
            runner._terminate_owned_group(Gone())
        finally:
            os.killpg = original
