import conftest_paths  # noqa: F401
import tempfile
import unittest
from unittest.mock import patch

from autotune_core.bench_capabilities import BinaryCapabilities
from autotune_core.models import BenchBinaryIdentity, BenchmarkTarget, ExecutionEnvironment
from autotune_core.orchestrator import AutoTuneOrchestrator, generate_profiles
from autotune_core.planner import BenchmarkStrategy, ParameterSpace, StageDefinition
from autotune_core.results import BenchmarkMeasurement, BenchmarkWorkload
from autotune_core.rules import ResolvedRules
from autotune_core.run_store import RunStore


class _Runner:
    def run_case(self, run_id, target, case, repetitions, timeout, cancellation, capabilities):
        if case.settings.get("threads") == 1:
            return "failed", (), {"status": "failed"}
        return "completed", (BenchmarkMeasurement(case.case_id, 0, None, 50.0, exit_code=0),), {"status": "completed"}


class TestOrchestrator(unittest.TestCase):
    def test_candidate_failure_is_partial_progress_not_terminal_run_failure(self):
        root = tempfile.mkdtemp()
        store = RunStore(root, instance_id="owner")
        store.create_run("run", {}, {})
        env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/bench", "b", "f", "v", "x"), "")
        strategy = BenchmarkStrategy((StageDefinition("s1", (ParameterSpace("threads", (1, 2)),),
                                                       (BenchmarkWorkload("tg", 0, 8, 0),), 1, 4),))
        capabilities = BinaryCapabilities(frozenset({"--repetitions", "--n-depth"}), frozenset({"json"}), True, True, "cap")
        outcome = AutoTuneOrchestrator(store, _Runner(), capability_probe=lambda _: capabilities).run(
            "run", strategy, ResolvedRules((), (), (), ()), (env,), BenchmarkTarget("/m", "f"), 1, 1)
        self.assertEqual(store.load_manifest("run")["status"], "completed")
        self.assertEqual(store.load_manifest("run")["progress"]["status"], "partial")
        self.assertFalse(any(profile.name == "low_memory" for profile in outcome.profiles))

    def test_last_case_cancellation_cannot_complete_the_run(self):
        class CancellingRunner(_Runner):
            def run_case(self, *args):
                args[5].cancel()
                return super().run_case(*args)
        from autotune_core.bench_runner import CancellationToken
        root, token = tempfile.mkdtemp(), CancellationToken()
        store = RunStore(root, instance_id="owner")
        store.create_run("run", {}, {})
        env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/bench", "b", "f", "v", "x"), "")
        strategy = BenchmarkStrategy((StageDefinition("s", (), (BenchmarkWorkload("tg", 0, 8, 0),), 1, 1),))
        caps = BinaryCapabilities(frozenset({"--repetitions", "--n-depth"}), frozenset({"json"}), True, True, "cap")
        self.assertIsNone(AutoTuneOrchestrator(store, CancellingRunner(), capability_probe=lambda _: caps).run(
            "run", strategy, ResolvedRules((), (), (), ()), (env,), BenchmarkTarget("/m", "f"), 1, 1, token))
        self.assertEqual(store.load_manifest("run")["status"], "cancelled")

    def test_unexpected_exception_marks_owned_run_failed(self):
        class BrokenRunner:
            def run_case(self, *args):
                raise RuntimeError("bug")
        root = tempfile.mkdtemp(); store = RunStore(root, instance_id="owner"); store.create_run("run", {}, {})
        env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/bench", "b", "f", "v", "x"), "")
        strategy = BenchmarkStrategy((StageDefinition("s", (), (BenchmarkWorkload("tg", 0, 8, 0),), 1, 1),))
        caps = BinaryCapabilities(frozenset({"--repetitions", "--n-depth"}), frozenset({"json"}), True, True, "cap")
        with self.assertRaisesRegex(RuntimeError, "bug"):
            AutoTuneOrchestrator(store, BrokenRunner(), capability_probe=lambda _: caps).run(
                "run", strategy, ResolvedRules((), (), (), ()), (env,), BenchmarkTarget("/m", "f"), 1, 1)
        self.assertEqual(store.load_manifest("run")["status"], "failed")

    def test_initial_plan_and_profile_generation_failures_cannot_complete_run(self):
        root = tempfile.mkdtemp(); store = RunStore(root, instance_id="owner"); store.create_run("initial", {}, {})
        env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/bench", "b", "f", "v", "x"), "")
        caps = BinaryCapabilities(frozenset({"--repetitions", "--n-depth"}), frozenset({"json"}), True, True, "cap")
        with self.assertRaises(ValueError):
            AutoTuneOrchestrator(store, _Runner(), capability_probe=lambda _: caps).run(
                "initial", BenchmarkStrategy(()), ResolvedRules((), (), (), ()), (env,), BenchmarkTarget("/m", "f"), 1, 1)
        self.assertEqual(store.load_manifest("initial")["status"], "failed")
        store.create_run("profiles", {}, {})
        strategy = BenchmarkStrategy((StageDefinition("s", (), (BenchmarkWorkload("tg", 0, 8, 0),), 1, 1),))
        with patch("autotune_core.orchestrator.generate_profiles", side_effect=RuntimeError("profiles")):
            with self.assertRaisesRegex(RuntimeError, "profiles"):
                AutoTuneOrchestrator(store, _Runner(), capability_probe=lambda _: caps).run(
                    "profiles", strategy, ResolvedRules((), (), (), ()), (env,), BenchmarkTarget("/m", "f"), 1, 1)
        self.assertEqual(store.load_manifest("profiles")["status"], "failed")

    def test_resource_busy_waits_without_skipping_the_candidate(self):
        class WaitingLease:
            attempts = 0
            def __init__(self, *args, **kwargs): pass
            def acquire(self, *args):
                WaitingLease.attempts += 1
                if WaitingLease.attempts == 1:
                    from autotune_core.resource_lease import ResourceBusyError
                    raise ResourceBusyError("busy")
                return self
            def release(self): pass
        root = tempfile.mkdtemp(); store = RunStore(root, instance_id="owner"); store.create_run("run", {}, {})
        env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/bench", "b", "f", "v", "x"), "")
        strategy = BenchmarkStrategy((StageDefinition("s", (), (BenchmarkWorkload("tg", 0, 8, 0),), 1, 1),))
        caps = BinaryCapabilities(frozenset({"--repetitions", "--n-depth"}), frozenset({"json"}), True, True, "cap")
        AutoTuneOrchestrator(store, _Runner(), capability_probe=lambda _: caps, resource_wait_seconds=0,
                             resource_lease_factory=WaitingLease).run(
            "run", strategy, ResolvedRules((), (), (), ()), (env,), BenchmarkTarget("/m", "f"), 1, 1)
        progress = store.load_manifest("run")["progress"]
        self.assertEqual(progress["counts"]["skipped"], 0)
        self.assertEqual(progress["counts"]["succeeded"], 1)

    def test_stage_history_is_durable_across_three_stage_run(self):
        root = tempfile.mkdtemp(); store = RunStore(root, instance_id="owner"); store.create_run("run", {}, {})
        env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/bench", "b", "f", "v", "x"), "")
        workload = (BenchmarkWorkload("tg", 0, 8, 0),)
        strategy = BenchmarkStrategy(tuple(StageDefinition(stage, (), workload, 1, 1) for stage in ("coarse", "refine", "validate")))
        caps = BinaryCapabilities(frozenset({"--repetitions", "--n-depth"}), frozenset({"json"}), True, True, "cap")
        AutoTuneOrchestrator(store, _Runner(), capability_probe=lambda _: caps).run(
            "run", strategy, ResolvedRules((), (), (), ()), (env,), BenchmarkTarget("/m", "f"), 1, 1)
        progress = store.load_manifest("run")["progress"]
        self.assertEqual(3, progress["stage_count"])
        self.assertEqual(["coarse", "refine", "validate"], [item["stage_id"] for item in progress["stages"]])
        self.assertEqual(["completed", "completed", "completed"], [item["status"] for item in progress["stages"]])
        self.assertEqual([1, 1, 1], [item["counts"]["succeeded"] for item in progress["stages"]])
