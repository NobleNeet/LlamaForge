import conftest_paths  # noqa: F401
import tempfile
import unittest

from autotune_core.bench_capabilities import BinaryCapabilities
from autotune_core.models import BenchBinaryIdentity, BenchmarkTarget, ExecutionEnvironment
from autotune_core.orchestrator import AutoTuneOrchestrator
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
        self.assertTrue(any(profile.name == "low_memory" and profile.evidence == "heuristic" for profile in outcome.profiles))
