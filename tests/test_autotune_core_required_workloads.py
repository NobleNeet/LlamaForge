import conftest_paths  # noqa: F401
import unittest

from autotune_core.models import BenchBinaryIdentity, ExecutionEnvironment
from autotune_core.planner import Candidate, StageDefinition, StagePlan
from autotune_core.results import BenchmarkCase, BenchmarkMeasurement, BenchmarkWorkload
from autotune_core.scoring import derive_request_latency, rank_candidates


class TestRequiredWorkloads(unittest.TestCase):
    def test_missing_required_tg_excludes_candidate_and_request_is_not_derived(self):
        environment = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/b", None, None, None, "x"), "now")
        candidate = Candidate("candidate", "cpu", {}, environment)
        pp, tg = BenchmarkWorkload("pp", 10, 0, 0), BenchmarkWorkload("tg", 0, 10, 0)
        cases = (BenchmarkCase("pp", "s", "candidate", "cpu", {}, pp, environment),
                 BenchmarkCase("tg", "s", "candidate", "cpu", {}, tg, environment))
        plan = StagePlan(StageDefinition("s", (), (pp, tg), 1, 1), (candidate,), cases)
        measurements = (BenchmarkMeasurement("pp", 0, 100.0, None, exit_code=0),)
        self.assertEqual(rank_candidates(plan, measurements), ())
        request = BenchmarkWorkload("request", 10, 10, 0)
        self.assertIsNone(derive_request_latency("request", request, "pp", pp, measurements, "tg", tg, ()))
