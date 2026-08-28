import conftest_paths  # noqa: F401
import unittest

from autotune_core.models import BenchBinaryIdentity, ExecutionEnvironment
from autotune_core.planner import Candidate, ParameterSpace, StageDefinition, StagePlan
from autotune_core.results import BenchmarkCase, BenchmarkMeasurement, BenchmarkWorkload
from autotune_core.scoring import aggregate_case_measurements, rank_candidates


class TestScoring(unittest.TestCase):
    def test_repetitions_aggregate_to_median_and_ignore_failed_runs(self):
        workload = BenchmarkWorkload("tg", 0, 128, 0)
        measurements = [
            BenchmarkMeasurement("case", 0, None, 10.0, exit_code=0),
            BenchmarkMeasurement("case", 1, None, 100.0, exit_code=1, error="failed"),
            BenchmarkMeasurement("case", 2, None, 30.0, exit_code=0),
            BenchmarkMeasurement("case", 3, None, 20.0, exit_code=0),
        ]
        self.assertEqual(aggregate_case_measurements(workload, measurements), 20.0)

    def test_pg_uses_token_mix_latency_not_raw_pp_tg_average(self):
        workload = BenchmarkWorkload("pg_native", 1000, 10, 0)
        measurements = [BenchmarkMeasurement("case", 0, 1000.0, 10.0, exit_code=0)]
        self.assertIsNone(aggregate_case_measurements(workload, measurements))

    def test_mixed_auto_uses_relative_workload_normalization(self):
        environment = ExecutionEnvironment("hardware", "hip", {},
            BenchBinaryIdentity("hip", "/bench", "b", "h", "v", "artifact"), "now")
        definition = StageDefinition("mixed", (), (BenchmarkWorkload("pp", 1000, 0, 0),
                                                     BenchmarkWorkload("tg", 0, 10, 0)), 1, 2)
        first = Candidate("a", "hip", {}, environment)
        second = Candidate("b", "hip", {}, environment)
        cases = (
            BenchmarkCase("a-pp", "mixed", "a", "hip", {}, definition.workloads[0], environment),
            BenchmarkCase("a-tg", "mixed", "a", "hip", {}, definition.workloads[1], environment),
            BenchmarkCase("b-pp", "mixed", "b", "hip", {}, definition.workloads[0], environment),
            BenchmarkCase("b-tg", "mixed", "b", "hip", {}, definition.workloads[1], environment),
        )
        plan = StagePlan(definition, (first, second), cases)
        measurements = (
            BenchmarkMeasurement("a-pp", 0, 1000.0, None, exit_code=0),
            BenchmarkMeasurement("a-tg", 0, None, 10.0, exit_code=0),
            BenchmarkMeasurement("b-pp", 0, 500.0, None, exit_code=0),
            BenchmarkMeasurement("b-tg", 0, None, 50.0, exit_code=0),
        )
        ranked = rank_candidates(plan, measurements, scoring_intent="auto")
        self.assertEqual(ranked[0].candidate_id, "b")
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            rank_candidates(plan, measurements, scoring_intent="throughput")
