import conftest_paths  # noqa: F401
import unittest

from autotune_core.models import BenchBinaryIdentity, ExecutionEnvironment
from autotune_core.orchestrator import generate_profiles
from autotune_core.planner import Candidate, StageDefinition, StagePlan, stage_outcome
from autotune_core.results import BenchmarkCase, BenchmarkMeasurement, BenchmarkWorkload
from autotune_core.scoring import derive_request_latencies


def _plan(workloads):
    env = ExecutionEnvironment("hw", "cpu", {}, BenchBinaryIdentity("cpu", "/b", None, None, None, "x"), "")
    candidates = (Candidate("one", "cpu", {"threads": 1}, env), Candidate("two", "cpu", {"threads": 2}, env))
    cases = tuple(BenchmarkCase("%s-%s" % (candidate.candidate_id, index), "s", candidate.candidate_id, "cpu", candidate.settings,
                                workload, env) for candidate in candidates for index, workload in enumerate(workloads) if workload.mode != "request")
    return StagePlan(StageDefinition("s", (), tuple(workloads), 2, 4), candidates, cases)


class TestPhase41(unittest.TestCase):
    def test_multiple_depth_specialist_profiles_use_relative_scores(self):
        workloads = (BenchmarkWorkload("pp", 8, 0, 0), BenchmarkWorkload("pp", 8, 0, 4096),
                     BenchmarkWorkload("tg", 0, 8, 0), BenchmarkWorkload("tg", 0, 8, 4096))
        plan = _plan(workloads)
        measurements = []
        for case in plan.cases:
            rate = 100.0 if case.candidate_id == "one" and case.workload.context_depth == 0 else 10.0
            measurements.append(BenchmarkMeasurement(case.case_id, 0, rate if case.workload.mode == "pp" else None,
                                                     rate if case.workload.mode == "tg" else None, exit_code=0))
        profiles = generate_profiles(plan, tuple(measurements), lambda candidate: None)
        self.assertEqual({profile.name for profile in profiles}, {"balanced", "fast_prefill", "fast_decode"})
        outcome = stage_outcome(plan, tuple(measurements))
        self.assertEqual(len(outcome.objective_scores["decode"]), 2)

    def test_derived_request_latency_requires_both_aggregates(self):
        pp, tg, request = BenchmarkWorkload("pp", 8, 0, 0), BenchmarkWorkload("tg", 0, 4, 0), BenchmarkWorkload("request", 8, 4, 0)
        plan = _plan((pp, tg, request))
        measurements = tuple(BenchmarkMeasurement(case.case_id, 0, 8.0 if case.workload.mode == "pp" else None,
                                                   4.0 if case.workload.mode == "tg" else None, exit_code=0) for case in plan.cases)
        derived = derive_request_latencies(plan, measurements)
        self.assertEqual(len(derived), 2)
        self.assertEqual(derived[0].latency_seconds, 2.0)
        self.assertEqual(derive_request_latencies(plan, tuple(item for item in measurements if "-1" not in item.case_id)), ())

    def test_missing_one_required_depth_makes_candidate_ineligible(self):
        workloads = (BenchmarkWorkload("tg", 0, 8, 0), BenchmarkWorkload("tg", 0, 8, 4096))
        plan = _plan(workloads)
        measurements = tuple(BenchmarkMeasurement(case.case_id, 0, None, 100.0, exit_code=0)
                             for case in plan.cases if case.case_id != "one-1")
        outcome = stage_outcome(plan, measurements)
        self.assertEqual([score.candidate_id for score in outcome.objective_scores["decode"]], ["two"])
