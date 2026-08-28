import conftest_paths  # noqa: F401
import unittest

from autotune_core.models import BenchBinaryIdentity, ExecutionEnvironment
from autotune_core.planner import BenchmarkStrategy, ParameterSpace, StageDefinition, initial_stage_plan, next_stage_plan, stage_outcome
from autotune_core.results import BenchmarkMeasurement, BenchmarkWorkload
from autotune_core.rules import ResolvedRules


def _environment(backend, build):
    return ExecutionEnvironment("hardware", backend, {"runtime": build},
                                BenchBinaryIdentity(backend, "/%s/bench" % build, build, build, "v1", "artifact"), "now")


class TestPlanner(unittest.TestCase):
    def test_only_first_stage_is_realized_then_outcome_selects_winners(self):
        strategy = BenchmarkStrategy((
            StageDefinition("coarse", (ParameterSpace("n-gpu-layers", (0, 99)),),
                            (BenchmarkWorkload("tg", 0, 64, 0),), top_k=1, max_candidates=8),
            StageDefinition("refine", (ParameterSpace("threads", (4, 8)),),
                            (BenchmarkWorkload("pp", 64, 0, 1024), BenchmarkWorkload("tg", 0, 64, 1024)),
                            top_k=1, max_candidates=8),
        ))
        rules = ResolvedRules((), (), (), ())
        initial = initial_stage_plan(strategy, rules, (_environment("hip", "hip-build"), _environment("vulkan", "vk-build")))
        self.assertEqual(initial.definition.stage_id, "coarse")
        self.assertEqual(len(initial.candidates), 4)
        self.assertEqual({case.execution_environment.bench_binary.build_id for case in initial.cases}, {"hip-build", "vk-build"})
        measurements = tuple(BenchmarkMeasurement(case.case_id, 0, None, 100.0 if case.backend == "hip" else 10.0, exit_code=0)
                             for case in initial.cases)
        refined = next_stage_plan(strategy, stage_outcome(initial, measurements), rules)
        self.assertEqual(refined.definition.stage_id, "refine")
        self.assertEqual(len(refined.candidates), 2)
        self.assertEqual({candidate.backend for candidate in refined.candidates}, {"hip"})
        self.assertEqual(len(refined.cases), 4)
