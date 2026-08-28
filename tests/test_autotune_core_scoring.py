import conftest_paths  # noqa: F401
import unittest

from autotune_core.results import BenchmarkMeasurement, BenchmarkWorkload
from autotune_core.scoring import aggregate_case_measurements


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
