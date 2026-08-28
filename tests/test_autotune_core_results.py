import conftest_paths  # noqa: F401
import unittest

from autotune_core.models import BenchBinaryIdentity, ExecutionEnvironment
from autotune_core.results import BenchmarkCase, BenchmarkMeasurement, BenchmarkWorkload, TuneResult, deserialize_result, serialize_result, validate_result


class TestResults(unittest.TestCase):
    def test_result_round_trip_and_reference_validation(self):
        environment = ExecutionEnvironment("hardware", "hip", {"rocm": "6.3"},
                                           BenchBinaryIdentity("hip", "/bench", "b1", "hash", "v1", "artifact"), "now")
        workload = BenchmarkWorkload("pg", 512, 128, 2048)
        case = BenchmarkCase("case", "stage-1", "candidate", "hip", {"threads": 8}, workload, environment)
        measurement = BenchmarkMeasurement("case", 1, 100.0, 50.0, ("/bench", "--json"), "stdout",
                                           {"samples": []}, "stderr", 0, "start", "finish", 4.0)
        result = TuneResult("r1", "m1", "hardware", (case,), (measurement,))
        self.assertEqual(validate_result(result), [])
        self.assertEqual(deserialize_result(serialize_result(result)), result)
        invalid = TuneResult("r1", "m1", "hardware", (), (BenchmarkMeasurement("missing", 0, None, None),))
        self.assertIn("unknown case", validate_result(invalid)[0])
