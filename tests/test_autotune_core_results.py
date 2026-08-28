import conftest_paths  # noqa: F401
import unittest

from autotune_core.results import BenchmarkCase, BenchmarkMeasurement, TuneResult, deserialize_result, serialize_result, validate_result


class TestResults(unittest.TestCase):
    def test_result_round_trip_and_reference_validation(self):
        result = TuneResult("r1", "m1", "e1", (BenchmarkCase("case", "hip", {"threads": 8}),),
                            (BenchmarkMeasurement("case", 100.0, 50.0),))
        self.assertEqual(validate_result(result), [])
        self.assertEqual(deserialize_result(serialize_result(result)), result)
        invalid = TuneResult("r1", "m1", "e1", (), (BenchmarkMeasurement("missing", None, None),))
        self.assertIn("unknown case", validate_result(invalid)[0])
