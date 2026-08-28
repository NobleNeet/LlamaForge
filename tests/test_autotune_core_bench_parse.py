import conftest_paths  # noqa: F401
import unittest

from autotune_core.bench_parse import BenchParseError, expand_record, parse_structured_output
from test_autotune_core_bench_argv import _case


class TestBenchParse(unittest.TestCase):
    def test_pg_native_expands_samples_to_measurements(self):
        case = _case(mode="pg_native")
        record = {"n_prompt": 32, "n_gen": 8, "n_depth": 0, "samples_ts": [12.0, 14.0], "samples_ns": [100, 200]}
        measurements = expand_record(record, case, ("bench",), 0, "start", "finish", 1.0)
        self.assertEqual([item.repetition for item in measurements], [0, 1])
        self.assertEqual([item.native_tokens_per_second for item in measurements], [12.0, 14.0])

    def test_jsonl_and_malformed_or_mismatched_rejection(self):
        self.assertEqual(len(parse_structured_output('{"n_prompt": 1}\n')), 1)
        with self.assertRaises(BenchParseError):
            parse_structured_output("log line\n")
        with self.assertRaises(BenchParseError):
            expand_record({"n_prompt": 1, "n_gen": 0, "n_depth": 0, "samples_ts": [1], "samples_ns": [1]},
                          _case(), (), 0, "s", "f", 1)

    def test_json_scalars_and_arrays_with_non_objects_are_rejected(self):
        for text in ("null", "1", '"x"', "[{}, null]", "[1]"):
            with self.assertRaises(BenchParseError, msg=text):
                parse_structured_output(text)

    def test_requested_repetition_count_must_match_samples(self):
        record = {"n_prompt": 32, "n_gen": 0, "n_depth": 0, "samples_ts": [1.0], "samples_ns": [1]}
        with self.assertRaisesRegex(BenchParseError, "requested repetitions"):
            expand_record(record, _case(), (), 0, "s", "f", 1, requested_repetitions=2)
