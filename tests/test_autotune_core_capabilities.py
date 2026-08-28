import conftest_paths  # noqa: F401
import subprocess
import unittest

from autotune_core.bench_capabilities import CapabilityProbeError, parse_binary_capabilities, probe_binary_capabilities
from autotune_core.bench_argv import build_bench_argv
from autotune_core.models import BenchBinaryIdentity, BenchmarkTarget, ExecutionEnvironment
from autotune_core.results import BenchmarkCase, BenchmarkWorkload


class TestCapabilities(unittest.TestCase):
    def test_help_selects_json_when_jsonl_is_unavailable(self):
        caps = parse_binary_capabilities("--output json --repetitions --n-depth --threads")
        self.assertEqual(caps.structured_output_format(), "json")
        env = ExecutionEnvironment("h", "cpu", {}, BenchBinaryIdentity("cpu", "/bench", None, None, None, "x"), "")
        case = BenchmarkCase("c", "s", "x", "cpu", {"threads": 2}, BenchmarkWorkload("pp", 4, 0, 0), env)
        self.assertIn("json", build_bench_argv(BenchmarkTarget("/m", "f"), case, 2, caps))

    def test_probe_timeout_isolated(self):
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 1)
        with self.assertRaises(CapabilityProbeError):
            probe_binary_capabilities("/bench", runner=timeout)
