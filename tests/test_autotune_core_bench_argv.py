import conftest_paths  # noqa: F401
import unittest

from autotune_core.bench_argv import build_bench_argv
from autotune_core.bench_knobs import UnsupportedBenchKnobError
from autotune_core.models import BenchBinaryIdentity, BenchmarkTarget, ExecutionEnvironment
from autotune_core.results import BenchmarkCase, BenchmarkWorkload


def _case(settings=None, mode="pp"):
    env = ExecutionEnvironment("hw", "hip", {}, BenchBinaryIdentity("hip", "/bin/llama-bench", "b", "f", "v", "artifact"), "now")
    return BenchmarkCase("case", "stage", "candidate", "hip", settings or {}, BenchmarkWorkload(mode, 32, 8, 0), env)


class TestBenchArgv(unittest.TestCase):
    def test_argv_is_data_not_shell_and_normalizes_knobs(self):
        target = BenchmarkTarget("/models/a;touch nope.gguf", "fingerprint")
        argv = build_bench_argv(target, _case({"n-gpu-layers": "all", "flash-attn": "ON", "tensor-split": [0.5, 0.5]}), 3)
        self.assertIn("/models/a;touch nope.gguf", argv)
        self.assertIn("-1", argv)
        self.assertIn("on", argv)
        self.assertIn("0.5/0.5", argv)
        self.assertNotIn("sh", argv)

    def test_server_only_knob_is_rejected(self):
        with self.assertRaises(UnsupportedBenchKnobError):
            build_bench_argv(BenchmarkTarget("/m.gguf", "f"), _case({"ctx-size": "4096"}), 1)

    def test_request_is_not_executable(self):
        with self.assertRaisesRegex(ValueError, "not executable"):
            build_bench_argv(BenchmarkTarget("/m.gguf", "f"), _case(mode="request"), 1)
