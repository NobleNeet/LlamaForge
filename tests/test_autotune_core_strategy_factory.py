import conftest_paths  # noqa: F401
import unittest

from autotune_core.bench_capabilities import BinaryCapabilities
from autotune_core.models import BenchBinaryIdentity, ExecutionEnvironment, NormalizedGGUF, PreparedEnvironment
from autotune_core.strategy_factory import build_production_strategy


class TestProductionStrategy(unittest.TestCase):
    def test_unsupported_flash_carries_parent_stream_and_request_is_derived(self):
        env = ExecutionEnvironment("h", "cpu", {}, BenchBinaryIdentity("cpu", "/b", None, "f", None, "x"), "")
        prepared = PreparedEnvironment(env, BinaryCapabilities(frozenset(), frozenset({"json"}), True, True, "c"), None, "f")
        gguf = NormalizedGGUF("/m", "llama", None, None, 8192, None, None, None, None, None, None, None, None, False)
        strategy = build_production_strategy(gguf, None, (prepared,), None)
        flash = strategy.stages[1].parameters[2]
        self.assertFalse(flash.applicable(type("C", (), {"execution_environment": env})()))
        self.assertTrue(any(workload.mode == "request" for workload in strategy.stages[2].workloads))
