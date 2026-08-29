import conftest_paths  # noqa: F401
import unittest

from autotune_core.bench_capabilities import BinaryCapabilities
from autotune_core.models import BenchBinaryIdentity, ExecutionEnvironment, NormalizedGGUF, PreparedEnvironment
from autotune_core.orchestrator import AutoTuneOrchestrator
from autotune_core.strategy_factory import build_production_strategy, representative_context, safe_proxy_depth


ALL_FLAGS = frozenset({"--n-gpu-layers", "--threads", "--batch-size", "--ubatch-size", "--flash-attn",
                       "--cache-type-k", "--cache-type-v"})


def _gguf(context=32768):
    return NormalizedGGUF("/m", "llama", None, None, context, None, None, None, None, None, None, None, None, False)


def _prepared(flags=ALL_FLAGS):
    env = ExecutionEnvironment("h", "cpu", {}, BenchBinaryIdentity("cpu", "/b", None, "f", None, "x"), "")
    return PreparedEnvironment(env, BinaryCapabilities(flags, frozenset({"json"}), True, True, "c"), None, "f")


class TestProductionStrategy(unittest.TestCase):
    def test_five_stage_order_and_bounded_case_budget(self):
        strategy = build_production_strategy(_gguf(), None, (_prepared(),), None)
        self.assertEqual(["coarse", "batch_probe", "flash_probe", "kv_probe", "validate"],
                         [stage.stage_id for stage in strategy.stages])
        self.assertEqual(24, sum(stage.max_candidates * sum(workload.mode != "request" for workload in stage.workloads)
                                 for stage in strategy.stages))
        self.assertEqual(("balanced",), strategy.stages[0].retention_objectives)
        self.assertFalse(strategy.stages[0].pareto_retention)
        self.assertEqual(1, strategy.stages[0].top_k)

    def test_workloads_and_native_probe_scoring_are_explicit(self):
        stages = {stage.stage_id: stage for stage in build_production_strategy(_gguf(), None, (_prepared(),), None).stages}
        self.assertEqual([("pp", 256, 0, 0), ("tg", 0, 32, 0)],
                         [(w.mode, w.prompt_tokens, w.generation_tokens, w.context_depth) for w in stages["coarse"].workloads])
        self.assertEqual(("pp", 2048, 0, 0), tuple(getattr(stages["batch_probe"].workloads[0], key)
                         for key in ("mode", "prompt_tokens", "generation_tokens", "context_depth")))
        self.assertEqual("throughput", stages["flash_probe"].scoring_intent)
        self.assertEqual("throughput", stages["kv_probe"].scoring_intent)
        self.assertEqual(("pp", "tg", "request"), tuple(item.mode for item in stages["validate"].workloads))

    def test_proxy_depth_reserves_pg_token_budget(self):
        strategy = build_production_strategy(_gguf(4096), None, (_prepared(),), None)
        representative = representative_context(_gguf(4096))
        flash, kv = strategy.stages[2].workloads[0], strategy.stages[3].workloads[0]
        self.assertEqual(3840, representative)
        self.assertLessEqual(flash.context_depth + flash.prompt_tokens + flash.generation_tokens, representative)
        self.assertLessEqual(kv.context_depth + kv.prompt_tokens + kv.generation_tokens, representative)
        self.assertEqual(1760, flash.context_depth)
        self.assertEqual(3296, kv.context_depth)
        self.assertEqual(4096, safe_proxy_depth(16384, 4096, 2048, 32))
        self.assertEqual(8192, safe_proxy_depth(16384, 8192, 512, 32))

    def test_unsupported_kv_flags_carry_the_parent_stream(self):
        strategy = build_production_strategy(_gguf(), None, (_prepared(frozenset()),), None)
        kv = strategy.stages[3]
        self.assertEqual((), kv.parameters[0].applicable_execution_fingerprints)
        self.assertEqual((), kv.parameters[1].applicable_execution_fingerprints)

    def test_equivalent_strategies_have_stable_fingerprint(self):
        self.assertEqual(AutoTuneOrchestrator.strategy_fingerprint(build_production_strategy(_gguf(), None, (_prepared(),), None)),
                         AutoTuneOrchestrator.strategy_fingerprint(build_production_strategy(_gguf(), None, (_prepared(),), None)))
