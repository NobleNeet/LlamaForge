import conftest_paths  # noqa: F401
import json
import os
import tempfile
import unittest

from autotune_core.models import EnvironmentSnapshot, NormalizedGGUF, PhysicalGPU
from autotune_core.rules import candidate_allowed, load_rule_set, resolve_rules


def _model():
    return NormalizedGGUF(
        path="m.gguf", architecture="llama", name=None, quantization=None,
        context_length=None, embedding_length=None, block_count=None,
        attention_head_count=None, attention_head_count_kv=None,
        sliding_window=None, expert_count=None, expert_used_count=None,
        nextn_predict_layers=None, has_nextn=False,
    )


def _environment():
    gpu = PhysicalGPU("amd:0", "AMD", "GPU", None, "gfx1151", False, ("hip", "vulkan"))
    return EnvironmentSnapshot("linux", {}, (gpu,), ("hip", "vulkan"), "now")


class TestRules(unittest.TestCase):
    def _rule_file(self, rules):
        path = os.path.join(tempfile.mkdtemp(), "rules.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "rules": rules}, handle)
        return path

    def test_constraints_narrow_and_cannot_be_reenabled_by_later_rules(self):
        rule_set = load_rule_set(self._rule_file([
            {"id": "base", "when": {"architectures": ["llama"]},
             "candidate_seeds": [{"backend": "hip", "batch": 512}],
             "hard_constraints": [{"key": "backend", "allowed_values": ["vulkan"]}]},
            {"id": "later", "when": {"backends": ["rocm"]},
             "candidate_seeds": [{"backend": "hip", "batch": 1024}],
             "hard_constraints": [{"key": "batch", "maximum": 768}],
             "exclusions": [{"backend": "vulkan", "batch": 1}]},
        ]))
        resolved = resolve_rules([rule_set], _model(), _environment())
        self.assertEqual(len(resolved.candidate_seeds), 2)
        self.assertFalse(candidate_allowed({"backend": "hip", "batch": 512}, resolved))
        self.assertFalse(candidate_allowed({"backend": "vulkan", "batch": 1024}, resolved))
        self.assertTrue(candidate_allowed({"backend": "vulkan", "batch": 512}, resolved))

    def test_loader_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            load_rule_set(self._rule_file([{"id": "x"}, {"id": "x"}]))
