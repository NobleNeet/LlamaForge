import conftest_paths  # noqa: F401
import unittest

from autotune_core.staleness import ProfileIdentity, stale_reasons


class TestStaleness(unittest.TestCase):
    def test_field_wise_identity_and_unrelated_backend_are_distinct(self):
        base = ProfileIdentity("m", "h", {"rocm": "1"}, {"build": "a"}, "cap", "rules", "strategy", "v1", "hip")
        self.assertEqual(stale_reasons(base, base), ())
        changed = ProfileIdentity("m", "h", {"rocm": "2"}, {"build": "a"}, "cap", "rules", "strategy", "v1", "hip")
        self.assertEqual(stale_reasons(base, changed), ("runtime_changed",))
        other = ProfileIdentity("m", "h", {"vulkan": "2"}, {"build": "b"}, "x", "rules", "strategy", "v1", "vulkan")
        self.assertEqual(stale_reasons(base, other), ("runtime_changed",))
        policy = ProfileIdentity("m", "h", {"rocm": "1"}, {"build": "a"}, "cap", "rules", "strategy", "phase4.1-v1", "hip")
        self.assertEqual(stale_reasons(base, policy), ("tuning_policy_changed",))
