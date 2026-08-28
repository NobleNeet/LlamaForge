import conftest_paths  # noqa: F401
import unittest

from autotune_core.models import TuneProfile
from autotune_core.profiles import resolve_effective_settings


class TestProfiles(unittest.TestCase):
    def test_explicit_knobs_override_selected_profile_without_mutating_it(self):
        profile = TuneProfile("p1", "model", "environment", {"ctx-size": "8192", "threads": "12"})
        result = resolve_effective_settings({"ctx-size": "4096", "flash-attn": "off"}, profile,
                                            {"threads": "16"})
        self.assertEqual(result, {"ctx-size": "8192", "flash-attn": "off", "threads": "16"})
        self.assertEqual(profile.settings["threads"], "12")
