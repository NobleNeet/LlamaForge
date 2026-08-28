import os
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))


class TestAutoTuneFrontendBoundary(unittest.TestCase):
    def test_models_owns_editor_and_autotune_has_no_circular_import(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        with open(os.path.join(ROOT, "web", "js", "autotune.js"), encoding="utf-8") as handle: autotune = handle.read()
        self.assertIn('from "./autotune.js"', models)
        self.assertNotIn('from "./models.js"', autotune)
        self.assertIn("data-autotune-panel", models)
        self.assertIn("bridge.stage", autotune)

    def test_autotune_live_region_is_separate_from_knob_grid(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        self.assertLess(models.index('class="ed-autotune"'), models.index('class="ed-knobs"'))
        self.assertIn("syncAutoTune(m)", models)
