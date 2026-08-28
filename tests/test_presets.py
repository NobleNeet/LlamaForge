import conftest_paths  # noqa: F401
import os, tempfile, unittest
import config


class TestPresets(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.dir, "config.json")
        self._orig = config.CONFIG
        config.CONFIG = self.cfg_path

    def tearDown(self):
        config.CONFIG = self._orig

    def test_empty_by_default(self):
        self.assertEqual(config.get_presets("m"), {})

    def test_save_and_read_back(self):
        config.save_preset("m", "coding", {"temp": "0.2", "top-p": "0.9"})
        presets = config.get_presets("m")
        self.assertEqual(presets["coding"], {"temp": "0.2", "top-p": "0.9", "n-gpu-layers": "99"})

    def test_blank_values_dropped(self):
        config.save_preset("m", "fast", {"temp": "0.7", "top-k": "  ", "n-gpu-layers": ""})
        self.assertEqual(config.get_presets("m")["fast"], {"temp": "0.7", "n-gpu-layers": "99"})

    def test_overwrite_existing(self):
        config.save_preset("m", "x", {"temp": "0.1"})
        config.save_preset("m", "x", {"temp": "0.9"})
        self.assertEqual(config.get_presets("m")["x"], {"temp": "0.9", "n-gpu-layers": "99"})

    def test_models_have_independent_preset_namespaces(self):
        config.save_preset("alpha", "coding", {"temp": "0.1"})
        config.save_preset("beta", "coding", {"temp": "0.9"})
        self.assertEqual(config.get_presets("alpha")["coding"]["temp"], "0.1")
        self.assertEqual(config.get_presets("beta")["coding"]["temp"], "0.9")

    def test_blank_name_rejected(self):
        with self.assertRaises(ValueError):
            config.save_preset("m", "  ", {"temp": "0.1"})

    def test_blank_model_rejected(self):
        with self.assertRaises(ValueError):
            config.save_preset("  ", "x", {"temp": "0.1"})

    def test_delete(self):
        config.save_preset("m", "gone", {"temp": "0.1"})
        self.assertTrue(config.delete_preset("m", "gone"))
        self.assertNotIn("gone", config.get_presets("m"))
        self.assertFalse(config.delete_preset("m", "gone"))

    def test_survives_non_dict_presets_field(self):
        config.save({**config.load(), "presets": "corrupt"})
        self.assertEqual(config.get_presets("m"), {})
        config.save_preset("m", "ok", {"temp": "0.5"})
        self.assertEqual(config.get_presets("m")["ok"], {"temp": "0.5", "n-gpu-layers": "99"})

    def test_legacy_global_presets_are_visible_until_model_writes(self):
        config.save({**config.load(), "presets": {"legacy": {"temp": "0.4", "n-gpu-layers": "99"}}})
        self.assertIn("legacy", config.get_presets("m"))
        config.save_preset("m", "mine", {"temp": "0.7"})
        self.assertIn("legacy", config.get_presets("m"))
        self.assertIn("mine", config.get_presets("m"))

    def test_model_bucket_deep_copies_legacy_presets(self):
        config.save({**config.load(), "presets": {"legacy": {"temp": "0.4", "top-k": "20"}}})
        config.save_preset("m", "mine", {"temp": "0.7"})
        saved = config.load()
        saved["model_presets"]["m"]["legacy"]["temp"] = "0.9"
        self.assertEqual(saved["presets"]["legacy"]["temp"], "0.4")


if __name__ == "__main__":
    unittest.main()
