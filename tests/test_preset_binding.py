"""Bind a preset as a model's default (issue #2).

A binding lives in config.json (preset_bindings: {model_id: preset_name}); the
preset's knobs are materialized into the model's models.ini section when bound
and re-materialized when the preset is edited. Hand edits win by write-order.
"""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest
from unittest import mock

import config, routes


class _ConfigTempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")

    def tearDown(self):
        config.CONFIG = self._saved


class BindingStorageTest(_ConfigTempCase):
    def setUp(self):
        super().setUp()
        config.save_preset("qwopus", "coding", {"temp": "0.2"})

    def test_bind_records_the_pair(self):
        config.bind_preset("qwopus", "coding")
        self.assertEqual(config.get_bindings(), {"qwopus": "coding"})

    def test_bind_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            config.bind_preset("qwopus", "no-such-preset")

    def test_unbind_removes_only_that_model(self):
        config.bind_preset("qwopus", "coding")
        config.save_preset("ornith", "coding", {"temp": "0.3"})
        config.bind_preset("ornith", "coding")
        self.assertTrue(config.unbind_preset("qwopus"))
        self.assertEqual(config.get_bindings(), {"ornith": "coding"})

    def test_unbind_absent_is_false(self):
        self.assertFalse(config.unbind_preset("ghost"))

    def test_bindings_for_preset(self):
        config.bind_preset("qwopus", "coding")
        config.save_preset("ornith", "coding", {"temp": "0.3"})
        config.bind_preset("ornith", "coding")
        config.save_preset("gemma", "chat", {"temp": "0.8"})
        config.bind_preset("gemma", "chat")
        self.assertEqual(sorted(config.bindings_for_preset("coding")), ["ornith", "qwopus"])

    def test_deleting_a_preset_drops_its_bindings(self):
        config.bind_preset("qwopus", "coding")
        config.delete_preset("qwopus", "coding")
        self.assertEqual(config.get_bindings(), {})

    def test_prune_binding_on_model_delete(self):
        config.bind_preset("qwopus", "coding")
        self.assertTrue(config.prune_binding("qwopus"))
        self.assertEqual(config.get_bindings(), {})


class BindMaterializeRouteTest(_ConfigTempCase):
    """Route-level: binding writes the preset's knobs; editing the preset
    re-materializes into every bound model."""

    def setUp(self):
        super().setUp()
        self.ini = os.path.join(self.tmp, "models.ini")
        cfg = config.load(); cfg["models_ini"] = self.ini; config.save(cfg)
        config.set_keys("qwopus", {"model": "/m/q.gguf"})
        config.save_preset("qwopus", "coding", {"temp": "0.2", "top-k": "20"})
        self.applied = []
        # capture materialization instead of touching a live router
        self._orig = routes._apply_knobs_and_reload
        routes._apply_knobs_and_reload = lambda mid, clean: self.applied.append((mid, clean)) or False

    def tearDown(self):
        routes._apply_knobs_and_reload = self._orig
        super().tearDown()

    def _post(self, fn, **body):
        req = mock.Mock(); req.body = body
        return fn(req)

    def test_bind_materializes_the_preset(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        self.assertEqual(config.get_bindings(), {"qwopus": "coding"})
        self.assertEqual(len(self.applied), 1)
        mid, clean = self.applied[0]
        self.assertEqual(mid, "qwopus")
        self.assertEqual(clean.get("temp"), "0.2")
        self.assertEqual(clean.get("n-gpu-layers"), "99")

    def test_editing_a_bound_preset_resyncs_every_model(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        config.set_keys("ornith", {"model": "/m/o.gguf"})
        config.save_preset("ornith", "coding", {"temp": "0.4", "top-k": "15"})
        self._post(routes.post_presets_bind, model="ornith", name="coding")
        self.applied.clear()
        self._post(routes.post_presets_save, model="qwopus", name="coding", settings={"temp": "0.9"})
        self.assertEqual(self.applied, [("qwopus", {"temp": "0.9", "n-gpu-layers": "99"})])

    def test_unbind_leaves_knobs_in_place(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        self.applied.clear()
        self._post(routes.post_presets_bind, model="qwopus", name="")   # unbind
        self.assertEqual(config.get_bindings(), {})
        self.assertEqual(self.applied, [], "unbind must not rewrite the section")

    def test_saving_an_unbound_preset_materializes_nothing(self):
        self._post(routes.post_presets_save, model="qwopus", name="coding", settings={"temp": "0.5"})
        self.assertEqual(self.applied, [])


if __name__ == "__main__":
    unittest.main()
