"""Route handlers that the old if-chain made unreachable from a test.

Each handler is now a plain function of Req -> (status, payload), so these run
with no socket and no live router.
"""
import conftest_paths  # noqa: F401
import os, tempfile, unittest
from unittest import mock

import config, routes
from routes import Req, ApiError


class ConfigAllowlistTest(unittest.TestCase):
    """/api/config used to be `cfg.update(body)` - an unfiltered merge that let
    a request set server_bin, which /api/schema then executes."""

    def setUp(self):
        self.saved = {}
        self.patch = mock.patch.object(
            config, "update", side_effect=lambda ch: (self.saved.update(ch),
                                                      dict(self.saved))[1])
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_accepts_the_keys_the_ui_sets(self):
        status, out = routes.post_config(Req(body={
            "theme": "dark", "cvd": True, "ui_mode": "advanced",
            "onboarded": True, "auto_load_model": "qwen", "wsl_distro": "Ubuntu",
            "llama_backend": "vulkan"}))
        self.assertEqual(status, 200)
        self.assertEqual(self.saved["theme"], "dark")
        self.assertEqual(self.saved["ui_mode"], "advanced")
        self.assertEqual(self.saved["llama_backend"], "vulkan")
        self.assertNotIn("rejected", out)

    def test_refuses_executable_path_keys(self):
        """The RCE path: set server_bin, then GET /api/schema runs it."""
        for key in ("server_bin", "llama_src", "build_dir", "models_ini",
                    "wiki_dir", "docs_dir"):
            with self.assertRaises(ApiError, msg=key) as cm:
                routes.post_config(Req(body={key: "/tmp/evil"}))
            self.assertEqual(cm.exception.status, 400)
            self.assertEqual(self.saved, {}, f"{key} was written")

    def test_refuses_keys_owned_by_other_routes(self):
        for key in ("router_api_key", "router_host", "presets", "cmake_flags"):
            with self.assertRaises(ApiError, msg=key):
                routes.post_config(Req(body={key: "x"}))
        self.assertEqual(self.saved, {})

    def test_rejects_ill_typed_values(self):
        for body in ({"ui_mode": "root"}, {"theme": "neon"}, {"cvd": "yes"},
                     {"vllm_port": 99999}, {"vllm_port": "8081"},
                     {"model_dirs": "not-a-list"}, {"onboarded": 1},
                     {"llama_backend": "metal"}):
            with self.assertRaises(ApiError, msg=str(body)):
                routes.post_config(Req(body=body))
        self.assertEqual(self.saved, {})

    def test_mixed_body_applies_good_keys_and_names_the_bad(self):
        status, out = routes.post_config(
            Req(body={"theme": "light", "server_bin": "/tmp/evil"}))
        self.assertEqual(status, 200)
        self.assertEqual(self.saved, {"theme": "light"})
        self.assertEqual(out["rejected"], ["server_bin"])

    def test_valid_ports_and_dirs_pass(self):
        routes.post_config(Req(body={"vllm_port": 8081,
                                     "model_dirs": ["/models", "/mnt/d"]}))
        self.assertEqual(self.saved["vllm_port"], 8081)
        self.assertEqual(self.saved["model_dirs"], ["/models", "/mnt/d"])

    def test_model_dirs_are_trimmed_and_deduped(self):
        routes.post_config(Req(body={"model_dirs": [" /models ", "", "/mnt/d", "/models"]}))
        self.assertEqual(self.saved["model_dirs"], ["/models", "/mnt/d"])


class SaveAndPresetTest(unittest.TestCase):
    """/api/save and /api/presets/apply share the write-then-reload sequence
    that was duplicated inline in two branches of the old if-chain."""

    def setUp(self):
        self.calls = []
        self.set_keys = mock.patch.object(config, "set_keys",
                                          side_effect=lambda *a, **k: self.calls.append(("set", a)))
        self.set_keys.start()
        self.addCleanup(self.set_keys.stop)

    def _router(self, loaded_ids=()):
        def fake(path, method="GET", body=None, timeout=30):
            self.calls.append(("router", path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": i, "status": {"value": "loaded"}}
                                      for i in loaded_ids]}
            return 200, {}
        return fake

    def test_save_blank_value_clears_the_key(self):
        with mock.patch.object(routes, "router", self._router()):
            status, out = routes.post_save(Req(body={
                "model": "m", "settings": {"ctx-size": "4096", "temp": "  "}}))
        self.assertEqual(status, 200)
        written = [c for c in self.calls if c[0] == "set"][0][1][1]
        self.assertEqual(written, {"ctx-size": "4096", "temp": None})

    def test_save_clears_alias_when_schema_renames_a_knob(self):
        fake_schema = {"groups": [{"name": "common", "knobs": [{
            "key": "n-gpu-layers", "aliases": ["n-gpu-layers", "gpu-layers"]}]}]}
        with mock.patch.object(routes, "schema", return_value=fake_schema), \
             mock.patch.object(routes, "router", self._router()):
            status, out = routes.post_save(Req(body={
                "model": "m", "settings": {"gpu-layers": "99"}}))
        written = [c for c in self.calls if c[0] == "set"][0][1][1]
        self.assertEqual(written, {"n-gpu-layers": "99", "gpu-layers": None})

    def test_save_unloads_a_running_model_before_reload(self):
        with mock.patch.object(routes, "router", self._router(loaded_ids=["m"])):
            status, out = routes.post_save(
                Req(body={"model": "m", "settings": {"ctx-size": "8192"}}))
        self.assertTrue(out["was_running"])
        paths = [c[1] for c in self.calls if c[0] == "router"]
        self.assertIn("/models/unload", paths)
        self.assertIn("/models?reload=1", paths)

    def test_save_does_not_unload_an_idle_model(self):
        with mock.patch.object(routes, "router", self._router(loaded_ids=[])):
            status, out = routes.post_save(
                Req(body={"model": "m", "settings": {"ctx-size": "8192"}}))
        self.assertFalse(out["was_running"])
        paths = [c[1] for c in self.calls if c[0] == "router"]
        self.assertNotIn("/models/unload", paths)

    def test_apply_unknown_preset_is_a_400(self):
        with mock.patch.object(config, "get_presets", return_value={}):
            with self.assertRaises(ApiError) as cm:
                routes.post_presets_apply(Req(body={"model": "m", "name": "nope"}))
        self.assertEqual(cm.exception.status, 400)

    def test_apply_preset_uses_the_same_reload_sequence_as_save(self):
        presets = {"coding": {"temp": "0.2", "ctx-size": ""}}
        with mock.patch.object(config, "get_presets", return_value=presets), \
             mock.patch.object(routes, "router", self._router(loaded_ids=["m"])):
            status, out = routes.post_presets_apply(
                Req(body={"model": "m", "name": "coding"}))
        self.assertEqual(status, 200)
        self.assertTrue(out["was_running"])
        written = [c for c in self.calls if c[0] == "set"][0][1][1]
        self.assertEqual(written, {"temp": "0.2", "ctx-size": None, "n-gpu-layers": "99"})


class ScanPruneTest(unittest.TestCase):
    """/api/scan/prune deletes models.ini sections - it must never remove one
    whose file is actually present."""

    def setUp(self):
        self.removed = []
        mock.patch.object(config, "remove_section",
                          side_effect=lambda s: (self.removed.append(s), True)[1]).start()
        self.addCleanup(mock.patch.stopall)

    def _no_router(self, *a, **k):
        return 599, {}

    def test_keeps_a_section_whose_file_reappeared(self):
        sections = {"gone": {"model": "/nope/a.gguf"},
                    "back": {"model": "/real/b.gguf"}}
        with mock.patch.object(config, "read_sections", return_value=sections), \
             mock.patch.object(routes, "router", self._no_router), \
             mock.patch.object(os.path, "exists",
                               side_effect=lambda p: p == "/real/b.gguf"):
            status, out = routes.post_scan_prune(Req(body={"ids": ["gone", "back"]}))
        self.assertEqual(out["removed"], ["gone"])
        self.assertEqual(self.removed, ["gone"])

    def test_ignores_unknown_ids(self):
        with mock.patch.object(config, "read_sections", return_value={}), \
             mock.patch.object(routes, "router", self._no_router):
            status, out = routes.post_scan_prune(Req(body={"ids": ["ghost"]}))
        self.assertEqual(out["removed"], [])

    def test_prune_removes_models_outside_scan_roots(self):
        sections = {"keep": {"model": "/models/a.gguf"},
                    "drop": {"model": "/other/b.gguf"}}
        with mock.patch.object(config, "read_sections", return_value=sections), \
             mock.patch.object(routes, "router", self._no_router), \
             mock.patch.object(os.path, "exists", return_value=True):
            status, out = routes.post_scan_prune(
                Req(body={"ids": ["keep", "drop"], "roots": ["/models"]}))
        self.assertEqual(out["removed"], ["drop"])
        self.assertEqual(self.removed, ["drop"])


class ScanRootsTest(unittest.TestCase):
    def test_scan_uses_explicit_roots_from_request(self):
        with mock.patch.object(routes.scanner, "scan", return_value=[{"id": "m"}]) as scan, \
             mock.patch.object(routes, "_scan_prune_candidates", return_value=[]), \
             mock.patch.object(routes, "_remove_models", return_value=[]), \
             mock.patch.object(routes.scanner, "list_drives", return_value=["/default"]):
            status, out = routes.post_scan(Req(body={"roots": [" /a ", "/b", "/a"]}))
        self.assertEqual(status, 200)
        self.assertEqual(out["roots"], ["/a", "/b"])
        scan.assert_called_once_with(["/a", "/b"])

    def test_scan_uses_saved_model_dirs_when_request_omits_roots(self):
        with mock.patch.object(routes, "cfg", return_value={"model_dirs": [" /m1 ", "/m2", "/m1"]}), \
             mock.patch.object(routes.scanner, "scan", return_value=[] ) as scan, \
             mock.patch.object(routes, "_scan_prune_candidates", return_value=[]), \
             mock.patch.object(routes, "_remove_models", return_value=[]), \
             mock.patch.object(routes.scanner, "list_drives", return_value=["/default"]):
            status, out = routes.post_scan(Req(body={}))
        self.assertEqual(status, 200)
        self.assertEqual(out["roots"], ["/m1", "/m2"])
        scan.assert_called_once_with(["/m1", "/m2"])

    def test_scan_falls_back_to_default_roots_when_no_dirs_are_saved(self):
        with mock.patch.object(routes, "cfg", return_value={"model_dirs": []}), \
             mock.patch.object(routes.scanner, "scan", return_value=[] ) as scan, \
             mock.patch.object(routes, "_scan_prune_candidates", return_value=[]), \
             mock.patch.object(routes, "_remove_models", return_value=[]), \
             mock.patch.object(routes.scanner, "list_drives", return_value=["/home/u", "/mnt"]):
            status, out = routes.post_scan(Req(body={}))
        self.assertEqual(status, 200)
        self.assertEqual(out["roots"], ["/home/u", "/mnt"])
        scan.assert_called_once_with(["/home/u", "/mnt"])

    def test_scan_rejects_bad_roots_shape(self):
        with self.assertRaises(ApiError) as cm:
            routes.post_scan(Req(body={"roots": "not-a-list"}))
        self.assertEqual(cm.exception.status, 400)

    def test_scan_removes_configured_models_outside_explicit_roots(self):
        with mock.patch.object(routes.scanner, "scan", return_value=[]), \
             mock.patch.object(routes, "_scan_prune_candidates",
                               return_value=[{"id": "drop", "model": "/other/m.gguf", "reason": "outside scan roots"}]), \
             mock.patch.object(routes, "_remove_models", return_value=["drop"]) as remove, \
             mock.patch.object(routes.scanner, "list_drives", return_value=["/default"]):
            status, out = routes.post_scan(Req(body={"roots": ["/models"]}))
        self.assertEqual(status, 200)
        self.assertEqual(out["removed"], ["drop"])
        remove.assert_called_once_with(["drop"], roots=["/models"])


class ScanMissingScopeTest(unittest.TestCase):
    def _router(self, *a, **k):
        return 200, {"data": [{"id": "drop", "status": {"value": "loaded"}}]}

    def test_missing_lists_out_of_scope_models_when_scan_dirs_are_saved(self):
        sections = {"keep": {"model": "/models/a.gguf"},
                    "drop": {"model": "/other/b.gguf"}}
        with mock.patch.object(routes, "cfg", return_value={"model_dirs": ["/models"]}), \
             mock.patch.object(config, "read_sections", return_value=sections), \
             mock.patch.object(routes, "router", side_effect=self._router), \
             mock.patch.object(os.path, "exists", return_value=True):
            status, out = routes.get_scan_missing(Req())
        self.assertEqual(status, 200)
        self.assertEqual(out["roots"], ["/models"])
        self.assertEqual(out["missing"], [{"id": "drop", "model": "/other/b.gguf",
                                           "reason": "outside scan roots", "loaded": True}])

    def test_missing_omits_out_of_scope_logic_when_no_scan_dirs_are_saved(self):
        sections = {"keep": {"model": "/models/a.gguf"},
                    "drop": {"model": "/other/b.gguf"}}
        with mock.patch.object(routes, "cfg", return_value={"model_dirs": []}), \
             mock.patch.object(config, "read_sections", return_value=sections), \
             mock.patch.object(routes, "router", return_value=(200, {"data": []})), \
             mock.patch.object(os.path, "exists", return_value=True):
            status, out = routes.get_scan_missing(Req())
        self.assertEqual(status, 200)
        self.assertEqual(out["missing"], [])


class AutotuneRefineCleaningTest(unittest.TestCase):
    def test_refine_drops_blank_knobs_before_loading(self):
        writes = []

        def fake_set_keys(section, updates, path=None):
            writes.append((section, dict(updates)))

        def fake_router(path, method="GET", body=None, timeout=30):
            if path == "/models/load":
                return 200, {}
            return 200, {"data": [{"id": "m", "status": {"value": "offline"}}]}

        with mock.patch.object(routes, "_find_model", return_value={"id": "m", "settings": {"model": "/m.gguf"}}), \
             mock.patch.object(routes, "_autotune_recommend",
                               return_value={"knobs": {"ctx-size": "65536"}}), \
             mock.patch.object(routes.autotune, "refine",
                               side_effect=lambda base, intent, load_fn, measure_fn: (
                                   load_fn(base), {"knobs": base, "measurements": {"candidates": [], "chosen_tok_s": 0.0}}
                               )[1]), \
             mock.patch.object(routes.config, "set_keys", side_effect=fake_set_keys), \
             mock.patch.object(routes, "router", side_effect=fake_router):
            out = routes._autotune_refine({"model": "m", "intent": "balanced",
                                           "knobs": {"ctx-size": "65536", "temp": ""}})
        self.assertNotIn("error", out)
        self.assertEqual(writes[0][1]["ctx-size"], "65536")
        self.assertIsNone(writes[0][1]["temp"])

    def test_refine_uses_intent_recommendation_over_current_conflicts(self):
        seen = {}

        def fake_refine(base, intent, load_fn, measure_fn):
            seen["base"] = dict(base)
            return {"knobs": base, "measurements": {"candidates": [], "chosen_tok_s": 0.0}}

        with mock.patch.object(routes, "_find_model", return_value={"id": "m", "settings": {"model": "/m.gguf"}}), \
             mock.patch.object(routes, "_autotune_recommend",
                               return_value={"knobs": {"ctx-size": "16384", "batch-size": "2048",
                                                       "ubatch-size": "512", "flash-attn": "on"}}), \
             mock.patch.object(routes.autotune, "refine", side_effect=fake_refine):
            out = routes._autotune_refine({"model": "m", "intent": "speed",
                                           "knobs": {"ctx-size": "150000", "batch-size": "4096",
                                                     "spec-type": "draft-mtp"}})
        self.assertNotIn("error", out)
        self.assertEqual(seen["base"]["ctx-size"], "16384")
        self.assertEqual(seen["base"]["batch-size"], "2048")
        self.assertEqual(seen["base"]["ubatch-size"], "512")
        self.assertEqual(seen["base"]["spec-type"], "draft-mtp")

    def test_refine_drops_previous_managed_knobs_when_intent_changes(self):
        seen = {}

        def fake_refine(base, intent, load_fn, measure_fn):
            seen["base"] = dict(base)
            return {"knobs": base, "measurements": {"candidates": [], "chosen_tok_s": 0.0}}

        with mock.patch.object(routes, "_find_model", return_value={"id": "m", "settings": {"model": "/m.gguf"}}), \
             mock.patch.object(routes, "_autotune_recommend",
                               return_value={"knobs": {"ctx-size": "65536", "cache-type-k": "q8_0",
                                                       "cache-type-v": "q8_0", "n-gpu-layers": "99"}}), \
             mock.patch.object(routes.autotune, "refine", side_effect=fake_refine):
            out = routes._autotune_refine({"model": "m", "intent": "balanced",
                                           "knobs": {"ctx-size": "16384", "cache-type-k": "f16",
                                                     "cache-type-v": "f16", "batch-size": "4096",
                                                     "ubatch-size": "512", "spec-type": "draft-mtp"}})
        self.assertNotIn("error", out)
        self.assertEqual(seen["base"]["ctx-size"], "65536")
        self.assertEqual(seen["base"]["cache-type-k"], "q8_0")
        self.assertEqual(seen["base"]["cache-type-v"], "q8_0")
        self.assertNotIn("batch-size", seen["base"])
        self.assertNotIn("ubatch-size", seen["base"])
        self.assertEqual(seen["base"]["spec-type"], "draft-mtp")

    def test_refine_load_clears_stale_managed_knobs_from_models_ini(self):
        writes = []

        def fake_set_keys(section, updates, path=None):
            writes.append((section, dict(updates)))

        def fake_router(path, method="GET", body=None, timeout=30):
            if path == "/models/load":
                return 200, {}
            return 200, {"data": [{"id": "m", "status": {"value": "offline"}}]}

        with mock.patch.object(routes, "_find_model", return_value={"id": "m", "settings": {"model": "/m.gguf"}}), \
             mock.patch.object(routes, "_autotune_recommend",
                               return_value={"knobs": {"ctx-size": "65536", "n-gpu-layers": "99"}}), \
             mock.patch.object(routes.autotune, "refine",
                               side_effect=lambda base, intent, load_fn, measure_fn: (
                                   load_fn(base), {"knobs": base, "measurements": {"candidates": [], "chosen_tok_s": 0.0}}
                               )[1]), \
             mock.patch.object(routes.config, "set_keys", side_effect=fake_set_keys), \
             mock.patch.object(routes, "router", side_effect=fake_router):
            out = routes._autotune_refine({"model": "m", "intent": "balanced",
                                           "knobs": {"ctx-size": "16384", "cache-type-k": "f16",
                                                     "cache-type-v": "f16", "batch-size": "4096",
                                                     "ubatch-size": "512", "spec-type": "draft-mtp"}})
        self.assertNotIn("error", out)
        self.assertEqual(writes[0][1]["ctx-size"], "65536")
        self.assertEqual(writes[0][1]["n-gpu-layers"], "99")
        self.assertIsNone(writes[0][1]["cache-type-k"])
        self.assertIsNone(writes[0][1]["cache-type-v"])
        self.assertIsNone(writes[0][1]["batch-size"])
        self.assertIsNone(writes[0][1]["ubatch-size"])
        self.assertEqual(writes[0][1]["spec-type"], "draft-mtp")

    def test_refine_load_reinjects_sticky_mtp_settings(self):
        writes = []

        def fake_set_keys(section, updates, path=None):
            writes.append((section, dict(updates)))

        def fake_router(path, method="GET", body=None, timeout=30):
            if path == "/models/load":
                return 200, {}
            return 200, {"data": [{"id": "m", "status": {"value": "offline"}}]}

        with mock.patch.object(routes, "_find_model", return_value={"id": "m", "settings": {"model": "/m.gguf"}}), \
             mock.patch.object(routes, "_autotune_recommend",
                               return_value={"knobs": {"ctx-size": "65536", "n-gpu-layers": "4"}}), \
             mock.patch.object(routes.autotune, "refine",
                               side_effect=lambda base, intent, load_fn, measure_fn: (
                                   load_fn({"ctx-size": "65536", "n-gpu-layers": "4"}),
                                   {"knobs": base, "measurements": {"candidates": [], "chosen_tok_s": 0.0}}
                               )[1]), \
             mock.patch.object(routes.config, "set_keys", side_effect=fake_set_keys), \
             mock.patch.object(routes, "router", side_effect=fake_router):
            out = routes._autotune_refine({"model": "m", "intent": "balanced",
                                           "knobs": {"spec-type": "draft-mtp",
                                                     "spec-draft-model": "/m/mtp.gguf"}})
        self.assertNotIn("error", out)
        self.assertEqual(writes[0][1]["spec-type"], "draft-mtp")
        self.assertEqual(writes[0][1]["spec-draft-model"], "/m/mtp.gguf")

    def test_refine_load_restores_blank_sticky_mtp_settings(self):
        writes = []

        def fake_set_keys(section, updates, path=None):
            writes.append((section, dict(updates)))

        def fake_router(path, method="GET", body=None, timeout=30):
            if path == "/models/load":
                return 200, {}
            return 200, {"data": [{"id": "m", "status": {"value": "offline"}}]}

        with mock.patch.object(routes, "_find_model", return_value={"id": "m", "settings": {"model": "/m.gguf"}}), \
             mock.patch.object(routes, "_autotune_recommend",
                               return_value={"knobs": {"ctx-size": "65536", "n-gpu-layers": "4"}}), \
             mock.patch.object(routes.autotune, "refine",
                               side_effect=lambda base, intent, load_fn, measure_fn: (
                                   load_fn({"ctx-size": "65536", "n-gpu-layers": "4",
                                            "spec-type": "", "spec-draft-model": ""}),
                                   {"knobs": base, "measurements": {"candidates": [], "chosen_tok_s": 0.0}}
                               )[1]), \
             mock.patch.object(routes.config, "set_keys", side_effect=fake_set_keys), \
             mock.patch.object(routes, "router", side_effect=fake_router):
            out = routes._autotune_refine({"model": "m", "intent": "balanced",
                                           "knobs": {"spec-type": "draft-mtp",
                                                     "spec-draft-model": "/m/mtp.gguf"}})
        self.assertNotIn("error", out)
        self.assertEqual(writes[0][1]["spec-type"], "draft-mtp")
        self.assertEqual(writes[0][1]["spec-draft-model"], "/m/mtp.gguf")

    def test_refine_uses_stored_sticky_settings_when_ui_omits_them(self):
        seen = {}

        def fake_refine(base, intent, load_fn, measure_fn):
            seen["base"] = dict(base)
            return {"knobs": base, "measurements": {"candidates": [], "chosen_tok_s": 0.0}}

        with mock.patch.object(routes, "_find_model", return_value={"id": "m", "settings": {"model": "/m.gguf"}}), \
             mock.patch.object(routes.config, "read_sections",
                               return_value={"m": {"model": "/m.gguf", "spec-type": "draft-mtp",
                                                   "spec-draft-model": "/m/mtp.gguf"}}), \
             mock.patch.object(routes, "_autotune_recommend",
                               return_value={"knobs": {"ctx-size": "65536", "n-gpu-layers": "4"}}), \
             mock.patch.object(routes.autotune, "refine", side_effect=fake_refine):
            out = routes._autotune_refine({"model": "m", "intent": "balanced",
                                           "knobs": {"ctx-size": "65536"}})
        self.assertNotIn("error", out)
        self.assertEqual(seen["base"]["spec-type"], "draft-mtp")
        self.assertEqual(seen["base"]["spec-draft-model"], "/m/mtp.gguf")


class HubAddTest(unittest.TestCase):
    def test_missing_file_is_a_400(self):
        with self.assertRaises(ApiError) as cm:
            routes.post_hub_add(Req(body={"path": "/definitely/not/here.gguf"}))
        self.assertEqual(cm.exception.status, 400)

    def test_registers_every_gguf_in_the_folder(self):
        tmp = tempfile.mkdtemp()
        for n in ("a.gguf", "b.gguf", "notes.txt"):
            open(os.path.join(tmp, n), "w").close()
        seen = {}
        with mock.patch.object(routes.scanner, "build_entries",
                               side_effect=lambda paths: (seen.setdefault("paths", paths),
                                                          [{"id": "a", "model": paths[0]}])[1]), \
             mock.patch.object(config, "set_keys"), \
             mock.patch.object(config, "apply_ctx_defaults"), \
             mock.patch.object(routes, "router", lambda *a, **k: (200, {})):
            status, out = routes.post_hub_add(
                Req(body={"path": os.path.join(tmp, "a.gguf")}))
        self.assertEqual(status, 200)
        self.assertEqual(sorted(os.path.basename(p) for p in seen["paths"]),
                         ["a.gguf", "b.gguf"])       # .txt not registered


class RouteTableTest(unittest.TestCase):
    def test_every_route_maps_to_a_callable(self):
        for table in (routes.GET_ROUTES, routes.POST_ROUTES):
            for path, handler in table.items():
                self.assertTrue(callable(handler), path)
                self.assertTrue(path.startswith("/"), path)

    def test_no_path_is_registered_twice_in_one_table(self):
        for table in (routes.GET_ROUTES, routes.POST_ROUTES):
            self.assertEqual(len(table), len(set(table)))

    def test_vllm_routes_are_all_under_the_gated_prefix(self):
        """server._vllm_gate short-circuits on /api/vllm/ - a vLLM route named
        anything else would run on Linux and fail confusingly."""
        for table in (routes.GET_ROUTES, routes.POST_ROUTES):
            for path, handler in table.items():
                if handler.__name__.startswith(("get_vllm", "post_vllm")):
                    self.assertTrue(path.startswith("/api/vllm/"), path)


if __name__ == "__main__":
    unittest.main()
