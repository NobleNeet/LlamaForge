"""Route handlers that the old if-chain made unreachable from a test.

Each handler is now a plain function of Req -> (status, payload), so these run
with no socket and no live router.
"""
import conftest_paths  # noqa: F401
import contextlib, os, tempfile, unittest
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
            "router_models_max": 4, "llama_backend": "vulkan"}))
        self.assertEqual(status, 200)
        self.assertEqual(self.saved["theme"], "dark")
        self.assertEqual(self.saved["ui_mode"], "advanced")
        self.assertEqual(self.saved["router_models_max"], 4)
        self.assertEqual(self.saved["llama_backend"], "vulkan")
        self.assertNotIn("rejected", out)

    def test_router_models_max_zero_means_unlimited_and_is_valid(self):
        """0 is llama.cpp's "no count limit", not an absent value - and the Setup
        field is where users type it, so it must survive the validator."""
        routes.post_config(Req(body={"router_models_max": 0}))
        self.assertEqual(self.saved, {"router_models_max": 0})

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
                     {"api_idle_unload_minutes": -1},
                     {"router_models_max": -1}, {"router_models_max": "4"},
                     # bool passes isinstance(v, int); a stray True must not
                     # become "keep exactly one model loaded".
                     {"router_models_max": True},
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
                                     "model_dirs": ["/models", "/mnt/d"],
                                     "api_idle_unload_minutes": 7}))
        self.assertEqual(self.saved["vllm_port"], 8081)
        self.assertEqual(self.saved["model_dirs"], ["/models", "/mnt/d"])
        self.assertEqual(self.saved["api_idle_unload_minutes"], 7)

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


class BuildUnloadTest(unittest.TestCase):
    """Pulling and rebuilding llama.cpp overwrites the running binary, so any
    model the router is holding should be unloaded first. These lock in that
    behaviour: the preload unload runs for the llamacpp build, is a genuine
    no-op when nothing is loaded, happens before the build starts, is skipped
    for the ikllama build (which has no router models) and never crashes the
    build when the router will not answer.
    """

    def setUp(self):
        self.timeline = []          # ordered ("unload", mid) / "build" events
        self._loaded = []

        def fake_router(path, method="GET", body=None, timeout=30):
            if path == "/models":
                return 200, {"data": [{"id": i, "status": {"value": "loaded"}}
                                      for i in self._loaded]}
            if path == "/models/unload" and body is not None:
                self.timeline.append(("unload", body["model"]))
            return 200, {}

        def record_build(*a, **k):
            self.timeline.append("build")
            return True

        self.builder = mock.Mock()
        self.builder.start.side_effect = record_build
        self.builder_ik = mock.Mock()
        self.builder_ik.start.side_effect = record_build

        for p in (
            mock.patch.object(routes, "router", side_effect=fake_router),
            mock.patch.object(routes, "BUILDER_LLAMA", self.builder),
            mock.patch.object(routes, "BUILDER_IKLLAMA", self.builder_ik),
            mock.patch.object(config, "update", mock.Mock()),
        ):
            p.start()
        self.addCleanup(mock.patch.stopall)

    def _recommend_cpu(self):
        return mock.patch.object(
            routes.hardware, "recommend",
            return_value={"selected_backend": "cpu", "cmake_flags": {}})

    def test_llamacpp_build_unloads_every_loaded_model_before_building(self):
        self._loaded = ["m1", "m2"]
        with mock.patch.object(routes.BuildManager, "validate_paths", return_value=""), \
             self._recommend_cpu():
            status, out = routes.post_build_start(Req(body={"target": "llamacpp"}))
        self.assertTrue(out["started"])
        # timeline mixes ("unload", mid) tuples with the "build" string event
        loaded = [entry for entry in self.timeline if isinstance(entry, tuple)]
        unloads = [mid for _event, mid in loaded]
        self.assertEqual(unloads, ["m1", "m2"])
        # the build is the last thing that happens - after all unloads
        self.assertEqual(self.timeline[-1], "build")

    def test_llamacpp_build_with_no_loaded_models_does_not_unload(self):
        self._loaded = []
        with mock.patch.object(routes.BuildManager, "validate_paths", return_value=""), \
             self._recommend_cpu():
            status, out = routes.post_build_start(Req(body={"target": "llamacpp"}))
        self.assertTrue(out["started"])
        # the only event is the build itself - nothing to unload
        self.assertEqual(self.timeline, ["build"])

    def test_ikkllama_build_does_not_unload_models(self):
        # ikllama predates router mode, so its rebuild must not touch models.
        self._loaded = ["m1"]
        with mock.patch.object(routes.BuildManager, "validate_paths", return_value=""), \
             self._recommend_cpu():
            status, out = routes.post_build_start(Req(body={"target": "ikkllama"}))
        self.assertTrue(out["started"])
        self.assertEqual(self.timeline, ["build"])

    def test_helper_reports_the_unloaded_ids_and_calls_the_hook(self):
        self._loaded = ["m1", "m2"]
        with mock.patch.object(routes, "MODEL_UNLOAD_HOOK") as hook:
            unloaded = routes._unload_all_models(source="build")
        self.assertEqual(unloaded, ["m1", "m2"])
        self.assertEqual(self.timeline, [("unload", "m1"), ("unload", "m2")])
        first = hook.call_args_list[0]
        self.assertEqual(first.kwargs["source"], "build")
        self.assertEqual(first.kwargs["backend"], "llamacpp")

    def test_helper_returns_empty_when_router_does_not_answer(self):
        def no_router(path, method="GET", body=None, timeout=30):
            return 599, {"error": "no router"}
        with mock.patch.object(routes, "router", side_effect=no_router), \
             mock.patch.object(routes, "MODEL_UNLOAD_HOOK") as hook:
            unloaded = routes._unload_all_models()
        self.assertEqual(unloaded, [])
        hook.assert_not_called()


class ModelLifecycleHookTest(unittest.TestCase):
    def test_load_and_unload_routes_call_hooks_on_success(self):
        seen = []
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append(path)
            if path in ("/models/load", "/models/unload"):
                return 200, {"ok": True}
            return 200, {}

        with mock.patch.object(routes, "router", side_effect=fake_router), \
             mock.patch.object(routes, "MODEL_LOAD_HOOK",
                               side_effect=lambda mid, source="", backend="": seen.append(("load", mid, source, backend))), \
             mock.patch.object(routes, "MODEL_UNLOAD_HOOK",
                               side_effect=lambda mid, source="", backend="": seen.append(("unload", mid, source, backend))):
            status, _ = routes.post_load(Req(body={"model": "m"}, path="/api/load"))
            self.assertEqual(status, 200)
            status, _ = routes.post_unload(Req(body={"model": "m"}, path="/api/unload"))
            self.assertEqual(status, 200)
        self.assertEqual(calls, ["/models?reload=1", "/models/load", "/models/unload"])
        self.assertEqual(seen, [("load", "m", "/api/load", "llamacpp"),
                                ("unload", "m", "/api/unload", "llamacpp")])

    def test_load_with_explicit_settings_uses_them_instead_of_rebinding_default_preset(self):
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            return 200, {"ok": True}

        with mock.patch.object(routes, "router", side_effect=fake_router), \
             mock.patch.object(config, "set_keys") as set_keys, \
             mock.patch.object(config, "read_sections",
                               side_effect=[{"m": {"ctx-size": "0", "batch-size": "4096"}},
                                            {"m": {"ctx-size": "0", "batch-size": "4096"}}]), \
             mock.patch.object(routes, "_prepare_model_for_load") as prepare:
            status, _ = routes.post_load(Req(body={"model": "m",
                                                   "settings": {"ctx-size": "0",
                                                                "batch-size": "4096"}},
                                            path="/api/load"))
        self.assertEqual(status, 200)
        set_keys.assert_called_once_with("m", {"ctx-size": "0", "batch-size": "4096"})
        prepare.assert_not_called()
        self.assertEqual(calls, [("/models?reload=1", "GET", None),
                                 ("/models/load", "POST", {"model": "m"})])


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


class LogRouteTest(unittest.TestCase):
    def test_llama_output_log_reads_stdout_only(self):
        with mock.patch.object(routes, "llama_output_log_tail", return_value="prefill\ntoken\n") as tail:
            status, out = routes.get_llama_output_log(Req())
        self.assertEqual(status, 200)
        self.assertEqual(out["log"], "prefill\ntoken\n")
        tail.assert_called_once_with(400)


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


class NetworkConfigTest(unittest.TestCase):
    def test_network_updates_router_port_and_restarts_on_new_port(self):
        saved = {}
        base = {"router_host": "127.0.0.1", "router_api_key": "", "router_port": 8080,
                "panel_host": "127.0.0.1", "panel_port": 8090, "router_models_max": 4}
        with mock.patch.object(routes, "cfg", return_value={"router_host": "127.0.0.1",
                                                            "router_api_key": "",
                                                            "router_port": 8080,
                                                            "panel_host": "127.0.0.1",
                                                            "panel_port": 8090,
                                                            "router_models_max": 4}), \
             mock.patch.object(config, "update",
                               side_effect=lambda ch: (saved.update(ch), dict(base, **saved))[1]), \
             mock.patch.object(routes, "_active_server_bin", return_value="/bin/llama-server"), \
             mock.patch.object(config, "ini_path", return_value="/tmp/models.ini"), \
             mock.patch.object(routes.router_ctl, "restart", return_value=(True, "")) as restart, \
             mock.patch.object(routes, "PANEL_RESTART") as panel_restart:
            status, out = routes.post_network(Req(body={"host": "0.0.0.0", "port": 9090,
                                                        "panel_host": "0.0.0.0", "api_key": "secret"}))
        self.assertEqual(status, 200)
        self.assertEqual(saved["router_port"], 9090)
        self.assertEqual(saved["router_host"], "0.0.0.0")
        self.assertEqual(saved["panel_host"], "0.0.0.0")
        self.assertEqual(out["port"], 9090)
        self.assertTrue(out["panel_restart_required"])
        restart.assert_called_once_with("/bin/llama-server", "/tmp/models.ini",
                                        9090, "0.0.0.0", "secret", routes.LOGDIR,
                                        # Applied from config.json: restarting the
                                        # router to change host/port must not
                                        # silently reset the loaded-model limit.
                                        models_max=4)
        panel_restart.assert_called_once_with("0.0.0.0", 8090)

    def test_network_keeps_existing_key_when_field_is_omitted(self):
        saved = {}
        base = {"router_host": "127.0.0.1", "router_api_key": "keepme", "router_port": 8080,
                "panel_host": "127.0.0.1", "panel_port": 8090}
        with mock.patch.object(routes, "cfg", return_value={"router_host": "127.0.0.1",
                                                            "router_api_key": "keepme",
                                                            "router_port": 8080,
                                                            "panel_host": "127.0.0.1",
                                                            "panel_port": 8090}), \
             mock.patch.object(config, "update",
                               side_effect=lambda ch: (saved.update(ch), dict(base, **saved))[1]), \
             mock.patch.object(routes, "_active_server_bin", return_value="/bin/llama-server"), \
             mock.patch.object(config, "ini_path", return_value="/tmp/models.ini"), \
             mock.patch.object(routes.router_ctl, "restart", return_value=(True, "")), \
             mock.patch.object(routes, "PANEL_RESTART") as panel_restart:
            status, out = routes.post_network(Req(body={"host": "127.0.0.1", "port": 8181}))
        self.assertEqual(status, 200)
        self.assertEqual(saved["router_api_key"], "keepme")
        self.assertEqual(saved["router_port"], 8181)
        self.assertEqual(saved["panel_host"], "127.0.0.1")
        self.assertFalse(out["panel_restart_required"])
        panel_restart.assert_not_called()

    def test_network_rejects_invalid_port(self):
        with self.assertRaises(ApiError) as cm:
            routes.post_network(Req(body={"host": "127.0.0.1", "port": 70000}))
        self.assertEqual(cm.exception.status, 400)

    def test_network_rejects_invalid_panel_host(self):
        with self.assertRaises(ApiError) as cm:
            routes.post_network(Req(body={"host": "127.0.0.1", "panel_host": "192.168.1.5"}))
        self.assertEqual(cm.exception.status, 400)

    def test_get_network_reports_the_configured_models_max(self):
        """The Setup field needs something to show; llama.cpp never reports its
        own --models-max back out, so the configured value is all there is."""
        with mock.patch.object(routes, "cfg", return_value={"router_host": "127.0.0.1",
                                                            "router_port": 8080,
                                                            "panel_host": "127.0.0.1",
                                                            "panel_port": 8090,
                                                            "router_api_key": "",
                                                            "router_models_max": 0}), \
             mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.9"), \
             mock.patch.object(routes.router_ctl, "is_running", return_value=True):
            status, out = routes.get_network(Req())
        self.assertEqual(status, 200)
        self.assertEqual(out["models_max"], 0)


class RouterRestartTest(unittest.TestCase):
    """`router_models_max` is read once, into the router's command line, so the
    only way to apply it is to restart - and a restart unloads every model. The
    route therefore has to name what it cost, and refuse rather than trade a
    working router for one that cannot start."""

    BASE = {"router_host": "127.0.0.1", "router_api_key": "", "router_port": 8080,
            "router_models_max": 4}

    def _run(self, loaded=("a", "b"), restart=(True, ""), router_mode=True,
             sbin="/bin/llama-server", exists=True):
        seen_hooks, restart_calls = [], []
        rows = [{"id": "default", "status": {"value": "loaded"}}] + \
               [{"id": mid, "status": {"value": "loaded"}} for mid in loaded] + \
              [{"id": "gone", "status": {"value": "offline"}}]

        def fake_restart(server_bin, ini, port, host, key, logdir, models_max=None):
            restart_calls.append(models_max)
            return restart

        with contextlib.ExitStack() as stack:
            for p in (mock.patch.object(routes, "cfg", return_value=dict(self.BASE)),
                      mock.patch.object(routes, "_active_server_bin", return_value=sbin),
                      # Whether the binary is on disk is its own axis: no test
                      # should depend on what this machine happens to have.
                      mock.patch.object(os.path, "exists", return_value=exists),
                      mock.patch.object(config, "ini_path", return_value="/tmp/models.ini"),
                      mock.patch.object(routes.router_ctl, "supports_router_mode",
                                        return_value=router_mode),
                      mock.patch.object(routes.router_ctl, "restart", side_effect=fake_restart),
                      mock.patch.object(routes, "router", return_value=(200, {"data": rows})),
                      mock.patch.object(routes, "MODEL_UNLOAD_HOOK", side_effect=lambda mid, source="", backend="": seen_hooks.append((mid, source)))):
                stack.enter_context(p)
            status, out = routes.post_router_restart(Req(body={}, path="/api/router/restart"))
        return status, out, restart_calls, seen_hooks

    def test_restart_applies_the_configured_limit_and_reports_the_cost(self):
        status, out, restart_calls, hooks = self._run(loaded=("a", "b"))
        self.assertEqual(status, 200)
        self.assertTrue(out["ok"])
        self.assertEqual(restart_calls, [4])          # from config.json, not the old default
        self.assertEqual(out["models_max"], 4)
        self.assertEqual(out["unloaded"], ["a", "b"])  # the UI confirms against this
        # These left with the process, not via /models/unload, so accounting
        # only stays honest if the hook hears about it here.
        self.assertEqual(hooks, [("a", "/api/router/restart"), ("b", "/api/router/restart")])

    def test_restart_refuses_a_binary_without_router_mode(self):
        """Stopping a healthy router to start one that dies would leave nothing."""
        status, out, restart_calls, _ = self._run(router_mode=False)
        self.assertEqual(status, 200)
        self.assertFalse(out["ok"])
        self.assertIn("--models-preset", out["error"])
        self.assertEqual(restart_calls, [])

    def test_restart_refuses_a_missing_binary(self):
        status, out, restart_calls, _ = self._run(exists=False)
        self.assertEqual(status, 200)
        self.assertFalse(out["ok"])
        self.assertEqual(restart_calls, [])

    def test_failed_restart_is_500_and_keeps_the_hooks_quiet(self):
        status, out, _, hooks = self._run(restart=(False, "port still busy"))
        self.assertEqual(status, 500)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "port still busy")
        self.assertEqual(hooks, [])   # nothing actually went away


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
