"""The engine abstraction.

These tests exist mostly to pin the contract a third engine (ik-llama) will have
to satisfy: the registry must route a model to its owner without the caller
knowing which engine that is, and each engine must answer the same verbs.
"""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import backends


class FakeDeps:
    """The handful of routes.py functions a backend reaches for."""

    def __init__(self, models=None, router_port=8080, globals_=None, config=None):
        self._models = models or []
        self._port = router_port
        self._globals = globals_ or {}
        self._config = config or {}
        self.router_calls = []
        self.mgr = mock.Mock()
        self.mgr.status.return_value = []
        self.dl = mock.Mock()

    def cfg(self):
        return dict(self._config, router_port=self._port)

    def ik_schema(self):
        return {"groups": [], "count": 0}

    def model_state(self):
        import copy
        return {"models": copy.deepcopy(self._models), "global": dict(self._globals)}

    def router(self, path, method="GET", body=None, timeout=30):
        self.router_calls.append((path, method, body))
        return 200, {}

    def schema(self):
        return {"groups": [], "count": 0}

    def vllm_schema(self):
        return {"groups": [], "count": 0}

    def vllm_mgr(self):
        return self.mgr

    def vllm_dl(self):
        return self.dl

    def _apply_knobs_and_reload(self, mid, knobs):
        self.router_calls.append(("apply", mid, knobs))
        return True

    def _prepare_model_for_load(self, mid):
        self.router_calls.append(("/models?reload=1", "GET", None))


class LlamaCppBackendTest(unittest.TestCase):
    def setUp(self):
        self.deps = FakeDeps(models=[
            {"id": "qwen", "status": "loaded"},
            {"id": "phi", "status": "offline"}], globals_={"ctx-size": "4096"})
        self.be = backends.LlamaCppBackend(self.deps)

    def test_rows_are_tagged_and_loaded_rows_get_an_endpoint(self):
        rows = self.be.list_models()
        self.assertTrue(all(r["backend"] == "llamacpp" for r in rows))
        loaded = next(r for r in rows if r["id"] == "qwen")
        self.assertEqual(loaded["endpoint"], "http://127.0.0.1:8080")
        self.assertNotIn("endpoint", next(r for r in rows if r["id"] == "phi"))

    def test_state_returns_models_and_globals_in_one_call(self):
        st = self.be.state()
        self.assertEqual(st["global"], {"ctx-size": "4096"})
        self.assertEqual(len(st["models"]), 2)

    def test_load_and_unload_hit_the_router(self):
        self.assertEqual(self.be.load("qwen"), (True, ""))
        self.assertEqual(self.be.unload("qwen"), (True, ""))
        paths = [c[0] for c in self.deps.router_calls]
        self.assertEqual(paths, ["/models?reload=1", "/models/load", "/models/unload"])

    def test_save_goes_through_the_shared_reload_sequence(self):
        out = self.be.save("qwen", {"ctx-size": "8192"})
        self.assertFalse(out["restarted"])     # llama.cpp unloads, doesn't restart
        self.assertTrue(out["was_running"])
        self.assertIn(("apply", "qwen", {"ctx-size": "8192"}), self.deps.router_calls)

    def test_delete_is_unsupported_and_says_why(self):
        with self.assertRaises(backends.Unsupported):
            self.be.delete("qwen")


class VllmBackendTest(unittest.TestCase):
    def setUp(self):
        self.deps = FakeDeps()
        self.be = backends.VllmBackend(self.deps)

    def test_unavailable_off_windows(self):
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            self.assertFalse(self.be.available())
        with mock.patch.object(backends.osplat, "IS_WIN", True):
            self.assertTrue(self.be.available())

    def test_rows_come_from_the_registry_and_are_tagged(self):
        with mock.patch.object(backends.vllm_registry, "models", return_value=["org/m"]), \
             mock.patch.object(backends.vllm_registry, "load",
                               return_value={"org/m": {"settings": {"x": "1"},
                                                       "size_bytes": 2 * 1024**3}}), \
             mock.patch.object(backends.vllm_registry, "effective_settings",
                               return_value={"max-model-len": "8192"}), \
             mock.patch.object(backends.osplat, "IS_WIN", True):
            rows = self.be.list_models()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["backend"], "vllm")
        self.assertEqual(rows[0]["status"], "offline")
        self.assertEqual(rows[0]["eff_ctx"], "8192")
        self.assertEqual(rows[0]["file_gib"], 2.0)

    def test_running_instance_maps_to_loaded_with_an_endpoint(self):
        self.deps.mgr.status.return_value = [
            {"model_id": "org/m", "state": "ready",
             "endpoint": "http://127.0.0.1:8081"}]
        with mock.patch.object(backends.vllm_registry, "models", return_value=["org/m"]), \
             mock.patch.object(backends.vllm_registry, "load", return_value={"org/m": {}}), \
             mock.patch.object(backends.vllm_registry, "effective_settings", return_value={}), \
             mock.patch.object(backends.osplat, "IS_WIN", True):
            rows = self.be.list_models()
        self.assertEqual(rows[0]["status"], "loaded")
        self.assertEqual(rows[0]["endpoint"], "http://127.0.0.1:8081")

    def test_load_rejects_an_unregistered_model(self):
        with mock.patch.object(backends.vllm_registry, "load", return_value={}):
            ok, err = self.be.load("nope")
        self.assertFalse(ok)
        self.assertIn("unknown", err)

    def test_save_restarts_a_running_model(self):
        self.deps.mgr.status.return_value = [{"model_id": "m", "state": "ready"}]
        with mock.patch.object(backends.vllm_registry, "set_settings") as setter, \
             mock.patch.object(backends.vllm_registry, "load", return_value={"m": {}}), \
             mock.patch.object(backends.vllm_registry, "effective_settings", return_value={}):
            out = self.be.save("m", {"max-model-len": "4096"})
        self.assertTrue(out["restarted"])
        setter.assert_called_once()
        self.deps.mgr.stop.assert_called_once_with("m")
        self.deps.mgr.start.assert_called_once()

    def test_save_does_not_restart_an_idle_model(self):
        self.deps.mgr.status.return_value = []
        with mock.patch.object(backends.vllm_registry, "set_settings"):
            out = self.be.save("m", {"max-model-len": "4096"})
        self.assertFalse(out["restarted"])
        self.deps.mgr.start.assert_not_called()

    def test_delete_removes_the_registry_entry_only_on_success(self):
        self.deps.dl.delete.return_value = (False, "busy")
        with mock.patch.object(backends.vllm_registry, "remove") as rm:
            ok, err = self.be.delete("m")
        self.assertFalse(ok)
        rm.assert_not_called()

        self.deps.dl.delete.return_value = (True, "")
        with mock.patch.object(backends.vllm_registry, "remove") as rm:
            ok, err = self.be.delete("m")
        self.assertTrue(ok)
        rm.assert_called_once_with("m")


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.deps = FakeDeps(models=[{"id": "qwen", "status": "offline"}])
        self.reg = backends.Registry(self.deps)

    def test_enabled_excludes_unavailable_engines(self):
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            names = [b.name for b in self.reg.enabled()]
        self.assertEqual(names, ["llamacpp"])

    def test_state_merges_engines_and_sorts_loaded_first(self):
        self.deps._models = [{"id": "b-model", "status": "offline"},
                             {"id": "a-model", "status": "loaded"}]
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            st = self.reg.state()
        self.assertEqual([m["id"] for m in st["models"]], ["a-model", "b-model"])

    def test_for_model_trusts_a_valid_hint_without_listing_models(self):
        with mock.patch.object(backends.osplat, "IS_WIN", True):
            be = self.reg.for_model("anything", hint="vllm")
        self.assertEqual(be.name, "vllm")

    def test_for_model_ignores_a_hint_for_an_unavailable_engine(self):
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            be = self.reg.for_model("qwen", hint="vllm")
        self.assertEqual(be.name, "llamacpp")

    def test_for_model_looks_the_id_up_without_a_hint(self):
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            self.assertEqual(self.reg.for_model("qwen").name, "llamacpp")

    def test_unknown_id_falls_back_to_llamacpp(self):
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            self.assertEqual(self.reg.for_model("ghost").name, "llamacpp")


class EngineSelectionTest(unittest.TestCase):
    """The engine question is answered from injected deps, never by reaching
    around them into the real config.json - otherwise a developer's own machine
    decides what the tests see."""

    def _reg(self, **cfg):
        return backends.Registry(FakeDeps(
            models=[{"id": "qwen", "status": "offline"}], config=cfg))

    def test_ikllama_is_unavailable_without_a_binary(self):
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            names = [b.name for b in self._reg().enabled()]
        self.assertEqual(names, ["llamacpp"])

    def test_ikllama_availability_reads_injected_config_not_the_real_one(self):
        """Point deps at a file that exists; the engine must light up from THAT,
        with no reference to the config.json on this machine."""
        with mock.patch.object(backends.os.path, "exists", return_value=True), \
             mock.patch.object(backends.osplat, "IS_WIN", False):
            reg = self._reg(ik_llama_server_bin="/nowhere/llama-server")
            names = [b.name for b in reg.enabled()]
        self.assertEqual(names, ["llamacpp", "ikllama"])

    def test_state_uses_the_active_engine_from_deps(self):
        with mock.patch.object(backends.os.path, "exists", return_value=True), \
             mock.patch.object(backends.osplat, "IS_WIN", False):
            reg = self._reg(active_engine="ikllama",
                            ik_llama_server_bin="/nowhere/llama-server")
            st = reg.state()
        self.assertEqual([m["backend"] for m in st["models"]], ["ikllama"])

    def test_state_lists_each_model_once_not_per_llama_engine(self):
        """Both llama engines drive the same router, so a model must not appear
        twice just because ik_llama happens to be built."""
        with mock.patch.object(backends.os.path, "exists", return_value=True), \
             mock.patch.object(backends.osplat, "IS_WIN", False):
            st = self._reg(ik_llama_server_bin="/nowhere/llama-server").state()
        self.assertEqual([m["id"] for m in st["models"]], ["qwen"])

    def test_an_unknown_active_engine_falls_back_instead_of_raising(self):
        """/api/state polls every few seconds; a typo in config.json must not
        turn the whole dashboard into a 500."""
        with mock.patch.object(backends.osplat, "IS_WIN", False):
            st = self._reg(active_engine="typo").state()
        self.assertEqual([m["backend"] for m in st["models"]], ["llamacpp"])

    def test_every_backend_satisfies_the_documented_verbs(self):
        """What a third engine has to implement."""
        for be in self._reg().all():
            for verb in ("available", "list_models", "schema", "load", "unload",
                         "save", "delete"):
                self.assertTrue(callable(getattr(be, verb, None)),
                                f"{be.name} is missing {verb}")
            self.assertTrue(be.name)


if __name__ == "__main__":
    unittest.main()
