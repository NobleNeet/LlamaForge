import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import stats
import routes, server


class ApiIdleUnloadTest(unittest.TestCase):
    def setUp(self):
        server._reset_api_idle_state()
        live_patch = mock.patch.object(stats.TRACKER, "live", {})
        live_patch.start()
        self.addCleanup(live_patch.stop)
        with server._PRESET_SYNC_LOCK:
            server._PRESET_SYNC_PENDING.clear()
        self.addCleanup(server._reset_api_idle_state)
        self.addCleanup(lambda: server._PRESET_SYNC_PENDING.clear())

    def _router(self, path, method="GET", body=None, timeout=30):
        if path == "/models":
            return 200, {"data": [{"id": "m", "status": {"value": "loaded"}}]}
        return 200, {}

    def test_reaper_discovers_resident_model_and_waits_full_timeout(self):
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=self._router):
            self.assertEqual(server._reap_api_idle_models(now=100), [])
            self.assertEqual(server._reap_api_idle_models(now=159), [])
            self.assertEqual(server._reap_api_idle_models(now=160), ["m"])

    def test_load_hook_starts_timer_without_inference(self):
        with mock.patch.object(server.time, "time", return_value=100):
            server._track_loaded_model("m", source="/api/load", backend="llamacpp")
            server._track_loaded_model("other", backend="vllm")
        self.assertEqual(server._API_IDLE_LAST, {"m": 100})
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=self._router):
            self.assertEqual(server._reap_api_idle_models(now=160), ["m"])

    def test_enable_timer_during_request_and_loading_preserves_inflight(self):
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 0}), \
             mock.patch.object(server.time, "time", return_value=100):
            server._track_api_model_begin("m")
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", return_value=(200, {"data": []})):
            self.assertEqual(server._reap_api_idle_models(now=200), [])
        self.assertEqual(server._API_IDLE_INFLIGHT, {"m": 1})
        with mock.patch.object(server.time, "time", return_value=210):
            server._track_loaded_model("m", backend="llamacpp")
        self.assertEqual(server._API_IDLE_INFLIGHT, {"m": 1})
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=self._router):
            self.assertEqual(server._reap_api_idle_models(now=300), [])
            with mock.patch.object(server.time, "time", return_value=310):
                server._track_api_model_end("m")
            self.assertEqual(server._reap_api_idle_models(now=369), [])
            self.assertEqual(server._reap_api_idle_models(now=370), ["m"])

    def test_disabled_timer_does_not_query_or_unload(self):
        server._API_IDLE_LAST["m"] = 100
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 0}), \
             mock.patch.object(routes, "router") as router:
            self.assertEqual(server._reap_api_idle_models(now=1000), [])
        router.assert_not_called()

    def test_router_activity_resets_timer_but_historical_throughput_does_not(self):
        server._API_IDLE_LAST["m"] = 100
        metrics = {"requests_processing": 1, "gen_per_sec": 20}
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=self._router), \
             mock.patch.object(stats.TRACKER, "live", {"models": {"m": metrics}}):
            self.assertEqual(server._reap_api_idle_models(now=200), [])
            metrics["requests_processing"] = 0
            self.assertEqual(server._reap_api_idle_models(now=259), [])
            self.assertEqual(server._reap_api_idle_models(now=260), ["m"])

    def test_request_start_during_router_query_prevents_unload(self):
        server._API_IDLE_LAST["m"] = 100

        def router(*args, **kwargs):
            server._track_api_model_begin("m")
            return self._router(*args, **kwargs)

        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=router) as call:
            self.assertEqual(server._reap_api_idle_models(now=200), [])
        call.assert_called_once_with("/models", timeout=3)

    def test_failed_unload_is_retried(self):
        server._API_IDLE_LAST["m"] = 100

        def router(path, *args, **kwargs):
            return (503, {}) if path == "/models/unload" else self._router(path)

        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=router):
            self.assertEqual(server._reap_api_idle_models(now=200), [])
        self.assertEqual(server._API_IDLE_LAST, {"m": 100})
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=self._router):
            self.assertEqual(server._reap_api_idle_models(now=215), ["m"])

    def test_reaper_unloads_loaded_model_after_idle_timeout(self):
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": "m", "status": {"value": "loaded"}}]}
            if path == "/models/unload":
                return 200, {"ok": True}
            return 200, {}

        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=fake_router), \
             mock.patch.object(server.time, "time", return_value=100.0):
            server._track_api_model_begin("m")
        with mock.patch.object(server.time, "time", return_value=110.0):
            server._track_api_model_end("m")

        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=fake_router):
            unloaded = server._reap_api_idle_models(now=171.0)
        self.assertEqual(unloaded, ["m"])
        self.assertIn(("/models/unload", "POST", {"model": "m"}), calls)

    def test_reaper_skips_models_with_inflight_requests(self):
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": "m", "status": {"value": "loaded"}}]}
            if path == "/models/unload":
                return 200, {"ok": True}
            return 200, {}

        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=fake_router), \
             mock.patch.object(server.time, "time", return_value=100.0):
            server._track_api_model_begin("m")

        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=fake_router):
            unloaded = server._reap_api_idle_models(now=200.0)
        self.assertEqual(unloaded, [])
        self.assertNotIn(("/models/unload", "POST", {"model": "m"}), calls)

    def test_reaper_forgets_models_that_are_no_longer_loaded(self):
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", return_value=(200, {"data": []})), \
             mock.patch.object(server.time, "time", return_value=100.0):
            server._track_api_model_begin("m")
        with mock.patch.object(server.time, "time", return_value=110.0):
            server._track_api_model_end("m")

        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", return_value=(200, {"data": []})):
            unloaded = server._reap_api_idle_models(now=200.0)
        self.assertEqual(unloaded, [])
        self.assertEqual(server._API_IDLE_LAST, {})
        self.assertEqual(server._API_IDLE_INFLIGHT, {})

    def test_preset_sync_defers_while_model_is_busy(self):
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": "m", "status": {"value": "loaded"}}]}
            return 200, {}

        with mock.patch.object(routes, "router", side_effect=fake_router), \
             mock.patch.object(stats.TRACKER, "live",
                               {"loaded_model": "m", "requests_processing": 1,
                                "gen_per_sec": 0.0, "router_up": True}):
            server._schedule_preset_sync("m", source="save")
        self.assertIn("m", server._PRESET_SYNC_PENDING)
        self.assertNotIn(("/models/unload", "POST", {"model": "m"}), calls)
        self.assertNotIn(("/models?reload=1", "GET", None), calls)

    def test_preset_sync_reloads_loaded_idle_model(self):
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": "m", "status": {"value": "loaded"}}]}
            return 200, {}

        with mock.patch.object(routes, "router", side_effect=fake_router), \
             mock.patch.object(stats.TRACKER, "live",
                               {"loaded_model": "m", "requests_processing": 0,
                                "gen_per_sec": 0.0, "router_up": True}):
            server._schedule_preset_sync("m", source="save")
        self.assertNotIn("m", server._PRESET_SYNC_PENDING)
        self.assertEqual(calls, [
            ("/models", "GET", None),
            ("/models/unload", "POST", {"model": "m"}),
            ("/models?reload=1", "GET", None),
            ("/models/load", "POST", {"model": "m"}),
        ])

    def test_preset_sync_busy_check_is_per_model(self):
        # `router_models_max > 1`: a second resident model must not be held
        # "busy" by the traffic of the first one.
        live = {"loaded_model": "a", "loaded_models": ["a", "b"],
                "models": {"a": {"requests_processing": 1, "gen_per_sec": 20.0},
                           "b": {"requests_processing": 0, "gen_per_sec": 0.0}},
                "requests_processing": 1, "gen_per_sec": 20.0, "router_up": True}
        with mock.patch.object(stats.TRACKER, "live", live):
            self.assertTrue(server._preset_sync_busy("a"))
            self.assertFalse(server._preset_sync_busy("b"))

    def test_preset_sync_defers_loaded_model_without_metrics(self):
        # A resident model whose metrics did not come back this poll cannot be
        # proven idle, so the unload is deferred instead of cutting a run short.
        live = {"loaded_model": "b", "loaded_models": ["b"], "models": {},
                "requests_processing": 0, "gen_per_sec": 0.0, "router_up": True}
        with mock.patch.object(stats.TRACKER, "live", live):
            self.assertTrue(server._preset_sync_busy("b"))
            self.assertFalse(server._preset_sync_busy("c"))   # not loaded at all

    def test_preset_sync_refreshes_cache_only_when_model_is_not_loaded(self):
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": "m", "status": {"value": "offline"}}]}
            return 200, {}

        with mock.patch.object(routes, "router", side_effect=fake_router), \
             mock.patch.object(stats.TRACKER, "live",
                               {"loaded_model": None, "requests_processing": 0,
                                "gen_per_sec": 0.0, "router_up": True}):
            server._schedule_preset_sync("m", source="save")
        self.assertNotIn("m", server._PRESET_SYNC_PENDING)
        self.assertEqual(calls, [
            ("/models", "GET", None),
            ("/models?reload=1", "GET", None),
        ])


if __name__ == "__main__":
    unittest.main()
