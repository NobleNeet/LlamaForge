import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import stats
import routes, server


class ApiIdleUnloadTest(unittest.TestCase):
    def setUp(self):
        server._reset_api_idle_state()
        with server._PRESET_SYNC_LOCK:
            server._PRESET_SYNC_PENDING.clear()
        self.addCleanup(server._reset_api_idle_state)
        self.addCleanup(lambda: server._PRESET_SYNC_PENDING.clear())

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
