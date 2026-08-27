import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import routes, server


class ApiIdleUnloadTest(unittest.TestCase):
    def setUp(self):
        server._reset_api_idle_state()
        self.addCleanup(server._reset_api_idle_state)

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


if __name__ == "__main__":
    unittest.main()
