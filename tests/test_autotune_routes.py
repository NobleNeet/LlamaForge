import conftest_paths  # noqa: F401
import unittest

import routes


class TestAutoTuneRoutes(unittest.TestCase):
    def test_routes_use_injected_lazy_service(self):
        class Service:
            def start(self, request): return {"run_id": "r", "status": "planned"}
            def status(self, run_id): return {"run_id": run_id, "status": "running"}
            def result(self, run_id): return {"run_id": run_id, "profiles": []}
            def list_runs(self, limit): return []
            def cancel(self, run_id): return {"run_id": run_id, "status": "running"}
        original = routes._AUTOTUNE_SERVICE
        try:
            routes._AUTOTUNE_SERVICE = Service()
            self.assertEqual(routes.post_autotune_start(routes.Req({"model_path": "/m"}))[0], 202)
            self.assertEqual(routes.get_autotune_status(routes.Req(qs={"run_id": "r"}))[1]["status"], "running")
        finally:
            routes._AUTOTUNE_SERVICE = original
