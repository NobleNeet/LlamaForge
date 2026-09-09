"""Daily schedule, request admission, and idle-only build integration."""
import conftest_paths  # noqa: F401
from datetime import datetime
import tempfile
import threading
import unittest
from unittest import mock

import build_schedule
from builder import BuildManager
import config
import routes
import server
import stats


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {"build_auto_update_enabled": True, "build_auto_update_time": "03:00"}
        self.run = mock.Mock(return_value="Build done")
        self.schedule = build_schedule.BuildSchedule(lambda: dict(self.cfg), self.cfg.update, self.run)
        self.now = datetime(2026, 9, 9, 3, 0, 10)

    def test_disabled_wrong_time_and_no_catch_up(self):
        self.cfg["build_auto_update_enabled"] = False
        self.schedule.tick(self.now)
        self.cfg["build_auto_update_enabled"] = True
        self.schedule.tick(datetime(2026, 9, 9, 2, 59))
        self.schedule.tick(datetime(2026, 9, 9, 3, 1))
        self.run.assert_not_called()

    def test_once_per_day_even_after_restart_or_clock_rollback(self):
        self.schedule.tick(self.now)
        restarted = build_schedule.BuildSchedule(lambda: self.cfg, self.cfg.update, self.run)
        restarted.tick(self.now)
        restarted.tick(datetime(2026, 9, 8, 3, 0))
        self.run.assert_called_once()
        restarted.tick(datetime(2026, 9, 10, 3, 0))
        self.assertEqual(self.run.call_count, 2)

    def test_active_request_skips_entire_day(self):
        with self.schedule.activity() as admitted:
            self.assertTrue(admitted)
            self.schedule.tick(self.now)
        self.schedule.tick(self.now)
        self.run.assert_not_called()
        self.assertIn("Skipped", self.cfg["build_auto_update_status"])

    def test_build_blocks_new_activity_and_exception_releases_gate(self):
        def run():
            with self.schedule.activity() as admitted:
                self.assertFalse(admitted)
            raise RuntimeError("offline")
        self.run.side_effect = run
        self.schedule.tick(self.now)
        self.assertEqual(self.cfg["build_auto_update_status"], "Failed: offline")
        with self.schedule.activity() as admitted:
            self.assertTrue(admitted)

    def test_validation(self):
        for value in ("00:00", "23:59", "03:00"):
            self.assertEqual(build_schedule.valid_time(value), value)
        for value in ("24:00", "03:60", "3:00", "12:00:01", None, 300, True, "03:00\n"):
            self.assertIsNone(build_schedule.valid_time(value))

    def test_config_api_persists_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "CONFIG", tmp + "/config.json"):
            _, result = routes.post_config(routes.Req(body=self.cfg))
            self.assertTrue(result["ok"])
            self.assertEqual(config.load()["build_auto_update_time"], "03:00")
            self.assertTrue(config.load()["build_auto_update_enabled"])
            for patch in ({"build_auto_update_time": "24:01"},
                          {"build_auto_update_enabled": "true"},
                          {"build_auto_update_last_date": "2026-09-09"}):
                with self.assertRaises(routes.ApiError):
                    routes.post_config(routes.Req(body=patch))


class IdleTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {"active_engine": "llamacpp"}
        self.models = [{"id": m, "status": {"value": "loaded"}} for m in ("a", "b")]
        self.metrics = {stats.M_REQ_PROCESSING: 0, "llamacpp:requests_deferred": 0}
        patches = [mock.patch.object(routes, "_AUTOTUNE_SERVICE", None),
                   mock.patch.object(routes, "VLLM_SUPPORTED", False),
                   mock.patch.object(routes.BUILDER_LLAMA, "state", {"running": False}),
                   mock.patch.object(routes.BUILDER_IKLLAMA, "state", {"running": False}),
                   mock.patch.object(routes.router_ctl, "is_running", return_value=True),
                   mock.patch.object(routes, "router", return_value=(200, {"data": self.models})),
                   mock.patch.object(stats.TRACKER, "_scrape", return_value=self.metrics)]
        self.mocks = [p.start() for p in patches]
        for patch in patches:
            self.addCleanup(patch.stop)

    def test_all_loaded_idle_models(self):
        self.assertEqual(routes._scheduled_build_idle(self.cfg), ("", True, ["a", "b"]))
        self.assertEqual(self.mocks[-1].call_count, 2)

    def test_processing_queued_missing_or_malformed_metrics_skip(self):
        for metrics in (None, {}, {stats.M_REQ_PROCESSING: 0},
                        dict(self.metrics, **{stats.M_REQ_PROCESSING: 1}),
                        dict(self.metrics, **{"llamacpp:requests_deferred": 1}),
                        dict(self.metrics, **{stats.M_REQ_PROCESSING: float("nan")})):
            self.mocks[-1].side_effect = [self.metrics, metrics]
            self.assertTrue(routes._scheduled_build_idle(self.cfg)[0])

    def test_loading_unknown_and_unreachable_skip(self):
        self.models[1]["status"]["value"] = "loading"
        self.assertTrue(routes._scheduled_build_idle(self.cfg)[0])
        self.mocks[-2].return_value = (503, {})
        self.assertTrue(routes._scheduled_build_idle(self.cfg)[0])
        self.mocks[-2].return_value = (200, {})
        self.assertTrue(routes._scheduled_build_idle(self.cfg)[0])

    def test_stopped_router_is_idle_but_wrong_engine_and_jobs_skip(self):
        self.mocks[-3].return_value = False
        self.assertEqual(routes._scheduled_build_idle(self.cfg), ("", False, []))
        self.assertTrue(routes._scheduled_build_idle({"active_engine": "ikllama"})[0])
        routes.BUILDER_IKLLAMA.state["running"] = True
        self.assertTrue(routes._scheduled_build_idle(self.cfg)[0])
        routes.BUILDER_IKLLAMA.state["running"] = False
        service = mock.Mock(_registry={"job": object()})
        service._registry_lock = threading.Lock()
        with mock.patch.object(routes, "_AUTOTUNE_SERVICE", service):
            self.assertTrue(routes._scheduled_build_idle(self.cfg)[0])

    def test_running_vllm_skips(self):
        with mock.patch.object(routes, "VLLM_SUPPORTED", True):
            self.assertIn("vLLM", routes._scheduled_build_idle(self.cfg)[0])


class ScheduledBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = {"llama_src": self.tmp.name, "build_dir": self.tmp.name + "/build"}
        patches = [mock.patch.object(routes, "cfg", return_value=self.cfg),
                   mock.patch.object(routes.hardware, "recommend", return_value={"selected_backend": "cpu", "cmake_flags": {}}),
                   mock.patch.object(routes.prereqs, "status", return_value={}),
                   mock.patch.object(routes, "_validate_build_backend", return_value=""),
                   mock.patch.object(config, "update"),
                   mock.patch.object(routes, "_scheduled_build_idle", return_value=("", True, ["a"])),
                   mock.patch.object(routes.router_ctl, "stop", return_value=True),
                   mock.patch.object(routes.BUILDER_LLAMA, "state", {"running": False, "phase": "done"}),
                   mock.patch.object(routes.BUILDER_LLAMA, "run_build"),
                   mock.patch.object(routes, "_bring_router_back")]
        self.mocks = [p.start() for p in patches]
        for patch in patches:
            self.addCleanup(patch.stop)
        self.addCleanup(self.reset_restore)

    def reset_restore(self):
        routes._PREBUILD_RUNNING, routes._PREBUILD_LOADED = False, []

    def test_stops_builds_with_pull_and_restores(self):
        def build(*args, **kwargs):
            self.mocks[6].assert_called_once()
            self.assertTrue(routes._PREBUILD_RUNNING)
            self.assertEqual(routes._PREBUILD_LOADED, ["a"])
            self.assertTrue(kwargs["pull"])
            self.assertTrue(kwargs["strict_pull"])
        self.mocks[8].side_effect = build
        _, result = routes.post_build_start(routes.Req(body={}), scheduled=True)
        self.assertTrue(result["started"])
        self.mocks[9].assert_called_once()

    def test_restore_attempted_on_failure(self):
        self.mocks[8].side_effect = RuntimeError("build error")
        with self.assertRaises(RuntimeError):
            routes.post_build_start(routes.Req(body={}), scheduled=True)
        self.mocks[9].assert_called_once()

    def test_busy_recheck_and_stop_failure_do_not_build(self):
        self.mocks[5].return_value = ("busy", False, [])
        _, result = routes.post_build_start(routes.Req(body={}), scheduled=True)
        self.assertFalse(result["started"])
        self.mocks[6].assert_not_called()
        self.mocks[5].return_value = ("", True, ["a"])
        self.mocks[6].return_value = False
        _, result = routes.post_build_start(routes.Req(body={}), scheduled=True)
        self.assertFalse(result["started"])
        self.mocks[8].assert_not_called()
        self.assertFalse(routes._PREBUILD_RUNNING)

    def test_http_post_gate_rejects_update_requests(self):
        handler = mock.Mock()
        with mock.patch.object(routes.BUILD_SCHEDULE, "updating", True):
            server.H.do_POST(handler)
        self.assertEqual(handler._send.call_args.args[0], 503)
        handler._do_POST.assert_not_called()


class BuilderTest(unittest.TestCase):
    def test_failed_pull_aborts_only_automatic_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            bm = BuildManager(tmp)
            with mock.patch.object(bm, "_stream", return_value=1) as stream:
                bm.run_build(tmp, tmp + "/build", {}, strict_pull=True)
            self.assertEqual(stream.call_count, 1)
            self.assertEqual(bm.state["phase"], "failed")
            self.assertFalse(bm.state["running"])
            self.assertIn("automatic rebuild cancelled", bm.tail())

    def test_start_claims_before_worker_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            bm = BuildManager(tmp)
            with mock.patch("builder.threading.Thread"):
                self.assertTrue(bm.start(tmp, tmp + "/build", {}))
                self.assertTrue(bm.state["running"])
                self.assertFalse(bm.start(tmp, tmp + "/build", {}))


if __name__ == "__main__":
    unittest.main()
