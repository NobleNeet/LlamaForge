"""Log maintenance: in-place trim semantics, config resolution, and the
all-models-unloaded trigger. The trim must never create generation files,
never replace the inode, never read a whole file into memory, and never
break the unload that triggered it."""
import conftest_paths  # noqa: F401
import json, os, tempfile, threading, unittest
from unittest import mock

import config, log_manager, routes, server


class TrimFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, name, data):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_under_limit_is_untouched(self):
        path = self._write("a.log", b"line1\nline2\n")
        before = open(path, "rb").read()
        res = log_manager.trim_file(path, 1024)
        self.assertFalse(res["trimmed"])
        self.assertEqual(open(path, "rb").read(), before)

    def test_exact_limit_is_untouched(self):
        data = b"x" * 100
        path = self._write("exact.log", data)
        res = log_manager.trim_file(path, 100)
        self.assertFalse(res["trimmed"])
        self.assertEqual(os.path.getsize(path), 100)

    def test_over_limit_keeps_the_newest_side(self):
        lines = [f"line{i}\n".encode() for i in range(1000)]
        path = self._write("big.log", b"".join(lines))
        # Cap that fits roughly the last 100 lines.
        cap = sum(len(l) for l in lines[-100:])
        res = log_manager.trim_file(path, cap)
        self.assertTrue(res["trimmed"])
        kept = open(path, "rb").read()
        self.assertTrue(kept.endswith(lines[-1]), "newest line must survive")
        self.assertNotIn(b"line0\n", kept, "oldest lines must be gone")
        self.assertLessEqual(len(kept), cap)

    def test_oldest_complete_lines_removed_first(self):
        # 10 lines of 10 bytes each; cap 35 -> keep the newest whole lines
        # that fit: 3 lines (30 bytes). The 4th-newest would need 40.
        data = b"".join(f"{i:09d}\n".encode() for i in range(10))
        path = self._write("lines.log", data)
        log_manager.trim_file(path, 35)
        self.assertEqual(open(path, "rb").read(), b"000000007\n000000008\n000000009\n")

    def test_partial_line_at_window_start_is_dropped(self):
        # The window starts mid-line; the retained data must begin at a
        # newline boundary, never with a partial line.
        data = b"first\n" + b"second\n" * 4 + b"third\n"   # 40 bytes
        path = self._write("partial.log", data)
        log_manager.trim_file(path, 15)
        kept = open(path, "rb").read()
        self.assertLessEqual(len(kept), 15)
        self.assertEqual(kept, b"second\nthird\n",
                        f"retained data must start at a line boundary: {kept[:30]!r}")

    def test_result_never_exceeds_limit(self):
        data = b"".join(f"line {i} padding padding\n".encode() for i in range(500))
        path = self._write("cap.log", data)
        for cap in (1, 7, 23, 64, 1000, 4096):
            log_manager.trim_file(path, cap)
            self.assertLessEqual(os.path.getsize(path), cap,
                                f"cap={cap} violated after trim")

    def test_single_oversized_line_is_dropped(self):
        # One line longer than the cap: the cap wins, the line goes.
        path = self._write("huge.log", b"x" * 5000)
        res = log_manager.trim_file(path, 1000)
        self.assertTrue(res["trimmed"])
        self.assertEqual(os.path.getsize(path), 0)

    def test_zero_means_unlimited(self):
        path = self._write("unlimited.log", b"y" * 10_000)
        res = log_manager.trim_file(path, 0)
        self.assertFalse(res["trimmed"])
        self.assertEqual(res["reason"], "unlimited")
        self.assertEqual(os.path.getsize(path), 10_000)

    def test_missing_file_is_not_an_error(self):
        res = log_manager.trim_file(os.path.join(self.tmp, "ghost.log"), 100)
        self.assertFalse(res["trimmed"])
        self.assertEqual(res["reason"], "missing")

    def test_inode_is_preserved(self):
        # Child processes keep these files open; the trim must not swap in a
        # new inode (os.replace would strand their writers on an unlinked one).
        path = self._write("inode.log", b"".join(f"line {i}\n".encode() for i in range(2000)))
        ino_before = os.stat(path).st_ino
        log_manager.trim_file(path, 4096)
        self.assertEqual(os.stat(path).st_ino, ino_before,
                         "in-place trim must keep the same inode")

    def test_append_mode_writer_sees_trimmed_content(self):
        # Simulate a child process holding the file open in append mode:
        # after the trim, its next write must land after the retained tail,
        # not at a stale offset (which would create a sparse hole).
        path = self._write("append.log", b"".join(f"{i:06d} old\n".encode() for i in range(500)))
        writer = open(path, "ab")
        self.addCleanup(writer.close)
        log_manager.trim_file(path, 700)
        writer.write(b"NEWLINE\n")
        writer.flush()
        content = open(path, "rb").read()
        self.assertTrue(content.endswith(b"NEWLINE\n"))
        self.assertNotIn(b"\x00", content, "no sparse hole from a stale write offset")
        self.assertLessEqual(len(content), 700 + len(b"NEWLINE\n"))


class ConfigResolutionTest(unittest.TestCase):
    def test_default_log_dir_is_repo_logs(self):
        c = dict(config.DEFAULTS)
        self.assertEqual(log_manager.effective_log_dir(c),
                         os.path.join(config.ROOT, "logs"))

    def test_absolute_log_dir_is_used_verbatim(self):
        d = tempfile.mkdtemp()
        self.assertEqual(log_manager.effective_log_dir({"log_dir": d}), d)

    def test_relative_log_dir_anchors_to_repo_root(self):
        got = log_manager.effective_log_dir({"log_dir": "mylogs"})
        self.assertEqual(got, os.path.normpath(os.path.join(config.ROOT, "mylogs")))

    def test_tilde_is_expanded(self):
        got = log_manager.effective_log_dir({"log_dir": "~/llamaforge-logs"})
        self.assertNotIn("~", got)
        self.assertTrue(os.path.isabs(got))

    def test_limit_defaults_when_unset(self):
        c = dict(config.DEFAULTS)
        self.assertEqual(log_manager.limit_bytes("router_stdout", c), 64 * log_manager.MB)
        self.assertEqual(log_manager.limit_bytes("build_custom", c), 32 * log_manager.MB)

    def test_user_override_wins(self):
        c = {"log_limits_mb": {"router_stdout": 5}}
        self.assertEqual(log_manager.limit_bytes("router_stdout", c), 5 * log_manager.MB)

    def test_zero_override_is_unlimited(self):
        c = {"log_limits_mb": {"router_stdout": 0}}
        self.assertEqual(log_manager.limit_bytes("router_stdout", c), 0)

    def test_garbage_value_falls_back_to_default(self):
        for bad in ("lots", -3, True, 1.5):
            c = {"log_limits_mb": {"router_stdout": bad}}
            self.assertEqual(log_manager.limit_bytes("router_stdout", c), 64 * log_manager.MB,
                             f"bad value {bad!r} must fall back to the default")

    def test_unknown_kind_defaults_to_zero(self):
        self.assertEqual(log_manager.limit_bytes("nonexistent_kind", {}), 0)

    def test_log_path_maps_kind_to_file(self):
        c = {"log_dir": "/tmp/lf-logs"}
        self.assertEqual(log_manager.log_path("router_stdout", c), "/tmp/lf-logs/router.out.log")
        self.assertEqual(log_manager.log_path("build_llamacpp", c), "/tmp/lf-logs/build.log")
        with self.assertRaises(KeyError):
            log_manager.log_path("build_custom", c)   # wildcard family, no single path


class TrimAllTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = {"log_dir": self.tmp,
                    "log_limits_mb": {"router_stdout": 0, "build_custom": 0}}
        # 0 MB would mean unlimited; use byte-level caps via patched limit_bytes
        # so the test does not need MB-sized files.
        self.limits = {}

        def fake_limit(kind, c=None):
            return self.limits.get(kind, 0)   # 0 = unlimited by default
        self._limit_patch = mock.patch.object(log_manager, "limit_bytes", side_effect=fake_limit)
        self._limit_patch.start()
        self.addCleanup(self._limit_patch.stop)

    def _write(self, name, data):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_build_custom_family_shares_one_cap(self):
        p1 = self._write("build-custom-aaa.log", b"a" * 500)
        p2 = self._write("build-custom-bbb.log", b"b" * 700)
        other = self._write("build.log", b"c" * 500)      # build_llamacpp, uncapped here
        self.limits["build_custom"] = 100
        res = log_manager.trim_all(self.cfg)
        self.assertLessEqual(os.path.getsize(p1), 100)
        self.assertLessEqual(os.path.getsize(p2), 100)
        self.assertEqual(os.path.getsize(other), 500, "non-custom build log untouched")
        self.assertIn(p1, res["trimmed"])
        self.assertIn(p2, res["trimmed"])

    def test_no_generation_files_created(self):
        self.limits["router_stdout"] = 10
        self._write("router.out.log", b"x" * 1000)
        log_manager.trim_all(self.cfg)
        leftovers = [n for n in os.listdir(self.tmp)
                    if n.endswith((".1", ".old", ".bak")) or ".log." in n]
        self.assertEqual(leftovers, [], "trim must not create rotated files")

    def test_missing_files_are_skipped_silently(self):
        res = log_manager.trim_all(self.cfg)     # empty dir: nothing exists
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["trimmed"], [])

    def test_per_file_failure_is_reported_not_raised(self):
        path = self._write("router.out.log", b"x" * 1000)
        self.limits["router_stdout"] = 10

        def boom(p, max_bytes):
            if os.path.basename(p) == "router.out.log":
                raise OSError("disk on fire")
            return {"path": p, "size": 0, "max": max_bytes,
                    "trimmed": False, "bytes_freed": 0, "reason": "within limit"}

        with mock.patch.object(log_manager, "trim_file", side_effect=boom):
            res = log_manager.trim_all(self.cfg)     # must not raise
        self.assertEqual(len(res["errors"]), 1)
        self.assertIn("router_stdout", res["errors"][0])

    def test_maybe_trim_all_never_raises(self):
        with mock.patch.object(log_manager, "trim_all", side_effect=RuntimeError("boom")):
            res = log_manager.maybe_trim_all()
        self.assertTrue(res["ran"])
        self.assertTrue(res["errors"])


class ConcurrencyTest(unittest.TestCase):
    def test_concurrent_triggers_collapse_to_one_sweep(self):
        running = threading.Event()
        release = threading.Event()
        calls = []

        def slow_trim(c=None, log=None):
            calls.append(1)
            running.set()
            release.wait(5)
            return {"ran": True, "trimmed": [], "errors": []}

        with mock.patch.object(log_manager, "trim_all", side_effect=slow_trim):
            t1 = threading.Thread(target=log_manager.maybe_trim_all)
            t1.start()
            running.wait(5)
            res2 = log_manager.maybe_trim_all()      # lock held -> skip
            release.set()
            t1.join(5)
        self.assertEqual(res2["ran"], False)
        self.assertEqual(res2["reason"], "busy")
        self.assertEqual(len(calls), 1)


class IdleTriggerTest(unittest.TestCase):
    """maybe_run_idle_maintenance fires only on resident -> all-unloaded."""

    def _router_models(self, statuses):
        def router(path, method="GET", body=None, timeout=30):
            if path == "/models":
                return 200, {"data": [{"id": mid, "status": {"value": st}}
                                     for mid, st in statuses.items()]}
            return 200, {}
        return router

    def test_trims_when_router_holds_nothing(self):
        with mock.patch.object(routes, "router", side_effect=self._router_models({})), \
             mock.patch.object(log_manager, "maybe_trim_all", return_value={"ran": True, "trimmed": ["/x"], "errors": []}) as trim:
            routes.maybe_run_idle_maintenance(source="test")
        trim.assert_called_once()

    def test_skips_when_a_model_is_loaded(self):
        with mock.patch.object(routes, "router", side_effect=self._router_models({"m": "loaded"})), \
             mock.patch.object(log_manager, "maybe_trim_all") as trim:
            routes.maybe_run_idle_maintenance(source="test")
        trim.assert_not_called()

    def test_skips_when_a_model_is_loading(self):
        with mock.patch.object(routes, "router", side_effect=self._router_models({"m": "loading"})), \
             mock.patch.object(log_manager, "maybe_trim_all") as trim:
            routes.maybe_run_idle_maintenance(source="test")
        trim.assert_not_called()

    def test_skips_when_vllm_is_live(self):
        fake_mgr = mock.Mock()
        fake_mgr.status.return_value = [{"model_id": "v", "state": "ready"}]
        with mock.patch.object(routes, "router", side_effect=self._router_models({})), \
             mock.patch.object(routes, "VLLM_SUPPORTED", True), \
             mock.patch.object(routes, "_VLLM", fake_mgr), \
             mock.patch.object(log_manager, "maybe_trim_all") as trim:
            routes.maybe_run_idle_maintenance(source="test")
        trim.assert_not_called()

    def test_trims_when_vllm_only_has_failed_instances(self):
        fake_mgr = mock.Mock()
        fake_mgr.status.return_value = [{"model_id": "v", "state": "failed"}]
        with mock.patch.object(routes, "router", side_effect=self._router_models({})), \
             mock.patch.object(routes, "VLLM_SUPPORTED", True), \
             mock.patch.object(routes, "_VLLM", fake_mgr), \
             mock.patch.object(log_manager, "maybe_trim_all", return_value={"ran": True, "trimmed": [], "errors": []}) as trim:
            routes.maybe_run_idle_maintenance(source="test")
        trim.assert_called_once()

    def test_dead_router_still_trims(self):
        # A router that will not answer holds nothing we manage.
        with mock.patch.object(routes, "router", return_value=(500, {})), \
             mock.patch.object(log_manager, "maybe_trim_all", return_value={"ran": True, "trimmed": [], "errors": []}) as trim:
            routes.maybe_run_idle_maintenance(source="test")
        trim.assert_called_once()

    def test_trigger_never_raises(self):
        with mock.patch.object(routes, "router", side_effect=RuntimeError("nope")), \
             mock.patch.object(log_manager, "maybe_trim_all") as trim:
            routes.maybe_run_idle_maintenance(source="test")   # must not raise
        trim.assert_not_called()

    def test_unload_route_triggers_after_success(self):
        req = routes.Req(body={"model": "m", "backend": "llamacpp"}, path="/api/models/unload")
        backend = mock.Mock()
        backend.unload.return_value = (True, "")
        backend.name = "llamacpp"
        with mock.patch.object(routes, "REGISTRY") as reg, \
             mock.patch.object(routes, "maybe_run_idle_maintenance") as trig:
            reg.for_model.return_value = backend
            routes.post_model_unload(req)
        trig.assert_called_once()

    def test_failed_unload_does_not_trigger(self):
        req = routes.Req(body={"model": "m", "backend": "llamacpp"}, path="/api/models/unload")
        backend = mock.Mock()
        backend.unload.return_value = (False, "busy")
        backend.name = "llamacpp"
        with mock.patch.object(routes, "REGISTRY") as reg, \
             mock.patch.object(routes, "maybe_run_idle_maintenance") as trig:
            reg.for_model.return_value = backend
            routes.post_model_unload(req)
        trig.assert_not_called()

    def test_reaper_triggers_after_unloading(self):
        # One model, idle past the timeout -> reaped -> maintenance asked.
        server._reset_api_idle_state()
        server._API_IDLE_LAST["m"] = 100
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=self._router_models({"m": "loaded"})), \
             mock.patch.object(routes, "maybe_run_idle_maintenance") as trig:
            unloaded = server._reap_api_idle_models(now=200)
        self.assertEqual(unloaded, ["m"])
        trig.assert_called_once()

    def test_reaper_without_unloads_does_not_trigger(self):
        server._reset_api_idle_state()
        with mock.patch.object(routes, "cfg", return_value={"api_idle_unload_minutes": 1}), \
             mock.patch.object(routes, "router", side_effect=self._router_models({})), \
             mock.patch.object(routes, "maybe_run_idle_maintenance") as trig:
            self.assertEqual(server._reap_api_idle_models(now=200), [])
        trig.assert_not_called()


class ConfigValidatorTest(unittest.TestCase):
    def test_log_dir_accepts_strings(self):
        self.assertEqual(routes._v_str("/var/log/llamaforge"), "/var/log/llamaforge")
        self.assertEqual(routes._v_str(""), "")

    def test_log_limits_accepts_known_kinds(self):
        v = routes._v_log_limits({"router_stdout": 10, "build_custom": 0})
        self.assertEqual(v, {"router_stdout": 10, "build_custom": 0})

    def test_log_limits_rejects_unknown_kinds(self):
        self.assertIsNone(routes._v_log_limits({"not_a_kind": 5}))

    def test_log_limits_rejects_bad_values(self):
        for bad in (-1, 1.5, True, "big", None):
            self.assertIsNone(routes._v_log_limits({"router_stdout": bad}),
                             f"value {bad!r} must be rejected")

    def test_empty_limits_clears_overrides(self):
        self.assertEqual(routes._v_log_limits({}), {})

    def test_config_route_accepts_new_keys(self):
        req = routes.Req(body={"log_dir": "/tmp/lf", "log_limits_mb": {"router_stdout": 7}})
        with mock.patch.object(config, "update", return_value=dict(config.DEFAULTS)) as upd:
            status, out = routes.post_config(req)
        self.assertEqual(status, 200)
        self.assertEqual(upd.call_args[0][0]["log_dir"], "/tmp/lf")
        self.assertEqual(upd.call_args[0][0]["log_limits_mb"], {"router_stdout": 7})
        self.assertNotIn("rejected", out)

    def test_config_route_rejects_unknown_log_key(self):
        req = routes.Req(body={"log_limits_mb": {"bogus": 1}})
        with mock.patch.object(config, "update", return_value=dict(config.DEFAULTS)):
            with self.assertRaises(routes.ApiError) as cm:
                routes.post_config(req)
        self.assertEqual(cm.exception.status, 400)

    def test_defaults_present_for_backward_compat(self):
        # A config that predates the feature: load() merges DEFAULTS, so both
        # keys exist with the legacy-behaving values.
        c = dict(config.DEFAULTS)
        self.assertEqual(c["log_dir"], "")
        self.assertEqual(c["log_limits_mb"], {})
        self.assertEqual(log_manager.effective_log_dir(c),
                         os.path.join(config.ROOT, "logs"))


class SummaryTest(unittest.TestCase):
    def test_summary_merges_defaults_with_overrides(self):
        c = {"log_dir": "", "log_limits_mb": {"router_stdout": 3}}
        s = log_manager.summary(c)
        self.assertEqual(s["log_dir"], "")
        self.assertEqual(s["effective_dir"], os.path.join(config.ROOT, "logs"))
        self.assertEqual(s["limits_mb"]["router_stdout"], 3)
        self.assertEqual(s["limits_mb"]["router_stderr"], 32)   # default
        self.assertEqual(s["limits_mb"]["build_custom"], 32)    # default

    def test_every_kind_has_a_summary_entry(self):
        s = log_manager.summary({})
        for kind in log_manager.DEFAULT_LIMITS_MB:
            self.assertIn(kind, s["limits_mb"])


if __name__ == "__main__":
    unittest.main()
