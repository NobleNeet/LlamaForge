import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import server
import stats


class MainStartupTest(unittest.TestCase):
    def test_main_starts_auto_load_and_idle_threads_without_shadowing_threading(self):
        started = []

        class FakeThread:
            def __init__(self, target=None, args=(), daemon=None, name=None):
                self.target = target
                self.args = args
                self.daemon = daemon
                self.name = name

            def start(self):
                started.append((self.name, self.target, self.args, self.daemon))

        class FakeHTTPD:
            def __init__(self, addr, handler):
                self.addr = addr
                self.handler = handler

            def serve_forever(self):
                return None

        cfg = {"panel_port": 8090, "panel_host": "127.0.0.1",
               "auto_load_model": "qwen", "server_bin": ""}

        with mock.patch.object(server.config, "migrate"), \
             mock.patch.object(server.routes, "cfg", return_value=cfg), \
             mock.patch.object(server.config, "ensure_models_ini", return_value=False), \
             mock.patch.object(server.argspec, "build_key_aliases", return_value={"keys": set(), "alias_to_key": {}}), \
             mock.patch.object(server.config, "sanitize_models_ini", return_value={"changed": False}), \
             mock.patch.object(server.config, "apply_ctx_defaults", return_value={"changed": False}), \
             mock.patch.object(stats.TRACKER, "start"), \
             mock.patch.object(server, "ThreadingHTTPServer", FakeHTTPD), \
             mock.patch.object(server.threading, "Thread", FakeThread):
            server.main()

        names = [name for name, _target, _args, _daemon in started]
        self.assertIn("auto-load", names)
        self.assertIn("api-idle-reaper", names)
        self.assertIn("preset-sync", names)


if __name__ == "__main__":
    unittest.main()
