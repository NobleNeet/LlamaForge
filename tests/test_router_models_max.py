"""The `--models-max` the router is started with.

llama.cpp's router evicts by loaded-model COUNT and never by free memory, so
`--models-max 1` is exactly why loading a second model always unloaded the
first one. `0` means unlimited and disables llama.cpp's own eviction
(llama.cpp/tools/server/server-models.cpp), which is what the memory-aware
admission policy needs - but the shipped default has to stay 1 until that
policy exists, or every existing install changes behaviour overnight.
"""
import conftest_paths  # noqa: F401
import os, tempfile, unittest
from unittest import mock

import config, router_ctl, routes
from routes import Req


class ResolveModelsMaxTest(unittest.TestCase):
    def test_absent_key_keeps_the_historical_one(self):
        self.assertEqual(router_ctl.resolve_models_max({}), 1)
        self.assertEqual(router_ctl.DEFAULT_MODELS_MAX, 1)

    def test_zero_is_unlimited_and_survives(self):
        self.assertEqual(router_ctl.resolve_models_max({"router_models_max": 0}), 0)

    def test_a_number_is_taken_as_is(self):
        self.assertEqual(router_ctl.resolve_models_max({"router_models_max": 4}), 4)

    def test_a_string_from_hand_edited_json_still_works(self):
        self.assertEqual(router_ctl.resolve_models_max({"router_models_max": "3"}), 3)

    def test_negative_would_kill_the_router_at_startup(self):
        self.assertEqual(router_ctl.resolve_models_max({"router_models_max": -1}), 1)

    def test_junk_falls_back_instead_of_raising(self):
        self.assertEqual(router_ctl.resolve_models_max({"router_models_max": "many"}), 1)
        self.assertEqual(router_ctl.resolve_models_max({"router_models_max": None}), 1)
        self.assertEqual(router_ctl.resolve_models_max(None), 1)

    def test_shipped_default_is_the_unchanged_behaviour(self):
        self.assertEqual(config.DEFAULTS["router_models_max"], 1)


class StartArgvTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        mock.patch.object(router_ctl.os.path, "exists", return_value=True).start()
        mock.patch.object(router_ctl, "is_running", return_value=False).start()
        self.popen = mock.patch.object(router_ctl.subprocess, "Popen").start()
        self.addCleanup(mock.patch.stopall)

    def _argv(self, **kw):
        ok, err = router_ctl.start("/bin/llama-server", "/tmp/models.ini", 8080,
                                   "127.0.0.1", "", self.dir, **kw)
        self.assertTrue(ok, err)
        return self.popen.call_args[0][0]

    def _models_max_of(self, argv):
        return int(argv[argv.index("--models-max") + 1])

    def test_default_is_unchanged(self):
        self.assertEqual(self._models_max_of(self._argv()), 1)

    def test_an_explicit_zero_is_forwarded_verbatim(self):
        self.assertEqual(self._models_max_of(self._argv(models_max=0)), 0)

    def test_a_junk_value_cannot_take_the_router_down(self):
        self.assertEqual(self._models_max_of(self._argv(models_max=-5)), 1)


class RestartUsesConfigTest(unittest.TestCase):
    """The dashboard must not restart the router back onto 1 behind the user's
    back: Setup and the engine switch both read the config they just saved."""

    def setUp(self):
        self.saved = {}
        self.base = {"router_port": 8080, "router_host": "127.0.0.1",
                     "router_api_key": "", "router_models_max": 0,
                     "models_ini": "/tmp/models.ini",
                     "server_bin": "/bin/llama-server", "active_engine": "llamacpp",
                     "ik_llama_server_bin": "/bin/ik/llama-server"}
        mock.patch.object(routes.config, "load",
                          side_effect=lambda: dict(self.base, **self.saved)).start()
        mock.patch.object(routes.config, "update",
                          side_effect=lambda ch: (self.saved.update(ch),
                                                  dict(self.base, **self.saved))[1]).start()
        mock.patch.object(routes.config, "ini_path", return_value="/tmp/models.ini").start()
        self.restart = mock.patch.object(routes.router_ctl, "restart",
                                         return_value=(True, "")).start()
        mock.patch.object(routes.router_ctl, "supports_router_mode", return_value=True).start()
        mock.patch.object(routes.os.path, "exists", return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def test_network_apply_keeps_the_configured_limit(self):
        routes.post_network(Req(body={"host": "127.0.0.1", "port": 8080,
                                      "api_key": "", "panel_host": "127.0.0.1"}))
        self.assertEqual(self.restart.call_args[1]["models_max"], 0)

    def test_engine_switch_keeps_the_configured_limit(self):
        routes.post_engine_switch(Req(body={"engine": "llamacpp"}))
        self.assertEqual(self.restart.call_args[1]["models_max"], 0)


if __name__ == "__main__":
    unittest.main()
