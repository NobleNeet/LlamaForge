import conftest_paths  # noqa: F401
import os, tempfile, unittest
from unittest import mock
import config, hub, routes


class _TmpConfig(unittest.TestCase):
    """config.json in a temp file, so settings written by a route never touch
    the developer's real one."""

    def setUp(self):
        self._cfg = config.CONFIG
        fd, self.tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(self.tmp, "w", encoding="utf-8") as f:
            f.write("{}")
        config.CONFIG = self.tmp

    def tearDown(self):
        config.CONFIG = self._cfg
        try:
            os.remove(self.tmp)
        except OSError:
            pass


class TestResolveDownloadDir(unittest.TestCase):
    """download_dir() is pure over the config it is handed, so the whole
    fallback chain is checkable without a config.json on disk."""

    def test_nothing_settles_on_the_repo_models_folder(self):
        self.assertEqual(routes.download_dir({}), os.path.join(routes.ROOT, "models"))

    def test_scan_root_used_when_set(self):
        c = {"model_dirs": ["/data/models", "/other"]}
        self.assertEqual(routes.download_dir(c),
                         os.path.join("/data/models", "LlamaForge-downloads"))

    def test_setting_beats_the_scan_root(self):
        c = {"model_dirs": ["/data/models"], "download_dir": "/mnt/disk2/gguf"}
        self.assertEqual(routes.download_dir(c), "/mnt/disk2/gguf")

    def test_default_helper_ignores_the_setting(self):
        c = {"model_dirs": ["/data/models"], "download_dir": "/mnt/disk2/gguf"}
        self.assertEqual(routes.default_download_dir(c),
                         os.path.join("/data/models", "LlamaForge-downloads"))

    def test_relative_and_tilde_paths_are_anchored(self):
        self.assertEqual(routes.download_dir({"download_dir": "gguf"}),
                         os.path.abspath("gguf"))
        self.assertEqual(routes.download_dir({"download_dir": "~/gguf"}),
                         os.path.join(os.path.expanduser("~"), "gguf"))

    def test_whitespace_only_setting_counts_as_unset(self):
        c = {"download_dir": "   ", "model_dirs": ["/data/models"]}
        self.assertEqual(routes.download_dir(c),
                         os.path.join("/data/models", "LlamaForge-downloads"))


class TestDownloadDirRoute(_TmpConfig):
    def test_config_route_accepts_a_path(self):
        status, out = routes.post_config(routes.Req(body={"download_dir": "/mnt/disk2/gguf"}))
        self.assertEqual(status, 200)
        self.assertEqual(config.load()["download_dir"], "/mnt/disk2/gguf")
        self.assertEqual(routes.download_dir(), "/mnt/disk2/gguf")

    def test_blank_clears_back_to_the_default(self):
        config.update({"download_dir": "/mnt/disk2/gguf"})
        routes.post_config(routes.Req(body={"download_dir": ""}))
        self.assertEqual(config.load()["download_dir"], "")
        self.assertEqual(routes.download_dir(), os.path.join(routes.ROOT, "models"))

    def test_non_string_is_refused(self):
        with self.assertRaises(routes.ApiError):
            routes.post_config(routes.Req(body={"download_dir": 17}))

    def test_hub_dir_reports_setting_default_and_effective(self):
        config.update({"download_dir": "/mnt/disk2/gguf"})
        _st, d = routes.get_hub_dir(routes.Req())
        self.assertEqual(d["custom"], "/mnt/disk2/gguf")
        self.assertEqual(d["dir"], "/mnt/disk2/gguf")
        self.assertEqual(d["default"], os.path.join(routes.ROOT, "models"))

    def test_the_ui_can_actually_ask(self):
        """/api/hub/dir registered, or the Discover box loads nothing and quietly
        falls back to guessing at the folder."""
        self.assertIs(routes.GET_ROUTES["/api/hub/dir"], routes.get_hub_dir)

    def test_hub_dir_flags_a_folder_outside_the_scan_roots(self):
        with tempfile.TemporaryDirectory() as roots, tempfile.TemporaryDirectory() as else_dir:
            config.update({"model_dirs": [roots], "download_dir": else_dir})
            _st, d = routes.get_hub_dir(routes.Req())
            self.assertFalse(d["scanned"])
            self.assertTrue(d["dir"].startswith(else_dir))

    def test_hub_dir_accepts_a_download_dir_inside_a_scan_root(self):
        with tempfile.TemporaryDirectory() as roots:
            config.update({"model_dirs": [roots]})
            _st, d = routes.get_hub_dir(routes.Req())
            self.assertTrue(d["scanned"])
            self.assertEqual(d["dir"], os.path.join(roots, "LlamaForge-downloads"))

    def test_hub_dir_treats_no_scan_roots_as_everywhere(self):
        """An empty `model_dirs` is not "scan nothing" - Setup walks the drive
        roots - so the note must not warn about a folder a scan would find."""
        with mock.patch.object(routes.scanner, "list_drives", return_value=["/mnt"]):
            config.update({"model_dirs": [], "download_dir": "/mnt/disk2/gguf"})
            _st, d = routes.get_hub_dir(routes.Req())
        self.assertTrue(d["scanned"])

    def test_download_lands_under_the_configured_folder(self):
        seen = {}

        def fake_start(repo, paths, dest):
            seen["dest"] = dest
            return False                      # never spawn the real worker

        real = routes.DOWNLOADS.start
        routes.DOWNLOADS.start = fake_start
        try:
            config.update({"download_dir": "/mnt/disk2/gguf"})
            _st, out = routes.post_hub_download(
                routes.Req(body={"repo": "acme/model", "path": "m.gguf", "shards": 1}))
        finally:
            routes.DOWNLOADS.start = real
        self.assertEqual(out["dest"], os.path.join("/mnt/disk2/gguf", "acme--model"))
        self.assertEqual(seen["dest"], out["dest"])


class TestProgressReportsDest(unittest.TestCase):
    """The download card shows the folder, including after a Resume - so the
    job's own state has to carry it."""

    def test_start_publishes_the_destination(self):
        dm = hub.DownloadManager()
        dm._run = lambda *a: None             # the worker thread is not the subject
        self.assertTrue(dm.start("acme/model", ["m.gguf"], "/mnt/disk2/gguf"))
        self.assertEqual(dm.progress()["dest"], "/mnt/disk2/gguf")

    def test_run_republishes_it_for_a_resumed_job(self):
        dm = hub.DownloadManager()
        dm._fetch = lambda url, dest: None    # no network, no bytes
        with tempfile.TemporaryDirectory() as d:
            dm._run("acme/model", ["m.gguf"], d)
            self.assertEqual(dm.progress()["dest"], d)
            self.assertEqual(dm.progress()["phase"], "done")


class TestDiscoverSaveToField(unittest.TestCase):
    """The Discover field is the only way to reach the setting from the UI, so
    pin the wiring a regression would have to break."""

    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "web", "js", "discover.js"), encoding="utf-8") as f:
            cls.js = f.read()

    def test_field_and_buttons_exist(self):
        for marker in ('id="hub-dir"', 'id="hub-dir-save"', 'id="hub-dir-default"',
                       'id="hub-dir-note"'):
            self.assertIn(marker, self.js)

    def test_saves_through_the_config_route_then_rereads_the_server(self):
        self.assertIn('api("/api/config", {download_dir: path})', self.js)
        self.assertIn('api("/api/hub/dir")', self.js)

    def test_download_card_shows_where_the_bytes_went(self):
        self.assertIn('id="dl-dest"', self.js)
        self.assertIn('$("#dl-dest").textContent = r.dest', self.js)
        self.assertIn('if (s.dest) $("#dl-dest").textContent = s.dest', self.js)

    def test_safetensors_mode_hides_the_gguf_folder_row(self):
        """A safetensors transfer lands in the vLLM cache inside WSL, so leaving
        the GGUF folder row visible would imply it steers those downloads too."""
        self.assertIn("function hubDirMode()", self.js)
        self.assertIn('row.style.display', self.js)
        self.assertIn('"safetensors" ? "none"', self.js)


if __name__ == "__main__":
    unittest.main()
