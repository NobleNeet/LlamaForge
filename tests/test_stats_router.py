import conftest_paths  # noqa: F401
import json, os, tempfile, unittest, urllib.error, urllib.parse
import stats


class RouterCase(unittest.TestCase):
    def setUp(self):
        self._orig = stats.STATS_FILE
        fd, self.path = tempfile.mkstemp(suffix=".json"); os.close(fd); os.unlink(self.path)
        stats.STATS_FILE = self.path
        self.tr = stats.StatsTracker()
        self.tr._poll_vllm = lambda: None   # keep the test off the network

    def tearDown(self):
        stats.STATS_FILE = self._orig
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _wire(self, prompt, gen, model="nomic", seen=None):
        """Emulate the llama.cpp router: bare /metrics 400s (needs a model
        name), /metrics?model= works, /models reports one loaded model."""
        def fake_get(path, timeout=4):
            if seen is not None:
                seen.append(path)
            if path == "/models":
                return json.dumps({"data": [
                    {"id": "default", "status": {"value": "unloaded"}},
                    {"id": model, "status": {"value": "loaded"}},
                ]})
            if path.startswith("/metrics?model="):
                return (f"llamacpp:prompt_tokens_total {prompt}\n"
                        f"llamacpp:tokens_predicted_total {gen}\n"
                        f"llamacpp:predicted_tokens_seconds 12.5\n")
            if path == "/metrics":
                raise urllib.error.HTTPError(path, 400, "model name missing", {}, None)
            raise AssertionError("unexpected path " + path)
        self.tr._get = fake_get


class TestRouterMetricsScrape(RouterCase):
    def test_router_up_and_tokens_attributed(self):
        self._wire(prompt=10, gen=20)
        self.tr.poll_once()                       # baseline
        self.assertTrue(self.tr.live["router_up"])
        self._wire(prompt=15, gen=60)             # counters advanced
        self.tr.poll_once()
        m = self.tr.data["models"]["nomic"]
        self.assertEqual(m["prompt"], 5)
        self.assertEqual(m["generated"], 40)
        self.assertGreater(self.tr.live["gen_per_sec"], 0)

    def test_scrape_includes_model_param_never_bare(self):
        seen = []
        self._wire(prompt=1, gen=1, seen=seen)
        self.tr.poll_once()
        self.assertIn("/models", seen)
        self.assertTrue(any(p.startswith("/metrics?model=") for p in seen),
                        f"never scraped with ?model=; saw {seen}")
        self.assertNotIn("/metrics", seen)        # bare form must not be used

    def test_router_up_with_no_model_loaded(self):
        def fake_get(path, timeout=4):
            if path == "/models":
                return json.dumps({"data": [{"id": "default", "status": {"value": "unloaded"}}]})
            raise AssertionError("should not scrape metrics with nothing loaded")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertTrue(self.tr.live["router_up"])   # up, just idle
        self.assertIsNone(self.tr.live["loaded_model"])

    def test_router_down_reports_offline(self):
        def fake_get(path, timeout=4):
            raise urllib.error.URLError("connection refused")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertFalse(self.tr.live["router_up"])


class TestMultiLoadedModels(RouterCase):
    """`router_models_max > 1` keeps several models resident at once.

    Each model runs in its own child process with its own counters, so every
    loaded model has to be scraped and diffed on its own. Attributing to a
    single "the" loaded model was the bug: models after the first one - newest
    loads among them - never appeared on the Stats tab at all.
    """

    def _wire(self, counters, loaded=("alpha", "beta")):
        """counters: {model id: (prompt_total, gen_total)}; loaded: ids the
        router currently reports as loaded (preset order, as llama.cpp returns)."""
        def fake_get(path, timeout=4):
            if path == "/models":
                return json.dumps({"data": [
                    {"id": "default", "status": {"value": "unloaded"}},
                ] + [{"id": mid, "status": {"value": "loaded" if mid in loaded else "unloaded"}}
                     for mid in ("alpha", "beta")]})
            if path.startswith("/metrics?model="):
                mid = urllib.parse.unquote(path.split("=", 1)[1])
                p, g = counters[mid]
                return (f"llamacpp:prompt_tokens_total {p}\n"
                        f"llamacpp:tokens_predicted_total {g}\n"
                        f"llamacpp:requests_processing {1 if g else 0}\n")
            raise AssertionError("unexpected path " + path)
        self.tr._get = fake_get

    def test_every_loaded_model_is_recorded(self):
        self._wire({"alpha": (10, 20), "beta": (5, 7)}, loaded=("alpha", "beta"))
        self.tr.poll_once()                                   # baseline both
        self._wire({"alpha": (12, 25), "beta": (9, 17)}, loaded=("alpha", "beta"))
        self.tr.poll_once()
        got = {mid: (m["prompt"], m["generated"])
               for mid, m in self.tr.data["models"].items()}
        self.assertEqual(got, {"alpha": (2, 5), "beta": (4, 10)})

    def test_newly_loaded_model_reaches_the_summary(self):
        """The reported symptom: a model used for the first time stayed absent."""
        self._wire({"alpha": (10, 20), "beta": (0, 0)}, loaded=("alpha",))
        self.tr.poll_once()
        self._wire({"alpha": (10, 20), "beta": (30, 4)}, loaded=("alpha", "beta"))
        self.tr.poll_once()                                   # beta: baseline only
        # beta's row exists from the moment the router reports it loaded, but its
        # first scrape only sets the baseline - there is no window to attribute.
        self.assertEqual(self.tr.data["models"]["beta"]["generated"], 0)
        self._wire({"alpha": (10, 20), "beta": (30, 14)}, loaded=("alpha", "beta"))
        self.tr.poll_once()
        ids = [row["id"] for row in self.tr.summary()["per_model"]]
        self.assertIn("beta", ids)
        beta = [r for r in self.tr.summary()["per_model"] if r["id"] == "beta"][0]
        self.assertEqual((beta["prompt"], beta["generated"]), (0, 10))
        # ... and the traffic that beta served did not land on alpha
        alpha = [r for r in self.tr.summary()["per_model"] if r["id"] == "alpha"][0]
        self.assertEqual((alpha["prompt"], alpha["generated"]), (0, 0))

    def test_live_reports_every_loaded_model_and_totals(self):
        self._wire({"alpha": (1, 2), "beta": (3, 4)})
        self.tr.poll_once()
        self.assertEqual(self.tr.live["loaded_models"], ["alpha", "beta"])
        self.assertEqual(self.tr.live["loaded_model"], "alpha")   # pre-multi-model key
        self.assertEqual(sorted(self.tr.live["models"]), ["alpha", "beta"])
        self.assertEqual(self.tr.live["requests_processing"], 2)  # both report one
        self.assertEqual(self.tr.live["prompt_per_sec"], 0.0)     # rates default to 0

    def test_unloaded_model_rebaselines_instead_of_diffing_stale_counters(self):
        self._wire({"alpha": (100, 100), "beta": (0, 0)}, loaded=("alpha",))
        self.tr.poll_once()
        self._wire({"alpha": (140, 140), "beta": (0, 0)}, loaded=("alpha",))
        self.tr.poll_once()                                     # +40 / +40
        self._wire({"alpha": (0, 0), "beta": (0, 0)}, loaded=())  # reloaded: counters reset
        self.tr.poll_once()
        self.assertEqual(self.tr._prev, {})                     # baseline forgotten
        self._wire({"alpha": (6, 6), "beta": (0, 0)}, loaded=("alpha",))
        self.tr.poll_once()                                     # baseline, no negative
        self._wire({"alpha": (9, 8), "beta": (0, 0)}, loaded=("alpha",))
        self.tr.poll_once()
        m = self.tr.data["models"]["alpha"]
        self.assertEqual((m["prompt"], m["generated"]), (43, 42))

    def test_failed_scrape_keeps_the_baseline_so_tokens_are_not_lost(self):
        self._wire({"alpha": (10, 20), "beta": (0, 0)}, loaded=("alpha",))
        self.tr.poll_once()
        good = self.tr._get

        def flaky(path, timeout=4):
            if path.startswith("/metrics?model="):
                raise urllib.error.HTTPError(path, 400, "boom", {}, None)
            return good(path, timeout)
        self.tr._get = flaky
        self.tr.poll_once()
        self.assertEqual(self.tr.live["models"], {})            # nothing scraped
        self._wire({"alpha": (10, 25), "beta": (0, 0)}, loaded=("alpha",))
        self.tr.poll_once()
        # the window spans two polls; both generations are still attributed
        self.assertEqual(self.tr.data["models"]["alpha"]["generated"], 5)


class TestRouterState(unittest.TestCase):
    def test_returns_every_loaded_id_in_router_order(self):
        tr = stats.StatsTracker()
        tr._get = lambda path, timeout=4: json.dumps({"data": [
            {"id": "beta",  "status": {"value": "loaded"}},
            {"id": "default", "status": {"value": "unloaded"}},
            {"id": "alpha", "status": {"value": "loaded"}},
            {"id": "gamma", "status": {"value": "loading"}},
        ]})
        up, loaded = tr._router_models()
        self.assertTrue(up)
        self.assertEqual(loaded, ["beta", "alpha"])


if __name__ == "__main__":
    unittest.main()
