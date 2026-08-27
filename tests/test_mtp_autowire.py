"""Auto-wire MTP draft models (issue #3).

scanner attaches an mtp-* sibling as a speculative draft model, enabling
spec-type=draft-mtp only when the sidecar declares NextN layers (the signal
llama.cpp gates on). Wiring is additive: spec-type is also the ngram-* selector,
so a re-scan must never wipe a hand-set speculative mode.
"""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest
from unittest import mock

import config, gguf, routes, scanner


class HasNextnTest(unittest.TestCase):
    def _write_gguf(self, kvs):
        """Minimal GGUF with the given (key, int_value) pairs. type 4 = uint32."""
        import struct
        path = os.path.join(self.tmp, "m.gguf")
        with open(path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))          # version
            f.write(struct.pack("<Q", 0))          # tensor count
            f.write(struct.pack("<Q", len(kvs)))   # kv count
            for k, v in kvs:
                kb = k.encode()
                f.write(struct.pack("<Q", len(kb))); f.write(kb)
                f.write(struct.pack("<I", 4))      # value type uint32
                f.write(struct.pack("<I", v))
        return path

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_true_when_nextn_layers_present(self):
        p = self._write_gguf([("gemma4-assistant.nextn_predict_layers", 2)])
        self.assertTrue(gguf.has_nextn(p))

    def test_false_when_zero_layers(self):
        p = self._write_gguf([("arch.nextn_predict_layers", 0)])
        self.assertFalse(gguf.has_nextn(p))

    def test_false_when_absent(self):
        p = self._write_gguf([("arch.block_count", 40)])
        self.assertFalse(gguf.has_nextn(p))

    def test_false_on_unreadable(self):
        self.assertFalse(gguf.has_nextn(os.path.join(self.tmp, "nope.gguf")))


class BuildEntriesMtpTest(unittest.TestCase):
    """Pure over fake paths; has_nextn is patched so no files are read."""

    def _entries(self, paths, nextn=False):
        with mock.patch.object(scanner, "_slug", side_effect=lambda s: s.lower()), \
             mock.patch("gguf.has_nextn", return_value=nextn), \
             mock.patch("gguf.metadata", return_value={}):
            return {e["id"]: e for e in scanner.build_entries(paths)}

    def test_sidecar_attaches_as_draft_model(self):
        paths = ["/m/model.gguf", "/m/mtp-model.gguf"]
        e = self._entries(paths, nextn=True)
        self.assertEqual(len(e), 1)                 # sidecar isn't a standalone model
        (entry,) = e.values()
        self.assertEqual(entry["draft_model"], "/m/mtp-model.gguf")
        self.assertTrue(entry.get("draft_mtp"))

    def test_attach_only_when_no_nextn(self):
        paths = ["/m/model.gguf", "/m/mtp-model.gguf"]
        (entry,) = self._entries(paths, nextn=False).values()
        self.assertEqual(entry["draft_model"], "/m/mtp-model.gguf")
        self.assertNotIn("draft_mtp", entry)        # inert until the user opts in

    def test_no_sidecar_no_draft_keys(self):
        (entry,) = self._entries(["/m/model.gguf"]).values()
        self.assertNotIn("draft_model", entry)
        self.assertNotIn("draft_mtp", entry)

    def test_sidecar_in_other_dir_not_attached(self):
        paths = ["/a/model.gguf", "/b/mtp-model.gguf"]
        (entry,) = self._entries(paths, nextn=True).values()
        self.assertNotIn("draft_model", entry)

    def test_duplicate_basenames_disambiguate_by_multiple_parent_dirs(self):
        paths = [
            "/models/org-a/release/model.gguf",
            "/alt/org-a/release/model.gguf",
            "/elsewhere/org-b/release/model.gguf",
        ]
        entries = scanner.build_entries(paths)
        ids = sorted(e["id"] for e in entries)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)
        self.assertIn("alt--org-a--release--model", ids)
        self.assertIn("models--org-a--release--model", ids)
        self.assertIn("org-b--release--model", ids)


class FindGgufsShardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_keeps_small_first_shard_when_set_total_exceeds_threshold(self):
        d = os.path.join(self.tmp, "Qwen3.8-Flash-Next-GGUF")
        os.makedirs(d, exist_ok=True)
        sizes = {
            "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf": 11 * 1024 * 1024,
            "Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf": 47 * 1024 * 1024,
            "Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf": 31 * 1024 * 1024,
        }
        for name, size in sizes.items():
            with open(os.path.join(d, name), "wb") as f:
                f.truncate(size)
        hits = sorted(os.path.basename(p) for p in scanner.find_ggufs([self.tmp], min_mb=50))
        self.assertEqual(hits, sorted(sizes))

    def test_drops_shard_set_when_total_stays_below_threshold(self):
        d = os.path.join(self.tmp, "tiny")
        os.makedirs(d, exist_ok=True)
        for idx in ("00001", "00002", "00003"):
            with open(os.path.join(d, f"m-{idx}-of-00003.gguf"), "wb") as f:
                f.truncate(10 * 1024 * 1024)
        self.assertEqual(scanner.find_ggufs([self.tmp], min_mb=50), [])


class ScanApplyMtpTest(unittest.TestCase):
    """Route-level: additive wiring that never clobbers a hand-set spec-type."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.ini = os.path.join(self.tmp, "models.ini")
        with open(config.CONFIG, "w") as f:
            json.dump({"models_ini": self.ini}, f)
        # keep the router + ctx pass out of the way
        self._r, self._c = routes.router, config.apply_ctx_defaults
        routes.router = lambda *a, **k: (200, {})
        config.apply_ctx_defaults = lambda *a, **k: {"changed": []}

    def tearDown(self):
        config.CONFIG = self._saved
        routes.router, config.apply_ctx_defaults = self._r, self._c

    def _apply(self, entries):
        req = mock.Mock()
        req.body = {"entries": entries}
        routes.post_scan_apply(req)

    def test_enables_draft_mtp_on_a_fresh_model(self):
        self._apply([{"id": "qwopus", "model": "/m/q.gguf",
                      "draft_model": "/m/mtp-q.gguf", "draft_mtp": True}])
        sect = config.read_sections()["qwopus"]
        self.assertEqual(sect["spec-draft-model"], "/m/mtp-q.gguf")
        self.assertEqual(sect["spec-type"], "draft-mtp")

    def test_does_not_overwrite_a_hand_set_spec_type(self):
        config.set_keys("ornith", {"model": "/m/o.gguf", "spec-type": "ngram-mod"})
        self._apply([{"id": "ornith", "model": "/m/o.gguf",
                      "draft_model": "/m/mtp-o.gguf", "draft_mtp": True}])
        sect = config.read_sections()["ornith"]
        self.assertEqual(sect["spec-type"], "ngram-mod", "clobbered a hand-set mode")
        # the draft model still attaches (it was absent), harmless while inert
        self.assertEqual(sect["spec-draft-model"], "/m/mtp-o.gguf")

    def test_attach_only_leaves_spec_type_unset(self):
        self._apply([{"id": "m", "model": "/m/m.gguf",
                      "draft_model": "/m/mtp-m.gguf"}])   # no draft_mtp
        sect = config.read_sections()["m"]
        self.assertEqual(sect["spec-draft-model"], "/m/mtp-m.gguf")
        self.assertNotIn("spec-type", sect)


if __name__ == "__main__":
    unittest.main()
