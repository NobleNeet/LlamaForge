"""Projector discovery and persistence across scan and download registration."""
import conftest_paths  # noqa: F401
import json
import os
import tempfile
import unittest
from unittest import mock

import config
import routes
import scanner


class ProjectorScanTest(unittest.TestCase):
    def test_model_families_attach_without_metadata_allowlist(self):
        for name in ("gemma-4-31B-it-QAT-Q4_0", "Qwen3.5-4B", "future-model"):
            with self.subTest(name=name), mock.patch(
                    "gguf.metadata", side_effect=AssertionError("must not inspect architecture")):
                model = f"/models/{name}.gguf"
                projector = f"/models/mmproj-{name}-BF16.gguf"
                entries = scanner.build_entries([model, projector])
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]["model"], model)
                self.assertEqual(entries[0]["mmproj"], projector)

    def test_missing_or_other_directory_projector_does_not_attach(self):
        for paths in (["/models/text.gguf"],
                      ["/models/text.gguf", "/other/mmproj.gguf"]):
            with self.subTest(paths=paths):
                (entry,) = scanner.build_entries(paths)
                self.assertNotIn("mmproj", entry)

    def test_multiple_projectors_are_not_selected_arbitrarily(self):
        paths = ["/models/model.gguf", "/models/mmproj-a.gguf", "/models/mmproj-b.gguf"]
        for order in (paths, list(reversed(paths))):
            (entry,) = scanner.build_entries(order)
            self.assertNotIn("mmproj", entry)

    def test_duplicate_projector_path_is_not_ambiguous(self):
        (entry,) = scanner.build_entries([
            "/models/model.gguf", "/models/mmproj.gguf", "/models/mmproj.gguf"])
        self.assertEqual(entry["mmproj"], "/models/mmproj.gguf")

    def test_projector_attaches_to_each_quantization_and_first_shard(self):
        paths = ["/models/m-Q4.gguf", "/models/m-Q8-00001-of-00002.gguf",
                 "/models/m-Q8-00002-of-00002.gguf", "/models/MMPROJ-F16.GGUF"]
        entries = scanner.build_entries(paths)
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertEqual(entry["mmproj"], paths[-1])

    def test_small_projector_is_included_but_small_models_are_filtered(self):
        with tempfile.TemporaryDirectory() as folder:
            sizes = {"gemma-4.gguf": 50 * 1024 * 1024,
                     "mmproj-gemma-4.gguf": 1024,
                     "small.gguf": 1024, "mmproj-empty.gguf": 0}
            for name, size in sizes.items():
                with open(os.path.join(folder, name), "wb") as stream:
                    stream.truncate(size)
            (entry,) = scanner.scan([folder])
            self.assertEqual(entry["mmproj"], os.path.join(folder, "mmproj-gemma-4.gguf"))
            self.assertEqual(entry["model"], os.path.join(folder, "gemma-4.gguf"))


class ProjectorRegistrationTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        cfg = os.path.join(self.folder, "config.json")
        with open(cfg, "w") as stream:
            json.dump({"models_ini": os.path.join(self.folder, "models.ini")}, stream)
        for patch in (mock.patch.object(config, "CONFIG", cfg),
                      mock.patch.object(config, "apply_ctx_defaults"),
                      mock.patch.object(routes, "router", return_value=(200, {}))):
            patch.start()
            self.addCleanup(patch.stop)

    def register(self, via, entry):
        if via == "scan":
            routes.post_scan_apply(routes.Req(body={"entries": [entry]}))
        else:
            with mock.patch.object(scanner, "build_entries", return_value=[entry]):
                routes._register_ggufs_beside([])
        self.assertEqual(routes.router.call_args, mock.call("/models?reload=1"))
        return config.read_sections()[entry["id"]]

    def test_scan_and_download_persist_detected_projector(self):
        for via in ("scan", "download"):
            with self.subTest(via=via):
                entry = {"id": via, "model": "/m/model.gguf", "mmproj": "/m/mmproj.gguf"}
                self.assertEqual(self.register(via, entry)["mmproj"], entry["mmproj"])

    def test_existing_model_without_projector_is_repaired(self):
        for via in ("scan", "download"):
            with self.subTest(via=via):
                config.set_keys(via, {"model": "/m/model.gguf"})
                entry = {"id": via, "model": "/m/model.gguf", "mmproj": "/m/mmproj.gguf"}
                self.assertEqual(self.register(via, entry)["mmproj"], entry["mmproj"])

    def test_existing_explicit_projector_survives_missing_or_different_detection(self):
        for via in ("scan", "download"):
            for detected in (None, "/m/mmproj-other.gguf"):
                with self.subTest(via=via, detected=detected):
                    config.set_keys(via, {"model": "/m/model.gguf", "mmproj": "/custom/projector.gguf"})
                    entry = {"id": via, "model": "/m/model.gguf", "mmproj": detected}
                    self.assertEqual(self.register(via, entry)["mmproj"], "/custom/projector.gguf")

    def test_changed_model_does_not_inherit_old_projector(self):
        for via in ("scan", "download"):
            for detected in (None, "/new/mmproj.gguf"):
                with self.subTest(via=via, detected=detected):
                    config.set_keys(via, {"model": "/old/model.gguf", "mmproj": "/old/mmproj.gguf"})
                    entry = {"id": via, "model": "/new/model.gguf", "mmproj": detected}
                    self.assertEqual(self.register(via, entry).get("mmproj"), detected)

    def test_scan_through_apply_writes_projector_to_router_preset(self):
        model = os.path.join(self.folder, "gemma-4-31B-it-QAT-Q4_0.gguf")
        projector = os.path.join(self.folder, "mmproj-gemma-4-31B-it-QAT-BF16.gguf")
        for path, size in ((model, 50 * 1024 * 1024), (projector, 1024)):
            with open(path, "wb") as stream:
                stream.truncate(size)
        (entry,) = scanner.scan([self.folder])
        section = self.register("scan", entry)
        self.assertEqual(section["model"], model)
        self.assertEqual(section["mmproj"], projector)
