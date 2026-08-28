import conftest_paths  # noqa: F401
import os
import tempfile
import unittest

from autotune_core.discovery import discover_gguf_sources, merge_model_sources, registered_model_source


class TestDiscovery(unittest.TestCase):
    def test_discovered_and_registered_sources_share_one_model_source_shape(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "nested", "model.gguf")
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as handle:
            handle.write(b"")
        discovered = discover_gguf_sources(directory)
        merged = merge_model_sources([registered_model_source("known", path)], discovered)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_kind, "registered")
        self.assertEqual(merged[0].registered_model_id, "known")
