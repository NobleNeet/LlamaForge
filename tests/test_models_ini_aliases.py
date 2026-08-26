import conftest_paths  # noqa: F401
import os
import tempfile
import unittest

import config


class ModelsIniAliasNormalizationTest(unittest.TestCase):
    def test_rewrites_gpu_layers_alias_to_canonical_key(self):
        fd, path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "version = 1\n"
                "\n"
                "[*]\n"
                "ctx-size = 65536\n"
                "\n"
                "[m]\n"
                "model = /tmp/model.gguf\n"
                "gpu-layers = 99\n"
            )
        out = config.normalize_known_aliases(path)
        self.assertEqual(out["changed"], ["m"])
        secs = config.read_sections(path)
        self.assertEqual(secs["m"]["n-gpu-layers"], "99")
        self.assertNotIn("gpu-layers", secs["m"])

    def test_keeps_canonical_value_when_both_keys_exist(self):
        fd, path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "version = 1\n"
                "\n"
                "[m]\n"
                "model = /tmp/model.gguf\n"
                "n-gpu-layers = 4\n"
                "gpu-layers = 99\n"
            )
        out = config.normalize_known_aliases(path)
        self.assertEqual(out["changed"], ["m"])
        secs = config.read_sections(path)
        self.assertEqual(secs["m"]["n-gpu-layers"], "4")
        self.assertNotIn("gpu-layers", secs["m"])


if __name__ == "__main__":
    unittest.main()
