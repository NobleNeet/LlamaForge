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

    def test_sanitize_drops_invalid_and_blank_keys_and_rebuilds_file(self):
        fd, path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "shell prompt garbage\n"
                "version = 1\n\n"
                "[*]\n"
                "ctx-size = 150000\n"
                "jinja = true\n"
                "load-on-startup = false\n\n"
                "[m]\n"
                "threads) = \n"
                "gpu-layers = 99\n"
                "spec-ngram-*-size-n = \n"
                "model = /tmp/model.gguf\n"
                "flash-attn = on"
            )
        out = config.sanitize_models_ini(
            path,
            valid_keys={"ctx-size", "jinja", "load-on-startup", "n-gpu-layers",
                        "model", "flash-attn"},
            alias_to_key={"gpu-layers": "n-gpu-layers", "n-gpu-layers": "n-gpu-layers"},
        )
        self.assertIn("__file__", out["changed"])
        secs = config.read_sections(path)
        self.assertEqual(secs["m"]["n-gpu-layers"], "99")
        self.assertNotIn("gpu-layers", secs["m"])
        self.assertNotIn("threads)", secs["m"])
        self.assertNotIn("spec-ngram-*-size-n", secs["m"])
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text.endswith("\n"))
        self.assertNotIn("shell prompt garbage", text)

    def test_sanitize_keeps_router_specific_model_keys_by_default(self):
        fd, path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "version = 1\n\n"
                "[m]\n"
                "model = /tmp/model.gguf\n"
                "mmproj = /tmp/mmproj.gguf\n"
                "spec-draft-model = /tmp/mtp.gguf\n"
                "embeddings = true\n"
            )
        out = config.sanitize_models_ini(path, valid_keys={"ctx-size"})
        self.assertEqual(out["changed"], [])
        secs = config.read_sections(path)
        self.assertEqual(secs["m"]["model"], "/tmp/model.gguf")
        self.assertEqual(secs["m"]["mmproj"], "/tmp/mmproj.gguf")
        self.assertEqual(secs["m"]["spec-draft-model"], "/tmp/mtp.gguf")
        self.assertEqual(secs["m"]["embeddings"], "true")


if __name__ == "__main__":
    unittest.main()
