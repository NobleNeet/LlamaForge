import conftest_paths  # noqa: F401
import unittest

import argspec


class CanonicalAliasTest(unittest.TestCase):
    def test_prefers_legacy_gpu_layers_key_when_both_aliases_exist(self):
        text = "--gpu-layers, --n-gpu-layers N  number of layers to store in VRAM\n"
        items = argspec.parse_help(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["key"], "n-gpu-layers")
        self.assertEqual(items[0]["aliases"], ["n-gpu-layers", "gpu-layers"])


if __name__ == "__main__":
    unittest.main()
