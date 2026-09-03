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

    def test_wrapped_removed_args_do_not_turn_into_fake_flags(self):
        text = """----- speculative params -----

--spec-draft-threads, -td, --threads-draft N
                                        number of threads to use during generation (default: same as
                                        --threads)
--spec-draft-cpu-strict, --cpu-strict-draft <0|1>
                                        Use strict CPU placement for draft model (default: same as
                                        --cpu-strict)
--spec-ngram-size-n N                   the argument has been removed. use the respective
                                        --spec-ngram-*-size-n or --spec-ngram-mod-n-match
"""
        keys = [it["key"] for it in argspec.parse_help(text)]
        self.assertIn("spec-draft-threads", keys)
        self.assertIn("spec-draft-cpu-strict", keys)
        self.assertIn("spec-ngram-size-n", keys)
        self.assertNotIn("threads)", keys)
        self.assertNotIn("cpu-strict)", keys)
        self.assertNotIn("spec-ngram-*-size-n", keys)

    def test_multi_value_positional_option_is_not_preset_editable(self):
        text = "--control-vector-layer-range START END  layer range to apply\n"
        items = argspec.parse_help(text)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["reserved"])

    def test_spec_type_is_dynamic_multi_enum(self):
        text = """----- speculative params -----

--spec-type none,draft-mtp,draft-dflash,ngram-cache,draft-future
    comma-separated list of types of speculative decoding to use
"""
        items = argspec.parse_help(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "enum")
        self.assertEqual(items[0]["options"],
                         ["none", "draft-mtp", "draft-dflash", "ngram-cache", "draft-future"])
        self.assertTrue(items[0]["multiple"])
        self.assertEqual(items[0]["separator"], ",")

    def test_single_enum_is_not_marked_multiple(self):
        text = "--pooling {none,mean,cls}  pooling strategy\n"
        items = argspec.parse_help(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "enum")
        self.assertFalse(items[0]["multiple"])

    def test_comma_separated_placeholder_is_marked_multiple_without_enum(self):
        text = "--tensor-split N0,N1,N2,...  fraction of the model to offload to each GPU\n"
        items = argspec.parse_help(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "str")
        self.assertTrue(items[0]["multiple"])


if __name__ == "__main__":
    unittest.main()
