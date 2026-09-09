import conftest_paths  # noqa: F401
import unittest

import argspec


class PlaceholderClassificationTest(unittest.TestCase):
    def test_open_ended_lists_are_strings(self):
        for names in ("dev1,dev2", "foo,bar"):
            for marker in ("..", "...", "…"):
                for delimiter in (",", "|"):
                    body = names.replace(",", delimiter) + delimiter + marker
                    for placeholder in (f"<{body}>", f"[{body}]", f"{{{body}}}", body):
                        with self.subTest(placeholder=placeholder):
                            self.assertEqual(argspec._classify(placeholder, ""), ("str", None))
                            item, = argspec.parse_help(f"--future-list {placeholder}  values to use\n")
                            self.assertEqual(item["type"], "str")
                            self.assertIsNone(item["options"])

    def test_open_ended_comma_lists_keep_multi_value_schema(self):
        for marker in ("..", "...", "…"):
            for desc in ("comma-separated list", "comma separated list", "values separated by commas"):
                for flag in ("-dev, --device", "--future-list"):
                    with self.subTest(marker=marker, desc=desc, flag=flag):
                        item, = argspec.parse_help(
                            f"{flag} <dev1,dev2,{marker}>\n    {desc} of devices to use\n"
                        )
                        self.assertEqual(item["type"], "str")
                        self.assertIsNone(item["options"])
                        self.assertTrue(item["multiple"])
                        self.assertEqual(item["separator"], ",")

    def test_finite_lists_remain_enums(self):
        for placeholder, options in (
            ("<none,layer,row>", ["none", "layer", "row"]),
            ("[auto|on|off]", ["auto", "on", "off"]),
        ):
            with self.subTest(placeholder=placeholder):
                self.assertEqual(argspec._classify(placeholder, ""), ("enum", options))
                item, = argspec.parse_help(f"--future-mode {placeholder}  mode to use\n")
                self.assertEqual(item["type"], "enum")
                self.assertEqual(item["options"], options)
                self.assertFalse(item["multiple"])
                self.assertEqual(item["separator"], "")

    def test_overrides_take_precedence_over_dynamic_enums(self):
        for key in ("tensor-split", "cache-type-k"):
            with self.subTest(key=key):
                item, = argspec.parse_help(f"--{key} <foo,bar>  values to use\n")
                self.assertEqual((item["type"], item["options"]), argspec.OVERRIDES[key])


if __name__ == "__main__":
    unittest.main()
