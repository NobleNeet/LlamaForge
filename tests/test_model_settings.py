import conftest_paths  # noqa: F401
import os
import shutil
import tempfile
import unittest

import config
import model_settings


MULTI_SCHEMA = {"groups": [{"name": "speculative", "knobs": [{
    "key": "spec-type",
    "aliases": ["spec-type"],
    "type": "enum",
    "options": ["none", "draft-dflash", "ngram-cache"],
    "multiple": True,
    "separator": ",",
}]}]}


class ModelSettingsTest(unittest.TestCase):
    def test_clean_settings_joins_multi_values_for_save(self):
        clean = model_settings.clean_settings(
            {"spec-type": ["draft-dflash", "ngram-cache"]},
            MULTI_SCHEMA,
        )
        self.assertEqual(clean["spec-type"], "draft-dflash,ngram-cache")

    def test_clean_settings_keeps_single_multi_value_compatible(self):
        clean = model_settings.clean_settings({"spec-type": "draft-dflash"}, MULTI_SCHEMA)
        self.assertEqual(clean["spec-type"], "draft-dflash")


class ModelIniRoundTripTest(unittest.TestCase):
    def test_models_ini_round_trip_preserves_comma_separated_value(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "models.ini")
        config.set_keys("m", {"spec-type": "draft-dflash,ngram-cache"}, path=path)
        self.assertEqual(config.read_sections(path)["m"]["spec-type"], "draft-dflash,ngram-cache")


if __name__ == "__main__":
    unittest.main()
