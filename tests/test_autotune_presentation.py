import tempfile
import unittest

import conftest_paths  # noqa: F401

from autotune_core.orchestrator import GeneratedProfile
from autotune_core.results import TuneResult
from autotune_core.staleness import ProfileIdentity
from autotune_service import (AutoTuneService, IncompatibleBackendError, ProfileStaleError,
                              PRODUCTION_REPETITIONS, PRODUCTION_TIMEOUT_SECONDS)
import model_settings


SCHEMA = {"groups": [{"knobs": [
    {"key": "n-gpu-layers", "aliases": []}, {"key": "threads", "aliases": ["threads-http"]},
]}]}


class _Fingerprint:
    def __init__(self, value): self.value = value


class TestAutoTunePresentation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model_fp = "model-fingerprint"
        self.config = {"llama_backend": "cuda"}
        self.sections = {"qwen": {"model": "/models/qwen.gguf", "threads": "8", "n-gpu-layers": "20"}}
        self.service = AutoTuneService(self.temp.name, config_loader=lambda: self.config,
            fingerprint=lambda path: _Fingerprint(self.model_fp), schema_loader=lambda: SCHEMA,
            sections_loader=lambda: self.sections)
        identity = ProfileIdentity(self.model_fp, "hardware", {}, {"backend": "cuda"}, "cap", "rules", "strategy", "score", "cuda")
        profiles = tuple(GeneratedProfile(name, "candidate", {"n-gpu-layers": "all", "threads": 4}, "measured", "benchmark", identity)
                         for name in ("balanced", "fast_prefill", "fast_decode"))
        self.run_id = "11111111-1111-4111-8111-111111111111"
        self.failed_id = "22222222-2222-4222-8222-222222222222"
        self.service.store.create_run(self.run_id, {}, {"model_name": "qwen"})
        self.service.store.acquire(self.run_id)
        self.service.store.record_result(self.run_id, TuneResult(self.run_id, self.model_fp, "hardware"), profiles)
        self.service.store.finish(self.run_id, "completed")

    def tearDown(self): self.temp.cleanup()

    def test_preview_materializes_profile_without_mutating_config(self):
        before = dict(self.sections["qwen"])
        preview = self.service.preview(self.run_id, "balanced", "qwen")
        self.assertEqual("99", preview["settings"]["n-gpu-layers"])
        self.assertEqual("20", next(x for x in preview["changes"] if x["key"] == "n-gpu-layers")["current"])
        self.assertEqual(before, self.sections["qwen"])
        self.assertNotIn("/models/qwen.gguf", str(preview))

    def test_auto_backend_allows_preview_but_explicit_mismatch_rejects(self):
        self.config["llama_backend"] = "auto"
        self.assertEqual("not_evaluated", self.service.preview(self.run_id, "balanced", "qwen")["staleness"]["state"])
        self.config["llama_backend"] = "hip"
        with self.assertRaises(IncompatibleBackendError): self.service.preview(self.run_id, "balanced", "qwen")

    def test_model_fingerprint_mismatch_is_stale(self):
        self.service.fingerprint = lambda path: _Fingerprint("changed")
        with self.assertRaises(ProfileStaleError): self.service.preview(self.run_id, "balanced", "qwen")

    def test_specialist_profiles_and_result_provenance_are_available(self):
        self.assertEqual("fast_prefill", self.service.preview(self.run_id, "fast_prefill", "qwen")["profile"])
        self.assertEqual("fast_decode", self.service.preview(self.run_id, "fast_decode", "qwen")["profile"])
        self.assertEqual("cuda", self.service.result(self.run_id)["profiles"][0]["provenance"]["backend"])

    def test_status_redacts_unknown_error_text(self):
        self.service.store.create_run(self.failed_id, {}, {"model_name": "qwen"})
        self.service.store.acquire(self.failed_id)
        self.service.store.fail(self.failed_id, "OSError", "/private/path command stderr")
        error = self.service.status(self.failed_id)["error"]
        self.assertEqual("internal_error", error["code"])
        self.assertNotIn("/private/path", error["message"])


class TestAutoTuneMaterialization(unittest.TestCase):
    def test_strategy_v2_service_execution_defaults(self):
        self.assertEqual(2, PRODUCTION_REPETITIONS)
        self.assertEqual(300, PRODUCTION_TIMEOUT_SECONDS)

    def test_all_is_materialized_but_partial_offload_is_not(self):
        self.assertEqual("99", model_settings.materialize_autotune_settings({"n-gpu-layers": "all"}, SCHEMA)["settings"]["n-gpu-layers"])
        self.assertEqual("20", model_settings.materialize_autotune_settings({"n-gpu-layers": 20}, SCHEMA)["settings"]["n-gpu-layers"])
        self.assertEqual("0", model_settings.materialize_autotune_settings({"n-gpu-layers": 0}, SCHEMA)["settings"]["n-gpu-layers"])

    def test_unsupported_knob_remains_visible_as_a_warning(self):
        result = model_settings.materialize_autotune_settings({"unsupported": "yes"}, SCHEMA)
        self.assertFalse(result["applicable"])
        self.assertEqual("unsupported_knob", result["warnings"][0]["code"])
