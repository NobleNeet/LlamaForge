import conftest_paths  # noqa: F401
import unittest

from autotune_core.backends import backend_display_name, canonical_backend_id
from autotune_core.bench_artifacts import resolve_bench_binary


class TestBenchArtifacts(unittest.TestCase):
    def test_canonical_backend_ids_keep_display_separate(self):
        self.assertEqual(canonical_backend_id("ROCm"), "hip")
        self.assertEqual(canonical_backend_id("hip"), "hip")
        self.assertEqual(backend_display_name("hip"), "AMD ROCm/HIP")

    def test_exact_artifact_beats_server_sibling_guess(self):
        config = {"server_bin": "/build/llama-server", "autotune_build_artifacts": [
            {"backend": "rocm", "build_id": "build-a", "llama_bench_bin": "/artifacts/bench-a"}
        ]}
        ref = resolve_bench_binary(config, "hip", "build-a", exists=lambda path: path == "/artifacts/bench-a")
        self.assertEqual(ref.path, "/artifacts/bench-a")
        self.assertEqual(ref.provenance, "artifact")

    def test_sibling_is_only_a_fallback(self):
        config = {"server_bin": "/build/llama-server"}
        ref = resolve_bench_binary(config, "vulkan", exists=lambda path: path == "/build/llama-bench")
        self.assertEqual(ref.provenance, "sibling_fallback")
