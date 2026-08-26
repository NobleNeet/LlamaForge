import conftest_paths  # noqa: F401
import unittest
import autotune

MIB = 1024 * 1024


def gpu(vram_mib, cc="8.6"):
    return {"vram_mib": vram_mib, "compute_cap": cc}


class TestRecommendCore(unittest.TestCase):
    def test_no_gpu_is_cpu_only(self):
        hw = {"gpus": [], "cpu": {"threads": 16, "cores": 8}}
        r = autotune.recommend({"block_count": 32}, hw, "balanced")
        self.assertEqual(r["knobs"]["n-gpu-layers"], "0")
        self.assertEqual(r["knobs"]["flash-attn"], "off")
        self.assertEqual(r["knobs"]["threads"], "16")

    def test_small_model_full_offload(self):
        hw = {"gpus": [gpu(24000)], "cpu": {"threads": 24, "cores": 12}}
        r = autotune.recommend({"block_count": 32}, hw, "balanced",
                               size_bytes=5 * 1024 * MIB)  # ~5 GB weights, fits 24 GB
        self.assertEqual(r["knobs"]["n-gpu-layers"], "99")
        self.assertEqual(r["knobs"]["flash-attn"], "on")

    def test_big_model_prefers_full_offload(self):
        hw = {"gpus": [gpu(8000)], "cpu": {"threads": 16, "cores": 8}}
        # Even when the model is obviously larger than VRAM, recommend max
        # offload and let llama.cpp decide whether the load fits.
        r = autotune.recommend({"block_count": 80}, hw, "balanced",
                               size_bytes=40 * 1024 * MIB)
        self.assertEqual(r["knobs"]["n-gpu-layers"], "99")

    def test_rationale_present_for_each_knob(self):
        hw = {"gpus": [gpu(24000)], "cpu": {"threads": 24, "cores": 12}}
        r = autotune.recommend({"block_count": 32}, hw, "balanced",
                               size_bytes=5 * 1024 * MIB)
        for k in r["knobs"]:
            self.assertIn(k, r["rationale"])
            self.assertTrue(r["rationale"][k])

    def test_unknown_meta_degrades_gracefully(self):
        hw = {"gpus": [gpu(24000)], "cpu": {"threads": 24, "cores": 12}}
        r = autotune.recommend({}, hw, "balanced", size_bytes=None)
        # no layers / no size known -> safe full-offload attempt, no crash
        self.assertIn("n-gpu-layers", r["knobs"])
        # rationale should indicate unknown, not falsely claim weights fit
        rationale = r["rationale"]["n-gpu-layers"].lower()
        self.assertIn("unknown", rationale)


if __name__ == "__main__":
    unittest.main()
