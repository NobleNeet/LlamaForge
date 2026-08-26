import conftest_paths  # noqa: F401
import unittest

import hardware


class TestHardwareParsing(unittest.TestCase):
    def test_parse_nvidia_csv_keeps_existing_shape(self):
        rows = hardware._parse_nvidia_csv("0, NVIDIA GeForce RTX 4090, 24564, 8.9\n")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vendor"], "NVIDIA")
        self.assertEqual(rows[0]["backend"], "cuda")
        self.assertEqual(rows[0]["vram_mib"], 24564)
        self.assertEqual(rows[0]["compute_cap"], "8.9")

    def test_rocminfo_arch_parses_gfx1151(self):
        text = "Agent 2\n  Name: gfx1151\n"
        self.assertEqual(hardware._rocminfo_arches(text), ["gfx1151"])

    def test_vulkan_devices_parse_summary_style(self):
        text = "GPU0 : AMD Radeon 8060S Graphics\nGPU1 : llvmpipe (LLVM 18.1.0)\n"
        out = hardware._vulkan_devices(text)
        self.assertEqual(out[0]["name"], "AMD Radeon 8060S Graphics")
        self.assertEqual(out[1]["name"], "llvmpipe (LLVM 18.1.0)")

    def test_software_vulkan_devices_are_flagged(self):
        self.assertTrue(hardware._is_software_vulkan_device("llvmpipe (LLVM 18.1.0)"))
        self.assertFalse(hardware._is_software_vulkan_device("AMD Radeon 8060S Graphics"))

    def test_amd_card_names_can_be_renamed_from_vulkan(self):
        base = [hardware._gpu_row(name="card1", vendor="AMD", index=0, total=1024, integrated=True, uma=True)]
        vk = [{"name": "Radeon 8060S Graphics", "vendor": "AMD", "driver": "AMD proprietary"}]
        out = hardware._rename_amd_base_with_vulkan_names(base, vk)
        self.assertEqual(out[0]["name"], "Radeon 8060S Graphics")

    def test_gpu_row_keeps_local_and_gtt_memory_fields(self):
        row = hardware._gpu_row(name="apu", vendor="AMD", total=122880, used=25095,
                                local_total=1024, local_used=814, gtt_total=122880, gtt_used=25095,
                                uma=True)
        self.assertEqual(row["local_memory_total_mib"], 1024)
        self.assertEqual(row["gtt_total_mib"], 122880)
        self.assertTrue(row["is_uma"])

    def test_merge_keeps_one_physical_gpu_for_hip_and_vulkan(self):
        base = [hardware._gpu_row(name="AMD Radeon 8060S Graphics", vendor="AMD",
                                  index=0, total=131072, integrated=True, uma=True)]
        hip = [hardware._gpu_row(name="AMD Radeon 8060S Graphics", vendor="AMD",
                                 backend="hip", arch="gfx1151")]
        vk = [hardware._gpu_row(name="AMD Radeon 8060S Graphics", vendor="AMD",
                                backend="vulkan")]
        out = hardware._merge_gpu_lists(base, hip)
        out = hardware._merge_gpu_lists(out, vk)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["backends"], ["hip", "vulkan"])
        self.assertEqual(out[0]["architecture"], "gfx1151")
        self.assertTrue(out[0]["is_uma"])


class TestRecommend(unittest.TestCase):
    def test_auto_prefers_cuda_then_hip_then_vulkan(self):
        rec = hardware.recommend(gpus=[{"name": "RTX", "backends": ["cuda"], "compute_cap": "8.9"}],
                                 cpu={"avx512_hint": False}, backend="auto")
        self.assertEqual(rec["selected_backend"], "cuda")
        self.assertEqual(rec["cmake_flags"]["GGML_CUDA"], "ON")

    def test_explicit_hip_emits_gpu_targets(self):
        gpu = {"name": "AMD Radeon 8060S Graphics", "vendor": "AMD", "backends": ["hip", "vulkan"],
               "architecture": "gfx1151", "is_uma": True}
        rec = hardware.recommend(gpus=[gpu], cpu={"avx512_hint": False}, backend="hip")
        self.assertEqual(rec["selected_backend"], "hip")
        self.assertEqual(rec["cmake_flags"]["GGML_HIP"], "ON")
        self.assertEqual(rec["cmake_flags"]["GPU_TARGETS"], "gfx1151")

    def test_explicit_vulkan_on_uma_keeps_gpu_runtime(self):
        gpu = {"name": "AMD Radeon 8060S Graphics", "vendor": "AMD", "backends": ["vulkan"],
               "is_uma": True}
        rec = hardware.recommend(gpus=[gpu], cpu={"avx512_hint": False}, backend="vulkan")
        self.assertEqual(rec["selected_backend"], "vulkan")
        self.assertEqual(rec["runtime"]["n-gpu-layers"], "99")
        self.assertIn("UMA GPU detected", " ".join(rec["notes"]))

    def test_cpu_backend_when_no_gpu(self):
        rec = hardware.recommend(gpus=[], cpu={"avx512_hint": False}, backend="auto")
        self.assertEqual(rec["selected_backend"], "cpu")
        self.assertEqual(rec["runtime"]["n-gpu-layers"], "0")

    def test_total_fit_vram_ignores_uma(self):
        gpus = [{"vram_mib": 24576}, {"vram_mib": 131072, "is_uma": True, "fit_vram_mib": None}]
        self.assertEqual(hardware.total_fit_vram_mib(gpus), 24576)


if __name__ == "__main__":
    unittest.main()
