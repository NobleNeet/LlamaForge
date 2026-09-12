import conftest_paths  # noqa: F401
import json
import unittest
from unittest import mock

import autotune as at
import autotune_hardware
import routes

G = 1024 ** 3
META = dict(architecture="test", block_count=48, embedding_length=5376,
            head_count=32, head_count_kv=8, context_length=131072,
            quantization="Q4_K_M")


def hw(vram=16, ram=64, uma=False):
    return {"gpus": [{"name": "test", "backends": ["hip", "vulkan"] if uma else ["cuda"],
                      "is_uma": uma, "memory_total_mib": vram * 1024,
                      "memory_free_mib": vram * 1024}],
            "ram_total_bytes": ram * G, "ram_available_bytes": ram * G,
            "cpu": {"cores": 16, "threads": 32}}


class StaticRecommendationTests(unittest.TestCase):
    def recommend(self, machine, size=20, meta=None):
        result = at.recommend(META if meta is None else meta, machine, size_bytes=size * G)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["default"], "balanced")
        for name in at.PRESETS:
            rec = result[name]
            self.assertEqual(set(rec["rationale"]), set(rec["knobs"]))
            if rec["applicable"]:
                m = rec["memory"]
                self.assertLessEqual(m["ram_used_bytes"] + m["ram_reserve_bytes"], m["ram_available_bytes"])
                self.assertLessEqual(m["gpu_used_bytes"] + m["gpu_reserve_bytes"], m["gpu_available_bytes"])
        return result

    def test_a_large_uma_preserves_quality_and_full_offload(self):
        r = self.recommend(hw(128, 128, True))
        for n in at.PRESETS:
            self.assertEqual(r[n]["knobs"]["n-gpu-layers"], "49")
            self.assertEqual(r[n]["knobs"]["cache-type-k"], "f16")
            self.assertLessEqual(int(r[n]["knobs"]["threads"]), 8)
        self.assertEqual(r["balanced"]["knobs"]["batch-size"], "2048")

    def test_b_discrete_partial_after_buffers(self):
        r = self.recommend(hw())
        for n in at.PRESETS:
            self.assertTrue(r[n]["applicable"])
            self.assertTrue(0 < int(r[n]["knobs"]["n-gpu-layers"]) < 48)
            m = r[n]["memory"]
            self.assertLess(m["gpu_weight_budget_bytes"], 16 * G - m["kv_bytes"])

    def test_c_safe_has_fewer_layers_on_small_gpu(self):
        r = self.recommend(hw(8))
        self.assertLess(int(r["safe"]["knobs"]["n-gpu-layers"]), int(r["aggressive"]["knobs"]["n-gpu-layers"]))

    def test_d_all_presets_full_with_large_vram(self):
        r = self.recommend(hw(64))
        for n in at.PRESETS:
            self.assertEqual(r[n]["knobs"]["n-gpu-layers"], "49")
            self.assertEqual(r[n]["knobs"]["cache-type-v"], "f16")

    def test_e_missing_and_malformed_metadata(self):
        for meta in ({}, {"block_count": "broken", "embedding_length": -2, "head_count": float('nan')}):
            r = self.recommend(hw(), 4, meta)
            self.assertEqual(r["balanced"]["confidence"], "low")
            self.assertEqual(r["balanced"]["knobs"]["n-gpu-layers"], "0")
        r = at.recommend({}, {})
        self.assertFalse(r["aggressive"]["applicable"])

    def test_f_moe_total_residency_active_bandwidth(self):
        dense = self.recommend(hw())
        moe = self.recommend(hw(), meta={**META, "expert_count": 64, "expert_used_count": 8})
        self.assertEqual(dense["geometry"]["weights"], moe["geometry"]["weights"])
        self.assertLess(moe["geometry"]["active_params"], moe["geometry"]["total_params"])
        self.assertEqual(dense["balanced"]["knobs"], moe["balanced"]["knobs"])
        self.assertGreater(moe["balanced"]["prediction"]["theoretical_decode_ceiling"], dense["balanced"]["prediction"]["theoretical_decode_ceiling"])

    def test_g_cpu_uses_physical_cores(self):
        machine = hw(); machine["gpus"] = []
        r = self.recommend(machine)
        for n in at.PRESETS:
            self.assertEqual(r[n]["knobs"]["n-gpu-layers"], "0")
            self.assertEqual(r[n]["knobs"]["threads"], "16")
            self.assertEqual(r[n]["knobs"]["batch-size"], "512")

    def test_no_compute_backend_is_not_a_usable_gpu(self):
        machine = hw(); machine["gpus"][0]["backends"] = []
        self.assertEqual(self.recommend(machine)["balanced"]["knobs"]["n-gpu-layers"], "0")

    def test_free_memory_and_backend_override(self):
        machine = hw(); machine["gpus"][0]["memory_free_mib"] = 2 * 1024
        low = self.recommend(machine)
        normal = self.recommend(hw())
        self.assertLess(int(low["balanced"]["knobs"]["n-gpu-layers"]), int(normal["balanced"]["knobs"]["n-gpu-layers"]))
        machine["backend"] = "cpu"
        self.assertEqual(self.recommend(machine)["balanced"]["knobs"]["n-gpu-layers"], "0")

    def test_multiple_gpu_capacity_not_pooled(self):
        machine = hw(); machine["gpus"] *= 2
        r = self.recommend(machine)
        self.assertEqual(r["hardware"]["gpu_available"], 16 * G)
        self.assertEqual(r["balanced"]["knobs"]["split-mode"], "none")

    def test_uma_not_double_counted_or_small_carveout_limited(self):
        machine = hw(16, 128, True)
        machine["gpus"][0].update(gtt_total_mib=96 * 1024, gtt_used_mib=0)
        r = self.recommend(machine)
        self.assertEqual(r["balanced"]["knobs"]["n-gpu-layers"], "49")
        machine["ram_available_bytes"] = 10 * G
        r = self.recommend(machine)
        self.assertFalse(r["aggressive"]["applicable"])
        self.assertLessEqual(r["hardware"]["gpu_available"], 10 * G)

    def test_unequal_layer_sizes_accumulated_from_tail(self):
        meta = {**META, "block_count": 4, "layer_weight_bytes": [G, G, G, 10 * G], "weight_bytes": 13 * G}
        r = self.recommend(hw(8), 13, meta)
        self.assertEqual(r["balanced"]["knobs"]["n-gpu-layers"], "0")
        reverse = self.recommend(hw(8), 13, {**meta, "layer_weight_bytes": [10 * G, G, G, G]})
        self.assertGreater(int(reverse["balanced"]["knobs"]["n-gpu-layers"]), 0)

    def test_no_fit_is_not_applicable_and_context_shrinks(self):
        r = self.recommend(hw(1, 2), 20)
        self.assertFalse(r["aggressive"]["applicable"])
        self.assertTrue(r["aggressive"]["warnings"])
        self.assertLess(int(r["aggressive"]["knobs"]["ctx-size"]), 8192)

    def test_context_never_exceeds_trained(self):
        r = self.recommend(hw(64), meta={**META, "context_length": 1000})
        self.assertEqual(r["balanced"]["knobs"]["ctx-size"], "1000")

    def test_quant_block_overhead_and_asymmetric_heads(self):
        g = at.model_geometry({**META, "key_length": 256, "value_length": 128}, 20 * G)
        m = at.estimate_memory(g, 8192, "q8_0", 512, 128, "cuda")
        self.assertAlmostEqual(m["kv_bytes"], 48 * 8192 * 8 * 384 * 34 / 32 * 1.12)

    def test_api_is_read_only_and_has_only_recommend_route(self):
        with mock.patch.object(routes.config, "read_sections", return_value={"m": {"model": "/m.gguf"}}), \
             mock.patch.object(routes.gguf, "inspect_model", return_value={**META, "file_size_bytes": 20 * G}), \
             mock.patch.object(routes.autotune_hardware, "snapshot", return_value=hw()), \
             mock.patch.object(routes, "cfg", return_value={}), \
             mock.patch.object(routes, "router", side_effect=AssertionError("router called")), \
             mock.patch.object(routes.config, "set_keys", side_effect=AssertionError("write called")):
            result = routes._autotune_recommend({"model": "m"})
        self.assertIn("balanced", result)
        endpoints = set(routes.GET_ROUTES) | set(routes.POST_ROUTES)
        self.assertEqual({p for p in endpoints if p.startswith('/api/autotune/')}, {'/api/autotune/recommend'})

    def test_debug_diagnostics(self):
        with self.assertLogs('autotune', 'DEBUG') as log:
            self.recommend(hw())
        self.assertIn('gpu_weight_budget_bytes', log.output[0])
        self.assertIn('threads=', log.output[0])


class HardwareSnapshotTests(unittest.TestCase):
    def test_apple_silicon_uses_existing_metal_budget(self):
        with mock.patch.object(autotune_hardware.osplat, 'IS_MAC', True), \
             mock.patch.object(autotune_hardware.platform, 'machine', return_value='arm64'), \
             mock.patch.object(autotune_hardware.hardware, 'detect_gpus', return_value=[]), \
             mock.patch.object(autotune_hardware.hardware, 'detect_cpu', return_value={'cores': 12}), \
             mock.patch.object(autotune_hardware.osplat, 'total_ram_bytes', return_value=128 * G), \
             mock.patch.object(autotune_hardware.osplat, 'available_ram_bytes', return_value=100 * G):
            s = autotune_hardware.snapshot()
        h = at.hardware_budget(s)
        self.assertTrue(h.uma)
        self.assertEqual(h.backend, 'metal')
        self.assertLessEqual(h.gpu_available, 100 * G)

    def test_detection_failure_is_optional(self):
        with mock.patch.object(autotune_hardware.hardware, 'detect_gpus', side_effect=RuntimeError), \
             mock.patch.object(autotune_hardware.hardware, 'detect_cpu', side_effect=RuntimeError), \
             mock.patch.object(autotune_hardware.osplat, 'total_ram_bytes', return_value=0), \
             mock.patch.object(autotune_hardware.osplat, 'available_ram_bytes', return_value=0), \
             mock.patch.object(autotune_hardware.osplat, 'IS_MAC', False):
            r = at.recommend({}, autotune_hardware.snapshot())
        self.assertEqual(r['balanced']['confidence'], 'low')


class AdditionalBoundaryTests(unittest.TestCase):
    def test_nvidia_snapshot_reads_free_memory_from_telemetry(self):
        gpu = {'vendor': 'NVIDIA', 'index': 0, 'backends': ['cuda'], 'memory_total_mib': 16384}
        with mock.patch.object(autotune_hardware.hardware, 'detect_gpus', return_value=[gpu]), \
             mock.patch.object(autotune_hardware.hardware, 'detect_gpu_telemetry', return_value=[
                 {'vendor': 'NVIDIA', 'index': 0, 'memory_free_mib': 4096}]), \
             mock.patch.object(autotune_hardware.hardware, 'detect_cpu', return_value={'cores': 8}), \
             mock.patch.object(autotune_hardware.osplat, 'total_ram_bytes', return_value=64 * G), \
             mock.patch.object(autotune_hardware.osplat, 'available_ram_bytes', return_value=32 * G), \
             mock.patch.object(autotune_hardware.osplat, 'IS_MAC', False):
            h = at.hardware_budget(autotune_hardware.snapshot())
        self.assertEqual(h.gpu_available, 4 * G)

    def test_unknown_model_does_not_contact_router(self):
        with mock.patch.object(routes.config, 'read_sections', return_value={}), \
             mock.patch.object(routes, 'router', side_effect=AssertionError('router contacted')):
            self.assertIn('error', routes._autotune_recommend({'model': 'missing'}))

    def test_partial_preset_binding_preserves_layer_count(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(routes.config, 'CONFIG', os.path.join(tmp, 'config.json')):
            routes.config.save({'models_ini': os.path.join(tmp, 'models.ini')})
            routes.config.set_keys('m', {'model': '/m.gguf'})
            routes.config.save_preset('m', 'balanced', {'n-gpu-layers': '17'})
            with mock.patch.object(routes, 'schema', return_value={}):
                _, out = routes.post_presets_bind(routes.Req(body={'model': 'm', 'name': 'balanced'}))
            self.assertEqual(out['settings']['n-gpu-layers'], '17')
            self.assertEqual(routes.config.read_sections()['m']['n-gpu-layers'], '17')


class BackendSelectionTest(unittest.TestCase):
    def test_mixed_devices_follow_configured_backend(self):
        gpus = [{"backends": ["cuda"], "memory_total_mib": 8 * 1024},
                {"backends": ["hip", "vulkan"], "memory_total_mib": 24 * 1024}]
        with mock.patch.object(autotune_hardware.hardware, 'detect_gpus', return_value=gpus), \
             mock.patch.object(autotune_hardware.hardware, 'detect_cpu', return_value={'cores': 8}), \
             mock.patch.object(autotune_hardware.osplat, 'total_ram_bytes', return_value=64 * G), \
             mock.patch.object(autotune_hardware.osplat, 'available_ram_bytes', return_value=32 * G), \
             mock.patch.object(autotune_hardware.osplat, 'IS_MAC', False):
            automatic = at.hardware_budget(autotune_hardware.snapshot())
            explicit = at.hardware_budget(autotune_hardware.snapshot({'llama_backend': 'vulkan'}))
        self.assertEqual(automatic.backend, 'cuda')
        self.assertEqual(automatic.gpu_available, 8 * G)
        self.assertEqual(explicit.backend, 'vulkan')
        self.assertEqual(explicit.gpu_available, 24 * G)
