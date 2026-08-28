import conftest_paths  # noqa: F401
import unittest
from unittest import mock

from autotune_core.environment import capture_environment, environment_fingerprint, execution_environment, execution_fingerprint
from autotune_core.models import BenchBinaryIdentity


class TestEnvironment(unittest.TestCase):
    def test_canonicalizes_hip_and_keeps_one_physical_gpu(self):
        rows = [{"vendor": "AMD", "name": "Radeon 8060S", "architecture": "gfx1151",
                 "memory_total_mib": 32768, "is_uma": True, "backends": ["rocm", "vulkan"]}]
        with mock.patch("autotune_core.environment.hardware.detect_gpus", return_value=rows), \
             mock.patch("autotune_core.environment.hardware.available_backends", return_value=["rocm", "vulkan"]), \
             mock.patch("autotune_core.environment.hardware.detect_cpu", return_value={"name": "CPU", "cores": 8}):
            snapshot = capture_environment()
        self.assertEqual(len(snapshot.physical_gpus), 1)
        self.assertEqual(snapshot.physical_gpus[0].available_backends, ("hip", "vulkan"))
        self.assertEqual(snapshot.available_backends, ("hip", "vulkan"))

    def test_fingerprint_excludes_capture_time(self):
        rows = [{"vendor": "NVIDIA", "name": "GPU", "vram_mib": 1024, "backends": ["cuda"]}]
        with mock.patch("autotune_core.environment.hardware.detect_gpus", return_value=rows), \
             mock.patch("autotune_core.environment.hardware.available_backends", return_value=["cuda"]), \
             mock.patch("autotune_core.environment.hardware.detect_cpu", return_value={"name": "CPU", "cores": 8}):
            first, second = capture_environment(), capture_environment()
        self.assertNotEqual(first.captured_at, "")
        self.assertEqual(environment_fingerprint(first), environment_fingerprint(second))

    def test_pci_identity_beats_enumeration_index(self):
        rows = [{"vendor": "AMD", "name": "GPU", "pci_bus_id": "0000:C5:00.0", "backends": ["hip"]}]
        with mock.patch("autotune_core.environment.hardware.detect_gpus", return_value=rows), \
             mock.patch("autotune_core.environment.hardware.available_backends", return_value=["hip"]), \
             mock.patch("autotune_core.environment.hardware.detect_cpu", return_value={}):
            snapshot = capture_environment()
        self.assertEqual(snapshot.physical_gpus[0].stable_id, "pci_bus_id:0000:c5:00.0")

    def test_execution_fingerprint_tracks_runtime_and_binary_not_time(self):
        binary = BenchBinaryIdentity("hip", "/a/llama-bench", "build-a", "hash-a", "v1", "artifact")
        first = execution_environment("hardware", "rocm", {"rocm": "6.3", "driver": "x"}, binary, "one")
        second = execution_environment("hardware", "hip", {"rocm": "6.3", "driver": "x"}, binary, "two")
        changed = execution_environment("hardware", "hip", {"rocm": "6.4", "driver": "x"}, binary, "two")
        self.assertEqual(execution_fingerprint(first), execution_fingerprint(second))
        self.assertNotEqual(execution_fingerprint(first), execution_fingerprint(changed))
