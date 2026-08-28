import conftest_paths  # noqa: F401
import tempfile
import unittest

from autotune_core.resource_lease import BenchmarkResourceLease, ResourceBusyError


class TestResourceLease(unittest.TestCase):
    def test_only_auto_tune_lease_owners_contend(self):
        root = tempfile.mkdtemp()
        first = BenchmarkResourceLease(root, "gpu-0", instance_id="one", pid=1).acquire()
        with self.assertRaises(ResourceBusyError):
            BenchmarkResourceLease(root, "gpu-0", instance_id="two", pid=2).acquire()
        first.release()
        BenchmarkResourceLease(root, "gpu-0", instance_id="two", pid=2).acquire().release()

    def test_hip_and_vulkan_share_the_same_physical_resource_identity(self):
        root = tempfile.mkdtemp()
        hip = BenchmarkResourceLease(root, "hardware-physical-gpu-1", instance_id="hip", pid=1).acquire()
        with self.assertRaises(ResourceBusyError):
            BenchmarkResourceLease(root, "hardware-physical-gpu-1", instance_id="vulkan", pid=2).acquire()
        hip.release()

    def test_expired_dead_owner_lease_is_recovered(self):
        root = tempfile.mkdtemp()
        stale = BenchmarkResourceLease(root, "gpu", instance_id="dead", pid=99999999, lease_seconds=1, clock=lambda: 0).acquire()
        self.assertEqual(BenchmarkResourceLease.reconcile_orphans(root, clock=lambda: 2), ["gpu"])
        BenchmarkResourceLease(root, "gpu", instance_id="live", pid=2).acquire().release()
