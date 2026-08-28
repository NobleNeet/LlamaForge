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
