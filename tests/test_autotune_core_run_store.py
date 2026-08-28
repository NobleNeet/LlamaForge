import conftest_paths  # noqa: F401
import json
import os
import tempfile
import unittest

from autotune_core.run_store import RunStore


class TestRunStore(unittest.TestCase):
    def test_artifact_is_atomic_before_manifest_reference(self):
        root = tempfile.mkdtemp()
        store = RunStore(root, instance_id="one", pid=10, hostname="host", clock=lambda: 10)
        store.create_run("run", {"x": 1}, {"model": "m"})
        store.write_invocation("run", "i1", {"ok": True})
        self.assertEqual(store.load_manifest("run")["invocation_ids"], [])
        store.reconcile_interrupted_runs(pid_alive=lambda path: False)
        store.record_invocation("run", "i1", {"ok": True})
        self.assertEqual(store.load_manifest("run")["invocation_ids"], ["i1"])
        with open(os.path.join(root, "runs", "run", "invocations", "i1.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"ok": True})

    def test_recovery_skips_foreign_active_owner_and_reclaims_stale_orphan(self):
        root = tempfile.mkdtemp()
        foreign = RunStore(root, instance_id="foreign", pid=22, hostname="other", clock=lambda: 0, lease_seconds=1)
        foreign.create_run("foreign", {}, {})
        foreign.mark_running("foreign")
        local = RunStore(root, instance_id="local", pid=33, hostname="host", clock=lambda: 10, lease_seconds=1)
        local.create_run("orphan", {}, {})
        local.mark_running("orphan")
        orphan = local.load_manifest("orphan")
        orphan["lease_expires_at"] = 0
        local._save_manifest(orphan)
        self.assertEqual(local.reconcile_interrupted_runs(pid_alive=lambda path: False), ["orphan"])
        self.assertEqual(local.load_manifest("foreign")["status"], "running")
        self.assertEqual(local.load_manifest("orphan")["status"], "interrupted")

    def test_recovery_reconciles_artifact_written_before_manifest_update(self):
        root = tempfile.mkdtemp()
        store = RunStore(root, instance_id="old", pid=10, hostname="host", clock=lambda: 10, lease_seconds=1)
        store.create_run("run", {}, {})
        store.mark_running("run")
        store.write_invocation("run", "crash-window", {"complete": True})
        manifest = store.load_manifest("run")
        manifest["lease_expires_at"] = 0
        store._save_manifest(manifest)
        self.assertEqual(store.reconcile_interrupted_runs(pid_alive=lambda path: False), ["run"])
        self.assertEqual(store.load_manifest("run")["invocation_ids"], ["crash-window"])
