import conftest_paths  # noqa: F401
import json
import os
import tempfile
import threading
import unittest

from autotune_core.run_store import ResultArtifactSchemaError, RunOwnershipError, RunStore


class TestRunStore(unittest.TestCase):
    def test_artifact_is_atomic_before_manifest_reference(self):
        root = tempfile.mkdtemp()
        store = RunStore(root, instance_id="one", pid=10, hostname="host", clock=lambda: 10)
        store.create_run("run", {"x": 1}, {"model": "m"})
        store.write_invocation("run", "i1", {"ok": True})
        self.assertEqual(store.load_manifest("run")["invocation_ids"], [])
        store.acquire("run")
        store.record_invocation("run", "i1", {"ok": True})
        self.assertEqual(store.load_manifest("run")["invocation_ids"], ["i1"])
        self.assertEqual({"ok": True}, store.load_invocation("run", "i1"))
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

    def test_foreign_live_lease_rejected_same_owner_renews_and_terminal_mutations_rejected(self):
        root = tempfile.mkdtemp()
        first = RunStore(root, instance_id="one", pid=11, hostname="host", clock=lambda: 10, lease_seconds=100)
        first.create_run("run", {}, {})
        first.acquire("run")
        second = RunStore(root, instance_id="two", pid=22, hostname="host", clock=lambda: 20, lease_seconds=100)
        with self.assertRaises(RunOwnershipError):
            second.acquire("run", pid_alive=lambda path: True)
        renewed = first.acquire("run")
        self.assertEqual(renewed["owner_instance_id"], "one")
        first.finish("run", "completed")
        with self.assertRaises(RunOwnershipError):
            first.acquire("run")
        with self.assertRaises(RunOwnershipError):
            first.heartbeat("run")

    def test_concurrent_acquire_allows_exactly_one_owner(self):
        root = tempfile.mkdtemp()
        first = RunStore(root, instance_id="one", pid=11, hostname="host")
        second = RunStore(root, instance_id="two", pid=22, hostname="host")
        first.create_run("run", {}, {})
        barrier = threading.Barrier(2)
        outcomes = {}

        def acquire(store):
            barrier.wait()
            try:
                store.acquire("run")
                outcomes[store.instance_id] = "acquired"
            except RunOwnershipError:
                outcomes[store.instance_id] = "rejected"

        threads = [threading.Thread(target=acquire, args=(store,)) for store in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(outcomes.values()), ["acquired", "rejected"])
        self.assertIn(first.load_manifest("run")["owner_instance_id"], {"one", "two"})

    def test_foreign_record_is_rejected_before_artifact_write(self):
        root = tempfile.mkdtemp()
        owner = RunStore(root, instance_id="owner", pid=11, hostname="host")
        foreign = RunStore(root, instance_id="foreign", pid=22, hostname="host")
        owner.create_run("run", {}, {})
        owner.acquire("run")
        with self.assertRaises(RunOwnershipError):
            foreign.record_invocation("run", "forbidden", {"ok": True})
        artifact = os.path.join(root, "runs", "run", "invocations", "forbidden.json")
        self.assertFalse(os.path.exists(artifact))

    def test_live_stale_lock_is_not_reclaimed_but_orphan_lock_is(self):
        root = tempfile.mkdtemp()
        store = RunStore(root, lock_wait_seconds=0, lock_stale_seconds=0)
        store.create_run("run", {}, {})
        lock = store._lock_path("run")
        with open(lock, "w", encoding="utf-8") as handle:
            json.dump({"hostname": store.hostname, "pid": os.getpid()}, handle)
        with self.assertRaises(RunOwnershipError):
            store.acquire("run")
        os.unlink(lock)
        with open(lock, "w", encoding="utf-8") as handle:
            json.dump({"hostname": store.hostname, "pid": 99999999}, handle)
        self.assertEqual(store.acquire("run")["status"], "running")

    def test_result_artifact_schema_is_versioned_and_strict(self):
        root = tempfile.mkdtemp()
        store = RunStore(root, instance_id="one", pid=10, hostname="host")
        store.create_run("run", {}, {})
        store.acquire("run")
        from autotune_core.results import TuneResult
        store.record_result("run", TuneResult("r", "m", "h"), ())
        self.assertEqual(store.load_result("run")["schema_version"], 1)
        path = os.path.join(root, "runs", "run", "result.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 999}, handle)
        with self.assertRaises(ResultArtifactSchemaError):
            store.load_result("run")
