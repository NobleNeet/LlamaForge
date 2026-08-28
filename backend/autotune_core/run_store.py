"""Owner-aware, artifact-first persistence for recoverable benchmark runs."""
import json
import os
import socket
import time
import uuid

import atomicio


_TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class RunOwnershipError(RuntimeError):
    pass


class RunStore:
    def __init__(self, root_dir, instance_id=None, pid=None, hostname=None, clock=time.time, lease_seconds=60):
        self.root_dir = os.path.abspath(root_dir)
        self.instance_id = instance_id or str(uuid.uuid4())
        self.pid = os.getpid() if pid is None else pid
        self.hostname = hostname or socket.gethostname()
        self.clock = clock
        self.lease_seconds = lease_seconds

    def _run_dir(self, run_id):
        return os.path.join(self.root_dir, "runs", run_id)

    def _manifest_path(self, run_id):
        return os.path.join(self._run_dir(run_id), "manifest.json")

    def _artifact_dir(self, run_id):
        return os.path.join(self._run_dir(run_id), "invocations")

    def create_run(self, run_id, plan, target):
        os.makedirs(self._artifact_dir(run_id), exist_ok=False)
        now = self.clock()
        manifest = {"run_id": run_id, "status": "planned", "plan": plan, "target": target,
                    "invocation_ids": [], "owner_instance_id": self.instance_id, "owner_pid": self.pid,
                    "hostname": self.hostname, "started_at": None, "heartbeat_at": now,
                    "lease_expires_at": now + self.lease_seconds}
        atomicio.write_json(self._manifest_path(run_id), manifest)
        return manifest

    def load_manifest(self, run_id):
        with open(self._manifest_path(run_id), encoding="utf-8") as handle:
            return json.load(handle)

    def _save_manifest(self, manifest):
        atomicio.write_json(self._manifest_path(manifest["run_id"]), manifest)

    def _is_owner(self, manifest):
        return (manifest.get("owner_instance_id") == self.instance_id and manifest.get("owner_pid") == self.pid
                and manifest.get("hostname") == self.hostname)

    def _lease_active(self, manifest, pid_alive):
        if manifest.get("hostname") != self.hostname:
            return True
        owner_pid = manifest.get("owner_pid")
        alive = isinstance(owner_pid, int) and bool(pid_alive("/proc/%s" % owner_pid)) if os.name != "nt" else False
        return alive or self.clock() <= float(manifest.get("lease_expires_at") or 0)

    def acquire(self, run_id, pid_alive=os.path.exists):
        manifest = self.load_manifest(run_id)
        if manifest.get("status") in _TERMINAL:
            raise RunOwnershipError("terminal run cannot be acquired")
        if manifest.get("status") == "running" and not self._is_owner(manifest):
            if self._lease_active(manifest, pid_alive):
                raise RunOwnershipError("run is owned by a live foreign lease")
            raise RunOwnershipError("expired orphan must be reconciled before acquisition")
        now = self.clock()
        manifest.update({"status": "running", "owner_instance_id": self.instance_id, "owner_pid": self.pid,
                         "hostname": self.hostname, "started_at": manifest.get("started_at") or now,
                         "heartbeat_at": now, "lease_expires_at": now + self.lease_seconds})
        self._save_manifest(manifest)
        return manifest

    mark_running = acquire

    def _require_owner(self, manifest):
        if not self._is_owner(manifest):
            raise RunOwnershipError("run mutation requires the owning instance")
        if manifest.get("status") != "running":
            raise RunOwnershipError("run is not active")

    def heartbeat(self, run_id):
        manifest = self.load_manifest(run_id)
        self._require_owner(manifest)
        now = self.clock()
        manifest.update({"heartbeat_at": now, "lease_expires_at": now + self.lease_seconds})
        self._save_manifest(manifest)
        return manifest

    def raw_paths(self, run_id, invocation_id):
        directory = self._artifact_dir(run_id)
        return (os.path.join(directory, invocation_id + ".stdout.tmp"), os.path.join(directory, invocation_id + ".stderr.tmp"),
                os.path.join(directory, invocation_id + ".stdout"), os.path.join(directory, invocation_id + ".stderr"))

    def finalize_raw_files(self, temporary_stdout, temporary_stderr, stdout_path, stderr_path):
        os.replace(temporary_stdout, stdout_path)
        os.replace(temporary_stderr, stderr_path)

    def write_invocation(self, run_id, invocation_id, artifact):
        """The invocation document is committed before its manifest reference."""
        atomicio.write_json(os.path.join(self._artifact_dir(run_id), invocation_id + ".json"), artifact)

    def record_invocation(self, run_id, invocation_id, artifact):
        self.write_invocation(run_id, invocation_id, artifact)
        manifest = self.load_manifest(run_id)
        self._require_owner(manifest)
        ids = set(manifest.get("invocation_ids") or ())
        ids.add(invocation_id)
        manifest["invocation_ids"] = sorted(ids)
        self._save_manifest(manifest)

    def finish(self, run_id, status):
        if status not in _TERMINAL:
            raise ValueError("invalid terminal run status")
        manifest = self.load_manifest(run_id)
        self._require_owner(manifest)
        manifest["status"] = status
        manifest["finished_at"] = self.clock()
        self._save_manifest(manifest)
        return manifest

    def release(self, run_id, status):
        """Release the active lease by committing an explicit terminal status."""
        return self.finish(run_id, status)

    def _artifact_ids(self, run_id):
        directory = self._artifact_dir(run_id)
        if not os.path.isdir(directory):
            return []
        return sorted(name[:-5] for name in os.listdir(directory) if name.endswith(".json"))

    def reconcile_interrupted_runs(self, pid_alive=os.path.exists):
        """Only reclaim locally orphaned, expired leases; never kill foreign PIDs."""
        runs_dir = os.path.join(self.root_dir, "runs")
        if not os.path.isdir(runs_dir):
            return []
        changed = []
        now = self.clock()
        for run_id in sorted(os.listdir(runs_dir)):
            try:
                manifest = self.load_manifest(run_id)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if manifest.get("status") != "running":
                continue
            foreign_host = manifest.get("hostname") != self.hostname
            owner_pid = manifest.get("owner_pid")
            owner_alive = isinstance(owner_pid, int) and bool(pid_alive("/proc/%s" % owner_pid)) if os.name != "nt" else False
            lease_expired = now > float(manifest.get("lease_expires_at") or 0)
            if foreign_host or owner_alive or not lease_expired:
                continue
            manifest["invocation_ids"] = sorted(set(manifest.get("invocation_ids") or ()).union(self._artifact_ids(run_id)))
            manifest["status"] = "interrupted"
            manifest["recovered_at"] = now
            self._save_manifest(manifest)
            changed.append(run_id)
        return changed
