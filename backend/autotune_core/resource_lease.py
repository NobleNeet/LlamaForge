"""Owner-aware Auto Tune-only device leases; never manage serving processes."""
import json
import os
import socket
import time
import uuid


class ResourceBusyError(RuntimeError):
    pass


class BenchmarkResourceLease:
    def __init__(self, root_dir, resource_id, instance_id=None, pid=None, hostname=None, lease_seconds=3600,
                 clock=time.time):
        self.root_dir, self.resource_id = os.path.abspath(root_dir), resource_id
        self.instance_id = instance_id or str(uuid.uuid4())
        self.pid = os.getpid() if pid is None else pid
        self.hostname = hostname or socket.gethostname()
        self.lease_seconds, self.clock = lease_seconds, clock
        # Resource IDs are identities, not user-controlled paths.
        safe_id = uuid.uuid5(uuid.NAMESPACE_URL, resource_id).hex
        self.path = os.path.join(self.root_dir, "resources", safe_id + ".lock")

    @staticmethod
    def _pid_alive(pid):
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def _reclaimable(self):
        try:
            with open(self.path, encoding="utf-8") as handle:
                owner = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if owner.get("hostname") != self.hostname or self._pid_alive(owner.get("pid")):
            return False
        return self.clock() > float(owner.get("lease_expires_at") or 0)

    def acquire(self, wait_seconds=0, cancellation=None, poll_interval=0.05):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        deadline = time.monotonic() + wait_seconds
        while True:
            if cancellation is not None and cancellation.cancelled():
                raise ResourceBusyError("resource wait cancelled")
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._reclaimable():
                    try:
                        os.unlink(self.path)
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise ResourceBusyError("benchmark resource is already leased: %s" % self.resource_id)
                time.sleep(poll_interval)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"resource_id": self.resource_id, "instance_id": self.instance_id, "pid": self.pid,
                           "hostname": self.hostname, "acquired_at": self.clock(),
                           "lease_expires_at": self.clock() + self.lease_seconds}, handle, sort_keys=True)
                handle.flush(); os.fsync(handle.fileno())
            return self

    def release(self):
        try:
            with open(self.path, encoding="utf-8") as handle:
                owner = json.load(handle)
            if owner.get("instance_id") == self.instance_id and owner.get("pid") == self.pid and owner.get("hostname") == self.hostname:
                os.unlink(self.path)
        except FileNotFoundError:
            pass

    @classmethod
    def reconcile_orphans(cls, root_dir, hostname=None, clock=time.time):
        """Reclaim only expired leases whose local owner PID is dead; never signal it."""
        directory = os.path.join(os.path.abspath(root_dir), "resources")
        if not os.path.isdir(directory):
            return []
        hostname, reclaimed = hostname or socket.gethostname(), []
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    owner = json.load(handle)
                expired = clock() > float(owner.get("lease_expires_at") or 0)
                alive = cls._pid_alive(owner.get("pid"))
                if owner.get("hostname") == hostname and expired and not alive:
                    os.unlink(path); reclaimed.append(owner.get("resource_id", name))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return reclaimed

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
