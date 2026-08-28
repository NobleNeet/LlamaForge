"""Auto Tune-only resource leases; never manage llama-server processes."""
import json
import os
import socket
import time
import uuid


class ResourceBusyError(RuntimeError):
    pass


class BenchmarkResourceLease:
    def __init__(self, root_dir, resource_id, instance_id=None, pid=None, hostname=None):
        self.root_dir, self.resource_id = os.path.abspath(root_dir), resource_id
        self.instance_id = instance_id or str(uuid.uuid4())
        self.pid = os.getpid() if pid is None else pid
        self.hostname = hostname or socket.gethostname()
        self.path = os.path.join(self.root_dir, "resources", resource_id + ".lock")

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ResourceBusyError("benchmark resource is already leased: %s" % self.resource_id) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"instance_id": self.instance_id, "pid": self.pid, "hostname": self.hostname,
                       "acquired_at": time.time()}, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        return self

    def release(self):
        try:
            with open(self.path, encoding="utf-8") as handle:
                owner = json.load(handle)
            if owner.get("instance_id") == self.instance_id and owner.get("pid") == self.pid:
                os.unlink(self.path)
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
