"""Isolated llama-bench execution with bounded-memory raw artifact capture."""
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone

from .bench_argv import build_bench_argv
from .bench_parse import BenchParseError, expand_record, parse_structured_output


class CancellationToken:
    def __init__(self):
        import threading
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def cancelled(self):
        return self._event.is_set()


class BenchRunner:
    def __init__(self, store, poll_interval=0.1, terminate_grace_seconds=2, max_parse_bytes=16 * 1024 * 1024,
                 clock=time.monotonic):
        self.store = store
        self.poll_interval = poll_interval
        self.terminate_grace_seconds = terminate_grace_seconds
        self.max_parse_bytes = max_parse_bytes
        self.clock = clock

    def _spawn(self, argv, stdout, stderr):
        kwargs = {"stdout": stdout, "stderr": stderr, "stdin": subprocess.DEVNULL, "shell": False}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(argv, **kwargs)

    def _terminate_owned_group(self, process):
        if process.poll() is not None:
            return
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        deadline = self.clock() + self.terminate_grace_seconds
        while process.poll() is None and self.clock() < deadline:
            time.sleep(self.poll_interval)
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)

    def _read_bounded(self, path):
        with open(path, "rb") as handle:
            data = handle.read(self.max_parse_bytes + 1)
        if len(data) > self.max_parse_bytes:
            raise BenchParseError("stdout exceeds structured parse limit")
        return data.decode("utf-8", "replace")

    def run_case(self, run_id, target, case, repetitions, timeout_seconds, cancellation=None):
        argv = build_bench_argv(target, case, repetitions)
        invocation_id = str(uuid.uuid4())
        self.store.mark_running(run_id)
        out_tmp, err_tmp, out_path, err_path = self.store.raw_paths(run_id, invocation_id)
        started_wall = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        started = self.clock()
        status, error, measurements, exit_code = "completed", None, (), None
        with open(out_tmp, "wb") as stdout, open(err_tmp, "wb") as stderr:
            process = self._spawn(argv, stdout, stderr)
            while process.poll() is None:
                elapsed = self.clock() - started
                if cancellation is not None and cancellation.cancelled():
                    status, error = "cancelled", "cancelled"
                    self._terminate_owned_group(process)
                    break
                if elapsed > timeout_seconds:
                    status, error = "failed", "timeout"
                    self._terminate_owned_group(process)
                    break
                self.store.heartbeat(run_id)
                time.sleep(self.poll_interval)
            exit_code = process.wait()
        self.store.finalize_raw_files(out_tmp, err_tmp, out_path, err_path)
        finished_wall = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        duration = self.clock() - started
        if status == "completed" and exit_code != 0:
            status, error = "failed", "llama-bench exited with %s" % exit_code
        if status == "completed":
            try:
                records = parse_structured_output(self._read_bounded(out_path))
                if len(records) != 1:
                    raise BenchParseError("one invocation must produce exactly one record")
                measurements = expand_record(records[0], case, argv, exit_code, started_wall, finished_wall, duration)
            except (OSError, BenchParseError) as exc:
                status, error = "failed", str(exc)
        artifact = {"invocation_id": invocation_id, "case_id": case.case_id, "status": status,
                    "argv": list(argv), "stdout_path": os.path.basename(out_path), "stderr_path": os.path.basename(err_path),
                    "exit_code": exit_code, "started_at": started_wall, "finished_at": finished_wall,
                    "duration_seconds": duration, "error": error,
                    "measurements": [measurement.__dict__ for measurement in measurements]}
        self.store.record_invocation(run_id, invocation_id, artifact)
        return status, measurements, artifact
