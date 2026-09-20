"""User-owned Git checkouts and explicit shell recipes, independent of CMake."""
import os
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

from builder import BuildManager

BUILTINS = {"llamacpp": "llama.cpp", "ikllama": "ik_llama"}
FIELDS = ("name", "repository", "branch", "source", "build", "server_binary", "build_command")


def expand(text, target, jobs=None):
    values = {"source": target["source"], "build": target["build"],
              "jobs": str(jobs or os.cpu_count() or 8)}
    # Replace only our tokens; shell braces, ${VAR}, and command syntax survive.
    return re.sub(r"(?<!\$)\{(source|build|jobs)\}", lambda m: values[m[1]], text)


def binary_path(target):
    path = expand(target["server_binary"], target)
    if re.search(r"\{[^{}]+\}", path):
        raise ValueError("Server Binary contains an unknown placeholder")
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(target["build"], path))


def validate(raw):
    if not isinstance(raw, dict):
        raise ValueError("Target must be an object")
    target = {}
    for field in FIELDS:
        value = raw.get(field, "")
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{field} is required and must be text without NUL characters")
        target[field] = value if field == "build_command" else value.strip()
    repository = target["repository"]
    url = urlsplit(repository)
    if not ((url.scheme in ("https", "http", "ssh", "git") and url.hostname and url.path not in ("", "/"))
            or re.fullmatch(r"[\w.-]+@[\w.-]+:[^\s]+", repository)):
        raise ValueError("Repository must be an HTTP(S), SSH, or Git repository URL")
    branch = target["branch"]
    if subprocess.run(["git", "check-ref-format", "--branch", branch],
                      capture_output=True).returncode or branch.startswith("-"):
        raise ValueError("Invalid branch name")
    for field in ("source", "build"):
        path = os.path.expanduser(target[field])
        if not os.path.isabs(path):
            raise ValueError(f"{field} must be an absolute path")
        target[field] = os.path.normpath(path)
        parent = path
        while not os.path.lexists(parent):
            parent = os.path.dirname(parent)
        if not os.path.isdir(parent):
            raise ValueError(f"{field} has a non-directory path component")
    src = target["source"]
    if os.path.exists(src):
        result = subprocess.run(["git", "-C", src, "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True)
        if result.returncode or os.path.realpath(result.stdout.strip()) != os.path.realpath(src):
            raise ValueError("Source must be the root of a Git checkout")
    binary_path(target)
    return target


class CustomBuildManager(BuildManager):
    def run_custom(self, target, _claimed=False):
        if not _claimed and not self._claim():
            return
        try:
            open(self.log_path, "w").close()
            self.state.pop("server_bin", None)
            target = validate(target)
            src, build = target["source"], target["build"]
            def run(cmd, cwd=None):
                rc = self._stream(cmd, cwd=cwd)
                if rc:
                    self.state["returncode"] = rc
                    raise RuntimeError(f"Command exited with code {rc}")
            self.state["phase"] = "pull"
            if not os.path.exists(src):
                run(["git", "clone", "--branch", target["branch"], "--", target["repository"], src])
            else:
                # Never change an existing checkout's remote or branch silently.
                origin = self._git(src, "remote", "get-url", "origin")
                if origin.returncode or origin.stdout.strip().rstrip("/") != target["repository"].rstrip("/"):
                    raise RuntimeError("Source origin differs from Repository; update the target or checkout explicitly")
                branch = self._git(src, "rev-parse", "--abbrev-ref", "HEAD")
                if branch.returncode or branch.stdout.strip() != target["branch"]:
                    raise RuntimeError("Source branch differs from Branch; checkout the requested branch explicitly")
                run(["git", "-C", src, "fetch", "origin"])
                run(["git", "-C", src, "pull", "--ff-only", "origin", target["branch"]])
            os.makedirs(build, exist_ok=True)
            self.state["phase"] = "backup"
            self.backup_binaries(build)
            binary = binary_path(target)
            # Also cover arbitrary output layouts outside bin/.
            if os.path.isfile(binary):
                backup = binary + f".backup-{time.time_ns()}"
                shutil.copy2(binary, backup)
                self._log(f"[backup] prior server binary -> {backup}")
            self.state["phase"] = "build"
            bash = shutil.which("bash")
            if not bash:
                raise RuntimeError("Custom builds require Bash (on Windows, install Git Bash and expose bash on PATH)")
            run([bash, "-e", "-o", "pipefail", "-c", expand(target["build_command"], target)], cwd=build)
            if not os.path.isfile(binary):
                raise RuntimeError(f"Server Binary not found: {binary}")
            self._record_built(binary)
            self.state.update(phase="done", returncode=0)
            self._log("=== BUILD OK ===")
        except Exception as exc:
            self.state.update(phase="failed", returncode=self.state.get("returncode") or 1)
            self._log(f"=== BUILD FAILED: {exc} ===")
        finally:
            self.state.update(running=False, finished=time.time())
            self._upd_cache.clear()

    def start_custom(self, target):
        if not self._claim():
            return False
        try:
            threading.Thread(target=self.run_custom, args=(dict(target),),
                             kwargs={"_claimed": True}, daemon=True).start()
        except Exception:
            self.state.update(running=False, phase="failed", returncode=1, finished=time.time())
            raise
        return True
