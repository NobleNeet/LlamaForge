"""Download safetensors repos into the WSL-side HuggingFace cache and manage
them from the dashboard (the 'hybrid' storage model: files live in WSL ext4 for
native load speed, but sizes/deletes are driven from the UI so the user never
opens a WSL shell).

Progress is polled by du -sb on the repo's cache dir vs the expected total.
Pure stdlib; all filesystem work happens inside WSL via wsl.py.
"""
import re, threading, time

import wsl

VENV = "~/.llamaforge/vllm-venv"
HF_CACHE = "~/.cache/huggingface/hub"

# `repo` reaches these from a request body. Every script below therefore binds
# it as $1 rather than interpolating it - a repo id of `x; rm -rf ~` used to be
# executed verbatim by delete_cmd's `rm -rf`.
DOWNLOAD_SCRIPT = f'exec {wsl.sh_path(VENV)}/bin/hf download "$1"'
DELETE_SCRIPT   = f'rm -rf -- {wsl.sh_path(HF_CACHE)}/"$1"'
DU_SCRIPT       = f'du -sb {wsl.sh_path(HF_CACHE)}/"$1" 2>/dev/null | cut -f1'


def cache_dirname(repo):
    """HF cache dir name for a repo, e.g. Qwen/Qwen3-8B -> models--Qwen--Qwen3-8B.

    Rejects anything that isn't a plain `org/name` repo id, so a crafted id can
    neither escape the cache directory via `..` nor smuggle a path separator."""
    repo = (repo or "").strip()
    if not _REPO_RE.match(repo):
        raise ValueError(f"invalid repo id: {repo!r}")
    return "models--" + repo.replace("/", "--")


# HuggingFace repo ids: "org/name" or a bare "name"; letters, digits, ._- only.
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def _du_bytes(distro, repo):
    try:
        dirname = cache_dirname(repo)
    except ValueError:
        return 0
    code, out, _err = wsl.run(DU_SCRIPT, dirname, distro=distro, timeout=15)
    try:
        return int(out.strip()) if code == 0 and out.strip() else 0
    except ValueError:
        return 0


class Manager:
    """One download at a time, streamed to a log; progress via du polling."""
    def __init__(self, distro):
        self.distro = distro
        self.lock = threading.Lock()
        self.state = {"running": False, "repo": "", "downloaded": 0,
                      "total": 0, "phase": "idle", "error": ""}

    def progress(self):
        with self.lock:
            if self.state["running"]:
                self.state["downloaded"] = _du_bytes(self.distro, self.state["repo"])
            return dict(self.state)

    def start(self, repo, expected_bytes):
        try:
            cache_dirname(repo)          # reject a bad id before spawning anything
        except ValueError:
            with self.lock:
                self.state.update(phase="failed", error=f"invalid repo id: {repo!r}")
            return False
        with self.lock:
            if self.state["running"]:
                return False
            self.state = {"running": True, "repo": repo, "downloaded": 0,
                          "total": expected_bytes, "phase": "downloading", "error": ""}
        threading.Thread(target=self._run, args=(repo,), daemon=True).start()
        return True

    def _run(self, repo):
        import os
        import log_manager
        try:
            logdir = log_manager.effective_log_dir()
            os.makedirs(logdir, exist_ok=True)
            with open(os.path.join(logdir, "vllm-download.log"), "w",
                      encoding="utf-8", errors="replace") as log:
                p = wsl.popen(DOWNLOAD_SCRIPT, repo,
                              stdout=log, stderr=log, distro=self.distro)
                rc = p.wait()
            self.state["phase"] = "done" if rc == 0 else "failed"
            if rc != 0:
                self.state["error"] = "hf download failed - see log"
        except Exception as e:
            self.state.update(phase="failed", error=str(e))
        finally:
            self.state["running"] = False

    def delete(self, repo):
        try:
            dirname = cache_dirname(repo)
        except ValueError as e:
            return False, str(e)
        code, _out, err = wsl.run(DELETE_SCRIPT, dirname,
                                  distro=self.distro, timeout=30)
        return code == 0, (err or "").strip()

    def wsl_path(self, repo):
        """The snapshot path vLLM can serve directly (skips a re-resolve)."""
        return f"{HF_CACHE}/{cache_dirname(repo)}"
