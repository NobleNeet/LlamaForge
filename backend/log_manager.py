"""Where LlamaForge's logs live and how big each one is allowed to get.

Two settings drive this module (see config.DEFAULTS):

* ``log_dir``       the log directory. "" = <repo root>/logs, the historical
                    location, so a config that predates this feature keeps
                    behaving exactly as it always did.
* ``log_limits_mb`` per-kind maximum size in megabytes, 0 = unlimited.
                    Keyed by *logical kind* (router_stdout, build_custom, ...)
                    rather than by file name, so renaming a log on disk never
                    silently drops the cap a user configured for it.

This is deliberately NOT generational rotation. No ``.1``, ``.old`` or backup
file is ever produced: an oversized log is shrunk in place, keeping its newest
lines and dropping its oldest, and every kind stays exactly one file forever.

In-place matters because child processes - llama-server, the vLLM serve
process, the panel's own restart - keep stdout/stderr descriptors open on
these files for their whole lifetime. The usual "write a temp file, then
os.replace() it over the original" trick would swap in a new inode and strand
those writers on an unlinked one: their output would quietly disappear from
the file the dashboard reads. So we open the file ``r+b``, read only the
retained tail, seek(0), write it back and truncate(). Every log here is
opened by its writer in append mode, which means the next write lands after
the retained content instead of at a stale offset, so nothing is lost.

Memory: a multi-gigabyte log is never read whole. We seek to
``size - max_bytes`` and read exactly that window in binary mode, then drop
everything up to the first newline so the retained region begins on a
complete-line boundary. Binary mode is not incidental - starting a decode in
the middle of a multi-byte UTF-8 sequence is exactly what we are avoiding.

Trimming is best-effort maintenance. A failure is reported to the caller and
never propagates into the model unload, router restart or build that
triggered it. A lock collapses the several concurrent triggers (multiple
unload paths, the idle reaper, build threads) into at most one sweep.

Pure stdlib.
"""
import glob, os, threading

import config

# Logical log kind -> file name inside the effective log directory. Every
# entry is a log LlamaForge itself opens or redirects a child process into,
# so every entry is one we are allowed to shrink.
LOG_FILES = {
    "router_stdout":   "router.out.log",      # llama.cpp router stdout
    "router_stderr":   "router.err.log",      # llama.cpp router stderr
    "panel_stdout":    "panel.out.log",       # the dashboard's own stdout
    "panel_stderr":    "panel.err.log",       # the dashboard's own stderr
    "vllm_stdout":     "vllm.out.log",       # vllm serve stdout
    "vllm_stderr":     "vllm.err.log",       # vllm serve stderr
    "vllm_setup":      "vllm-setup.log",     # vLLM install/update job
    "vllm_download":   "vllm-download.log",  # safetensors downloads
    "build_llamacpp":  "build.log",          # llama.cpp build
    "build_ikllama":   "build-ikllama.log",  # ik_llama build
}

# Custom builds get one manager - and one log - per user-defined target, named
# build-custom-<uuid>.log. They are not one kind per target: they all share a
# single "build_custom" cap, so N saved targets cannot multiply the budget.
CUSTOM_BUILD_KIND = "build_custom"
CUSTOM_BUILD_GLOB = "build-custom-*.log"

# Default caps in MB, where 0 means unlimited. A kind the user's
# log_limits_mb does not mention falls back to the value here, so adding a
# kind in a later release cannot leave an existing install uncapped by
# accident. This dict is the single source of truth for the numbers;
# config.DEFAULTS stores only the user's overrides on top of it.
DEFAULT_LIMITS_MB = {
    "router_stdout":  64,
    "router_stderr":  32,
    "panel_stdout":   32,
    "panel_stderr":   32,
    "vllm_stdout":    64,
    "vllm_stderr":    32,
    "vllm_setup":     16,
    "vllm_download":  16,
    "build_llamacpp": 32,
    "build_ikllama":  32,
    "build_custom":   32,
}

MB = 1024 * 1024

# One sweep at a time. A trigger that finds the lock held skips rather than
# queueing: a sweep is already running (or has just run), so a second one
# back-to-back would add nothing but I/O contention with the unload that
# triggered it.
_trim_lock = threading.Lock()


def effective_log_dir(c=None):
    """The directory logs live in, always absolute.

    "" (or a missing key) = <repo root>/logs, exactly where logs have always
    gone. A relative value is anchored to the repo root through config._abs,
    the same rule every other configured path follows, so a detached child
    launched from elsewhere resolves it identically. ``~`` is expanded.
    """
    c = config.load() if c is None else c
    raw = (c.get("log_dir") or "").strip()
    if not raw:
        return os.path.join(config.ROOT, "logs")
    return config._abs(os.path.expanduser(raw))


def limit_bytes(kind, c=None):
    """The cap for `kind` in bytes; 0 = unlimited.

    A kind missing from the user's log_limits_mb falls back to
    DEFAULT_LIMITS_MB. A non-integer or negative value is treated as the
    default rather than trusted, so a hand-edited config.json cannot make a
    log unbounded by accident (only an explicit 0 does that).
    """
    c = config.load() if c is None else c
    limits = c.get("log_limits_mb")
    default = DEFAULT_LIMITS_MB.get(kind, 0)
    if not isinstance(limits, dict) or kind not in limits:
        return default * MB
    val = limits[kind]
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        return default * MB
    return val * MB


def log_path(kind, c=None):
    """Absolute path of a fixed-name log kind. Raises KeyError for an unknown
    kind (build_custom has no single path; see custom_build_paths)."""
    if kind not in LOG_FILES:
        raise KeyError(f"unknown log kind: {kind}")
    return os.path.join(effective_log_dir(c), LOG_FILES[kind])


def custom_build_paths(c=None):
    """Every build-custom-*.log currently on disk, sorted for stable order."""
    return sorted(glob.glob(os.path.join(effective_log_dir(c), CUSTOM_BUILD_GLOB)))


def _all_targets(c=None):
    """(path, kind) pairs for every log this module manages.

    Fixed-name kinds come from LOG_FILES; the custom-build family is expanded
    from disk so a target added after startup is still covered, and each file
    carries the shared build_custom cap.
    """
    d = effective_log_dir(c)
    targets = [(os.path.join(d, name), kind) for kind, name in LOG_FILES.items()]
    targets += [(p, CUSTOM_BUILD_KIND) for p in custom_build_paths(c)]
    return targets


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def summary(c=None):
    """Everything the Setup UI needs about log storage in one payload:
    the configured directory, the effective one, and the effective per-kind
    caps in MB (defaults merged with the user's overrides), so the form can
    show real numbers instead of only what happens to be stored."""
    c = config.load() if c is None else c
    return {
        "log_dir": (c.get("log_dir") or ""),
        "effective_dir": effective_log_dir(c),
        "limits_mb": {kind: limit_bytes(kind, c) // MB for kind in DEFAULT_LIMITS_MB},
    }


def trim_file(path, max_bytes):
    """Shrink one log file to at most `max_bytes`, in place.

    Returns a dict describing the outcome:
        {"path", "size", "max", "trimmed", "bytes_freed", "reason"?}

    max_bytes 0 means unlimited and is a no-op. A missing file is not an
    error - it reports trimmed=False. The file keeps its inode throughout:
    read the retained tail, seek(0), write it back, truncate.

    Retention rule: keep the newest `max_bytes` of content, but only whole
    lines. We read a window of max_bytes from the end and drop the leading
    partial line, so the result is always <= max_bytes and always starts at
    a line boundary. If a single line is longer than max_bytes, the whole
    window is one partial line and everything is dropped: the size cap wins
    over keeping an oversized line.
    """
    if not max_bytes or max_bytes <= 0:
        return {"path": path, "size": _size(path), "max": 0, "trimmed": False,
                "bytes_freed": 0, "reason": "unlimited"}
    size = _size(path)
    if size is None:
        return {"path": path, "size": None, "max": max_bytes, "trimmed": False,
                "bytes_freed": 0, "reason": "missing"}
    if size <= max_bytes:
        return {"path": path, "size": size, "max": max_bytes, "trimmed": False,
                "bytes_freed": 0, "reason": "within limit"}

    # Read only the last max_bytes bytes, in binary, so a multi-GB file never
    # lands in memory and a mid-UTF-8 start never has to be decoded.
    with open(path, "r+b") as f:
        f.seek(size - max_bytes)
        window = f.read(max_bytes)
        cut = window.find(b"\n")
        if cut < 0:
            # The whole window is one line with no newline: it is longer than
            # the cap, so keeping any of it would exceed max_bytes. Drop it all.
            retained = b""
        else:
            retained = window[cut + 1:]
        # Rewrite from the front and cut the tail off the SAME inode.
        f.seek(0)
        f.write(retained)
        f.truncate(len(retained))
    return {"path": path, "size": size, "max": max_bytes, "trimmed": True,
            "bytes_freed": size - len(retained)}


def trim_all(c=None, log=None):
    """Trim every managed log to its configured cap.

    Best-effort: a per-file failure is logged and skipped, never raised, so
    the unload/restart/build that triggered this sweep cannot be broken by a
    log we could not shrink. Returns a summary dict:
        {"ran": True, "trimmed": [paths], "errors": [str]}
    """
    c = config.load() if c is None else c
    summary = {"ran": True, "trimmed": [], "errors": []}
    for path, kind in _all_targets(c):
        try:
            res = trim_file(path, limit_bytes(kind, c))
        except Exception as e:      # permission, IO, a vanished file mid-sweep
            msg = f"could not trim {kind} ({path}): {e}"
            summary["errors"].append(msg)
            if log:
                try:
                    log(msg)
                except Exception:
                    pass
            continue
        if res.get("trimmed"):
            summary["trimmed"].append(path)
    return summary


def maybe_trim_all(log=None):
    """Run a trim sweep unless one is already running.

    The single entry point every unload path calls. Takes the lock without
    blocking: a concurrent trigger returns immediately with ran=False rather
    than piling up sweeps. Never raises.
    """
    if not _trim_lock.acquire(blocking=False):
        return {"ran": False, "trimmed": [], "errors": [], "reason": "busy"}
    try:
        return trim_all(log=log)
    except Exception as e:          # belt and braces: maintenance never breaks a caller
        msg = f"log maintenance failed: {e}"
        if log:
            try:
                log(msg)
            except Exception:
                pass
        return {"ran": True, "trimmed": [], "errors": [msg]}
    finally:
        _trim_lock.release()
