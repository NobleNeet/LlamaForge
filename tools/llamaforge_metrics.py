#!/usr/bin/env python3
"""llamaforge_metrics.py -- LlamaForge / llama-server metrics collector.

Parses the two logs produced by LlamaForge and turns them into self-contained,
append-only JSONL events grouped into daily files:

  * stdout log  (llama-server side):  ``[<port>] <t> <sev> <type> <message>``
  * stderr log  (LlamaForge router): ``<t> <sev> srv proxy_request: ...``

The collector is intentionally a "structured raw-data generator", not an
analysis tool: the JSONL it produces is the primary dataset and may be loaded
into a RDB, Elasticsearch/Kibana, DuckDB/Polars, converted to Parquet, etc.

Standard library only.  See the accompanying specification document.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - very old interpreters
    ZoneInfo = None

SCHEMA_VERSION = 1
DEFAULT_STATE_FILE = "metrics.state.json"
DEFAULT_POLL_INTERVAL = 0.25

EVENT_TYPES = (
    "model_loaded", "inference_completed", "model_unloaded",
    "model_load_started", "model_load_failed", "server_exited",
    "inference_incomplete", "parser_warning",
)

# Regexes against the llama-server stdout log lines.
LINE_RE = re.compile(r"^\[(\d+)\]\s+(\S+)\s+(.*)$", re.DOTALL)
LOAD_MODEL_RE = re.compile(r"load_model:\s*loading model\s*'([^']+)'")
INITIALIZE_RE = re.compile(
    r"n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*(\d+),\s*kv_unified\s*=\s*'([^']*)'"
)
LISTEN_RE = re.compile(r"listening on\s+\S+:(\d+)")
EXIT_RE = re.compile(r"exit command received")
CANCEL_RE = re.compile(r"cancel task,\s*id_task\s*=\s*(\d+)")
MTP_PATH_RE = re.compile(r"target model\s*'([^']+)'")
SLOT_TASK_RE = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(-?\d+)")
PREFILL_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*"
    r"\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
)
DECODE_RE = re.compile(
    r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*"
    r"\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
)
TOTAL_RE = re.compile(r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens")
GRAPHS_RE = re.compile(r"graphs reused\s*=\s*(\d+)")
DRAFT_RE = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s+\(\s*(\d+)\s+accepted\s*/"
    r"\s*(\d+)\s+generated\),\s+mean len\s*=\s*([0-9.]+)"
)
RELEASE_RE = re.compile(r"stop processing:\s*n_tokens\s*=\s*(\d+),\s*truncated\s*=\s*(\d+)")
CLI_ARG_RE = re.compile(r"--([A-Za-z0-9-]+)\s+(\S+)")
# llama-server prints "llama_server: model loaded" once the model is ready.
SERVER_MODEL_LOADED_RE = re.compile(r"llama_server:\s*model loaded$")
# LlamaForge router prints the CLI args used to spawn a server instance.
ARGS_BLOCK_HEADER_RE = re.compile(
    r"spawning\s+server\s+instance\s+with\s+args:", re.IGNORECASE)
# A single CLI flag with an optional (same-line) value.
CLI_ARG_TOKEN_RE = re.compile(r"--([A-Za-z0-9-]+)(?:\s+(\S+))?")



def derive_alias(filename: str) -> str:
    """Best-effort model alias derived from a model filename.

    LlamaForge records models as ``ornith-1.5-35b-a3b-mtp.i1-q4-k-m`` which is
    the model filename (without extension) lower-cased with ``_`` replaced by
    ``-`` -- matching the alias used in the router stderr ``proxy_request``
    lines so the two sources can be correlated.
    """
    stem = os.path.splitext(filename)[0]
    return stem.lower().replace("_", "-")


def _add_cli_arg(cli: dict, flag: str, value: str) -> None:
    """Record a raw CLI flag/value pair.

    Repeated flags (e.g. multiple ``--x``) are collapsed into a list so both
    single-valued and repeated options round-trip cleanly.
    """
    if flag in cli:
        existing = cli[flag]
        if isinstance(existing, list):
            existing.append(value)
        else:
            cli[flag] = [existing, value]
    else:
        cli[flag] = value


def _normalize_cli_params(cli: dict) -> "dict | None":
    """Build the typed ``parameters`` view from raw CLI args.

    Only keys whose llama.cpp meaning is known are coerced to typed values and
    emitted. Unknown / future flags are intentionally dropped here (they are
    preserved verbatim in ``cli_parameters`` so nothing is ever lost).
    """
    int_keys = {"ctx-size": "ctx_size",
                "n-gpu-layers": "n_gpu_layers",
                "spec-draft-n-max": "spec_draft_n_max"}
    norm = {}
    for raw_flag, value in cli.items():
        key = raw_flag[2:]  # strip leading "--"
        if key in int_keys:
            if isinstance(value, list):
                # keep the last known value; lists are not typed
                value = value[-1]
            try:
                norm[int_keys[key]] = int(value)
            except (TypeError, ValueError):
                pass
        elif key == "flash-attn":
            norm["flash_attn"] = value
        elif key == "spec-type":
            norm["spec_type"] = [value] if isinstance(value, str) else value
    return norm or None


def dedup_key(event: dict) -> str:
    """Stable content hash used to avoid re-writing the same event.

    Volatile fields (random ids, wall-clock times) are excluded so a re-run over
    identical logs produces identical keys and can be de-duplicated.
    """
    volatile = {"event_id", "timestamp", "observed_at", "load_session_id"}
    stable = {k: v for k, v in event.items() if k not in volatile}
    canonical = json.dumps(stable, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_line(line: str, lineno: int):
    """Parse one llama-server stdout line into a :class:`LogLine`.

    Returns ``None`` when the line lacks the expected ``[port] <t> ...`` prefix
    (such lines are ignored, never fatal -- spec section 29).
    """
    m = LINE_RE.match(line)
    if not m:
        return None
    port = int(m.group(1))
    time_raw = m.group(2)
    toks = m.group(3).split(None, 2)
    if not toks:
        return None
    if len(toks) == 1:
        sev, msgtype, message = None, toks[0], ""
    elif len(toks) == 2:
        sev, msgtype, message = None, toks[0], toks[1]
    else:
        # severity is a single letter (D/I/W/E); otherwise the first token is
        # already the message-type.
        if len(toks[0]) == 1 and toks[0] in "DIWE":
            sev, msgtype, message = toks[0], toks[1], toks[2]
        else:
            sev, msgtype, message = None, toks[0], toks[1]
    return LogLine(port=port, time_raw=time_raw, sev=sev, msgtype=msgtype,
                   message=message.strip(), raw=line.rstrip("\n"), lineno=lineno)


@dataclass
class LogLine:
    port: int
    time_raw: str
    sev: "str | None"
    msgtype: str
    message: str
    raw: str
    lineno: int


@dataclass
class ErrLine:
    time_raw: str
    sev: str
    service: str
    message: str
    raw: str
    lineno: int


@dataclass
class LoadSession:
    session_id: str
    port: int
    path: str
    filename: str
    alias: str
    ctx_size: "int | None" = None
    n_slots: "int | None" = None
    kv_unified: "str | None" = None
    mtp: bool = False
    cli: dict = field(default_factory=dict)
    # Normalized typed view of the CLI args (known keys only).
    cli_norm: dict = field(default_factory=dict)
    # Relative-time (min.sec.mmm.us) markers for load duration calc; absolute
    # timestamps are never available from the logs.
    load_start_raw: "str | None" = None
    load_finish_raw: "str | None" = None
    unloaded: bool = False


class Collector:
    """Stateful collector that correlates the two logs and emits events."""

    def __init__(self, stdout_path, stderr_path, output_dir, tz,
                 verbose=False, emit_incomplete=False, include_raw=False,
                 state_file=None):
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.output_dir = Path(output_dir)
        self.tz = tz
        self.verbose = verbose
        self.emit_incomplete = emit_incomplete
        self.include_raw = include_raw
        self.state_file = Path(state_file) if state_file else None

        self.sessions: dict[str, LoadSession] = {}
        self.sessions_by_port: dict[int, LoadSession] = {}
        self.tasks: dict[tuple[int, int], dict] = {}
        # router stderr: port -> model alias (from proxy_request lines)
        self.alias_by_port: dict[int, str] = {}

        self.emitted_keys: set[str] = set()
        self.emitted: list[dict] = []
        self.stats: dict[str, int] = defaultdict(int)
        # router stderr: CLI args block ("spawning server instance with args:")
        # parsing state. The block is parsed here (stderr is processed before
        # stdout) and later attached to the matching load session.
        self._in_args_block: bool = False
        self._args_pending: "str | None" = None
        self._args_skip_binary: bool = False
        self._pending_cli: dict = {}
        self._pending_cli_model: "str | None" = None
        self._pending_blocks: list = []
        self._pending_unloads: list[dict] = []

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def daily_path(self, day: str) -> Path:
        return self.output_dir / ("metrics-%s.jsonl" % day)

    def _strip_internal(self, ev: dict) -> dict:
        return {k: v for k, v in ev.items() if not k.startswith("_")}

    def emit(self, ev: dict) -> bool:
        """Register an event; returns True if it was newly written."""
        self.stats["events_total"] += 1
        key = dedup_key(ev)
        if key in self.emitted_keys:
            self.stats["events_dedup"] += 1
            return False
        self.emitted_keys.add(key)
        self.emitted.append(ev)
        self.stats["events_" + ev["event_type"]] += 1
        if ev["event_type"] == "parser_warning":
            self.stats["warnings"] += 1
        return True

    def _base_event(self, event_type, line):
        observed = self.now()
        ev = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            # Absolute event time: only settable when the log carries an
            # absolute date. llama-server logs expose process-relative time
            # only, so this stays null and is never guessed (spec section 4).
            "timestamp": None,
            "observed_at": observed.isoformat(),
            "host": {"hostname": socket.gethostname()},
        }
        if line is not None:
            source = {"line": line.lineno, "relative_time": line.time_raw}
            if isinstance(line, LogLine):
                source["stdout_log"] = os.path.basename(self.stdout_path)
            elif isinstance(line, ErrLine):
                source["stderr_log"] = os.path.basename(self.stderr_path)
            ev["source"] = source
        if self.include_raw and getattr(line, "raw", None):
            ev["_raw"] = line.raw
        return ev

    def _warn(self, msg):
        if self.verbose:
            print("[warn] %s" % msg, file=sys.stderr)

    def _session_for(self, port):
        s = self.sessions_by_port.get(port)
        if s is not None:
            return s
        alias = self.alias_by_port.get(port)
        s = LoadSession(session_id=str(uuid.uuid4()), port=port, path=None,
                        filename=None, alias=alias)
        self.sessions[s.session_id] = s
        self.sessions_by_port[port] = s
        return s

    def session_parameters(self, session):
        """Summarize the runtime parameters known for a loaded session.

        The actual CLI args (parsed from the router stderr "spawning server
        instance with args:" block) take precedence for keys they cover; the
        runtime init line fills in the rest (n_slots, kv_cache, and ctx_size as
        a fallback when --ctx-size was not provided). Returns None when nothing
        is known yet.
        """
        params = {}
        # Actual CLI-provided, typed parameters win when present.
        for norm_key, value in (session.cli_norm or {}).items():
            params[norm_key] = value
        if "ctx_size" not in params and session.ctx_size is not None:
            params["ctx_size"] = session.ctx_size
        if session.n_slots is not None:
            params["n_slots"] = session.n_slots
        if session.kv_unified is not None:
            params["kv_cache"] = session.kv_unified
        if session.mtp:
            params["mtp"] = True
        return params or None

    def _attach_pending_cli(self, session) -> None:
        """Attach router CLI args (parsed in stderr) to a load session.

        Correlation is best-effort: a finalized args block whose ``--model``
        path matches the session's model path is preferred; otherwise the first
        args block lacking a model id is attached. Each block is consumed once.
        Unknown/future flags are preserved verbatim in ``cli_parameters``.
        """
        chosen = None
        if session.path:
            for blk in self._pending_blocks:
                if blk["model"] and (
                    blk["model"] == session.path
                    or session.path.endswith(blk["model"])
                    or os.path.basename(blk["model"]) in session.filename
                ):
                    chosen = blk
                    break
        if chosen is None:
            for blk in self._pending_blocks:
                if not blk["model"]:
                    chosen = blk
                    break
        if chosen is None:
            return
        self._pending_blocks.remove(chosen)
        session.cli = dict(chosen["cli"])
        session.cli_norm = _normalize_cli_params(chosen["cli"])
        self._warn("attached %d CLI args to port %d"
                   % (len(session.cli), session.port))

    # -- stdout handlers --------------------------------------------------
    def on_load_started(self, line: LogLine) -> None:
        """Handle ``load_model: loading model '<path>'`` (the load START).

        Emits a ``model_load_started`` event and remembers the relative-time
        marker so the load duration can be computed once ``model_loaded`` fires.
        """
        m = LOAD_MODEL_RE.search(line.message)
        if not m:
            return
        path = m.group(1)
        # A non-empty port that already had a loaded model means that model is
        # being unloaded to make way for the new one (port reuse).
        existing = self.sessions_by_port.get(line.port)
        if existing is not None:
            ev = self._base_event("model_unloaded", line)
            ev["load_session_id"] = existing.session_id
            ev["model"] = {"alias": existing.alias, "path": existing.path,
                           "filename": existing.filename}
            ev["server"] = {"port": existing.port}
            self.emit(ev)
        session = self._session_for(line.port)
        session.port = line.port
        session.path = path
        session.filename = os.path.basename(path)
        session.alias = derive_alias(session.filename)
        session.load_start_raw = line.time_raw
        # Attach any router CLI args parsed from stderr (see
        # _process_err_line); correlate by --model path when possible.
        self._attach_pending_cli(session)

        ev = self._base_event("model_load_started", line)
        ev["load_session_id"] = session.session_id
        ev["model"] = {"alias": session.alias, "path": session.path,
                       "filename": session.filename}
        ev["server"] = {"port": session.port}
        params = self.session_parameters(session)
        if params:
            ev["parameters"] = params
        if session.cli:
            ev["cli_parameters"] = dict(session.cli)
        self.emit(ev)

    def on_initialize(self, line: LogLine) -> None:
        """Capture runtime params from ``load_model: initializing, ...``.

        No event is emitted for this line -- it is a pure state update that
        fills in ctx_size / n_slots / kv_cache for the session.
        """
        s = self._session_for(line.port)
        m = INITIALIZE_RE.search(line.message)
        if m:
            s.n_slots = int(m.group(1))
            s.ctx_size = int(m.group(2))
            s.kv_unified = m.group(3)

    def on_model_loaded(self, line: LogLine) -> None:
        """Handle ``llama_server: model loaded`` (the load COMPLETE).

        Emits a ``model_loaded`` event. This fires strictly after
        ``model_load_started`` because llama-server logs the "loading model"
        line before the "model loaded" line, so the emitted sequence is ordered
        as: model_load_started -> init -> model_loaded -> inference.
        """
        s = self._session_for(line.port)
        s.load_finish_raw = line.time_raw
        ev = self._base_event("model_loaded", line)
        ev["load_session_id"] = s.session_id
        ev["model"] = {"alias": s.alias, "path": s.path,
                       "filename": s.filename}
        ev["server"] = {"port": s.port}
        params = self.session_parameters(s)
        if params:
            ev["parameters"] = params
        if s.cli:
            ev["cli_parameters"] = dict(s.cli)
        self.emit(ev)

    def handle_mtp(self, line: LogLine) -> None:
        s = self._session_for(line.port)
        if MTP_PATH_RE.search(line.message):
            if not s.mtp:
                s.mtp = True
                self._warn("model %s: multi-token path (draft) enabled"
                           % (line.port,))

    def on_cancel(self, line: LogLine) -> None:
        m = CANCEL_RE.search(line.message)
        if not m:
            return
        task = int(m.group(1))
        s = self.sessions_by_port.get(line.port)
        st = self.tasks.get((line.port, task))
        if st is None or s is None:
            return
        if not self._is_complete(st):
            self._emit_incomplete(line, st, s,
                                  ["prompt", "eval", "total"])

    def on_server_exit(self, line: LogLine) -> None:
        s = self.sessions_by_port.get(line.port)
        if s is None:
            s = self._session_for(line.port)
        if not s.unloaded:
            pending = self._consume_pending_unload(s.alias)
            extra = {}
            unload_line = line
            if pending is not None:
                unload_line = pending["line"]
                if pending["exit_status"] is not None:
                    extra["exit_status"] = pending["exit_status"]
            self._emit_model_unloaded(s, unload_line, extra)
        ev = self._base_event("server_exited", line)
        ev["load_session_id"] = s.session_id
        ev["server"] = {"port": s.port}
        self.emit(ev)

    def _emit_model_unloaded(self, session, line, extra=None) -> None:
        if session.unloaded:
            return
        ev = self._base_event("model_unloaded", line)
        ev["load_session_id"] = session.session_id
        ev["model"] = {"alias": session.alias}
        if session.path is not None:
            ev["model"]["path"] = session.path
        if session.filename is not None:
            ev["model"]["filename"] = session.filename
        ev["server"] = {"port": session.port}
        if extra:
            ev.update(extra)
        self.emit(ev)
        session.unloaded = True

    def _consume_pending_unload(self, alias):
        if alias is None:
            return None
        for index, item in enumerate(self._pending_unloads):
            if item["alias"] == alias and not item.get("consumed"):
                item["consumed"] = True
                return item
        return None

    def _parse_timing(self, m):
        return {
            "tokens": int(m.group(2)),
            "time_ms": float(m.group(1)),
            "ms_per_token": float(m.group(3)),
            "tokens_per_second": float(m.group(4)),
        }

    def _parse_total(self, m):
        # total time has no per-token breakdown: only tokens + time_ms.
        return {"tokens": int(m.group(2)), "time_ms": float(m.group(1))}

    def _parse_draft(self, m):
        # "draft acceptance = 0.71250 ( 114 accepted / 160 generated),
        #   mean len = 1.71" -> groups: rate, accepted, generated, mean_len.
        # Schema v1 field names (fixed).
        return {
            "acceptance_rate": float(m.group(1)),
            "accepted": int(m.group(2)),
            "generated": int(m.group(3)),
            "mean_len": float(m.group(4)),
        }


    # -- dispatch helpers -------------------------------------------------
    def _task_state(self, port, message):
        m = SLOT_TASK_RE.search(message)
        if not m:
            return None
        task = int(m.group(2))
        st = self.tasks.get((port, task))
        if st is None:
            st = {
                "port": port, "task": task, "slot": None,
                "prompt": None, "eval": None, "total": None,
                "graphs": None, "draft": None, "truncated": None,
                "raw": [], "complete": False,
            }
            self.tasks[(port, task)] = st
        return st

    def _is_complete(self, st):
        return all(st.get(k) is not None for k in ("prompt", "eval", "total"))

    def _emit_incomplete(self, line, st, session, missing):
        ev = self._base_event("inference_incomplete", line)
        ev["load_session_id"] = session.session_id
        ev["model"] = {"alias": session.alias}
        ev["inference"] = {"task_id": st["task"], "slot_id": st["slot"]}
        ev["complete"] = False
        ev["missing_fields"] = missing
        ev["server"] = {"port": st["port"]}
        self.emit(ev)

    def _emit_complete(self, st, line):
        s = self.sessions_by_port.get(st["port"])
        if s is None:
            s = self._session_for(st["port"])
        ev = self._base_event("inference_completed", line)
        ev["load_session_id"] = s.session_id
        # Non-normalize model info from the load session.
        ev["model"] = {"alias": s.alias, "path": s.path,
                       "filename": s.filename}
        # Inherited load-session parameters (typed) + raw CLI params.
        params = self.session_parameters(s)
        if params:
            ev["parameters"] = params
        if s.cli:
            ev["cli_parameters"] = dict(s.cli)
        ev["server"] = {"port": st["port"]}
        ev["inference"] = {"task_id": st["task"], "slot_id": st["slot"]}

        ev["prefill"] = dict(st["prompt"])
        ev["decode"] = dict(st["eval"])
        ev["context"] = {"tokens_at_decode_start": st["prompt"]["tokens"]}
        if st["draft"] is not None:
            ev["speculative"] = dict(st["draft"])
        runtime = {}
        if st["truncated"] is not None:
            runtime["truncated"] = bool(st["truncated"])
        runtime["graphs_reused"] = st["graphs"]
        ev["runtime"] = runtime
        ev["total"] = {
            "tokens": st["total"]["tokens"],
            "time_ms": st["total"]["time_ms"],
        }
        self.emit(ev)

    def on_release(self, st, line):
        m = RELEASE_RE.search(line.message)
        if m:
            st["truncated"] = int(m.group(2))
            st["complete"] = True
        if self._is_complete(st):
            self._emit_complete(st, line)

    # -- process one stdout line -----------------------------------------
    def process_out_line(self, line: LogLine) -> None:
        msg = line.message
        if "exit command received" in msg:
            self.on_server_exit(line)
            return
        # ``load_model: loading model '/path'`` (srv) -- distinct from the
        # ``load_model: initializing, n_slots = ...`` line handled below.
        # Emits ``model_load_started`` (the load START marker).
        if LOAD_MODEL_RE.search(msg):
            self.on_load_started(line)
            return
        if "n_slots" in msg:
            self.on_initialize(line)
            return
        # ``llama_server: model loaded`` (srv) -- the load COMPLETE marker,
        # emitted strictly after model_load_started.
        if SERVER_MODEL_LOADED_RE.search(msg):
            self.on_model_loaded(line)
            return
        if MTP_PATH_RE.search(msg):
            self.handle_mtp(line)
            return

        st = self._task_state(line.port, msg)
        if st is None:
            return
        if "cancel task" in msg:
            self.on_cancel(line)
        if line.msgtype == "slot":
            m = SLOT_TASK_RE.search(line.message)
            if m:
                st["slot"] = int(m.group(1))
            pm = PREFILL_RE.search(line.message)
            if pm:
                st["prompt"] = self._parse_timing(pm)
                return
            dm = DECODE_RE.search(line.message)
            if dm:
                st["eval"] = self._parse_timing(dm)
                return
            tm = TOTAL_RE.search(line.message)
            if tm:
                st["total"] = self._parse_total(tm)
                return
            gm = GRAPHS_RE.search(line.message)
            if gm:
                st["graphs"] = int(gm.group(1))
            elif DRAFT_RE.search(line.message):
                st["draft"] = self._parse_draft(DRAFT_RE.search(line.message))
            elif "stop processing" in line.message:
                self.on_release(st, line)

    # -- stdout pass ------------------------------------------------------
    def run_once(self):
        with open(self.stdout_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = parse_line(raw, lineno)
                if line is None:
                    continue
                self.process_out_line(line)

    # -- stderr (router) pass --------------------------------------------
    ERR_LINE_RE = re.compile(r"^(\S+)\s+([DIWE])\s+(\S+)\s+(.*)$")
    ERR_PROXY_RE = re.compile(
        r"proxy_reques\w*:\s*proxying request to model\s+(\S+)\s+on port\s+(\d+)"
    )
    ERR_UNLOAD_RE = re.compile(
        r"(?:unload|operator\(\)):\s*stopping model instance name=(\S+)"
    )
    ERR_EXIT_STATUS_RE = re.compile(
        r"operator\(\):\s*instance name=(\S+)\s+exited with status\s+(-?\d+)"
    )

    def process_err_line(self, raw: str, lineno: int) -> None:
        m = self.ERR_LINE_RE.match(raw)
        if not m:
            # Within an args block, bare "--flag value" lines may not carry the
            # standard time/sev prefix -- still try to collect them.
            if self._in_args_block:
                self._collect_args_line(raw)
            return
        line = ErrLine(
            time_raw=m.group(1),
            sev=m.group(2),
            service=m.group(3),
            message=m.group(4),
            raw=raw.rstrip("\n"),
            lineno=lineno,
        )
        pm = self.ERR_PROXY_RE.search(line.message)
        if pm:
            alias = pm.group(1)
            port = int(pm.group(2))
            if self.alias_by_port.get(port) != alias:
                self.alias_by_port[port] = alias
                self._warn("port %d -> alias %s" % (port, alias))
        um = self.ERR_UNLOAD_RE.search(line.message)
        if um:
            self._pending_unloads.append({
                "alias": um.group(1),
                "line": line,
                "exit_status": None,
                "consumed": False,
            })
        em = self.ERR_EXIT_STATUS_RE.search(line.message)
        if em:
            alias = em.group(1)
            exit_status = int(em.group(2))
            matched = False
            for item in reversed(self._pending_unloads):
                if item["alias"] == alias and item["exit_status"] is None and not item.get("consumed"):
                    item["line"] = line
                    item["exit_status"] = exit_status
                    matched = True
                    break
            if not matched:
                self._pending_unloads.append({
                    "alias": alias,
                    "line": line,
                    "exit_status": exit_status,
                    "consumed": False,
                })
        self._collect_err_args(line.message, raw)

    # -- router CLI args ("spawning server instance with args:") ----------
    def _collect_err_args(self, message: str, raw: str) -> None:
        if self._in_args_block:
            content = raw
            if "load:" in content:
                # The router prefixes each arg with "<time> I srv  load:  "
                content = content.split("load:", 1)[1]
            # The line immediately after the header is the server executable
            # path (no "--" flag) -- skip it; guard so a leading "--flag" line
            # is never mistaken for the binary.
            if self._args_skip_binary and "--" not in content:
                self._args_skip_binary = False
                return
            self._collect_args_line(content)
            return
        if ARGS_BLOCK_HEADER_RE.search(message):
            self._in_args_block = True
            self._pending_cli = {}
            self._pending_cli_model = None
            self._args_pending = None
            self._args_skip_binary = True

    def _collect_args_line(self, content: str) -> None:
        """Consume one args-block line (flag and value on separate lines).

        A CLI flag must appear at the *start* of the argument content, so a model
        path value (which begins with "/" and may itself contain "--", e.g.
        ``mradermacher--Ornith-1.5``) is never mistaken for a flag.
        """
        stripped = content.strip()
        m = CLI_ARG_TOKEN_RE.match(stripped)
        is_flag_line = m is not None and stripped.startswith("--")

        if self._args_pending is not None:
            if is_flag_line:
                # The pending flag is a boolean (a new flag follows) -> record it.
                _add_cli_arg(self._pending_cli, "--" + self._args_pending, True)
                self._args_pending = None
            else:
                # This line carries the pending flag's value.
                if stripped:
                    self._record_cli_arg(self._args_pending,
                                         stripped.split(None, 1)[0])
                self._args_pending = None
                return

        if not stripped:
            return  # blank line: ignore
        if not is_flag_line:
            # Non-empty line without a leading flag -> the args block has ended.
            self._finalize_args_block()
            return
        # Leading flag: value may be on the same line or the next one.
        if m.group(2):
            self._record_cli_arg(m.group(1), m.group(2))
        else:
            self._args_pending = m.group(1)

    def _record_cli_arg(self, flag: str, value) -> None:
        if value is not None:
            v = value.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
        else:
            v = ""
        _add_cli_arg(self._pending_cli, "--" + flag, v)
        if flag == "model":
            self._pending_cli_model = v or None

    def _finalize_args_block(self) -> None:
        if self._pending_cli or self._pending_cli_model:
            self._pending_blocks.append({
                "cli": dict(self._pending_cli),
                "model": self._pending_cli_model,
            })
        self._in_args_block = False
        self._args_pending = None

    def process_err_once(self):
        with open(self.stderr_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                self.process_err_line(raw, lineno)
            if self._in_args_block:
                self._finalize_args_block()

    def finalize_pending_unloads(self) -> None:
        if not self._pending_unloads:
            return
        sessions_by_alias = defaultdict(list)
        for session in self.sessions.values():
            if session.alias is not None:
                sessions_by_alias[session.alias].append(session)
        for item in self._pending_unloads:
            if item.get("consumed"):
                continue
            queue = sessions_by_alias.get(item["alias"], [])
            session = None
            while queue:
                candidate = queue.pop(0)
                if not candidate.unloaded:
                    session = candidate
                    break
            if session is None:
                continue
            item["consumed"] = True
            extra = {}
            if item["exit_status"] is not None:
                extra["exit_status"] = item["exit_status"]
            self._emit_model_unloaded(session, item["line"], extra)

    # -- file manager -----------------------------------------------------
    def _day_from_ts(self, iso: str) -> str:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz).date().isoformat()

    def _day_from_event(self, ev: dict) -> str:
        """Day (YYYY-MM-DD) for daily-file routing.

        Prefers the absolute event timestamp when the log provides one, and
        otherwise falls back to the observation time -- which in `once` mode is
        the parse time.
        """
        return self._day_from_ts(ev.get("timestamp")
                                 or ev.get("observed_at") or "")

    def _write_file(self, ev, day):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.daily_path(day)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._strip_internal(ev),
                                ensure_ascii=False,
                                separators=(",", ":")) + "\n")

    def _state_path(self):
        if self.state_file is not None:
            return self.state_file
        return self.output_dir / DEFAULT_STATE_FILE

    def _load_state(self):
        p = self._state_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_seen(self):
        return set(self._load_state().get("seen_keys", []))

    def _save_state(self, seen, watchers=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        p = self._state_path()
        tmp = p.with_suffix(".tmp")
        data = {
            "seen_keys": sorted(seen),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        if watchers:
            for name, watcher in watchers.items():
                if watcher is not None:
                    data[name] = watcher.snapshot()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, p)

    def write_emitted(self, append=True):
        """Write all emitted events to daily files (append + dedup gating).

        ``append=True`` gates each event against the previously persisted
        ``seen`` set so re-running over identical logs only writes new content.
        The current run's keys are only persisted *after* writing, so they are
        never used to dedupe themselves.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seen = self._load_seen() if append else set()
        written = 0
        for ev in self.emitted:
            key = dedup_key(ev)
            if append and key in seen:
                continue
            day = self._day_from_event(ev)
            self._write_file(ev, day)
            written += 1
            self.emitted_keys.add(key)
        if append:
            self._save_state(self.emitted_keys)
        return written

    def flush(self):
        """Append not-yet-written emitted events to their daily files."""
        for ev in self.emitted[self._flushed:]:
            self._write_file(ev, self._day_from_event(ev))
        self._flushed = len(self.emitted)
        self._save_state(self.emitted_keys)


class FileWatcher:
    """Tail a file by inode/offset, handling truncation and rotation."""

    def __init__(self, path, tz, verbose=False, start_offset=None,
                 start_lineno=0, from_start=False):
        self.path = path if path == "-" else Path(path)
        self.tz = tz
        self.verbose = verbose
        self._pos = 0
        self._lineno = start_lineno
        self._inode = None
        self._fh = None
        self._stdin_iter = None
        self._start_offset = start_offset
        self._from_start = from_start
        self._open()

    def _open(self):
        if self.path == "-":
            self._stdin_iter = iter(sys.stdin)
            self._fh = None
        else:
            self._fh = open(self.path, "r", encoding="utf-8")
            self._stdin_iter = None
            st = self.path.stat()
            self._inode = st.st_ino
            if self._from_start:
                self._pos = 0
            elif self._start_offset is not None and st.st_size >= self._start_offset:
                self._pos = self._start_offset
            else:
                self._pos = st.st_size
            self._fh.seek(self._pos)
            self._start_offset = None

    def _reset_generation(self):
        """Reset position and line counter after rotation or truncation."""
        self._pos = 0
        self._lineno = 0
        if self._fh is not None:
            self._fh.close()
        self._open()

    def read_new(self):
        """Return list of ``(content, lineno)`` for newly available complete lines.

        ``lineno`` is absolute within the current file generation and is reset on
        rotation/truncation, so a catch-up read re-numbers identically to the
        initial pass -- keeping dedup keys stable so re-read lines are correctly
        de-duplicated rather than re-emitted. A trailing line lacking a newline
        is buffered until more data completes it.
        """
        result: list[tuple[str, int]] = []
        if self.path == "-":
            while True:
                nxt = next(self._stdin_iter, None)
                if nxt is None or not nxt.endswith("\n"):
                    break  # wait for a complete line
                self._lineno += 1
                result.append((nxt.rstrip("\n"), self._lineno))
            return result
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return result
        if st.st_ino != self._inode or st.st_size < self._pos:
            self._reset_generation()
        self._fh.seek(self._pos)
        data = self._fh.read()
        if not data:
            return result
        lines = data.splitlines(keepends=True)
        if lines and not data.endswith("\n"):
            # Incomplete trailing line: rewind so it is re-read with more data.
            self._fh.seek(-len(lines[-1].encode("utf-8")), io.SEEK_CUR)
            lines = lines[:-1]
        for raw in lines:
            self._lineno += 1
            result.append((raw.rstrip("\n"), self._lineno))
        self._pos = self._fh.tell()
        return result

    def snapshot(self):
        if self.path == "-":
            return {
                "path": "-",
                "inode": None,
                "offset": None,
                "lineno": self._lineno,
            }
        return {
            "path": str(self.path),
            "inode": self._inode,
            "offset": self._pos,
            "lineno": self._lineno,
        }


def resolve_tz(tz):
    if tz in (None, "UTC", "utc"):
        return timezone.utc
    if tz == "local":
        return datetime.now().astimezone().tzinfo
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz)
        except Exception:
            pass
    m = re.match(r"^([+-])(\d{2}):?(\d{2})$", tz)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        return timezone(sign * timedelta(hours=int(m.group(2)),
                                         minutes=int(m.group(3))))
    raise SystemExit("invalid --tz: %r" % tz)


def _iter_file(path):
    if path == "-":
        yield from enumerate(sys.stdin, start=1)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            yield from enumerate(fh, start=1)


def build_collector(args):
    tz = resolve_tz(args.timezone)
    out_dir = Path(args.output_dir) if args.output_dir else Path(".")
    c = Collector(args.stdout_log, args.stderr_log, out_dir, tz,
                  verbose=args.verbose, emit_incomplete=args.emit_incomplete,
                  include_raw=args.include_raw, state_file=args.state_file)
    if args.stderr_log:
        c.process_err_once()
    for _lineno, raw in _iter_file(args.stdout_log):
        line = parse_line(raw, _lineno)
        if line is not None:
            c.process_out_line(line)
    c.finalize_pending_unloads()
    return c


def cmd_once(args):
    c = build_collector(args)
    written = c.write_emitted(append=args.append)
    print(json.dumps({"emitted": len(c.emitted), "written": written,
                      "dedup": c.stats.get("events_dedup", 0),
                      "events": dict(c.stats)}, indent=2))
    return 0


def cmd_watch(args):
    tz = resolve_tz(args.timezone)
    out_dir = Path(args.output_dir) if args.output_dir else Path(".")
    # NOTE: intentionally do NOT run the full sync here. The FileWatchers below
    # catch up from the start of each log with correct absolute line numbers, so
    # the collector state is built exactly once (re-synchronising would re-run
    # stateful handlers like ``on_load_started`` and emit duplicate load events).
    c = Collector(args.stdout_log, args.stderr_log, out_dir, tz,
                  verbose=args.verbose, emit_incomplete=args.emit_incomplete,
                  include_raw=args.include_raw, state_file=args.state_file)
    c._flushed = 0
    if args.append:
        c.emitted_keys |= c._load_seen()
    c.flush()  # create the output dir and persist (empty) state before tailing

    stop = threading.Event()

    def _stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    state = c._load_state()
    out_state = {} if args.from_start else state.get("stdout", {})
    err_state = {} if args.from_start else state.get("stderr", {})
    out_w = FileWatcher(
        args.stdout_log,
        c.tz,
        args.verbose,
        start_offset=out_state.get("offset"),
        start_lineno=out_state.get("lineno", 0),
        from_start=args.from_start,
    )
    err_w = FileWatcher(
        args.stderr_log,
        c.tz,
        args.verbose,
        start_offset=err_state.get("offset"),
        start_lineno=err_state.get("lineno", 0),
        from_start=args.from_start,
    ) if args.stderr_log else None

    print("# watching stdout=%r stderr=%r (pid %d)" %
          (args.stdout_log, args.stderr_log, os.getpid()), file=sys.stderr)
    while not stop.is_set():
        if err_w is not None:
            for raw, lineno in err_w.read_new():
                c.process_err_line(raw, lineno)
        data = out_w.read_new()
        for raw, lineno in data:
            line = parse_line(raw, lineno)
            if line is not None:
                c.process_out_line(line)
        c.finalize_pending_unloads()
        c.flush()
        c._save_state(c.emitted_keys, {"stdout": out_w, "stderr": err_w})
        stop.wait(args.poll_interval)
    print("# stopped; wrote %d events" % len(c.emitted), file=sys.stderr)
    return 0


def _add_common_args(p):
    p.add_argument("--stdout-log", "-o",
                   help="llama-server の stdout ログ（'-' で標準入力）")
    p.add_argument("--stderr-log", "-e",
                   help="LlamaForge router の stderr ログ")
    p.add_argument("--output-dir", default="./metrics",
                   help="日次 JSONL ファイルの出力ディレクトリ")
    p.add_argument("--timezone", "--tz", dest="timezone", default=None,
                   help="タイムゾーン：UTC（既定値）、local、IANA 名、"
                        "または +/-HH:MM")
    p.add_argument("--state-file", default=None,
                   help="watch モードのオフセット・dedup 状態を保存する JSON ファイル")
    p.add_argument("--append", dest="append", action="store_true",
                   default=True, help="追記モード（既定値）。同じログを再実行しても"
                                      "重複を書かない")
    p.add_argument("--no-append", dest="append", action="store_false",
                   help="日次ファイルを最初から書き直す（dedup を無効化）")
    p.add_argument("--emit-incomplete", action="store_true",
                   help="未完了の inference を 'inference_incomplete' として出力する")
    p.add_argument("--include-raw", action="store_true",
                   help="イベントに生ログを付与する")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="詳細ログを stderr に出力する")
    p.add_argument("--poll-interval", type=float,
                   default=DEFAULT_POLL_INTERVAL,
                   help="watch モードでのファイル監視ポーリング間隔（秒）")
    p.add_argument("--from-start", action="store_true",
                   help="watch モードで既存ログを先頭から処理してから監視する")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LlamaForge / llama-server のログを解析し、"
                    "日次ローテートする JSONL イベントに変換します。"
                    "標準ライブラリのみを使用します。",
        epilog=(
            "使用例:\n"
            "  一度だけ処理する（バッチ実行）:\n"
            "    python3 tools/llamaforge_metrics.py once \\\n"
            "      --stdout-log logs/router.out.log \\\n"
            "      --stderr-log logs/router.err.log \\\n"
            "      --output-dir /tmp/metrics --timezone UTC\n"
            "\n"
            "  監視する（監視モード、Ctrl-C で停止）:\n"
            "    python3 tools/llamaforge_metrics.py watch \\\n"
            "      --stdout-log logs/router.out.log \\\n"
            "      --stderr-log logs/router.err.log \\\n"
            "      --output-dir /tmp/metrics --poll-interval 1 -v\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(
        dest="mode", metavar="コマンド", required=False)
    once = sub.add_parser(
        "once", help="ログを一度だけ処理して終了する（バッチ実行）")
    _add_common_args(once)
    once.set_defaults(func=cmd_once)
    watch = sub.add_parser(
        "watch", help="ログを継続的に監視して出力する（監視モード）")
    _add_common_args(watch)
    watch.set_defaults(func=cmd_watch)

    args = parser.parse_args(argv)

    # 引数なしの場合はガイドを表示して終了する（エラーにしない）
    if args.mode is None:
        parser.print_help(sys.stdout)
        return 0

    if args.mode == "watch" and not args.stdout_log:
        parser.error("--stdout-log は watch モードで必須です")
    if not args.stdout_log and not args.stderr_log:
        parser.error(
            "少なくとも --stdout-log か --stderr-log のいずれかが必須です")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
