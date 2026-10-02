"""Incremental, PID-bound llama.cpp run history; aggregate accounting is separate."""
import copy
from collections import deque
import json
import hashlib
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

import atomicio
import log_manager
import router_ctl

LOG = logging.getLogger(__name__)
HISTORY_FILE = str(Path(__file__).resolve().parent.parent / 'stats-history.json')
LIMITS = (10, 25, 50, 100)
RETENTION = 1000
STARTUP_WINDOW = 30


def empty_store():
    return {'version': 1, 'next_id': 1, 'configs': {}, 'runs': {}}


def engine_identity(executable):
    """Only use a checkout containing the executable, never the dashboard repo."""
    name = 'llama.cpp'
    lower = executable.lower()
    if 'strix' in lower:
        name = 'strix-llama'
    elif 'ik_llama' in lower or 'ik-llama' in lower:
        name = 'ik_llama'
    commit = None
    root = Path(__file__).resolve().parent.parent
    for parent in Path(executable).resolve().parents:
        if parent == root or (parent / 'backend' / 'stats.py').exists():
            break
        if (parent / '.git').exists():
            if name == 'llama.cpp' and parent.name not in ('llama.cpp', 'llama-cpp'):
                name = parent.name
            try:
                commit = subprocess.check_output(
                    ['git', '-C', str(parent), 'rev-parse', '--short=7', 'HEAD'],
                    text=True, timeout=2, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.SubprocessError):
                pass
            break
    return name, commit


# Intentionally small vocabulary. Unknown fork arguments remain intact.
OPTIONS = {
    '--ctx-size': ('performance', 'Context (tok)'), '-c': ('performance', 'Context (tok)'),
    '--batch-size': ('performance', 'Batch (tok)'), '-b': ('performance', 'Batch (tok)'),
    '--ubatch-size': ('performance', 'UBatch (tok)'), '-ub': ('performance', 'UBatch (tok)'),
    '--threads': ('performance', 'Threads'), '-t': ('performance', 'Threads'),
    '--threads-batch': ('performance', 'Batch threads'),
    '--gpu-layers': ('performance', 'GPU Layers'), '-ngl': ('performance', 'GPU Layers'),
    '--n-gpu-layers': ('performance', 'GPU Layers'),
    '--cache-type-k': ('performance', 'KV Cache K'), '-ctk': ('performance', 'KV Cache K'),
    '--cache-type-v': ('performance', 'KV Cache V'), '-ctv': ('performance', 'KV Cache V'),
    '--flash-attn': ('performance', 'Flash Attention'), '-fa': ('performance', 'Flash Attention'),
    '--spec-type': ('speculative', 'Type'), '--draft-max': ('speculative', 'Draft Max (tok)'),
    '--draft-min': ('speculative', 'Draft Min (tok)'), '--draft-p-min': ('speculative', 'Draft P Min'),
    '--spec-draft-adaptive': ('speculative', 'Adaptive'),
    '--model': ('models', 'Main'), '-m': ('models', 'Main'),
    '--model-draft': ('models', 'Draft'), '-md': ('models', 'Draft'),
    '--mmproj': ('models', 'MMProj'),
}
SECRET = re.compile(r'(?:api[-_]?key|(?:^|[-_])(?:hf[-_]?)?token(?:$|[-_])|password|secret|credential)', re.I)


def snapshot(argv):
    argv = list(argv)
    hide = False
    for i, arg in enumerate(argv):
        if hide:
            argv[i] = '[REDACTED]'
            hide = False
        elif arg.startswith('-') and SECRET.search(arg.split('=', 1)[0]):
            if '=' in arg:
                argv[i] = arg.split('=', 1)[0] + '=[REDACTED]'
            else:
                hide = True
    groups, other = {}, []
    i = 1
    while i < len(argv):
        flag, sep, value = argv[i].partition('=')
        if flag not in OPTIONS:
            other.append(argv[i])
            i += 1
            continue
        if not sep:
            if i + 1 < len(argv) and (not argv[i + 1].startswith('-') or re.fullmatch(r'-\d+', argv[i + 1])):
                i += 1
                value = argv[i]
            else:
                value = 'on'
        group, label = OPTIONS[flag]
        groups.setdefault(group, {})[label] = value
        i += 1
    name, commit = engine_identity(argv[0])
    models = groups.get('models', {})
    return dict(argv=argv, launch_command=subprocess.list2cmdline(argv) if router_ctl.osplat.IS_WIN else shlex.join(argv),
                normalized_options=groups, other_options=other,
                engine_name=name, engine_commit=commit, engine_executable=argv[0],
                main_model=models.get('Main'), draft_model=models.get('Draft'), mmproj=models.get('MMProj'))


class Follower:
    """Start at EOF; subsequent new files start at zero. Partial bytes stay buffered."""
    def __init__(self, path):
        self.path = path
        self.identity = None
        self.offset = 0
        self.partial = b''
        self.anchor = b''
        self.backlog = False
        self.rewound = False
        self.line_offsets = []
        self.baseline()

    def baseline(self):
        self.partial = b''
        try:
            with open(self.path, 'rb') as f:
                st = os.fstat(f.fileno())
                self.identity = (st.st_dev, st.st_ino)
                self.offset = st.st_size
                f.seek(max(0, self.offset - 64))
                self.anchor = f.read(64)
        except OSError:
            self.identity, self.offset, self.anchor = None, 0, b''

    def read(self):
        self.rewound = False
        try:
            with open(self.path, 'rb') as f:
                st = os.fstat(f.fileno())
                identity = (st.st_dev, st.st_ino)
                f.seek(max(0, self.offset - len(self.anchor)))
                changed = f.read(len(self.anchor)) != self.anchor
                self.rewound = identity == self.identity and (st.st_size < self.offset or changed)
                if identity != self.identity or st.st_size < self.offset or changed:
                    self.offset, self.partial = 0, b''
                self.identity = identity
                f.seek(self.offset)
                data = f.read(1024 * 1024)
                self.offset = f.tell()
                self.backlog = self.offset < st.st_size
                f.seek(max(0, self.offset - 64))
                self.anchor = f.read(min(64, self.offset))
            lines = (self.partial + data).split(b'\n')
            self.partial = lines.pop()
            self.line_offsets = []
            end = self.offset - len(self.partial) - sum(len(line) + 1 for line in lines)
            for line in lines:
                self.line_offsets.append(end)
                end += len(line) + 1
            return [line.decode('utf-8', errors='replace').rstrip('\r') for line in lines]
        except OSError:
            return []


class RunHistory:
    def __init__(self, path=None, resolve_pid=None, clock=time.time, process_identity=None):
        self.path = path or HISTORY_FILE
        self.resolve_pid = resolve_pid or router_ctl.listening_pid
        self.process_identity = process_identity or router_ctl.process_identity
        self.identities = {}
        self.clock = clock
        self.lock = threading.RLock()
        try:
            with open(self.path, encoding='utf-8') as f:
                self.data = json.load(f)
            if (not isinstance(self.data, dict) or self.data.get('version') != 1
                    or not isinstance(self.data.get('configs'), dict)
                    or not isinstance(self.data.get('runs'), dict)
                    or type(self.data.get('next_id')) is not int or self.data['next_id'] < 1
                    or any(not isinstance(c, dict) or not isinstance(c.get('model_id'), str)
                           or not isinstance(c.get('argv'), list) for c in self.data['configs'].values())
                    or any(not isinstance(rs, list) or any(not isinstance(r, dict)
                           or str(r.get('load_config_id')) not in self.data['configs'] for r in rs)
                           for rs in self.data['runs'].values())):
                raise ValueError('unsupported history')
        except (OSError, ValueError, TypeError):
            self.data = empty_store()
        self.pending = []
        self.capture = None
        self.active = {}
        self.blocks = {}
        self.slot_keys = {}
        self.completed = set()
        self.completed_order = deque()
        self.retired_pids = set()
        self.skip_offsets = {}
        self.followers = {}
        self.router_pid = None
        self.router_identity = None
        self.dirty = False
        self._model_ids = frozenset(self.data['runs']) | frozenset(c['model_id'] for c in self.data['configs'].values())

    def close(self, pid):
        if pid in self.active:
            self.retired_pids.add(pid)
        self.active.pop(pid, None)
        self.identities.pop(pid, None)
        for key in list(self.blocks):
            if key[0] == pid:
                self.blocks.pop(key)
        self.slot_keys = {key: value for key, value in self.slot_keys.items() if key[0] != pid}
        self.completed = {key for key in self.completed if key[0] != pid}
        self.completed_order = deque(key for key in self.completed_order if key[0] != pid)

    def stderr(self, line):
        spawn = re.search(r'spawning server instance with name=(.+?) on port (\d+)', line)
        if spawn:
            if self.capture and self.capture.get('reading'):
                self.capture['ready'] = True
            self.capture = None
            mid, port = spawn.group(1), int(spawn.group(2))
            for pid, cid in list(self.active.items()):
                cfg = self.data['configs'][str(cid)]
                if cfg['model_id'] == mid or cfg['child_port'] == port:
                    self.close(pid)
            self.pending = [p for p in self.pending if p['model_id'] != mid and p['child_port'] != port]
            session = dict(model_id=mid, child_port=port, loaded_at=self.clock(), argv=[], ready=False)
            self.pending.append(session)
            self.capture = session
            return
        if 'spawning server instance with args:' in line and self.capture:
            self.capture['reading'] = True
            return
        if self.capture and self.capture.get('reading'):
            arg = re.search(r'load:   (.*)$', line)
            if arg:
                self.capture['argv'].append(arg.group(1))
                self.capture['last_arg_at'] = self.clock()
                return
            self.capture['ready'] = True
            self.capture = None
        exit_model = re.search(r'instance name=(.+?) exited with status', line)
        if exit_model:
            mid = exit_model.group(1)
            self.pending = [p for p in self.pending if p['model_id'] != mid]
            for pid, cid in list(self.active.items()):
                if self.data['configs'][str(cid)]['model_id'] == mid:
                    self.close(pid)
        exit_pid = re.search(r'\[(\d+)\].*(?:exited|terminated)', line)
        if exit_pid:
            self.close(int(exit_pid.group(1)))

    def bind(self):
        now = self.clock()
        for pid, cid in list(self.active.items()):
            cfg = self.data['configs'][str(cid)]
            identity = self.process_identity(pid)
            if self.resolve_pid(cfg['child_port']) != pid or (self.identities.get(pid) is not None and identity != self.identities[pid]):
                self.close(pid)
        for session in list(self.pending):
            if now - session['loaded_at'] > STARTUP_WINDOW:
                LOG.warning('Stats history: child port %s for %s could not be bound', session['child_port'], session['model_id'])
                self.pending.remove(session)
                if self.capture is session:
                    self.capture = None
                continue
            if not session['ready'] or not session['argv']:
                continue
            pid = self.resolve_pid(session['child_port'])
            if not pid:
                continue
            # A PID cannot belong to two live child sessions.
            if pid in self.active:
                LOG.warning('Stats history: ambiguous child PID %s', pid)
                self.close(pid)
                self.pending.remove(session)
                continue
            cid = self.data['next_id']
            self.data['next_id'] += 1
            cfg = {k: session[k] for k in ('model_id', 'child_port', 'loaded_at')}
            cfg.update(snapshot(session['argv']), id=cid, child_pid=pid)
            self.data['configs'][str(cid)] = cfg
            self.active[pid] = cid
            self.identities[pid] = self.process_identity(pid)
            # A reused PID's unread output can be from the previous process.
            # Discard only that PID up to this cursor; other models keep flowing.
            follower = self.followers.get('router_stdout')
            if pid in self.retired_pids and follower:
                try:
                    st = os.stat(follower.path)
                    self.skip_offsets[pid] = ((st.st_dev, st.st_ino), st.st_size)
                except OSError:
                    pass
            self.pending.remove(session)
            if self.capture is session:
                self.capture = None
            self._model_ids = self._model_ids | {cfg['model_id']}
            self.dirty = True

    def stdout(self, line, offset=None):
        prefix = re.match(r'^\[(\d+)\]', line)
        if not prefix:
            return
        pid = int(prefix.group(1))
        marker = self.skip_offsets.get(pid)
        if offset is not None and marker:
            follower = self.followers['router_stdout']
            if marker[0] == follower.identity and offset < marker[1]:
                return
        if pid not in self.active:
            LOG.debug('Stats history: skipping timing from unbound child PID %s', pid)
            return
        task = re.search(r'\btask\s+(\d+)', line)
        slot = re.search(r'\bid\s+(\d+)\s*\|', line)
        if not task and not slot:
            return  # no safe request identity
        if task:
            key = (pid, int(task.group(1)))
        else:
            slot_key = (pid, int(slot.group(1)))
            if 'prompt eval time' in line:
                # Timestamped prompt-line identity makes slot-only forks safe
                # against rotation replay without inventing a task ID.
                fingerprint = hashlib.sha256(line.encode('utf-8')).hexdigest()
                self.slot_keys[slot_key] = (pid, ('slot', slot_key[1], fingerprint))
            key = self.slot_keys.get(slot_key)
            if key is None:
                return
        if key in self.completed:
            return
        timing = re.search(r'(prompt eval|eval|total) time\s*=\s*(\d+(?:\.\d+)?) ms\s*/\s*(\d+) tokens', line)
        mtp = re.search(r'draft acceptance\s*=\s*(\d+(?:\.\d+)?)\s*\(\s*(\d+) accepted\s*/\s*(\d+) generated\)(?:, mean len\s*=\s*(\d+(?:\.\d+)?))?', line)
        if not timing and not mtp:
            return
        block = self.blocks.setdefault(key, dict(model_id=self.data['configs'][str(self.active[pid])]['model_id'],
             load_config_id=self.active[pid], child_pid=pid, task_id=int(task.group(1)) if task else None,
             prompt_tokens=None, prompt_eval_ms=None, pp_tps=None, generated_tokens=None,
             eval_ms=None, tg_tps=None, total_ms=None, mtp_acceptance=None, mtp_accepted=None,
             mtp_generated=None, mtp_mean_len=None))
        block['_updated'] = self.clock()
        if timing:
            kind, ms, tokens = timing.groups()
            tps = re.search(r'(\d+(?:\.\d+)?) tokens per second', line)
            if kind == 'total':
                block.update(total_ms=float(ms), timestamp=self.clock())
            else:
                fields = ('prompt_tokens', 'prompt_eval_ms', 'pp_tps') if kind == 'prompt eval' else ('generated_tokens', 'eval_ms', 'tg_tps')
                block.update(zip(fields, (int(tokens), float(ms), float(tps.group(1)) if tps else None)))
        if mtp:
            ratio, accepted, generated, mean = mtp.groups()
            block.update(mtp_acceptance=float(ratio), mtp_accepted=int(accepted), mtp_generated=int(generated), mtp_mean_len=float(mean) if mean else None)

    def finish(self, force=False):
        for key, block in list(self.blocks.items()):
            if not force and self.clock() - block['_updated'] < 1:
                continue
            if all(block[k] is not None for k in ('prompt_eval_ms', 'eval_ms', 'total_ms')):
                run = {k: v for k, v in block.items() if not k.startswith('_')}
                self.data['runs'].setdefault(run['model_id'], []).append(run)
                self.completed.add(key)
                self.completed_order.append(key)
                while len(self.completed_order) > 8192:
                    self.completed.discard(self.completed_order.popleft())
                self.blocks.pop(key)
                self.dirty = True
            elif self.clock() - block['_updated'] > 60:
                self.blocks.pop(key)
        self.prune()

    def prune(self):
        for mid, runs in self.data['runs'].items():
            self.data['runs'][mid] = sorted(runs, key=lambda r: r['timestamp'])[-RETENTION:]
        keep = set(self.active.values()) | {r['load_config_id'] for runs in self.data['runs'].values() for r in runs}
        self._model_ids = frozenset(self.data['runs']) | frozenset(
            c['model_id'] for cid, c in self.data['configs'].items() if int(cid) in keep)
        for cid in list(self.data['configs']):
            if int(cid) not in keep:
                del self.data['configs'][cid]
                self.dirty = True

    def model_ids(self):
        # Published immutable snapshot: aggregate refresh never waits on OS
        # process lookup, Git lookup, or history disk writes.
        return self._model_ids

    def recent(self, model, limit=10):
        if not isinstance(model, str) or not model or len(model) > 1024 or any(ord(c) < 32 for c in model):
            raise ValueError('invalid model identifier')
        if type(limit) is not int or limit not in LIMITS:
            raise ValueError('limit must be 10, 25, 50 or 100')
        with self.lock:
            runs = list(reversed(self.data['runs'].get(model, [])))[:limit]
            ids = {str(r['load_config_id']) for r in runs}
            return copy.deepcopy({'runs': runs, 'configs': {cid: self.data['configs'][cid] for cid in ids}})

    def config(self, cid, model):
        if not isinstance(cid, str) or not re.fullmatch(r'[1-9]\d{0,18}', cid):
            raise ValueError('invalid config identifier')
        self.recent(model)
        with self.lock:
            cfg = self.data['configs'].get(cid)
            return copy.deepcopy(cfg) if cfg and cfg['model_id'] == model else None

    def flush(self):
        if self.dirty:
            atomicio.write_json(self.path, self.data, indent=None)
            self.dirty = False

    def reset(self):
        with self.lock:
            self.data = empty_store()
            self.active.clear()
            self.identities.clear()
            self.pending.clear()
            self.capture = None
            self.blocks.clear()
            self.slot_keys.clear()
            self.completed.clear()
            self.completed_order.clear()
            self._model_ids = frozenset()
            self.retired_pids.clear()
            self.skip_offsets.clear()
            for follower in self.followers.values():
                follower.baseline()
            self.dirty = True
            self.flush()

    def baseline(self):
        self.followers = {kind: Follower(log_manager.log_path(kind)) for kind in ('router_stderr', 'router_stdout')}

    def poll_once(self):
        with self.lock:
            if not self.followers:
                self.baseline()
            for kind, follower in list(self.followers.items()):
                path = log_manager.log_path(kind)
                if path != follower.path:
                    self.followers[kind] = Follower(path)
                    for pid in list(self.active):
                        self.close(pid)
                    self.pending.clear()
                    self.blocks.clear()
                    self.slot_keys.clear()
                    self.capture = None
            import config
            router_pid = self.resolve_pid(config.load()['router_port'])
            router_identity = self.process_identity(router_pid) if router_pid else None
            if self.router_pid != router_pid or (self.router_identity is not None and router_identity != self.router_identity):
                for pid in list(self.active):
                    self.close(pid)
                self.pending.clear()
                self.capture = None
            self.router_pid = router_pid
            self.router_identity = router_identity
            err = self.followers['router_stderr']
            for line in err.read():
                self.stderr(line)
            # The block is complete only when no partial line/unread bytes remain.
            if (self.capture and self.capture.get('reading') and not err.partial
                    and self.clock() - self.capture.get('last_arg_at', self.clock()) >= 1
                    and os.path.getsize(err.path) == err.offset):
                self.capture['ready'] = True
            self.bind()
            out = self.followers['router_stdout']
            lines = out.read()
            if out.rewound:
                for pid, marker in list(self.skip_offsets.items()):
                    if marker[0] == out.identity:
                        # A truncate/regrow invalidates old byte positions.
                        self.skip_offsets[pid] = (out.identity, os.path.getsize(out.path))
            for line, offset in zip(lines, out.line_offsets):
                self.stdout(line, offset=offset)
            if not out.partial and not out.backlog:
                self.finish()
            self.flush()

    def start(self):
        self.baseline()
        def loop():
            while True:
                try:
                    self.poll_once()
                except Exception:
                    LOG.exception('Stats history ingestion failed')
                time.sleep(1)
        threading.Thread(target=loop, daemon=True, name='stats-history').start()
