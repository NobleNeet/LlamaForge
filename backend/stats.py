"""Usage statistics for LlamaForge.

The dashboard never sees inference traffic (clients hit the llama.cpp router
directly), and llama.cpp's own Prometheus counters reset on restart and keep no
per-model history. So this module runs a background poller that scrapes the
router's `/metrics`, diffs each model's token counters and attributes the delta
to the model it came from, and persists per-model + daily totals to stats.json.
The diff is per model - not "whatever model we happen to look at" - because the
router keeps every model up to `router_models_max` resident, so more than one
model can be serving traffic at the same time: scraping only one of them (the
old single-model assumption) left the others off the Stats tab entirely, newest
loads first among them. Pure stdlib.
"""
import json, os, re, threading, time, urllib.request, urllib.parse
from datetime import date

import atomicio, config

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_FILE = os.path.join(ROOT, "stats.json")

# Prometheus metric names from `llama-server --metrics`. Centralized so a future
# llama.cpp rename is a one-line fix; any missing metric degrades to 0.
M_PROMPT_TOTAL   = "llamacpp:prompt_tokens_total"
M_GEN_TOTAL      = "llamacpp:tokens_predicted_total"
M_PROMPT_PER_SEC = "llamacpp:prompt_tokens_seconds"
M_GEN_PER_SEC    = "llamacpp:predicted_tokens_seconds"
M_REQ_PROCESSING = "llamacpp:requests_processing"

# vLLM Prometheus counters (different names than llama.cpp). Any missing -> 0.
VLLM_PROMPT_TOTAL = "vllm:prompt_tokens_total"
VLLM_GEN_TOTAL    = "vllm:generation_tokens_total"


def vllm_token_totals(metrics):
    """(prompt_total, gen_total) from parsed vLLM /metrics."""
    return (metrics.get(VLLM_PROMPT_TOTAL, 0.0),
            metrics.get(VLLM_GEN_TOTAL, 0.0))


POLL_SECS  = 5       # how often we scrape the router
FLUSH_SECS = 15      # min interval between stats.json writes
DAILY_KEEP = 30      # retain ~a month of daily buckets (UI toggles 14/30 days)

_METRIC_RE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")


def _parse_metrics(text):
    """Prometheus text -> {name: value}, summing across any label sets."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            pass
    return out


def _empty():
    return {"models": {}, "daily": {}, "first_seen": time.time()}


class StatsTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = self._load()
        self._prev = {}              # model id -> (prompt_total, gen_total) last seen
        self._vprev = None           # (prompt, gen) from last vLLM poll
        self._vprev_model = None
        self._idle = {}              # model id -> was generation idle last poll
        self._dirty = False
        self._last_flush = 0.0
        # `loaded_model` stays the first loaded id for callers that predate
        # multi-model support; `loaded_models` is the full list and `models` the
        # per-model live rates (see poll_once).
        self.live = {"prompt_per_sec": 0.0, "gen_per_sec": 0.0,
                     "requests_processing": 0, "loaded_model": None,
                     "loaded_models": [], "models": {},
                     "router_up": False}

    # ---------- persistence ----------
    def _load(self):
        try:
            with open(STATS_FILE, encoding="utf-8") as f:
                d = json.load(f)
            d.setdefault("models", {})
            d.setdefault("daily", {})
            d.setdefault("first_seen", time.time())
            return d
        except Exception:
            return _empty()

    def _flush(self, force=False):
        now = time.time()
        if not force and (not self._dirty or now - self._last_flush < FLUSH_SECS):
            return
        try:
            atomicio.write_json(STATS_FILE, self.data, indent=None)
            self._dirty = False
            self._last_flush = now
        except Exception:
            pass

    # ---------- router access ----------
    def _base(self):
        return f"http://127.0.0.1:{config.load()['router_port']}"

    def _get(self, path, timeout=4):
        with urllib.request.urlopen(self._base() + path, timeout=timeout) as r:
            return r.read().decode(errors="replace")

    def _router_models(self):
        """(router_up, [loaded model id, ...]) - one /models call decides both.
        The router is 'up' whenever /models answers, whether or not a model is
        loaded. (Its /metrics is per-model and 400s without a model name, so
        /metrics can't be used to judge liveness.)

        Every loaded id is returned, not just the first: with
        `router_models_max > 1` the router keeps several models resident at once
        and traffic to any of them is real usage. Picking one made the others -
        typically the model the user just loaded - invisible on the Stats tab.
        """
        try:
            data = json.loads(self._get("/models"))
        except Exception:
            return (False, [])
        loaded = []
        for m in data.get("data", []):
            mid = m.get("id")
            if mid and mid != "default" and m.get("status", {}).get("value") == "loaded":
                loaded.append(mid)
        return (True, loaded)

    def _scrape(self, mid):
        """Parsed /metrics for one loaded model, or None if the scrape failed.

        None must not be read as "counters went to zero": the baseline is kept so
        the next good scrape still attributes the tokens generated in between.
        """
        try:
            return _parse_metrics(self._get("/metrics?model=" + urllib.parse.quote(mid)))
        except Exception:
            return None

    # ---------- accumulation (call under self.lock) ----------
    def _model(self, mid):
        return self.data["models"].setdefault(
            mid, {"prompt": 0, "generated": 0, "loaded_secs": 0, "gen_secs": 0,
                  "runs": 0, "last_used": 0})

    def _record_tokens(self, mid, dp, dg):
        m = self._model(mid)
        m["prompt"] += int(dp)
        m["generated"] += int(dg)
        if dg > 0:   # generation was active this poll window -> feeds avg tok/s
            m["gen_secs"] = m.get("gen_secs", 0) + POLL_SECS
        m["last_used"] = time.time()
        day = self.data["daily"].setdefault(date.today().isoformat(),
                                            {"prompt": 0, "generated": 0})
        day["prompt"] += int(dp)
        day["generated"] += int(dg)
        for d in sorted(self.data["daily"])[:-DAILY_KEEP]:
            self.data["daily"].pop(d, None)
        self._dirty = True

    def _poll_model(self, mid, metrics):
        """Diff one loaded model's counters against its own previous scrape.

        A model seen for the first time only sets its baseline: there is no
        window to attribute yet, and its counters start at whatever the child
        process had already served before we looked.
        """
        p = metrics.get(M_PROMPT_TOTAL, 0.0)
        g = metrics.get(M_GEN_TOTAL, 0.0)
        prev = self._prev.get(mid)
        self._prev[mid] = (p, g)
        if prev is None:
            self._idle[mid] = True
            return
        dp, dg = p - prev[0], g - prev[1]
        if dp < 0 or dg < 0:           # counter reset (that model's child restarted)
            self._idle[mid] = True
            return
        if dp or dg:
            self._record_tokens(mid, dp, dg)
        if dg > 0 and self._idle.get(mid, True):   # a fresh burst ~= one run
            self._model(mid)["runs"] += 1
            self._dirty = True
        self._idle[mid] = (dg == 0)

    # ---------- polling ----------
    def _poll_vllm(self):
        """Scrape vLLM's /metrics (if a model is loaded there) and attribute
        token deltas the same way as llama.cpp. Best-effort; silent on failure.
        Call under self.lock."""
        try:
            port = config.load().get("vllm_port", 8081)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as r:
                metrics = _parse_metrics(r.read().decode(errors="replace"))
        except Exception:
            self._vprev = None
            return
        p, g = vllm_token_totals(metrics)
        model = None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                data = json.loads(r.read().decode())
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            model = ids[0] if ids else None
        except Exception:
            model = None
        if self._vprev is not None and model and model == self._vprev_model:
            dp, dg = p - self._vprev[0], g - self._vprev[1]
            if dp < 0 or dg < 0:
                dp = dg = 0
            if dp or dg:
                self._record_tokens(model, dp, dg)
        self._vprev = (p, g)
        self._vprev_model = model

    def poll_once(self):
        # One /models call tells us both whether the router is up and which
        # models are loaded. The router's /metrics is per-model and 400s without
        # a model name, so we must know the names before scraping - scraping
        # bare /metrics (the old bug) made every poll look like the router down.
        up, loaded = self._router_models()
        if not up:
            with self.lock:
                self.live.update(router_up=False, prompt_per_sec=0.0,
                                 gen_per_sec=0.0, requests_processing=0,
                                 loaded_model=None, loaded_models=[], models={})
                self._prev = {}            # re-baseline on next good poll
                self._idle = {}
                self._poll_vllm()          # vLLM runs independently of this router
                self._flush()
            return

        # Scrape every loaded model, not just one of them. Each model runs in
        # its own child process with its own counters, so a delta seen on model
        # X belongs to X - which is what lets a model that was never used before
        # (or several at once) show up on the Stats tab.
        scraped = {}
        for mid in loaded:
            metrics = self._scrape(mid)
            if metrics is not None:
                scraped[mid] = metrics

        with self.lock:
            rates = {}
            for mid, metrics in scraped.items():
                rates[mid] = {
                    "prompt_per_sec":      metrics.get(M_PROMPT_PER_SEC, 0.0),
                    "gen_per_sec":         metrics.get(M_GEN_PER_SEC, 0.0),
                    "requests_processing": int(metrics.get(M_REQ_PROCESSING, 0.0)),
                }
            self.live.update(
                router_up=True,
                loaded_models=loaded,
                loaded_model=(loaded[0] if loaded else None),
                models=rates,
                prompt_per_sec=sum(r["prompt_per_sec"] for r in rates.values()),
                gen_per_sec=sum(r["gen_per_sec"] for r in rates.values()),
                requests_processing=sum(r["requests_processing"] for r in rates.values()),
            )
            if loaded:
                for mid in loaded:
                    self._model(mid)["loaded_secs"] += POLL_SECS
                self._dirty = True
            for mid in loaded:
                if mid in scraped:
                    self._poll_model(mid, scraped[mid])
            # Drop the baselines of models the router no longer holds loaded, so
            # reloading one starts from its fresh child counters instead of
            # diffing against numbers from the process that just exited.
            for mid in [m for m in self._prev if m not in loaded]:
                self._prev.pop(mid, None)
            for mid in [m for m in self._idle if m not in loaded]:
                self._idle.pop(mid, None)
            self._poll_vllm()
            self._flush()

    def run_forever(self):
        while True:
            try:
                self.poll_once()
            except Exception:
                pass
            time.sleep(POLL_SECS)

    def start(self):
        threading.Thread(target=self.run_forever, daemon=True, name="stats-poller").start()

    # ---------- read side (for the API) ----------
    def reset(self):
        """Zero the whole store (user-initiated from the Stats tab)."""
        with self.lock:
            self.data = _empty()
            self._prev = {}            # dict, see __init__
            self._idle = {}
            self._vprev = self._vprev_model = None
            self._dirty = True
            self._flush(force=True)

    def summary(self):
        with self.lock:
            models = self.data["models"]
            per_model = [{
                "id": mid,
                "prompt": m["prompt"], "generated": m["generated"],
                "tokens": m["prompt"] + m["generated"],
                "loaded_secs": m["loaded_secs"], "runs": m["runs"],
                # avg generation speed over windows where generation was active;
                # gen_secs is missing in stats.json files written before v2
                "avg_tps": round(m["generated"] / m["gen_secs"], 1)
                           if m.get("gen_secs") else 0,
                "last_used": m["last_used"],
            } for mid, m in models.items()]
            per_model.sort(key=lambda x: x["tokens"], reverse=True)
            tot_p = sum(m["prompt"] for m in models.values())
            tot_g = sum(m["generated"] for m in models.values())
            # "Inference time" is the sum of per-model loaded time. With
            # `router_models_max > 1` two models can be resident at once and both
            # accrue, so that sum can exceed wall-clock - each row's own loaded
            # time stays exact.
            tot_secs = sum(m["loaded_secs"] for m in models.values())
            most = per_model[0]["id"] if per_model and per_model[0]["tokens"] > 0 else None
            daily = [{"date": d, **v} for d, v in sorted(self.data["daily"].items())][-DAILY_KEEP:]
            return {
                "totals": {
                    "prompt": tot_p, "generated": tot_g, "tokens": tot_p + tot_g,
                    "loaded_hours": round(tot_secs / 3600, 1),
                    "models_used": sum(1 for m in models.values()
                                       if m["prompt"] + m["generated"] > 0),
                    "most_used": most,
                    "total_runs": sum(m["runs"] for m in models.values()),
                },
                "per_model": per_model,
                "daily": daily,
                "live": dict(self.live),
            }


TRACKER = StatsTracker()
