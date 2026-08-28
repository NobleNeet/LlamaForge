"""Inference engines behind one interface.

llama.cpp and vLLM grew as parallel universes: separate registries (models.ini
vs vllm_models.json), separate schema builders, separate hub clients, separate
process control, and a duplicated route for every verb (/api/load and
/api/vllm/load, /api/save and /api/vllm/save, ...). They met in exactly one
place - merge_vllm_models - and only on the read side. Every caller that wanted
to act on "a model" had to know which engine it belonged to first.

That does not survive a third engine. ik-llama is next on the roadmap and needs
the same process-managed treatment vLLM has, which would mean a third column in
every one of those pairs and a three-way branch at each call site.

So: a Backend is the set of verbs the dashboard needs from an engine, and the
engines are implementations of it. `for_model(id)` answers "who owns this?", so
routes can dispatch on the model rather than on the URL. The per-engine modules
keep doing the real work - this is the seam that lets callers stop caring which
one is on the other side.

A Backend implements:

    name            str, matches the "backend" field in a model row
    available()     is this engine usable on this machine at all
    list_models()   [model row]  - the shape /api/state already returns
    schema()        knob schema for the editor
    load(id)        -> (ok, error)
    unload(id)      -> (ok, error)
    save(id, knobs) -> {"restarted": bool} - persist, apply to a live model
    delete(id)      -> (ok, error); raises Unsupported when it can't

Pure stdlib.
"""
import os

import osplat, vllm_ctl, vllm_registry, wsl


class Unsupported(Exception):
    """The engine does not offer this verb (e.g. llama.cpp can't delete a model
    from the dashboard - the file is the user's, not ours)."""


class Backend:
    """Interface documentation; implementations don't inherit from it, they just
    satisfy it. Kept as a class so the contract has one obvious home."""
    name = ""

    def available(self):     return True
    def list_models(self):   raise NotImplementedError
    def schema(self):        raise NotImplementedError
    def load(self, mid):     raise NotImplementedError
    def unload(self, mid):   raise NotImplementedError
    def save(self, mid, knobs):  raise NotImplementedError
    def delete(self, mid):   raise Unsupported(f"{self.name} cannot delete models")


class LlamaCppBackend:
    """GGUF models served by llama.cpp's own multi-model router.

    The router owns process lifecycle and hot-reload, so this backend is mostly
    a translator between models.ini and the router's HTTP API."""
    name = "llamacpp"

    def __init__(self, deps):
        self._d = deps          # routes module functions, injected to stay testable

    def available(self):
        return True

    def state(self):
        """Rows plus the inherited [*] globals, from one router round-trip.
        /api/state polls every few seconds, so this stays a single call."""
        base = self._d.model_state()
        endpoint = f"http://127.0.0.1:{self._d.cfg()['router_port']}"
        for m in base["models"]:
            m["backend"] = self.name
            if m.get("status") == "loaded":
                m["endpoint"] = endpoint
        return base

    def list_models(self):
        return self.state()["models"]

    def schema(self):
        return self._d.schema()

    def load(self, mid):
        self._d._prepare_model_for_load(mid)
        code, res = self._d.router("/models/load", "POST", {"model": mid})
        if code == 200:
            return True, ""
        return False, (res or {}).get("error", "load failed")

    def unload(self, mid):
        code, res = self._d.router("/models/unload", "POST", {"model": mid})
        if code == 200:
            return True, ""
        return False, (res or {}).get("error", "unload failed")

    def save(self, mid, knobs):
        was_running = self._d._apply_knobs_and_reload(mid, knobs)
        # llama.cpp reads args at load time, so a live model is unloaded rather
        # than restarted - the user reloads when ready.
        return {"restarted": False, "was_running": was_running}

    def delete(self, mid):
        raise Unsupported("llama.cpp models are files on disk; remove them with "
                          "Setup > prune instead")


class VllmBackend:
    """safetensors / AWQ / GPTQ / FP8 models served by `vllm serve` inside WSL2.

    vLLM has no router and no hot reload: one process serves one model, and a
    knob change means a restart. This backend owns that process."""
    name = "vllm"

    def __init__(self, deps):
        self._d = deps

    def available(self):
        return osplat.IS_WIN          # vLLM rides WSL2

    def list_models(self):
        live = {i["model_id"]: i for i in self._d.vllm_mgr().status()} \
            if self.available() else {}
        rows = []
        for mid in vllm_registry.models():
            inst = live.get(mid)
            status = STATE_MAP.get(inst["state"], "offline") if inst else "offline"
            entry = vllm_registry.load().get(mid, {})
            row = {"id": mid, "backend": self.name, "status": status,
                   "failed": bool(inst and inst["state"] == "failed"),
                   "modalities": ["text"], "in_ini": True,
                   "settings": entry.get("settings", {}),
                   "eff_ctx": vllm_registry.effective_settings(mid).get("max-model-len", "?"),
                   "file_gib": round(entry.get("size_bytes", 0) / 1024**3, 2)
                               if entry.get("size_bytes") else None}
            if inst and status == "loaded":
                row["endpoint"] = inst["endpoint"]
            rows.append(row)
        return rows

    def schema(self):
        return self._d.vllm_schema()

    def _ref(self, mid):
        entry = vllm_registry.load().get(mid, {})
        return entry.get("wsl_path") or entry.get("repo") or mid

    def load(self, mid):
        if not vllm_registry.load().get(mid):
            return False, f"unknown vLLM model: {mid}"
        flags = vllm_ctl.settings_to_flags(vllm_registry.effective_settings(mid))
        return self._d.vllm_mgr().start(mid, self._ref(mid), flags)

    def unload(self, mid):
        self._d.vllm_mgr().stop(mid)
        return True, ""

    def save(self, mid, knobs):
        mgr = self._d.vllm_mgr()
        running = any(i["model_id"] == mid and i["state"] in ("ready", "loading")
                      for i in mgr.status())
        vllm_registry.set_settings(mid, knobs)
        if running:                    # no hot reload; bounce the process
            mgr.stop(mid)
            flags = vllm_ctl.settings_to_flags(vllm_registry.effective_settings(mid))
            mgr.start(mid, self._ref(mid), flags)
        return {"restarted": running, "was_running": running}

    def delete(self, mid):
        ok, err = self._d.vllm_dl().delete(mid)
        if ok:
            vllm_registry.remove(mid)
        return ok, err


STATE_MAP = {"ready": "loaded", "loading": "loading", "starting": "loading",
             "failed": "offline", "stopped": "offline"}


class IkLlamaBackend(LlamaCppBackend):
    """GGUF models served by ik_llama (ikawrakow's llama.cpp fork).

    It speaks the same router API as llama.cpp, so it inherits everything except
    which binary is on the other end and which knob schema that binary reports."""
    name = "ikllama"

    def available(self):
        sbin = self._d.cfg().get("ik_llama_server_bin", "")
        return bool(sbin and os.path.exists(sbin))

    def schema(self):
        return self._d.ik_schema()


# Engines that drive llama.cpp's router. Only one can own the router at a time,
# so `state()` reports the active one and skips its sibling rather than listing
# the same models twice.
LLAMA_FAMILY = ("llamacpp", "ikllama")


class Registry:
    """The set of engines this install can use, and the lookup that lets a route
    act on a model without knowing which engine owns it."""

    def __init__(self, deps):
        self._d = deps
        self._backends = [LlamaCppBackend(deps), IkLlamaBackend(deps), VllmBackend(deps)]

    def active_engine(self):
        """Which llama-family engine owns the router right now.

        Read through the injected deps like everything else here, and validated:
        /api/state polls this every few seconds, so a stray value in config.json
        must degrade to the default rather than 500 the whole dashboard."""
        name = self._d.cfg().get("active_engine", "llamacpp")
        return name if name in LLAMA_FAMILY else "llamacpp"

    def all(self):
        return list(self._backends)

    def enabled(self):
        return [b for b in self._backends if b.available()]

    def get(self, name):
        for b in self._backends:
            if b.name == name:
                return b
        raise KeyError(name)

    def supports(self, name):
        try:
            return self.get(name).available()
        except KeyError:
            return False

    def state(self):
        """{"models": [...], "global": {...}} across every usable engine, ordered
        loaded-first then by id - the order the model list has always used.
        `global` stays the active engine's [*] section; it is a models.ini concept."""
        active = self.active_engine()
        base_backend = self.get(active)
        base = base_backend.state()
        rows = list(base["models"])
        for b in self.enabled():
            if b is base_backend:
                continue
            # The idle llama-family sibling would re-list the same router rows.
            if b.name in LLAMA_FAMILY:
                continue
            rows.extend(b.list_models())
        rows.sort(key=lambda m: (m["status"] != "loaded", m["id"]))
        return {"models": rows, "global": base.get("global", {})}

    def list_models(self):
        return self.state()["models"]

    def for_model(self, mid, hint=""):
        """Which engine owns this model id.

        `hint` is the row's own `backend` field, which the dashboard already
        has - taking it avoids listing every engine's models just to route one
        click. It is only trusted if it names a usable engine; otherwise we look
        the id up, and fall back to llama.cpp so ids that predate the registry
        resolve the way they always did.

        A llama-family hint is snapped to whichever of those engines is active:
        an open tab can hold a row tagged with the engine that was current when
        it rendered, and honouring that stale tag would edit knobs against the
        wrong binary's schema."""
        if hint in LLAMA_FAMILY:
            return self.get(self.active_engine())
        if hint and self.supports(hint):
            return self.get(hint)
        for b in self.enabled():
            if any(m["id"] == mid for m in b.list_models()):
                return b
        return self.get("llamacpp")
