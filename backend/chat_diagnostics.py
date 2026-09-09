"""Read-only Test Chat projections of router-owned runtime information."""
import re
import docs


def render_message(content):
    # Reuse Help's escaped Markdown renderer. Diagnostics must not load remote
    # images supplied by a model, so omit image elements from the rendered text.
    return re.sub(r"<img\b[^>]*>", "[image omitted]", docs.render(content))


_SECRET = re.compile(r"(?:api[-_]?key|(?:hf|auth|access|bearer)[-_]?token|password|secret|credential)", re.I)


def runtime_args(args):
    """Preserve argv boundaries; omit authentication values from diagnostics."""
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return []
    out, hide_next = [], False
    for arg in args:
        if hide_next:
            out.append("[redacted]")
            hide_next = False
        elif arg.startswith("-") and _SECRET.search(arg.split("=", 1)[0]):
            flag, sep, _ = arg.partition("=")
            out.append(flag + "=[redacted]" if sep else flag)
            hide_next = not sep
        else:
            out.append(arg)
    return out


def arg_value(args, *names):
    for i, arg in enumerate(args):
        for name in names:
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
            if arg == name and i + 1 < len(args):
                return args[i + 1]
    return None


def port_number(value):
    try:
        port = int(value)
        return port if 1 <= port <= 65535 else None
    except (ValueError, TypeError):
        return None


def snapshot(deps, model):
    c = deps.cfg()
    rows = deps.REGISTRY.state().get("models", [])
    models = [{k: row.get(k) for k in ("id", "backend", "status", "failed")}
              for row in rows]
    chosen = next((row for row in rows if row.get("id") == model), None)
    engine = c.get("active_engine", "llamacpp")
    result = {"models": models, "model": model, "engine": engine,
              "router_port": c.get("router_port"), "panel_port": c.get("panel_port"),
              "status": "unknown", "argv": [], "port": None, "pid": None,
              "context_capacity": None, "available": False}
    if not chosen:
        return result
    result.update(status=chosen.get("status", "unknown"),
                  backend=chosen.get("backend", "llamacpp"))
    if result["backend"] != "llamacpp":
        result["note"] = "The existing panel Chat Completion proxy serves the llama-family router only."
        return result
    result["available"] = True
    st, data = deps.router("/models", timeout=3)
    result["router_up"] = st == 200
    if st != 200:
        result.update(status="offline", error="Router unavailable")
        return result
    row = next((r for r in data.get("data", []) if r.get("id") == model), None)
    if not row:
        result["status"] = "offline"
        return result
    status = row.get("status", {})
    result["status"] = "error" if status.get("failed") else status.get("value", "unknown")
    # An unloaded model's args may be a future preset, never call those live.
    if result["status"] == "loaded":
        args = runtime_args(status.get("args"))
        result["argv"] = args
        result["argv_source"] = "Router /models status.args (running instance)"
        result["port"] = port_number(arg_value(args, "--port"))
        if result["port"]:
            result["pid"] = deps.router_ctl._pid_on_port(result["port"])
        # /props describes effective runtime context, unlike saved models.ini.
        from urllib.parse import quote
        code, props = deps.router("/props?model=" + quote(model, safe=""), timeout=3)
        if code == 200 and isinstance(props, dict):
            settings = props.get("default_generation_settings", {})
            result["context_capacity"] = settings.get("n_ctx") if isinstance(settings, dict) else None
        result["model_path"] = arg_value(args, "--model", "-m")
        result["alias"] = arg_value(args, "--alias", "-a")
    return result
