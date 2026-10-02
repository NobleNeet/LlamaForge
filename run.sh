#!/usr/bin/env bash
# LlamaForge one-click runner (Linux / macOS).
# Reads config.json, starts the llama.cpp router + the LlamaForge backend,
# then opens the dashboard in your browser. Safe to run repeatedly.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cfg="$here/config.json"

# config.json is per-machine and deliberately not in the repo. Without this the
# first run died on a raw Python traceback from getcfg that said nothing about
# config.example.json sitting right next to it.
if [ ! -f "$cfg" ]; then
  if [ ! -f "$here/config.example.json" ]; then
    echo "config.json is missing and config.example.json was not found in $here." >&2
    echo "Re-clone the repo, or create config.json by hand." >&2
    exit 1
  fi
  cp "$here/config.example.json" "$cfg"
  echo "config.json not found - created one from config.example.json."
  echo "Set your llama.cpp paths and model folders in the dashboard's Setup tab."
fi

getcfg() { python3 -c "import json;print(json.load(open('$cfg')).get('$1',''))"; }

listening() { lsof -ti "tcp:$1" -sTCP:LISTEN >/dev/null 2>&1; }

router_port="$(getcfg router_port)"
panel_port="$(getcfg panel_port)"
panel_host="$(getcfg panel_host)"; [ -n "$panel_host" ] || panel_host=127.0.0.1
logdir="$here/logs"
mkdir -p "$logdir"

# Resolve the active binary/registry and sanitize with the shared Python path.
# A failed schema probe reports a warning while the dashboard still opens.
if ! listening "$router_port"; then
  PYTHONPATH="$here/backend" python3 -m runtime_registry
else
  owner="$(lsof -ti "tcp:$router_port" -sTCP:LISTEN 2>/dev/null | head -1)"
  owner_name="$(ps -p "${owner:-0}" -o comm= 2>/dev/null || true)"
  case "$owner_name" in
    *llama*|"") ;;
    *) echo "port $router_port is already in use by '$owner_name' (PID $owner)."
       echo "The router was not started. Stop that process, or change router_port in Setup." ;;
  esac
fi

# 2. LlamaForge backend (dashboard)
if ! listening "$panel_port"; then
  (cd "$here/backend" && nohup python3 server.py \
    >>"$logdir/panel.out.log" 2>>"$logdir/panel.err.log" </dev/null &)
  echo "started LlamaForge dashboard on $panel_host:$panel_port"
fi

# 3. open the dashboard
sleep 2
url="http://127.0.0.1:$panel_port/"
if [ "$(uname)" = "Darwin" ]; then open "$url"; else xdg-open "$url" >/dev/null 2>&1 || echo "open $url"; fi
