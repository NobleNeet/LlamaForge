---
title: HTTP API
section: reference
order: 3
---

# HTTP API

The LlamaForge dashboard backend (`backend/server.py`) listens on `panel_port` (default `8090`) and serves the web UI, the `/api/*` management API, and two agent-facing, provider-compatible chat endpoints. All routes below are read directly from `do_GET`/`do_POST` in `backend/server.py`.

`/api/*` request/response bodies are JSON. POST handlers read the body with `json.loads(self.rfile.read(n) or "{}")`, so a POST with no body is treated as `{}`.

> [!NOTE]
> If vLLM support isn't available on the host, every route is first checked by `_vllm_gate()`; requests to `/api/vllm/*` paths return an error response instead of reaching the normal handler.

## Agent-facing endpoints

These are the endpoints external coding agents (Claude Code, Codex, etc.) talk to — see `backend/agentsetup.py` for the config generators that point agents at them.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/messages` | Anthropic Messages API-compatible endpoint. Requires `x-api-key` auth (`_shim_auth_ok`) and `anthropic_shim_enabled: true` in `config.json` (the default). Supports `"stream": true` (SSE) via `_anthropic_stream`. Internally translated to the OpenAI-shaped request and forwarded to the router (`_anthropic_messages` -> `_router_openai`). |
| POST | `/v1/messages/count_tokens` | Anthropic-compatible token-count estimate for a would-be `/v1/messages` request. Same auth/enable gating as `/v1/messages`. |
| POST | `/v1/chat/completions` | OpenAI Chat Completions-compatible endpoint. Requires auth via `_shim_auth_ok`. Injects the active wiki context profile as a system message (`_inject_openai_system`) before forwarding to the router. Supports `"stream": true`. |
| POST | `/api/load` | Load a model into the router. Body: `{"model": "<id>"}`. Proxies to the router's `/models/load`. |
| POST | `/api/unload` | Unload a model from the router. Body: `{"model": "<id>"}`. Proxies to the router's `/models/unload`. |
| POST | `/api/unload_all` | Unload every currently loaded/loading model (except the router's `default` entry). |

## Model and preset management

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | Full dashboard state: models (llama.cpp + vLLM merged), GPU telemetry, platform, current `config.json`, and onboarding status. |
| GET | `/api/schema` | The knob schema (available `llama-server` flags), built from `llama-server --help`. |
| POST | `/api/save` | Save per-model knob overrides into `models.ini` (`config.set_keys`). Reloads the running model if it was loaded. |
| GET | `/api/presets` | List a model's saved knob presets from `config.json` (query param `model`). |
| POST | `/api/presets/save` | Save a named knob preset for one model. |
| POST | `/api/presets/delete` | Delete a named preset from one model. |
| POST | `/api/presets/apply` | Apply one model's saved preset knobs back onto that same model, with the same reload behavior as `/api/save`. |
| POST | `/api/presets/bind` | Bind one of a model's own presets as that model's default (materializes its knobs; `name: ""` unbinds). Re-saving a bound preset re-syncs that same model. |
| GET | `/api/model/metadata` | GGUF metadata for a model id (query param `model`). |
| GET | `/api/model/diag` | Diagnostic read of the router log against a model's merged (`[*]` + per-model) settings (query param `model`). |
| POST | `/api/autotune/recommend` | Recommend knob values for a model given hardware constraints. Body: `{model, intent}` where `intent` is `balanced`, `speed`, `context`, or `coding`. Returns `{knobs, reasons}`. |
| POST | `/api/autotune/refine` | Auto-generate knob recommendations, benchmark candidates with real completion requests (~200 tokens), and return the fastest config. Body: `{model, intent}` (knobs optional; generated if omitted). Returns `{knobs, measurements: {candidates: [{knobs, tok_s}], chosen_tok_s}}`. |
| GET | `/api/scan/missing` | List `models.ini` entries whose GGUF file no longer exists on disk. |
| POST | `/api/scan` | Scan directories (`model_dirs` by default) for GGUF files. |
| POST | `/api/scan/apply` | Register scanned entries into `models.ini` and reapply ctx-size defaults. |
| POST | `/api/scan/prune` | Remove `models.ini` sections whose file is missing (unloading first if loaded). |

## Build, setup, and hardware

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/setup` | Prerequisite tool status (git/cmake/ninja/compiler plus CUDA / ROCm-HIP / Vulkan as applicable) plus hardware recommendation. |
| POST | `/api/setup/install` | Install a missing prerequisite tool (Windows/macOS only). |
| GET | `/api/gpus` | Live GPU telemetry via the available platform backend (`nvidia-smi`, Linux AMD detection, or Apple Silicon unified-memory telemetry). |
| GET | `/api/build/info` | Current commit, available updates, the selected backend, and recommended/saved CMake flags (per `target`: `llamacpp` or `ikllama`). |
| GET | `/api/build/log` | Tail of the build log plus builder state (`phase` includes `done_warnings` for a partial success). |
| POST | `/api/build/start` | Start (re)building the target engine with the given (or saved/recommended) CMake flags. |
| POST | `/api/engine/switch` | Point the router at `llamacpp` or `ikllama` (sets `active_engine`); refused if the target binary has no router mode. |

## VRAM prediction

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/vram/predict` | Predict whether a model quant will fit your GPU and at what approximate speed. Query params: `repo` (HF repo id), `quant` (e.g. `Q4_K_M`). Returns `{regime, tok_s, model_size_bytes, active_size_bytes}`. Factors in MoE active-vs-total parameters and GPU memory bandwidth (with Setup overrides). |

## Model Hub (download)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/hub/search` | Search Hugging Face for GGUF repos, annotated with what's already installed. |
| POST | `/api/hub/files` | List a repo's files/quantizations, sized against available VRAM. |
| POST | `/api/hub/download` | Start downloading a model (and optional mmproj) from the Hub. Returns `{started, dest}`, where `dest` is the folder the files are written into (a per-repo subfolder of config `download_dir`). |
| GET | `/api/hub/progress` | Current download progress. Includes `dest`, so the dashboard can keep naming the folder across a pause/Resume. |
| GET | `/api/hub/dir` | Where Discover saves GGUFs: `{dir, custom, default, scanned}` — the folder the backend resolves, the raw `download_dir` setting, what the default would be, and whether the folder is under a scan root. |
| POST | `/api/hub/cancel` / `/api/hub/pause` / `/api/hub/resume` | Control the active download. |
| POST | `/api/hub/add` | Register a finished manual/download-dir GGUF into `models.ini`. |

## Network and router control

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/network` | Current router host/port, whether an API key is set, LAN IP, the configured `router_models_max` (`models_max`), and whether the router is running. |
| POST | `/api/network` | Update `router_host`/`router_port`/`router_api_key` plus `panel_host` in `config.json`, restart the router, and automatically restart the dashboard too when its own bind host changes. |
| POST | `/api/router/restart` | Restart the llama.cpp router so settings it reads only at startup — `router_models_max` — take effect. **Unloads every loaded model**; returns `{ok, models_max, router_port, unloaded: [ids]}`. Refused with `ok: false` (nothing stopped) when the active binary has no router mode. |
| GET | `/api/router/log` | Tail of the router's log. |
| GET | `/api/stats` | Usage stats summary. |
| POST | `/api/stats/reset` | Reset usage stats. |
| POST | `/api/config` | Merge the request body into `config.json` and save. |

> [!NOTE]
> There is no `GET /api/config` route; current config is read via `GET /api/state`'s `config` field instead.

## vLLM (WSL) management

Only reachable when vLLM support is available on the host (`_vllm_gate`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/vllm/setup` | vLLM/WSL install status plus any running setup job. |
| POST | `/api/vllm/setup/install` | Kick off the vLLM install script inside WSL. |
| POST | `/api/vllm/update` | Run the vLLM update script inside WSL. |
| GET | `/api/vllm/log` | Tail of the vLLM process log. |
| GET | `/api/vllm/schema` | vLLM knob schema. |
| GET | `/api/vllm/version` | Installed vs. latest PyPI vLLM version. |
| POST | `/api/vllm/load` / `/api/vllm/unload` | Load/unload a registered vLLM model. |
| POST | `/api/vllm/save` | Save a vLLM model's settings, restarting it if currently running. |
| POST | `/api/vllm/hub/search` | Search Hugging Face for vLLM-servable repos. |
| POST | `/api/vllm/hub/info` | Repo info sized against available VRAM. |
| POST | `/api/vllm/hub/download` | Start a vLLM model download. |
| GET | `/api/vllm/hub/progress` | Download progress. |
| POST | `/api/vllm/hub/register` | Register a downloaded model into the vLLM registry. |
| POST | `/api/vllm/delete` | Delete a downloaded vLLM model and deregister it. |

## Agent config and context wiki

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/agent/config` | Generate connection config (endpoint, key, model) for a named coding agent (query params `agent`, `model`, `small`, `inject`). |
| POST | `/api/agent/apply` | Write the generated agent config to disk on the machine running the dashboard. |
| GET | `/api/wiki/docs` | List context-wiki documents. |
| GET | `/api/wiki/doc` | Read a single document (query param `name`). |
| POST | `/api/wiki/doc` | Create/update a document. |
| POST | `/api/wiki/doc/delete` | Delete a document. |
| GET | `/api/wiki/profiles` | List saved context profiles. |
| POST | `/api/wiki/profile` | Save a named profile (its doc list + description). |
| POST | `/api/wiki/profile/delete` | Delete a profile. |
| GET | `/api/wiki/preview` | Preview the composed text for a profile (query param `profile`). |
| POST | `/api/wiki/active` | Set the active profile for a model. |
| POST | `/api/wiki/export` | Export composed context (e.g. to an agent's own file format). |

## Docs viewer

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/docs` | Manifest of documentation pages (sections/order/titles). |
| GET | `/api/docs/page` | Rendered HTML for one page (query param `slug`); 404 if the slug doesn't exist. |
| GET | `/docs/img/<name>` | Serve an image referenced from a docs page, path-safety-checked by `docs._safe_img`. |

## Static / UI

| Method | Path | Purpose |
|---|---|---|
| GET | `/` or `/index.html` | The dashboard's single HTML page. |
| GET | `/web/js/<name>.js` | A frontend ES module. Confined to `web/`; anything outside 404s. |

## Engine-agnostic model verbs

These dispatch on the model's own backend, so the same call works for a
llama.cpp GGUF and a vLLM safetensors repo. The body takes an optional
`backend` hint (`"llamacpp"` / `"vllm"`); without it the id is looked up.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/models/load` | Load `{model}` on whichever engine owns it. |
| POST | `/api/models/unload` | Unload `{model}`. |
| POST | `/api/models/save` | Persist `{settings}` and apply them (restarting the process if that engine has no hot reload). |
| POST | `/api/models/delete` | Delete the model's files, where the engine supports it (vLLM only; llama.cpp returns 400). |

The engine-specific paths (`/api/load`, `/api/vllm/load`, …) remain as aliases.

## Request requirements

The dashboard binds `127.0.0.1`, which keeps it off your network but leaves it
reachable by any page in your browser. Every request is therefore checked:

- `Host` must name this loopback service, and `Origin` — when present — must
  match it. Anything else gets **403**. This blocks both cross-site requests and
  DNS rebinding.
- `POST` bodies must be `application/json`. A form content type gets **415**,
  which is what stops a cross-site `<form>` from forging a state change.
- `POST /api/config` only accepts an allowlist of user-facing keys. See
  [Security](https://github.com/dadwritestech/LlamaForge/blob/master/SECURITY.md).

See also [config.json Reference](config.md) for the settings `/api/config` and `/api/network` write, and [models.ini Format](models-ini.md) for the file `/api/save`, `/api/scan/apply`, `/api/scan/prune`, and `/api/hub/add` mutate.
