---
title: models.ini Format
section: reference
order: 2
---

# models.ini Format

`models.ini` is an INI file passed straight to `llama-server` via `--models-preset <path>` (see `backend/router_ctl.py` `start()`). LlamaForge never translates it into a different format for the router — the dashboard reads and rewrites it directly (`backend/config.py` `read_sections()` / `set_keys()`), and `llama-server` parses it itself when the router starts. `config.json`'s `models_ini` key points at the file (default: `<repo root>/models.ini`).

The file is created for you if it's absent: `config.ensure_models_ini()` runs on startup (and both launch scripts do the same before starting the router) and writes a minimal file with a `[*]` section, because `llama-server` refuses to start without one. A **relative** `models_ini` (the shipped default is `./models.ini`) is anchored to the repo root before it's handed to the detached router, so launching from another directory can't point the router at an empty registry. When the active engine is **ik_llama**, a separate `-ikllama` sibling registry is used (e.g. `models-ikllama.ini`), since the two binaries accept different flags — see [Build & Update](build.md).

## Format

- One `[section]` per model. The section name is the model's id, used everywhere in the UI and API (`/api/load`, `/api/unload`, the `model` field in `/v1/messages` and `/v1/chat/completions`).
- An optional `[*]` section holds global defaults applied to every model that doesn't override them.
- Keys inside a section are `key = value` lines. A key name is the corresponding `llama-server` command-line flag with the leading `--` stripped and remaining dashes kept — `ctx-size` is `--ctx-size`, `n-cpu-moe` is `--n-cpu-moe`.
- `;` starts an inline or full-line comment; `config.read_sections()` strips it before returning values.
- The file must not carry a BOM — `_write()` in `backend/config.py` always writes without one, and sections are read with `encoding="utf-8-sig"` so a BOM left by another editor is tolerated on read but never re-added on write.

## Real example

From the project's own `models.ini`:

```ini
[*]
ctx-size = 150000

[qwen3.6-35b-a3b-ud-q4-k-xl]
model = D:/models/LlamaForge-downloads/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
mmproj = D:/models/LlamaForge-downloads/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F16.gguf
n-cpu-moe = 37

[qwen3-coder-next-q4-k-m]
model = D:/models/lmstudio-community/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf

[gpt-oss-20b-mxfp4]
model = D:/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf
ctx-size = 100000
```

`qwen3-coder-next-q4-k-m` has no `ctx-size` override, so it inherits `150000` from `[*]`. `gpt-oss-20b-mxfp4` sets its own `ctx-size = 100000`, overriding the global. `qwen3.6-35b-a3b-ud-q4-k-xl` adds `mmproj` (a multimodal projector file) and `n-cpu-moe` (MoE experts offloaded to CPU) on top of `model`.

## Common keys

| Key | Maps to | Purpose |
|---|---|---|
| `model` | `--model` | Path to the GGUF model file. Required per section. |
| `mmproj` | `--mmproj` | Path to a multimodal projector GGUF, for vision-capable models. |
| `ctx-size` | `--ctx-size` | Context window size, in tokens. |
| `n-cpu-moe` | `--n-cpu-moe` | Number of MoE expert layers offloaded to CPU (mixed CPU/GPU inference for MoE models). |
| `embeddings` | `--embeddings` | Set to `true` to mark/run the model in embeddings mode. |
| `spec-draft-model` | `--spec-draft-model` | Path to a speculative draft model (e.g. an auto-attached `mtp-*` sidecar). |
| `spec-type` | `--spec-type` | Speculative-decoding type, e.g. `draft-mtp` or an `ngram-*` mode. |

Beyond these, any `llama-server` flag can be set as a key — the dashboard's Advanced UI mode (`ui_mode: "advanced"` in `config.json`) exposes the full set (roughly 220 flags), discovered at runtime by parsing `llama-server --help` (`backend/argspec.py` `build_schema()`).

## Automatic ctx-size defaults

`config.apply_ctx_defaults()` keeps `ctx-size` values sane across the file: it sets `[*] ctx-size = 150000` (the `gguf.CTX_FULL` baseline), and for any model whose GGUF-reported trained context length is below that, writes an explicit per-model `ctx-size` override capped to what the model actually supports (never over-extending it). Models whose trained length can't be read are left untouched, and a model that already supports the full 150000 has any smaller per-model override removed so it falls back to the global. This runs on server startup and after scan/download operations that add new models.

## Auto-wired vision projectors

Keep each model family and its matching `mmproj*.gguf` in the same directory.
Scanning or registering a download attaches the projector to the model's
`mmproj` setting without relying on an architecture allowlist. This also works
for multiple quantizations of the same model and sharded GGUFs. Nonempty
projectors smaller than the normal 50 MB scan threshold are included.

Exactly one distinct projector in the directory is attached automatically. If
there are several, set `mmproj` explicitly in the model section or keep the
desired model/projector pair in its own directory. Keep unrelated model families
in separate directories; sharing a directory does not establish compatibility.
Rescans and download registration preserve an existing `mmproj` setting when
the model path has not changed.

For models already registered without a projector, click **Scan for GGUF models**
in Setup. The scan saves missing projector settings for existing model paths
immediately, even when there are no new models to add. The results list the
updated models. Existing projector and tuning settings are preserved. Unload
and reload any previously loaded model to use the new setting. New models still
require the **Add models to config** action.

## Auto-wired MTP draft models

When a scan finds an `mtp-*` GGUF next to a model, it attaches it the way `mmproj` is attached: the sibling's path is written as `spec-draft-model` on the parent section. It additionally sets `spec-type = draft-mtp` **only** when the sidecar declares NextN layers (`gguf.has_nextn()` — the property llama.cpp gates MTP on), so a sidecar the current build can't use is attached but left inert rather than breaking the load. This wiring is **additive**: because `spec-type` is also how you select `ngram-*` speculation, a re-scan never overwrites a `spec-type` (or `spec-draft-model`) you already set by hand.

## Editing

Prefer the dashboard over hand-editing: `POST /api/save` (per-model knobs) and the scan/download flows call `config.set_keys()`, which updates or removes individual keys in place while preserving comments and the position of every other line. Hand edits are safe too — `read_sections()` is tolerant of blank lines, comments, and section order.

See also [config.json Reference](config.md) for the `models_ini` path setting, and [HTTP API](api.md) for the endpoints that read and mutate this file.
