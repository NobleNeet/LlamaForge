---
title: models.ini Format
section: reference
order: 2
---

# models.ini Format

`models.ini` is an INI file passed straight to `llama-server` via `--models-preset <path>` (see `backend/router_ctl.py` `start()`). LlamaForge never translates it into a different format for the router — the dashboard reads and rewrites the active registry directly (`backend/config.py` `read_sections()` / `set_keys()`), and `llama-server` parses it itself when the router starts.

The built-in llama.cpp runtime uses `config.json`'s `models_ini` path (default: `<repo root>/models.ini`). Other llama-family runtimes use independent registries because forks may add, remove, rename, or reinterpret command-line flags. A fork-specific option must never be passed to another runtime merely because both binaries are llama.cpp-compatible.

The registry mapping is:

- built-in **llama.cpp** → `models_ini` (normally `models.ini`);
- **ik_llama** → `ik_llama_models_ini`, or a derived `-ikllama` sibling when unset (for example `models-ikllama.ini`);
- each **Custom Build Target** → that target's optional `models_ini`, or a derived sibling named from the target's stable ID when unset (for example target `custom-strix` → `models-custom-strix.ini`).

Custom Target registry identity is based on the stable target ID, never the display name. Renaming a target therefore does not change or orphan its model settings.

The active registry is selected together with the active runtime/build target. All llama-family model operations that read or mutate a registry — model listing, load configuration, knob editing, preset application, scan/download registration, automatic `ctx-size` updates, model deletion metadata, and router restart — operate on that selected registry only. Switching runtimes does not merge or copy settings from another runtime's registry.

## Registry creation and isolation

A registry is created if it is absent and must contain at least a `[*]` section because `llama-server` refuses to start router mode without one. A relative configured path is anchored to the repository root before it is handed to the detached router.

The built-in llama.cpp registry remains the baseline registry. When a Custom Build Target is activated for the first time and its registry does not yet exist, LlamaForge copies the current built-in llama.cpp registry to the target's resolved registry path before starting that target. This gives the target the same model inventory and current baseline settings without making the files permanently shared. After creation, the Custom Target registry is authoritative for that target and is never automatically re-synced from built-in llama.cpp.

If the target registry already exists, LlamaForge uses that registry rather than copying or merging settings from another target. Switching away and back restores that target's own settings, subject only to the active binary schema refresh described below. A Custom Target may therefore safely persist fork-only keys such as `spec-draft-adaptive` while that option continues to exist in that fork.

Removing a Custom Build Target removes registration only. Its derived or explicitly configured registry file remains on disk, like the target's source/build directories and binaries; LlamaForge must not delete model configuration implicitly.

For upgrades from the older shared-registry design, an already-active Custom Build Target whose dedicated registry does not yet exist is initialized from the existing built-in registry before the target is restarted. This preserves the settings that were previously shared. The one-time copy itself does not guess which keys are fork-specific. After registries are separated, each registry is refreshed against its own runtime's current option schema when that runtime is selected, so stale keys inherited during migration are removed from the runtime that no longer supports them.

## Runtime schema refresh and automatic sanitization

The selected `llama-server` binary is the source of truth for both the Web UI's editable load-option list and the option keys that may remain in that runtime's registry. LlamaForge derives the current option schema from the selected binary's `--help` output through `backend/argspec.py`; the registry sanitizer and the knob editor must use the same binary metadata so they cannot disagree about which options exist.

LlamaForge refreshes the selected registry against the selected binary schema at these points:

- on panel startup, before the active router is started;
- whenever the user switches between built-in llama.cpp, ik_llama, or a Custom Build Target, before the destination router is started;
- after a build/update replaces the binary that will continue as the active runtime, before that rebuilt binary is started again.

For a runtime/build-target switch, the required order is:

1. resolve and validate the destination binary;
2. resolve or initialize the destination registry;
3. obtain the destination binary's current flag schema;
4. sanitize **only the destination registry** using that schema;
5. make the same schema the source for the Web UI knob list;
6. stop/switch/restart the router against the sanitized destination registry.

Sanitization is intentionally runtime-local. Selecting built-in llama.cpp may remove a Strix-only key from built-in `models.ini`, but it must not edit the Strix Custom Target's `models-<target-id>.ini`. Selecting Strix later sanitizes the Strix registry against the Strix binary instead. No normal switch copies settings between registries.

A registry key is retained when it is either:

- a canonical option exposed by the selected binary's current schema;
- a recognized alias that can be rewritten to the current canonical key; or
- LlamaForge-owned/preset metadata explicitly allowed in `models.ini` even when it is not a normal editable CLI knob, such as `model`, `mmproj`, `spec-draft-model`, `embeddings`, and `load-on-startup`.

When a successful schema refresh finds a stored key that is no longer valid for that runtime, LlamaForge removes it from that runtime's registry. This covers both fork-specific contamination and ordinary version drift: if a future llama.cpp or fork release removes or renames an option, the stale setting is removed or canonicalized before the new binary parses the registry. Blank values and malformed registry content may be cleaned at the same time.

Destructive sanitization must never be based on an incomplete or unavailable schema. If LlamaForge cannot obtain a reliable option schema during an explicit runtime/build-target switch, the switch is refused **before the currently working router is stopped**, and the destination registry is left unchanged. During panel startup, schema-discovery failure leaves the registry unchanged and is reported as a warning/error rather than guessing that unknown keys are obsolete.

The refresh is idempotent: once a registry matches the current binary schema, repeating startup or selecting the same runtime again produces no further changes.

## Format

- One `[section]` per model. The section name is the model's id, used everywhere in the UI and API (`/api/load`, `/api/unload`, the `model` field in `/v1/messages` and `/v1/chat/completions`).
- An optional `[*]` section holds global defaults applied to every model that doesn't override them.
- Keys inside a section are `key = value` lines. A key name is the corresponding `llama-server` command-line flag with the leading `--` stripped and remaining dashes kept — `ctx-size` is `--ctx-size`, `n-cpu-moe` is `--n-cpu-moe`.
- `;` starts an inline or full-line comment; `config.read_sections()` strips it before returning values.
- The file must not carry a BOM — `_write()` in `backend/config.py` always writes without one, and sections are read with `encoding="utf-8-sig"` so a BOM left by another editor is tolerated on read but never re-added on write.

## Real example

From the project's own built-in `models.ini`:

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
model = D:/models/lmstudio-community/gpt-oss-120b-GGUF/gpt-oss-20b-MXFP4.gguf
ctx-size = 100000
```

`qwen3-coder-next-q4-k-m` has no `ctx-size` override, so it inherits `150000` from `[*]`. `gpt-oss-20b-mxfp4` sets its own `ctx-size = 100000`, overriding the global. `qwen3.6-35b-a3b-ud-q4-k-xl` adds `mmproj` (a multimodal projector file) and `n-cpu-moe` (MoE experts offloaded to CPU) on top of `model`.

A Custom Target fork can independently add keys that only that binary supports:

```ini
[gemma-4-31b-it]
model = /models/gemma-4-31b-it.gguf
spec-draft-adaptive = true
```

That key remains in the Custom Target's registry while the selected fork continues to expose it. If a later version of that same fork removes the option, the next schema refresh removes the stale key from that fork's registry.

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

Beyond these, any flag exposed by the **currently selected** `llama-server` binary can be set as a key. The dashboard's Advanced UI mode (`ui_mode: "advanced"` in `config.json`) discovers the active binary's full flag set at runtime by parsing `llama-server --help` (`backend/argspec.py` `build_schema()`). A key supported only by one fork belongs only in that fork's registry.

## Automatic ctx-size defaults

`config.apply_ctx_defaults()` keeps `ctx-size` values sane in the selected registry: it sets `[*] ctx-size = 150000` (the `gguf.CTX_FULL` baseline), and for any model whose GGUF-reported trained context length is below that, writes an explicit per-model `ctx-size` override capped to what the model actually supports (never over-extending it). Models whose trained length can't be read are left untouched, and a model that already supports the full 150000 has any smaller per-model override removed so it falls back to the global. This runs on server startup and after scan/download operations that add new models.

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

Prefer the dashboard over hand-editing: `POST /api/save` (per-model knobs) and the scan/download flows call `config.set_keys()` against the selected runtime's registry, updating or removing individual keys in place while preserving comments and the position of every other line. Hand edits are safe too — `read_sections()` is tolerant of blank lines, comments, and section order.

See also [config.json Reference](config.md) for registry path settings, [Build & Update](build.md) for runtime/build-target switching, and [HTTP API](api.md) for the endpoints that read and mutate the selected registry.