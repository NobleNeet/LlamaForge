---
title: Models & Tuning
section: guides
order: 1
---

# Models & Tuning

Tune every `llama-server` flag on a per-model basis, save named presets per model, and compare settings across models — all from the Models tab, without hand-editing `models.ini`.

## What it does

Each row in the Models list is a section of `models.ini` (or an auto-discovered model not yet added to it). Expanding a row opens a live knob editor built from the running `llama-server` binary's own `--help` output, so the available flags always match the binary you actually built — not a hardcoded list that can drift out of date across llama.cpp versions.

`backend/argspec.py` (`build_schema()`) runs `<server_bin> --help`, parses the column-aligned help text into typed, grouped knobs (bool, int, float, enum, path, string), and caches the result per `(server_bin path, binary mtime)` — the cache self-invalidates automatically if you repoint `server_bin` at a different binary or rebuild it (`backend/server.py` `schema()`). A handful of flags the router itself owns (`host`, `port`, `model`, `hf-repo`, and similar) are filtered out of the editor as `RESERVED` — they aren't safe to set per model.

The browser-side Models view must follow the same runtime identity immediately. A successful switch between built-in llama.cpp, ik_llama, or any Custom Build Target invalidates the currently cached llama-family schema in the SPA, fetches `/api/schema` again for the newly active binary, refreshes `/api/state`, and rebuilds any open model knob editor from that new schema. The user must not need to press F5 or otherwise reload the page before the Models tab reflects the newly active runtime.

For this purpose, runtime identity is not only `active_engine`. Built-in llama.cpp and a Custom Build Target both use `active_engine = "llamacpp"`, so a change of `active_llamacpp_build_target` / active `server_bin` is also a runtime-schema change and must trigger the same client-side invalidation. A normal polling refresh must continue preserving in-progress edits, but an explicit successful runtime/build-target switch is a deliberate schema boundary and therefore invalidates the old knob grid.

The persistent runtime badge shown in the page header must represent the **effective active runtime/build**, not only the backend family. When built-in llama.cpp is active it displays `llama.cpp`; when ik_llama is active it displays `ik_llama`; when a registered Custom Build Target is active it displays that target's configured display `name` (for example `strix-llama.cpp_vulkan`) rather than the generic `llama.cpp` family label. If the active llama.cpp-compatible binary is external/unregistered, the badge must use the same resolved active-build identity/fallback used by Build / Update rather than falsely presenting it as built-in llama.cpp. The badge must refresh immediately after a successful runtime/build-target switch in the current SPA session; F5 must not be required.

The editor has two density levels, controlled by `ui_mode` in `config.json`:

- **Lite** shows a curated set of common knobs (`n-gpu-layers`, `ctx-size`, `cache-type-k`/`cache-type-v`, `flash-attn`, `batch-size`, `ubatch-size`, `threads`, `tensor-split`, `temp`, `top-p`).
- **Advanced** exposes every flag the binary reports via `--help`, grouped under the section headers from the help text itself.

Because the flag count is entirely a function of your `llama-server` build, LlamaForge does not hardcode a number for it — it is whatever `--help` reports at the time the schema is built.

## How to use it

1. Open the **Models** tab and click a model's row to expand its editor.
2. Set the knobs you want to override. Unset fields inherit from the `[*]` global-defaults section of `models.ini`.
3. Click **Save + Reload**. This writes the changed keys into the model's `models.ini` section (`config.set_keys()`), and if the model is currently loaded, unloads it first, then tells the router to reload (`router("/models?reload=1")`) so the next load picks up the new settings — no dashboard or router restart required.
4. To save a set of knobs for later on that same model, click **Save current +** in the preset bar to name the model's current settings as a preset. Presets are model-scoped: each model row only shows its own saved presets. Delete a preset with the `×` on its chip.
   - If that model already has a bound default preset, **Overwrite bound** saves the current knob values back into that existing preset name instead of creating a new one.
   - Clicking a preset chip sets it as that model's **default** (`POST /api/presets/bind`) and immediately loads that preset's values into the editor. The bound chip is highlighted; clicking the same chip again clears the default binding and leaves the current knobs in place. Re-saving a bound preset re-syncs that same model.
5. To compare settings across models, click **Compare** above the model list, tick the checkbox on two or more rows, then open the comparison. The table lists every knob key any selected model has explicitly set and highlights cells that differ between models; a blank cell marked "inherit" means that model falls back to the `[*]` default.
6. Open **Static presets** to review **Safe / Balanced / Aggressive** recommendations, automatically calculated for the selected GGUF. Balanced is selected initially. Choosing a preset or clicking **Apply preset to editor** fills the fields; you can still edit every value before **Save + Reload**. **Recalculate presets** refreshes the current memory estimate. No model is loaded during recommendation.

## Delete a model

The **Models available on this server** section supports permanent deletion of a model from disk.

The delete control is intentionally not placed in the always-visible model row. Expand the target model first; **Delete** appears at the far right of that model's action row, after the normal load/save/client actions. It is a destructive action and should use danger styling without competing visually with the primary load controls.

Clicking **Delete** must not remove anything immediately. Show a confirmation dialog that identifies the model and the filesystem path or paths that will be affected, makes clear that the operation is permanent, and requires an explicit destructive confirmation such as **Delete permanently**. **Cancel** remains the safe default action.

Deletion is defined in terms of the selected model, not simply "delete the parent directory":

- If the target model is the only model payload in its containing model directory, delete that directory as a whole.
- If the same directory contains another model, another quantization of the same model, or any other model payload that must remain usable, delete only the files that belong to the selected model and keep the shared directory.
- A sharded GGUF is one model payload. Deleting it removes every shard in that shard set, not only the first shard recorded in the registry.
- An `mmproj` file may be deleted together with the model only when LlamaForge can determine safely that it belongs exclusively to that model. An `mmproj` that may be shared or whose ownership is ambiguous must be preserved.
- After deleting only the selected model files from a shared directory, remove the directory as well if and only if it has become empty.
- The fact that a path is the selected model file's parent directory is never, by itself, sufficient justification for recursive directory deletion.

Discover's existing on-disk download layout is retained. In particular, multiple GGUF quantizations downloaded from the same Hugging Face repository may continue to live in the same repository-named directory. The delete implementation must therefore handle shared directories safely rather than changing the download directory structure.

The UI should treat deletion as a model-level operation regardless of whether the physical deletion resolves to a whole directory, a single GGUF, or a shard set. After a successful deletion the Models list must refresh so the deleted model no longer appears as available on the server.

## Screenshot

![Models tab](docs/img/models.png)

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Live flag schema | `backend/argspec.py` `build_schema()` | Parses `<server_bin> --help`; cached by `(server_bin, mtime)`; refreshes automatically after a rebuild or binary change. |
| Runtime switch UI refresh | Build/Update activation + Models SPA state | A successful runtime/build-target switch invalidates the browser's llama-family schema cache, refetches `/api/schema` and `/api/state`, and rebuilds the knob editor from the new active binary without requiring F5. Runtime identity includes the active Custom Build Target / `server_bin`, not only `active_engine`. |
| Header runtime badge | Global page header + resolved active build identity | Shows `llama.cpp` for built-in llama.cpp, `ik_llama` for ik_llama, and the registered Custom Build Target's configured display name when a custom llama.cpp-compatible build is active. External/unregistered builds use the same resolved fallback identity as Build / Update. Updates immediately after a successful switch without F5. |
| Reserved flags | `backend/argspec.py` `RESERVED` | Router-owned flags (`host`, `port`, `model`, `hf-repo`, etc.) are excluded from the per-model editor. |
| Hot reload | `POST /api/save` (`backend/server.py`) | Writes knobs via `config.set_keys()`, unloads the model if running, then calls the router's `/models?reload=1` so `models.ini` is re-read live. |
| Presets | `POST /api/presets/save` / `/apply` / `/delete` / `/bind` | Named knob sets stored per model in `config.json`'s `model_presets` key. The Models UI uses **bind** to make a chip the model's default, record the pairing in `preset_bindings`, materialize those knobs into that same model, and refresh the editor with the preset's values. |
| Compare | Models tab, Compare toggle (`web/js/models.js` `openCompare()`) | Client-side diff of `settings` across two or more selected models; no separate endpoint. |
| Static presets | `POST /api/autotune/recommend` | GGUF tensor/header geometry + hardware memory budget → Safe / Balanced / Aggressive, including settings, rationale, confidence, memory estimates and fit status. |
| Model deletion | Models tab, expanded model action row | Permanent model-level deletion with explicit confirmation. Delete the whole containing directory only when it contains no other model payload; otherwise delete only the selected model's file set, including all shards. Preserve ambiguous/shared `mmproj` files and remove a shared directory only after it becomes empty. Discover's current repository-based download layout is unchanged. |
| UI density | `ui_mode` in `config.json` (`"lite"` / `"advanced"`) | Lite = curated knob subset; advanced = the full parsed schema. |

## Troubleshooting

If the editor shows "Could not read knobs from `llama-server --help`", `server_bin` in `config.json` is missing, wrong, or the binary failed to run (missing DLLs is common on Windows). Fix the path from the Setup tab or `config.json` directly — the schema is retried automatically on the next open, no restart needed. If `--help` runs but returns no arguments, the help text format wasn't recognized; check the binary is actually `llama-server` and not a different tool.

If Build / Update reports a successful runtime/build-target switch but the Models editor still shows knobs from the previous runtime, that is a UI synchronization failure. A successful switch is required to refresh the schema and editor within the current SPA session; F5 is not part of the normal workflow.

See also [models.ini Format](models-ini.md) for the on-disk file this editor writes to, and [config.json Reference](config.md) for `server_bin` and `ui_mode`.

See [Static AutoTune](autotune.md) for memory assumptions and limitations.
