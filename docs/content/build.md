---
title: Build & Update
section: guides
order: 3
---

# Build & Update

Check whether your local llama.cpp checkout is behind upstream, and rebuild it with CMake flags auto-detected for your machine's GPU and CPU — without hand-typing `cmake` commands.

## What it does

The Build tab reports two independent things: how your local `llama_src` checkout compares to upstream `github.com/ggml-org/llama.cpp`, and what CMake flags this machine should build with.

**Current vs. upstream.** `backend/builder.py`'s `BuildManager.current_commit()` reads the local checkout's HEAD (`git log -1`) and branch. `check_updates()` runs `git fetch --quiet origin` against `origin/master`, then `git rev-list --count HEAD..origin/master` to report how many commits you're behind, plus the latest upstream commit's hash and subject. This check is cached for 15 minutes (`UPDATE_TTL = 900` seconds) so opening the Build tab doesn't `git fetch` GitHub on every visit; a failed fetch is retried sooner (`UPDATE_TTL_FAIL = 60` seconds). The **Check Update** button bypasses the cache with `force=1`, and a finished build clears the cache outright so the next check is fresh.

**Auto-detected build flags.** `backend/hardware.py`'s `recommend()` inspects your GPU(s) and CPU, then returns a dict of `{cmake_flags, notes, runtime, gpus, cpu}`:

- On **macOS**, it returns `GGML_METAL=ON` and `GGML_NATIVE=ON`, with a note that Metal uses unified memory as VRAM. No CUDA branch runs on Mac.
- If one or more **NVIDIA GPUs** are detected, it sets `GGML_CUDA=ON`, and if compute capabilities were readable, `CMAKE_CUDA_ARCHITECTURES` to the sorted, deduplicated list of detected architectures (e.g. `86;89`). It also sets `GGML_CUDA_FA_ALL_QUANTS=ON` to enable flash attention across all quantized KV cache combinations.
- On **Linux with AMD GPUs**, it can emit `GGML_HIP=ON` with `GPU_TARGETS=<gfx...>` when ROCm/HIP is available, or `GGML_VULKAN=ON` when Vulkan is available. `GPU_TARGETS` is only set when a gfx architecture was detected (for example `gfx1151` from `rocminfo`).
- If no acceleration backend is selected or available, it configures a CPU-only build.
- On Windows/Linux either way, it always sets `GGML_NATIVE=ON`. If the CPU looks like it supports AVX-512 (`avx512_hint` — on Linux read from real CPU flags; on Windows a name-based heuristic for Ryzen 7/9 (7000- and 9000-series), Xeon, and Threadripper parts), it additionally sets `GGML_AVX512=ON`, `GGML_AVX512_VNNI=ON`, `GGML_AVX512_VBMI=ON`, and `GGML_AVX512_BF16=ON`.
- It also returns a `runtime` recommendation (not a CMake flag, but a suggested per-model default): `n-gpu-layers=99` and `flash-attn=on` when a GPU (or Mac Metal) is present, `n-gpu-layers=0` and `flash-attn=off` on CPU-only machines.

These flags are shown as pills on the Build tab and used as the default for a rebuild. The Build tab also has a backend selector (`auto`, `cuda`, `hip`, `vulkan`, `cpu`) saved in `config.json` as `llama_backend`.

A rebuild (`BuildManager.run_build()`) runs in a background thread: it first validates that `llama_src`/`build_dir` are set and the source exists (an unset path is reported plainly instead of running `cmake -B  -S ` and surfacing CMake's own error), optionally `git pull --ff-only origin`, backs up the current binaries directory (`bin/Release` on Windows/MSVC, `bin` elsewhere) to a timestamped copy before touching anything, runs `cmake -B <build_dir> -S <llama_src> -DCMAKE_BUILD_TYPE=Release` with your flags, then `cmake --build <build_dir> --config Release --parallel <jobs>`. Output streams to a log file the dashboard polls live. On success it records where `llama-server` actually landed into `config.json`, so a first build doesn't leave you hand-fixing `server_bin`.

**"Built, with warnings."** `cmake --build` returns a single exit code for the whole build, so a failing non-essential target (the npm/`sharp` UI-asset step on Windows is the common one) used to be reported as a hard **BUILD FAILED** even though `llama-server` itself built. Now, if the build step fails but a *freshly built* `llama-server` is present (freshness proven by comparing its mtime to the build's start, so a stale binary from a previous build can't mask a real compile failure), the build is reported as **built, with warnings** — the binary is recorded and usable, and the failing step is left in the log for you to read. A build that produced no fresh binary is still a hard failure.

**Two llama-family engines.** The Build tab has a **Build Target** selector. Building the `ik_llama` target uses its own `ik_llama_src` / `ik_llama_build_dir` / `ik_llama_cmake_flags` and produces a separate binary. A **Switch engine** control (`POST /api/engine/switch`) points the router at the chosen engine by setting `active_engine`. The switch is gated on a capability probe (`router_ctl.supports_router_mode()`): because LlamaForge drives the router as `<server_bin> --models-preset ...`, a binary whose `llama-server` predates router mode is **refused** with an explanation rather than being switched to and taking the router down. Each engine reads its own `models.ini` (ik_llama uses a `-ikllama` sibling), and the knob editor reflects whichever engine is active.

## How to use it

1. Open the **Build** tab. **Current Build** shows your checkout's commit, branch, and date; **Upstream** shows whether you're behind `origin/master` and by how many commits.
2. Click **Check Update** to force a fresh upstream check instead of waiting for the 15-minute cache.
3. Review the **Acceleration Backend** selector and the **Build Flags** pills — these are auto-detected for your GPU/CPU and chosen backend.
4. Leave **git pull first** checked (it's checked by default when you're behind) to pull the latest commits before building, or uncheck it to rebuild the current checkout as-is.
5. Click **Pull latest & Rebuild** (or **Rebuild current** if already up to date). The build runs in the background; watch progress and any errors in the Build Log panel below, which polls live until the build finishes or fails.
6. If vLLM is installed, the same tab shows its installed vs. latest PyPI version, with an **Update vLLM** button when a newer release is available.

## Custom Build Targets

Choose **+ Add Target** in **Build / Update**, enter Name, Repository, Branch,
Source, Build, Server Binary, and Build Command, then **Validate** and **Save Target**.
Source and Build must be absolute paths. Source may be absent, or the root of an
existing Git checkout. Repository accepts HTTP(S), SSH (including `git@host:path`),
and Git URLs. Validation checks inputs locally; it does not test remote availability.
Neither Validate nor Save clones a repository or executes the Build Command.

**Pull & Build** clones a missing Source at the specified Branch. For an existing
checkout it fetches origin and pulls the specified branch with `--ff-only`.
The checkout's origin and current branch must match the saved Repository and Branch;
a mismatch fails with an explanation, without changing remotes or switching branches.
A failed fetch, pull, or clone stops the build.

The Build directory is created and used as the command's initial working directory.
Only the saved Build Command is executed; no CMake flags or fork-specific options
are generated. Multiline commands run using `bash -e -o pipefail`; ordinary command
or pipeline failures stop execution. Bash's normal conditional exceptions (for
example commands used in `if` or `||`) still apply. Windows custom builds require
Git Bash available as `bash` on PATH; built-in Windows builds retain their existing
CMake execution. You may use `cd` within the recipe.

```bash
cmake -S "{source}" -B . -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel {jobs}
```

`{source}` and `{build}` expand to the saved paths; `{jobs}` is the detected logical
CPU count (fallback 8). Substitution is textual: quote paths as appropriate for your
shell command. Other shell braces and `${variables}` are preserved.
Server Binary supports `{source}` and `{build}`; relative output paths resolve
against Build. A zero exit code without the specified binary is a failed build.
Git output and command stdout/stderr stream to Build Log. Prior binary directories
and the specified existing server binary are copied to timestamped backups.

Commands run with the LlamaForge process's permissions. Save only commands you
have reviewed; LlamaForge does not extract commands from repository documentation.
**Pull & Build** never changes the active binary. After building, click
**Use this build** to activate it explicitly as a llama.cpp-compatible build.

**Check Update** compares HEAD with `origin/<branch>` using the existing cache;
it never executes the recipe. **Edit** keeps the target ID stable and does not
move or delete old directories. **Remove** removes registration only: source,
build, binaries, and Git files remain on disk. Editing/removing a running target
is rejected. Custom targets are not part of the llama.cpp automatic update schedule.

### Use this build

Select a Custom Target and click **Use this build**. The file must exist, be
executable, and pass the existing `--models-preset` router compatibility probe.
Missing binaries produce “Server Binary does not exist. Build this target first.”
Validation failures leave the current runtime and configuration unchanged.

Activation keeps `active_engine = "llamacpp"` and changes `server_bin` to the
resolved target binary. A running router's loaded model IDs are captured before
stopping it. LlamaForge starts the new binary, waits for router readiness, then
attempts to restore the previous llama.cpp models using the existing model reload
helper. Switching from ik_llama does not copy its separate model registry.
A stopped router is started by activation. Builds must finish before activation.

If startup fails or readiness times out, the previous configuration is restored.
Any partially started new router is stopped, and the old router is restarted if
it was previously running; its models are reloaded on a best-effort basis.
Recovery failures are reported separately, so a failed rollback is never presented
as a successful activation. Model reload itself is best-effort, not transactional.

The UI shows **Active runtime**, **Active build**, and the configured binary path.
A Custom Target whose resolved binary is already active displays **Active**.
Editing or removing a target does not change the active binary; if no registered
target matches it afterward, the UI shows **External / unregistered build**.

To return, select the built-in **llama.cpp** target and click **Switch to llama.cpp**.
LlamaForge preserves the previously selected default binary (including an explicit
external `server_bin`) for this operation. Subsequent built-in builds update that
saved build path without replacing an active Custom Target. Automatic llama.cpp
updates skip while a Custom Target is selected; they resume under the existing
schedule after switching back. ik_llama remains a separate, unchanged runtime.

### Configuration and API

Existing configurations need no migration. A separate `custom_build_targets`
object maps generated stable IDs to target records:

```json
{
  "custom_build_targets": {
    "custom-example": {
      "id": "custom-example",
      "name": "My fork",
      "repository": "https://github.com/example/llama.cpp",
      "branch": "main",
      "source": "/home/user/fork",
      "build": "/home/user/fork/build",
      "server_binary": "{build}/bin/llama-server",
      "build_command": "cmake -S \"{source}\" -B .\ncmake --build . --parallel {jobs}"
    }
  }
}
```

Two optional configuration keys default to empty strings for older installations:

- `llama_builtin_server_bin`: saved default llama.cpp binary for switching back.
- `active_llamacpp_build_target`: selected custom ID, `llamacpp` after switching
  back, or empty for legacy/default selection. This never participates in runtime
  backend dispatch; `active_engine` remains `llamacpp` or `ikllama`.

- `GET /api/build/targets`: built-in and custom target records (`builtin` flag),
  plus `active_build` with `id`, `name`, and `server_bin`.
- `POST /api/build/activate`: `{ "target": "custom-ID" }`; validates and activates
  a saved Custom Target. Unknown IDs and built-in IDs return HTTP 400; running
  builds or unsafe current-router state return 409. Startup failure returns 500
  with `rollback_ok` and `rollback_error`. Success returns `active_engine` and
  `active_build`. Returning to built-in uses the existing
  `POST /api/engine/switch` with `{ "engine": "llamacpp" }`.
- `POST /api/build/targets/validate`: target fields; returns normalized fields and
  resolved binary path without saving.
- `POST /api/build/targets/save`: target fields; omit `id` to create, include an
  existing custom `id` to edit. Returns the saved target.
- `POST /api/build/targets/remove`: `{ "id": "..." }`; removes registration only.
- Existing `/api/build/info`, `/api/build/log`, and `/api/build/start` accept the
  custom target ID. Start uses the saved recipe, ignoring ad-hoc command/flag input.
  Unknown target IDs return HTTP 400, rather than falling back to llama.cpp.

## Daily automatic updates

In **Build / Update → Automatic Update · llama.cpp**, enable **Pull latest & rebuild automatically when idle**, choose a daily time, and click **Save schedule**. The default is disabled, with 03:00 preselected. The displayed timezone is the server's local timezone. LlamaForge must be running during the scheduled minute; the browser may be closed. Missed times are not caught up.

The scheduler checks once per local day. Active requests, a build, another active engine, a running vLLM server, or unavailable activity information skip that day's update. The last check and its reason appear in the card. It checks every loaded model's current processing and queued request counts using the upstream [llama.cpp metrics endpoint](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-metrics-prometheus-compatible-metrics-exporter); missing metrics are treated as unknown, not idle.

During an automatic update, new panel API operations return HTTP 503 and the llama.cpp router is stopped, so inference is temporarily unavailable. After the build, LlamaForge attempts to restore the router and previously loaded models, including after build failure. Clients connected directly to the router should retry connections during this maintenance window; their activity is checked immediately before stopping the router, but a direct request can still arrive between that check and the stop. Existing external `server_bin` paths are preserved. Only llama.cpp is updated. A failed git pull cancels the automatic rebuild; inspect **Build Log · llama.cpp** for build results.

## Screenshot

![Build tab](docs/img/build.png)

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Current commit | `BuildManager.current_commit()` | `git log -1` (hash, subject, date) + current branch on `llama_src`. |
| Upstream check | `BuildManager.check_updates()` | `git fetch` + `git rev-list --count HEAD..origin/master`; cached 15 min (60s on failure), bypassed with `force=1`. |
| Recommended flags | `hardware.recommend()` | Returns `cmake_flags`, human-readable `notes`, a `runtime` suggestion, and the detected `gpus`/`cpu`. |
| macOS flags | `hardware.recommend()` | `GGML_METAL=ON`, `GGML_NATIVE=ON`. |
| NVIDIA GPU flags | `hardware.recommend()` | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=<archs>` (if readable), `GGML_CUDA_FA_ALL_QUANTS=ON`. |
| AMD ROCm/HIP flags | `hardware.recommend()` | `GGML_HIP=ON`, plus `GPU_TARGETS=<gfx arch>` when detected. |
| Vulkan flags | `hardware.recommend()` | `GGML_VULKAN=ON`. |
| CPU-only flags | `hardware.recommend()` | No accelerator flags; `GGML_NATIVE=ON` still set. |
| AVX-512 flags | `hardware.recommend()` | `GGML_AVX512`, `GGML_AVX512_VNNI`, `GGML_AVX512_VBMI`, `GGML_AVX512_BF16` — all `ON` when `avx512_hint` is true. |
| Runtime suggestion | `hardware.recommend()["runtime"]` | `n-gpu-layers=99`, `flash-attn=on` with a GPU/Metal; `n-gpu-layers=0`, `flash-attn=off` CPU-only. |
| Rebuild | `BuildManager.run_build()` | Validates paths, optional `git pull --ff-only`, backs up prior binaries, `cmake` configure + build (Release, parallel jobs = CPU count by default), records the built `server_bin`. |
| Partial success | `BuildManager.run_build()` | Build-step failure with a fresh `llama-server` present → `done_warnings` (amber "built with warnings"); no fresh binary → hard `failed`. |
| Engine target / switch | Build tab selector, `POST /api/engine/switch` | Builds `llama.cpp` or `ik_llama`; switching sets `active_engine`, refused if the target binary has no router mode. |

## Troubleshooting

If the upstream status shows "check failed," `git fetch` couldn't reach GitHub (network issue, or `llama_src` isn't a valid checkout) — check `llama_src` in `config.json` points at a real git clone. If a build fails, the Build Log panel shows the failing `cmake` step's output; prior binaries are always backed up first (to a `bin-backup-<timestamp>` folder next to the build output) so a bad build doesn't leave you without a working server. If flag detection looks wrong, verify the backend-specific tools (`nvidia-smi`, `rocminfo` / `hipconfig`, `vulkaninfo`) are on `PATH` and working.

See also [Models & Tuning](models.md) for the flags the resulting `llama-server` binary exposes, and [config.json Reference](config.md) for `llama_src`, `build_dir`, `server_bin`, and `cmake_flags`.
