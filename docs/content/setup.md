---
title: Setup
section: guides
order: 4
---

# Setup

Check build prerequisites, install what is missing with your permission, scan your drives for GGUF models already on disk, and prune registry entries whose files are gone.

## What it does

The Setup tab reports on four independent things, each backed by its own backend module.

**Prerequisite detection.** `backend/prereqs.py` checks for four command-line tools — `git`, `cmake`, `ninja`, and `python` (via `shutil.which`, running `--version` to confirm and capture the installed version) — plus the platform C++ compiler and backend-specific acceleration toolchains. CUDA detection reads `nvcc --version` (or `$CUDA_PATH`) and is skipped on macOS. On Linux, ROCm/HIP is reported from `hipconfig` + `rocminfo`, and Vulkan from `vulkaninfo` + `glslc`.

**Installing missing tools.** Which package manager runs depends on the OS, verified directly in `prereqs.py`:

- **Windows:** `install()` tries `winget install --id <pkg> -e --accept-source-agreements --accept-package-agreements` first; if `winget` isn't on `PATH` or the install fails, it falls back to `choco install <pkg> -y`.
- **macOS:** `install()` requires Homebrew and runs `brew install <pkg>` — no `sudo` needed.
- **Linux:** the dashboard never runs `sudo`. `install()` returns `ok=False` with the exact command to paste into a terminal (`sudo apt-get install -y <pkg>`, `sudo dnf install -y <pkg>`, or `sudo pacman -S --noconfirm <pkg>`, picked by `osplat.linux_pkg_manager()` probing `apt-get` → `dnf` → `pacman` in that order).

Each tool in `prereqs.TOOLS` carries its own `winget`/`choco`/`brew`/`pkg` identifiers and a fallback download URL, used whichever path applies. `_can_auto_install()` gates the **Install** button: on Windows it needs `winget` or `choco` present; on macOS it needs `brew`; on Linux the button never appears — only the hint.

After a successful install, the running process's `PATH` is refreshed from the registry (`osplat.refresh_path()`, Windows-only) and the tool is re-probed, so a freshly installed `ninja` or `cmake` is detected immediately. Only if it still isn't visible does the result ask you to restart LlamaForge — the old behavior (always MISSING until a restart) is gone.

**Drive scan for GGUFs.** `backend/scanner.py`'s `scan()` (via `POST /api/scan`) walks a set of root directories looking for `.gguf` files at least 50 MB, skipping recycle bins, `.git`, and `.trash` folders. In the Setup tab you can provide any number of explicit scan directories; if that list is empty, the roots fall back to `list_drives()`: on Windows, every fixed drive letter (`GetLogicalDrives` + `GetDriveTypeW == DRIVE_FIXED`, so removable/network drives are excluded); on macOS, `$HOME` plus `/Volumes` if it exists; on Linux, `$HOME` plus any of `/mnt`, `/media`, `/srv`, `/data` that exist. Multi-shard GGUF sets collapse to their first shard, `mmproj*` files attach to their sibling model instead of appearing standalone, `mtp-*` speculative draft sidecars attach as a `spec-draft-model` (see [models.ini Format](models-ini.md)), and files with `embed` in the name are flagged as embedding endpoints. Confirmed results are written into `models.ini` via `POST /api/scan/apply`.

**Registry prune.** `POST /api/scan/prune` ("Check for deleted models") takes a list of model IDs, re-checks each one's `model` path in `models.ini` against disk, and — only for entries whose file no longer exists — unloads it from the router if currently loaded, then removes its section from `models.ini` with `config.remove_section()`. An entry whose file has reappeared since the check is left alone. This edits `models.ini` only; it never touches files on disk.

The tab also surfaces `hardware.recommend()`'s detected CPU/GPU/backends (shared with the Build tab), including AMD architecture strings such as `gfx1151` when available, lets you pick a **favourite model to auto-load on launch** (`auto_load_model` in `config.json`), and — on Windows — the vLLM/WSL2 install flow described in [vLLM Backend](vllm.md).

**Models kept loaded at once.** Under **Startup**, a **Models kept loaded at once** field edits `router_models_max` in `config.json`. At the shipped default of `1` loading a second model always unloads the first, because `llama-server` evicts by **count** and never inspects free memory; a higher number lets several models stay resident, and `0` switches llama.cpp's own count-based eviction off entirely. The limit is baked into the router's command line at startup, so saving this field restarts the router through `POST /api/router/restart` — after a confirmation that names the models about to be unloaded.

## How to use it

1. Open the **Setup** tab. **Prerequisites** lists Git, CMake, Ninja, Python, your C++ compiler, and acceleration-specific toolchains such as CUDA, ROCm/HIP, and Vulkan when applicable.
2. Click **Install** next to a missing tool to install it with your OS's package manager (Windows: winget, falling back to choco; macOS: Homebrew). On Linux, copy the shown command into a terminal yourself — the dashboard never runs `sudo`.
3. Review **Detected Hardware** — your CPU and any GPUs found, shared with the Build tab's flag recommendations.
4. Under **Scan Drives for Models**, add zero or more explicit directories to search. Leave the list empty to use the platform-default roots.
5. Click **Scan for GGUF models** to walk those directories for `.gguf` files; review the results and apply the ones you want registered.
6. Click **Check for deleted models** to find registry entries whose backing file no longer exists on disk, then prune the ones you confirm.
7. Optionally pick a model under **Startup** to auto-load when LlamaForge launches.
8. Optionally raise **Models kept loaded at once** (or set `0`) under **Startup** to hold more than one model resident. Confirm the restart when prompted — it unloads whatever is loaded, and per-model usage stats are only exact at `1`.

## Screenshot

![Setup tab](docs/img/setup.png)

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Prereq detection | `prereqs.status()` | `git`, `cmake`, `ninja`, `python` via `shutil.which` + `--version`; C++ compiler plus CUDA / ROCm-HIP / Vulkan toolchains detected separately. |
| Windows install | `prereqs.install()` | `winget install --id <id> -e ...`, falling back to `choco install <id> -y` if winget is absent or fails. |
| macOS install | `prereqs.install()` | `brew install <pkg>` — requires Homebrew, no sudo. |
| Linux install | `prereqs.install()` | Never runs sudo; returns the exact `apt-get`/`dnf`/`pacman` command to run yourself. |
| Drive scan roots | `config.model_dirs` or `scanner.list_drives()` | User-specified directories from Setup take precedence; empty falls back to Windows fixed drives, macOS `$HOME` + `/Volumes`, or Linux `$HOME` + existing `/mnt` `/media` `/srv` `/data`. |
| GGUF discovery | `scanner.find_ggufs()` | `.gguf` files ≥ 50 MB; skips recycle bin/`.git`/`.trash`; collapses multi-shard sets; attaches `mmproj` and `mtp-*` siblings; flags `embed` files. |
| PATH refresh after install | `osplat.refresh_path()` | Windows-only; re-reads PATH from the registry so a just-installed tool is detected without a restart. |
| Registry prune | `POST /api/scan/prune` | Removes a `models.ini` section only if its `model` file no longer exists on disk; unloads it from the router first if loaded. |
| Auto-load | `config.json: auto_load_model` | Model ID to load automatically once the router is ready after launch. |
| Loaded-model limit | `router_models_max` → `POST /api/config` + `POST /api/router/restart` | How many models `llama-server` may hold loaded at once (`--models-max`); `0` = unlimited, and llama.cpp's count-based eviction is what a higher number raises. The limit is read at router startup only, so saving restarts the router — which unloads every loaded model, after a confirmation naming them. |

## Troubleshooting

If **Install** doesn't appear for a missing tool, no supported package manager was found for your OS (Windows without winget or choco, macOS without Homebrew) — use the tool's download URL shown next to it instead. If a Windows install reports "winget failed" then falls through to choco, check the combined output shown in the toast/log for the underlying error (often a source-agreement prompt or an already-installed conflicting version). If the drive scan finds nothing on Windows, confirm your models sit on a fixed (non-removable, non-network) drive — `list_drives()` skips those by design. If **Check for deleted models** reports an entry you know still exists, verify the `model` path in `models.ini` matches the file's real location; a moved file reads as "deleted" until you re-scan and re-apply it.

See also [Build & Update](build.md) for the same hardware detection used to pick CMake flags, and [vLLM Backend](vllm.md) for the WSL2 install flow shown at the bottom of this tab on Windows.
