"""Detect and (with consent) install build prerequisites.

Windows: winget -> choco. macOS: Homebrew (no sudo needed). Linux: we detect
apt/dnf/pacman but never run sudo from the GUI - install() returns the exact
command for the user to run instead. Never installs without an explicit request.
Acceleration backends are reported separately so missing Vulkan does not make a
CPU or CUDA setup look broken.
"""
import os, shutil, subprocess, glob

import osplat

# name -> detection + per-platform package ids
TOOLS = {
    "git": {
        "cmd": "git", "args": ["--version"],
        "winget": "Git.Git", "choco": "git", "brew": "git", "pkg": "git",
        "url": "https://git-scm.com/downloads",
    },
    "cmake": {
        "cmd": "cmake", "args": ["--version"],
        "winget": "Kitware.CMake", "choco": "cmake", "brew": "cmake", "pkg": "cmake",
        "url": "https://cmake.org/download/",
    },
    "ninja": {
        "cmd": "ninja", "args": ["--version"],
        "winget": "Ninja-build.Ninja", "choco": "ninja", "brew": "ninja",
        "pkg": "ninja-build",
        "url": "https://github.com/ninja-build/ninja/releases",
    },
    "python": {
        "cmd": "python", "alt_cmd": "python3", "args": ["--version"],
        "winget": "Python.Python.3.12", "choco": "python", "brew": "python@3.12",
        "pkg": "python3",
        "url": "https://www.python.org/downloads/",
    },
}

def _which(cmd):
    return shutil.which(cmd)

def _tool_exe(spec):
    return _which(spec["cmd"]) or _which(spec.get("alt_cmd", ""))

def _version(spec):
    exe = _tool_exe(spec)
    if not exe:
        return None
    try:
        out = subprocess.run([exe] + spec["args"], capture_output=True, text=True, timeout=8)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        return exe

def find_msvc():
    """MSVC's cl.exe lives inside a VS install, not on PATH. Locate via vswhere."""
    vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
    if os.path.exists(vswhere):
        try:
            out = subprocess.run([vswhere, "-latest", "-products", "*",
                                  "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                  "-property", "installationPath"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                return {"present": True, "path": out}
        except Exception:
            pass
    # fallback: look for cl.exe under common VS dirs
    for base in (r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio"):
        hits = glob.glob(base + r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe")
        if hits:
            return {"present": True, "path": hits[0]}
    return {"present": False, "path": "",
            "url": "https://visualstudio.microsoft.com/downloads/ (Build Tools for Visual Studio -> Desktop development with C++)"}

def find_compiler():
    """The platform C++ compiler, labeled for the UI."""
    if osplat.IS_WIN:
        r = find_msvc()
        r["label"] = "MSVC (C++ compiler)"
        return r
    for cc in ("clang++", "g++"):
        exe = _which(cc)
        if exe:
            return {"present": True, "path": exe,
                    "label": "C++ compiler (clang++/g++)"}
    url = ("run: xcode-select --install" if osplat.IS_MAC
           else "install g++ or clang++ via your package manager")
    return {"present": False, "path": "", "url": url,
            "label": "C++ compiler (clang++/g++)"}

def find_cuda():
    if osplat.IS_MAC:                    # Metal instead; row hidden in the UI
        return {"present": False, "applicable": False}
    base = os.environ.get("CUDA_PATH") or ""
    nvcc = _which("nvcc")
    nvcc_name = "nvcc.exe" if osplat.IS_WIN else "nvcc"
    if nvcc or (base and os.path.exists(os.path.join(base, "bin", nvcc_name))):
        ver = ""
        try:
            src = nvcc or os.path.join(base, "bin", nvcc_name)
            out = subprocess.run([src, "--version"], capture_output=True, text=True, timeout=8).stdout
            import re
            m = re.search(r"release ([\d.]+)", out)
            ver = m.group(1) if m else ""
        except Exception:
            pass
        return {"present": True, "applicable": True, "version": ver, "path": base or nvcc}
    return {"present": False, "applicable": True,
            "url": "https://developer.nvidia.com/cuda-downloads (needed only for NVIDIA GPU builds)"}


def _tool_present(cmd, args=None, timeout=8):
    exe = _which(cmd)
    if not exe:
        return {"present": False, "path": "", "version": ""}
    version = ""
    try:
        out = subprocess.run([exe] + (args or ["--version"]), capture_output=True,
                             text=True, timeout=timeout)
        version = ((out.stdout or out.stderr).strip().splitlines() or [""])[0]
    except Exception:
        version = exe
    return {"present": True, "path": exe, "version": version}


def find_rocm():
    if not osplat.IS_LINUX:
        return {"present": False, "applicable": False}
    hip = _tool_present("hipconfig", ["--version"])
    rocm = _tool_present("rocminfo", [], timeout=15)
    if rocm["present"] and not rocm["version"]:
        rocm["version"] = "present"
    tools = {
        "hipconfig": hip,
        "rocminfo": rocm,
        "amd-smi": _tool_present("amd-smi", ["version"]),
        "rocm-smi": _tool_present("rocm-smi", ["--showproductname"]),
    }
    present = hip["present"] and rocm["present"]
    return {
        "present": present,
        "applicable": True,
        "version": hip["version"] or rocm["version"],
        "tools": tools,
        "hint": osplat.linux_install_hint(osplat.linux_pkg_manager(), "rocminfo hipcc"),
        "url": "https://rocm.docs.amd.com/",
    }


def find_vulkan():
    if osplat.IS_MAC:
        return {"present": False, "applicable": False}
    info = _tool_present("vulkaninfo", ["--summary"], timeout=15)
    glslc = _tool_present("glslc", ["--version"])
    present = info["present"] and glslc["present"]
    hint = ""
    if osplat.IS_LINUX:
        pm = osplat.linux_pkg_manager()
        pkgs = {"apt-get": "libvulkan-dev glslc spirv-headers",
                "dnf": "vulkan-loader-devel shaderc spirv-headers-devel",
                "pacman": "vulkan-headers shaderc spirv-headers"}
        hint = osplat.linux_install_hint(pm, pkgs.get(pm, "libvulkan-dev glslc spirv-headers"))
    return {
        "present": present,
        "applicable": True,
        "version": info["version"] or glslc["version"],
        "tools": {
            "vulkaninfo": info,
            "glslc": glslc,
        },
        "hint": hint,
        "url": "https://vulkan.lunarg.com/sdk/home",
    }

def installers():
    if osplat.IS_WIN:
        return {"winget": bool(_which("winget")), "choco": bool(_which("choco"))}
    if osplat.IS_MAC:
        return {"brew": bool(_which("brew"))}
    pm = osplat.linux_pkg_manager()
    return {pm: True} if pm else {}

def _can_auto_install(spec):
    if osplat.IS_WIN:
        return bool(_which("winget") or _which("choco"))
    if osplat.IS_MAC:
        return bool(_which("brew") and spec.get("brew"))
    return False                         # linux: hint only, never sudo from the GUI

def status():
    """Full prerequisite report."""
    tools = {}
    for name, spec in TOOLS.items():
        v = _version(spec)
        tools[name] = {"present": v is not None, "version": v or "",
                       "winget": spec["winget"], "choco": spec["choco"],
                       "installable": _can_auto_install(spec),
                       "hint": osplat.linux_install_hint(osplat.linux_pkg_manager(),
                                                         spec["pkg"]) if osplat.IS_LINUX else "",
                       "url": spec["url"]}
    rocm = find_rocm()
    vulkan = find_vulkan()
    return {
        "tools": tools,
        "msvc": find_compiler(),         # key kept for UI compat; now platform-generic
        "cuda": find_cuda(),
        "rocm": rocm,
        "vulkan": vulkan,
        "installers": installers(),
        "platform": osplat.current(),
    }

def _post_install(spec, log):
    """Make a just-installed tool visible without restarting, and say so plainly
    when it still isn't.

    The Setup tab kept showing a successfully installed ninja as MISSING: winget
    updates the registry PATH, but _which() reads the PATH this process
    inherited at launch. Refresh from the registry, re-probe, and only ask for a
    restart if that genuinely wasn't enough.
    """
    osplat.refresh_path()
    if _tool_exe(spec):
        return True, log
    return True, (log + "\n\nInstalled, but not on this process's PATH yet - "
                        "restart LlamaForge for it to be detected.")

def install(name):
    """Install one tool by name. Returns (ok, log)."""
    spec = TOOLS.get(name)
    if not spec:
        return False, f"unknown tool: {name}"
    if osplat.IS_MAC:
        if not _which("brew"):
            return False, f"Homebrew not found. Install it (https://brew.sh) or manually: {spec['url']}"
        r = subprocess.run(["brew", "install", spec["brew"]], capture_output=True, text=True)
        if r.returncode == 0:
            return _post_install(spec, (r.stdout + r.stderr) or "installed via brew")
        return False, (r.stdout + r.stderr) or "brew failed"
    if osplat.IS_LINUX:
        hint = osplat.linux_install_hint(osplat.linux_pkg_manager(), spec["pkg"])
        return False, (f"Run this in a terminal (the dashboard never runs sudo):\n  {hint}"
                       if hint else f"No known package manager. Install manually: {spec['url']}")
    if _which("winget"):
        r = subprocess.run(["winget", "install", "--id", spec["winget"], "-e",
                            "--accept-source-agreements", "--accept-package-agreements"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return _post_install(spec, r.stdout or "installed via winget")
        log = "winget failed:\n" + (r.stdout + r.stderr)
    else:
        log = "winget not available\n"
    if _which("choco"):
        r = subprocess.run(["choco", "install", spec["choco"], "-y"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return _post_install(spec, log + "\ninstalled via choco")
        return False, log + "\nchoco failed:\n" + (r.stdout + r.stderr)
    return False, log + f"\nNo package manager. Install manually: {spec['url']}"

if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2))
