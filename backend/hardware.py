"""Detect CPU/GPU hardware and recommend llama.cpp build/runtime settings.

The core distinction is:
  * physical GPU devices (vendor/name/arch/memory/UMA)
  * available llama.cpp acceleration backends on this machine (cuda/hip/vulkan)

One physical AMD iGPU may expose both HIP and Vulkan; that is still one device.
This module keeps the legacy `detect_gpus()` return shape usable by existing
callers while attaching richer fields for backend-neutral handling.
"""
import os
import re
import subprocess

import osplat


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout or ""
    except Exception:
        return ""


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return ""


def _read_int(path):
    txt = _read(path)
    try:
        return int(txt)
    except Exception:
        return None


def _read_temp_c(path):
    raw = _read_int(path)
    if raw is None:
        return None
    if raw > 1000:
        return int(raw / 1000)
    return raw


def detect_ram_gb():
    """Total system RAM in GB (SI, /1e9 to match GPU vendor GB units), 0 if unknown."""
    return round(osplat.total_ram_bytes() / 1e9, 1)


def _detect_cpu_windows():
    out = _run(["powershell", "-NoProfile", "-Command",
                "$c=Get-CimInstance Win32_Processor|Select-Object -First 1;"
                "'{0}|{1}|{2}' -f $c.Name,$c.NumberOfCores,$c.NumberOfLogicalProcessors"])
    info = {"name": "", "cores": None, "threads": None}
    parts = out.strip().split("|")
    if len(parts) == 3:
        info["name"] = parts[0].strip()
        info["cores"] = int(parts[1]) if parts[1].strip().isdigit() else None
        info["threads"] = int(parts[2]) if parts[2].strip().isdigit() else None
    n = info["name"].lower()
    info["avx512_hint"] = any(x in n for x in ["ryzen 7 9", "ryzen 9 9", "ryzen 7 7",
                                               "ryzen 9 7", "xeon", "threadripper"])
    return info


def detect_cpu():
    if osplat.IS_LINUX:
        c = osplat.linux_cpu()
        return {"name": c["name"], "cores": c["cores"], "threads": c["threads"],
                "avx512_hint": c["avx512"]}
    if osplat.IS_MAC:
        c = osplat.mac_cpu()
        return {"name": c["name"], "cores": c["cores"], "threads": c["threads"],
                "avx512_hint": False}
    return _detect_cpu_windows()


def _dedupe_backends(items):
    out = []
    for b in items or []:
        if b and b not in out:
            out.append(b)
    return out


def _gpu_row(name="", vendor="", backend=None, index=0, arch="", total=None,
             used=None, util=None, temp=None, integrated=False, uma=False,
             compute_cap="", local_total=None, local_used=None,
             gtt_total=None, gtt_used=None):
    free = None
    if total is not None and used is not None:
        free = max(0, total - used)
    return {
        "index": index,
        "device_index": index,
        "name": name or "GPU",
        "vendor": vendor or "",
        "backend": backend or "cpu",
        "backends": _dedupe_backends([backend] if backend else []),
        "architecture": arch or "",
        "compute_cap": compute_cap or "",
        "vram_mib": total,
        "fit_vram_mib": None if uma else total,
        "memory_total_mib": total,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "local_memory_total_mib": local_total,
        "local_memory_used_mib": local_used,
        "gtt_total_mib": gtt_total,
        "gtt_used_mib": gtt_used,
        "utilization": util,
        "temperature": temp,
        "used": used if used is not None else 0,
        "total": total if total is not None else 0,
        "util": util if util is not None else 0,
        "temp": temp if temp is not None else 0,
        "is_integrated": bool(integrated),
        "is_uma": bool(uma),
    }


def _parse_nvidia_csv(text):
    gpus = []
    for ln in text.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 3:
            continue
        cc = f[3] if len(f) > 3 and f[3] and f[3] != "[N/A]" else ""
        gpus.append(_gpu_row(name=f[1], vendor="NVIDIA", backend="cuda",
                             index=int(f[0]), total=int(f[2]) if f[2].isdigit() else None,
                             compute_cap=cc))
    return gpus


def _detect_nvidia():
    if osplat.IS_MAC:
        return []
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    return _parse_nvidia_csv(out)


def _parse_nvidia_telemetry(text):
    gpus = []
    for ln in text.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 6:
            gpus.append(_gpu_row(name=f[1], vendor="NVIDIA", backend="cuda",
                                 index=int(f[0]),
                                 total=int(f[3]) if f[3].isdigit() else None,
                                 used=int(f[2]) if f[2].isdigit() else None,
                                 util=int(f[4]) if f[4].isdigit() else None,
                                 temp=int(f[5]) if f[5].isdigit() else None))
    return gpus


def _nvidia_telemetry():
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"], timeout=8)
    return _parse_nvidia_telemetry(out)


def _parse_lspci(text):
    out = []
    for ln in text.splitlines():
        lo = ln.lower()
        if "vga compatible controller" not in lo and "3d controller" not in lo:
            continue
        vendor = ""
        if "advanced micro devices" in lo or " amd/" in lo or "amd" in lo:
            vendor = "AMD"
        elif "nvidia" in lo:
            vendor = "NVIDIA"
        elif "intel" in lo:
            vendor = "Intel"
        name = ln.split(":", 2)[-1].strip() if ":" in ln else ln.strip()
        if vendor:
            out.append({"vendor": vendor, "name": name})
    return out


def _sysfs_gpu_cards(base="/sys/class/drm"):
    rows = []
    if not os.path.isdir(base):
        return rows
    idx = 0
    for ent in sorted(os.listdir(base)):
        if not re.fullmatch(r"card\d+", ent):
            continue
        devdir = os.path.join(base, ent, "device")
        vendor_id = _read(os.path.join(devdir, "vendor")).lower()
        if vendor_id not in ("0x1002", "0x10de", "0x8086"):
            continue
        vendor = {"0x1002": "AMD", "0x10de": "NVIDIA", "0x8086": "Intel"}[vendor_id]
        name = _read(os.path.join(devdir, "product_name")) or _read(os.path.join(devdir, "label")) or ent
        total = _read_int(os.path.join(devdir, "mem_info_vram_total"))
        used = _read_int(os.path.join(devdir, "mem_info_vram_used"))
        gtt = _read_int(os.path.join(devdir, "mem_info_gtt_total"))
        gtt_used = _read_int(os.path.join(devdir, "mem_info_gtt_used"))
        util = _read_int(os.path.join(devdir, "gpu_busy_percent"))
        temp = None
        hwmons = os.path.join(devdir, "hwmon")
        if os.path.isdir(hwmons):
            for hm in sorted(os.listdir(hwmons)):
                cand = _read_temp_c(os.path.join(hwmons, hm, "temp1_input"))
                if cand is not None:
                    temp = cand
                    break
        local_total = int(total / (1024 * 1024)) if total is not None else None
        local_used = int(used / (1024 * 1024)) if used is not None else None
        gtt_total = int(gtt / (1024 * 1024)) if gtt is not None else None
        gtt_used = int(gtt_used / (1024 * 1024)) if gtt_used is not None else None
        # AMD APUs can expose a small local VRAM carve-out (e.g. 1 GiB) plus a
        # much larger GTT/shared-memory aperture. Treat those as UMA for the UI.
        is_uma = bool(gtt_total and (local_total is None or gtt_total > max(local_total * 4, 4096)))
        if is_uma and gtt_total is not None:
            total = gtt_total
            used = gtt_used if gtt_used is not None else local_used
        else:
            total = local_total
            used = local_used
        rows.append(_gpu_row(name=name, vendor=vendor, index=idx, total=total, used=used,
                             util=util, temp=temp,
                             integrated=is_uma or vendor == "Intel", uma=is_uma,
                             local_total=local_total, local_used=local_used,
                             gtt_total=gtt_total, gtt_used=gtt_used))
        idx += 1
    return rows


def _rocminfo_arches(text):
    out = []
    for ln in text.splitlines():
        m = re.search(r"\bName:\s*(gfx[0-9a-z]+)\b", ln, re.I)
        if m:
            arch = m.group(1)
            if arch not in out:
                out.append(arch)
    return out


def _vulkan_devices(text):
    name = None
    driver = ""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        m = re.search(r"deviceName\s*=\s*(.+)", s, re.I)
        if not m:
            m = re.search(r"GPU\d+\s*:\s*(.+)", s)
        if m:
            if name:
                out.append({"name": name, "driver": driver})
            name, driver = m.group(1).strip(), ""
            continue
        m = re.search(r"driverName\s*=\s*(.+)", s, re.I)
        if m:
            driver = m.group(1).strip()
    if name:
        out.append({"name": name, "driver": driver})
    return out


def _is_software_vulkan_device(name, driver=""):
    blob = f"{name} {driver}".lower()
    return any(tok in blob for tok in ("llvmpipe", "lavapipe", "swiftshader", "software rasterizer"))


def _rename_amd_base_with_vulkan_names(base, vk_rows):
    amd_base = [g for g in base if g.get("vendor") == "AMD"]
    amd_vk = [g for g in vk_rows if g.get("vendor") == "AMD" and not _is_software_vulkan_device(
        g.get("name", ""), g.get("driver", "")
    )]
    if not amd_base or len(amd_vk) != len(amd_base):
        return base
    out = [dict(g) for g in base]
    renamed = 0
    for row in out:
        if row.get("vendor") != "AMD":
            continue
        vk = amd_vk[renamed]
        if re.fullmatch(r"card\d+", row.get("name", ""), re.I) or not row.get("name"):
            row["name"] = vk.get("name", row.get("name", ""))
        renamed += 1
    return out


def _merge_gpu_lists(primary, extra):
    out = [dict(g) for g in primary]
    for add in extra:
        best = None
        for cur in out:
            same_vendor = (cur.get("vendor") or "").lower() == (add.get("vendor") or "").lower()
            if same_vendor and (cur.get("name", "").lower() in add.get("name", "").lower() or
                                add.get("name", "").lower() in cur.get("name", "").lower()):
                best = cur
                break
        if best is None:
            out.append(dict(add))
            continue
        best["backends"] = _dedupe_backends((best.get("backends") or []) + (add.get("backends") or []))
        best["backend"] = best["backends"][0] if best.get("backends") else best.get("backend", "cpu")
        for key in ("architecture", "memory_total_mib", "memory_used_mib", "memory_free_mib",
                    "vram_mib", "fit_vram_mib", "utilization", "temperature", "used", "total",
                    "util", "temp"):
            if (best.get(key) is None or best.get(key) == "") and add.get(key) not in (None, ""):
                best[key] = add.get(key)
        best["is_integrated"] = best.get("is_integrated") or add.get("is_integrated")
        best["is_uma"] = best.get("is_uma") or add.get("is_uma")
    for i, g in enumerate(out):
        g["index"] = g["device_index"] = i
    return out


def _detect_amd_linux():
    base = [g for g in _sysfs_gpu_cards() if g.get("vendor") == "AMD"]
    if not base:
        for idx, row in enumerate(_parse_lspci(_run(["lspci"]))):
            if row["vendor"] == "AMD":
                base.append(_gpu_row(name=row["name"], vendor="AMD", index=idx))
    rocminfo = _run(["rocminfo"], timeout=20)
    arches = _rocminfo_arches(rocminfo)
    hip = []
    if base and (rocminfo or _run(["hipconfig", "-l"], timeout=10).strip()):
        for i, g in enumerate(base):
            row = dict(g)
            row["backends"] = _dedupe_backends((row.get("backends") or []) + ["hip"])
            row["backend"] = "hip"
            if not row.get("architecture") and arches:
                row["architecture"] = arches[min(i, len(arches) - 1)]
            hip.append(row)
    vk = []
    vk_info = _run(["vulkaninfo", "--summary"], timeout=20) or _run(["vulkaninfo"], timeout=20)
    for dev in _vulkan_devices(vk_info):
        name = dev.get("name", "")
        vendor = "AMD" if "radeon" in name.lower() or "amd" in name.lower() else ""
        if _is_software_vulkan_device(name, dev.get("driver", "")):
            continue
        vk.append(_gpu_row(name=name, vendor=vendor, backend="vulkan"))
    base = _rename_amd_base_with_vulkan_names(base, vk)
    merged = _merge_gpu_lists(base, hip)
    merged = _merge_gpu_lists(merged, vk)
    for g in merged:
        if g.get("vendor") == "AMD":
            g["is_integrated"] = bool(g.get("is_integrated") or "graphics" in g.get("name", "").lower())
            if g.get("is_uma") and g.get("fit_vram_mib") is None:
                g["fit_vram_mib"] = None
    return merged


def detect_gpus():
    if osplat.IS_MAC:
        return []
    gpus = _detect_nvidia()
    if osplat.IS_LINUX:
        gpus = _merge_gpu_lists(gpus, _detect_amd_linux())
    for g in gpus:
        if not g.get("backends"):
            if g.get("vendor") == "NVIDIA":
                g["backends"] = ["cuda"]
                g["backend"] = "cuda"
            else:
                g["backends"] = []
                g["backend"] = "cpu"
    return gpus


def detect_gpus_verbose():
    return detect_gpus()


def detect_gpu_telemetry():
    if osplat.IS_MAC:
        return osplat.mac_gpu_telemetry()
    rows = _nvidia_telemetry()
    if osplat.IS_LINUX:
        rows = _merge_gpu_lists(rows, _detect_amd_linux())
    return rows or [{"error": "GPU telemetry unavailable"}]


def available_backends(gpus=None):
    gpus = detect_gpus() if gpus is None else gpus
    out = []
    for g in gpus:
        out.extend(g.get("backends") or [])
    return _dedupe_backends(out)


def total_fit_vram_mib(gpus=None):
    gpus = detect_gpus() if gpus is None else gpus
    total = 0
    for g in gpus:
        fit = g.get("fit_vram_mib")
        if fit is None and not g.get("is_uma"):
            fit = g.get("vram_mib")
        if fit:
            total += fit
    return total


def has_uma_gpu(gpus=None):
    gpus = detect_gpus() if gpus is None else gpus
    return any(g.get("is_uma") for g in gpus)


def _choose_backend(gpus, requested="auto"):
    avail = available_backends(gpus)
    if requested and requested != "auto":
        return requested
    for cand in ("cuda", "hip", "vulkan"):
        if cand in avail:
            return cand
    return "cpu"


def recommend(gpus=None, cpu=None, backend="auto"):
    """Return build/runtime recommendation for this machine."""
    gpus = detect_gpus() if gpus is None else gpus
    cpu = detect_cpu() if cpu is None else cpu
    flags, notes = {}, []
    avail = available_backends(gpus)

    if osplat.IS_MAC:
        flags["GGML_METAL"] = "ON"
        flags["GGML_NATIVE"] = "ON"
        notes.append("Apple Silicon detected - Metal build (uses unified memory as VRAM).")
        return {"cmake_flags": flags, "notes": notes,
                "runtime": {"n-gpu-layers": "99", "flash-attn": "on"},
                "gpus": gpus, "cpu": cpu, "selected_backend": "metal",
                "available_backends": ["metal"]}

    selected = _choose_backend(gpus, backend)
    if selected == "cuda":
        archs = sorted({g["compute_cap"].replace(".", "") for g in gpus if g.get("compute_cap")})
        flags["GGML_CUDA"] = "ON"
        if archs:
            flags["CMAKE_CUDA_ARCHITECTURES"] = ";".join(archs)
            notes.append(f"CUDA build for arch(s) {', '.join(archs)} ({len(gpus)} GPU(s)).")
        flags["GGML_CUDA_FA_ALL_QUANTS"] = "ON"
        notes.append("Enabled flash-attention for all quant KV combos.")
    elif selected == "hip":
        flags["GGML_HIP"] = "ON"
        archs = sorted({g.get("architecture") for g in gpus if g.get("architecture")})
        if archs:
            flags["GPU_TARGETS"] = ";".join(archs)
            notes.append(f"ROCm/HIP build for arch(s) {', '.join(archs)}.")
        else:
            notes.append("ROCm/HIP build selected; GPU target arch not detected, so llama.cpp will auto-target the local system.")
    elif selected == "vulkan":
        flags["GGML_VULKAN"] = "ON"
        notes.append("Vulkan build selected.")
    else:
        notes.append("No supported GPU backend selected - configuring a CPU-only build.")

    flags["GGML_NATIVE"] = "ON"
    if cpu.get("avx512_hint"):
        for f in ("GGML_AVX512", "GGML_AVX512_VNNI", "GGML_AVX512_VBMI", "GGML_AVX512_BF16"):
            flags[f] = "ON"
        notes.append("Enabled AVX-512 (+VNNI/VBMI/BF16) for this CPU.")

    if selected == "cpu":
        runtime = {"n-gpu-layers": "0", "flash-attn": "off"}
    else:
        runtime = {"n-gpu-layers": "99", "flash-attn": "on"}
        if has_uma_gpu(gpus):
            notes.append("UMA GPU detected - build is GPU-accelerated, but VRAM fit estimates stay conservative.")
    return {"cmake_flags": flags, "notes": notes, "runtime": runtime,
            "gpus": gpus, "cpu": cpu, "selected_backend": selected,
            "available_backends": avail}


if __name__ == "__main__":
    import json
    print(json.dumps(recommend(), indent=2))
