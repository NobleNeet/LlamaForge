"""Read-only hardware adapter for static recommendation; no inference probes."""
import platform

import hardware
import osplat


def snapshot(cfg=None):
    cfg = cfg or {}
    def best_effort(fn, fallback):
        try:
            return fn()
        except Exception:
            return fallback
    gpus = best_effort(hardware.detect_gpus, [])
    # Detection supplies backend identity; telemetry supplies current free VRAM.
    # Match vendor/index, never add the two rows as separate physical devices.
    if any(g.get("vendor") == "NVIDIA" for g in gpus):
        telemetry = best_effort(hardware.detect_gpu_telemetry, [])
        for g in gpus:
            for row in telemetry:
                if (g.get("vendor"), g.get("index")) == (row.get("vendor"), row.get("index")):
                    for key in ("memory_total_mib", "memory_free_mib", "memory_used_mib"):
                        if row.get(key) is not None:
                            g[key] = row[key]
    cpu = best_effort(hardware.detect_cpu, {})
    total = best_effort(osplat.total_ram_bytes, 0)
    available = best_effort(osplat.available_ram_bytes, 0)
    # General detection deliberately returns [] on macOS; use existing Metal
    # budget without changing its semantics for other LlamaForge consumers.
    if osplat.IS_MAC and platform.machine().lower() in ("arm64", "aarch64"):
        gpus = [{"name": "Apple Silicon", "index": 0, "is_uma": True,
                 "backends": ["metal"], "memory_total_mib": total * osplat.METAL_BUDGET / 1024**2}]
    requested = cfg.get("llama_backend", "auto")
    # Match LlamaForge's configured/automatic build backend before choosing a
    # physical GPU; a larger GPU on another backend is not interchangeable.
    selected = ("metal" if requested == "auto" and any("metal" in g.get("backends", []) for g in gpus)
                else hardware._choose_backend(gpus, requested))
    return {"gpus": gpus, "cpu": cpu, "ram_total_bytes": total,
            "ram_available_bytes": available or None,
            "bandwidths": cfg.get("vram_bandwidths") or {},
            "backend": selected}
