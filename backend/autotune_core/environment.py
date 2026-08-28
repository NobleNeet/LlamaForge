"""Hardware snapshots adapted from LlamaForge's existing detector."""
import hashlib
import json
import platform as platform_module
from datetime import datetime, timezone

import hardware

from .backends import canonical_backend_id
from .models import ExecutionEnvironment, HardwareIdentity, PhysicalGPU


def _canonical_backends(values):
    return tuple(sorted({item for value in values or ()
                         for item in [canonical_backend_id(value)] if item}))


def _stable_gpu_id(row, index, vendor, name, architecture):
    """Prefer hardware-provided identities without requiring detector changes."""
    for key in ("pci_bus_id", "pci_bus", "bus_id", "uuid", "device_uuid", "device_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return "%s:%s" % (key, value.strip().lower())
    return "%s:index-%s:%s:%s" % (vendor.lower(), index, architecture or "unknown", name.lower())


def capture_environment():
    """Capture stable hardware facts while retaining collection time separately."""
    gpu_rows = hardware.detect_gpus()
    gpus = []
    for index, row in enumerate(gpu_rows):
        vendor = str(row.get("vendor") or "")
        name = str(row.get("name") or "GPU")
        architecture = str(row.get("architecture") or "") or None
        backends = _canonical_backends(row.get("backends") or [row.get("backend")])
        memory_mib = row.get("memory_total_mib", row.get("vram_mib"))
        memory_bytes = int(memory_mib * 1024 * 1024) if isinstance(memory_mib, (int, float)) else None
        stable_id = _stable_gpu_id(row, index, vendor, name, architecture)
        gpus.append(PhysicalGPU(stable_id, vendor, name, memory_bytes, architecture,
                                bool(row.get("is_uma")), backends))
    cpu = dict(hardware.detect_cpu() or {})
    available = _canonical_backends(hardware.available_backends(gpu_rows))
    if not available:
        available = ("cpu",)
    return HardwareIdentity(
        platform=platform_module.system().lower(),
        cpu=cpu,
        physical_gpus=tuple(gpus),
        available_backends=available,
        captured_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )


def environment_fingerprint(snapshot):
    """Fingerprint only reproducibility-relevant facts, never capture timestamps."""
    payload = {
        "platform": snapshot.platform,
        "cpu": {key: snapshot.cpu.get(key) for key in ("name", "cores", "threads", "avx512_hint")},
        "physical_gpus": [{
            "stable_id": gpu.stable_id,
            "vendor": gpu.vendor,
            "name": gpu.name,
            "memory_bytes": gpu.memory_bytes,
            "architecture": gpu.architecture,
            "is_uma": gpu.is_uma,
            "available_backends": list(gpu.available_backends),
        } for gpu in snapshot.physical_gpus],
        "available_backends": list(snapshot.available_backends),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256-env-v1:" + hashlib.sha256(encoded).hexdigest()


def execution_fingerprint(environment):
    """Fingerprint a backend/build runtime without volatile observation fields."""
    binary = environment.bench_binary
    payload = {
        "hardware_fingerprint": environment.hardware_fingerprint,
        "backend": environment.backend,
        "runtime": dict(environment.runtime),
        "bench_binary": {
            "backend": binary.backend,
            "build_id": binary.build_id,
            "file_fingerprint": binary.file_fingerprint,
            "version_text": binary.version_text,
            "provenance": binary.provenance,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256-exec-v1:" + hashlib.sha256(encoded).hexdigest()


def execution_environment(hardware_fingerprint, backend, runtime, bench_binary, captured_at=None):
    """Build a case-scoped environment; caller supplies detected runtime facts."""
    canonical = canonical_backend_id(backend)
    if not canonical:
        raise ValueError("unknown backend")
    return ExecutionEnvironment(
        hardware_fingerprint=hardware_fingerprint,
        backend=canonical,
        runtime=dict(runtime or {}),
        bench_binary=bench_binary,
        captured_at=captured_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
