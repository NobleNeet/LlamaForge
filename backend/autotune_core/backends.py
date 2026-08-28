"""Auto Tune's backend vocabulary, isolated from external spelling variants."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BackendDescriptor:
    identifier: str
    display_name: str


_BACKENDS = {
    "cpu": BackendDescriptor("cpu", "CPU"),
    "cuda": BackendDescriptor("cuda", "NVIDIA CUDA"),
    "hip": BackendDescriptor("hip", "AMD ROCm/HIP"),
    "vulkan": BackendDescriptor("vulkan", "Vulkan"),
    "metal": BackendDescriptor("metal", "Apple Metal"),
}
_ALIASES = {
    "rocm": "hip",
    "rocm/hip": "hip",
    "amd rocm": "hip",
    "amd hip": "hip",
    "nvidia": "cuda",
    "apple": "metal",
}


def canonical_backend_id(value: object) -> Optional[str]:
    """Return an Auto Tune backend ID, or None for unknown/unavailable values."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    return key if key in _BACKENDS else None


def backend_descriptor(value: object) -> Optional[BackendDescriptor]:
    identifier = canonical_backend_id(value)
    return _BACKENDS.get(identifier) if identifier else None


def backend_display_name(value: object) -> str:
    descriptor = backend_descriptor(value)
    return descriptor.display_name if descriptor else "Unknown backend"
