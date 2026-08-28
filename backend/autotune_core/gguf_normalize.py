"""Auto Tune-specific GGUF adapter.

The existing gguf module remains the general model-card API.  This reader only
walks the compact header and tensor directory required for tuning inputs.
"""
import hashlib
import os
import struct
from typing import Dict, Optional

import gguf

from .models import FastFingerprint, NormalizedGGUF


_SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                   6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_SCALAR_SIZES = {kind: struct.calcsize(fmt) for kind, fmt in _SCALAR_FORMATS.items()}
_STRING = 8
_ARRAY = 9
_TENSOR_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "IQ4_NL_4", 35: "IQ4_NL_8", 36: "TQ1_0", 37: "TQ2_0",
}


def _read_exact(handle, length):
    value = handle.read(length)
    if len(value) != length:
        raise EOFError("truncated GGUF")
    return value


def _u32(handle):
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _u64(handle):
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _string(handle):
    length = _u64(handle)
    if length > 1024 * 1024:
        raise ValueError("GGUF string is too large")
    return _read_exact(handle, length).decode("utf-8", "replace")


def _skip_string(handle):
    handle.seek(_u64(handle), 1)


def _scalar(handle, kind):
    return struct.unpack(_SCALAR_FORMATS[kind], _read_exact(handle, _SCALAR_SIZES[kind]))[0]


def _skip_value(handle, kind):
    if kind in _SCALAR_SIZES:
        handle.seek(_SCALAR_SIZES[kind], 1)
    elif kind == _STRING:
        _skip_string(handle)
    elif kind == _ARRAY:
        element_kind, count = _u32(handle), _u64(handle)
        if element_kind == _STRING:
            for _ in range(count):
                _skip_string(handle)
        elif element_kind in _SCALAR_SIZES:
            handle.seek(_SCALAR_SIZES[element_kind] * count, 1)
        else:
            raise ValueError("unsupported GGUF array type")
    else:
        raise ValueError("unsupported GGUF value type")


def _read_tuning_header(path: str):
    """Return scalar header KVs and tensor type counts without reading payloads."""
    with open(path, "rb") as handle:
        if _read_exact(handle, 4) != b"GGUF" or _u32(handle) < 2:
            raise ValueError("not a supported GGUF")
        tensor_count, kv_count = _u64(handle), _u64(handle)
        if tensor_count > 10_000_000 or kv_count > 1_000_000:
            raise ValueError("unreasonable GGUF counts")
        scalars = {}
        for _ in range(kv_count):
            key, kind = _string(handle), _u32(handle)
            if kind in _SCALAR_SIZES:
                scalars[key] = _scalar(handle, kind)
            else:
                _skip_value(handle, kind)
        types: Dict[str, int] = {}
        for _ in range(tensor_count):
            _string(handle)
            dimensions = _u32(handle)
            if dimensions > 16:
                raise ValueError("unreasonable tensor dimensions")
            handle.seek(dimensions * 8, 1)
            tensor_type = _u32(handle)
            _u64(handle)  # tensor offset
            label = _TENSOR_TYPES.get(tensor_type, "TYPE_%d" % tensor_type)
            types[label] = types.get(label, 0) + 1
    return scalars, types


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def normalize_gguf(path: str) -> NormalizedGGUF:
    """Normalize available tuning facts; unreadable optional facts become empty."""
    public = gguf.metadata(path) or {}
    try:
        scalars, tensor_types = _read_tuning_header(path)
    except (OSError, EOFError, ValueError, struct.error):
        scalars, tensor_types = {}, {}
    architecture = public.get("architecture") or scalars.get("general.architecture")

    def field(suffix):
        return _as_int(scalars.get("%s.%s" % (architecture, suffix))) if architecture else None

    nextn_layers = field("nextn_predict_layers")
    return NormalizedGGUF(
        path=os.path.abspath(path),
        architecture=architecture if isinstance(architecture, str) else None,
        name=public.get("name") if isinstance(public.get("name"), str) else None,
        quantization=public.get("quantization") if isinstance(public.get("quantization"), str) else None,
        context_length=_as_int(public.get("context_length")),
        embedding_length=_as_int(public.get("embedding_length")),
        block_count=_as_int(public.get("block_count")),
        attention_head_count=_as_int(public.get("head_count")),
        attention_head_count_kv=field("attention.head_count_kv"),
        sliding_window=field("attention.sliding_window"),
        expert_count=_as_int(public.get("expert_count")),
        expert_used_count=_as_int(public.get("expert_used_count")),
        nextn_predict_layers=nextn_layers,
        has_nextn=bool(gguf.has_nextn(path) or (nextn_layers and nextn_layers > 0)),
        tensor_type_summary=dict(sorted(tensor_types.items())),
        metadata=dict(public),
    )


def fast_fingerprint(path: str, window_bytes: int = 65536) -> FastFingerprint:
    """Hash four deterministic windows, avoiding a full read of multi-GB GGUFs.

    The digest covers file size, sampled offsets, and each sample's digest.  It
    intentionally excludes path and mtime, so it is stable across moves/copies.
    """
    size = os.path.getsize(path)
    window_bytes = max(1, int(window_bytes))
    starts = {0, max(0, size - window_bytes), max(0, size // 3 - window_bytes // 2),
              max(0, (size * 2) // 3 - window_bytes // 2)}
    digest = hashlib.sha256()
    digest.update(b"llamaforge-fast-fingerprint-v1\0")
    digest.update(str(size).encode("ascii"))
    with open(path, "rb") as handle:
        for offset in sorted(starts):
            handle.seek(offset)
            sample = handle.read(window_bytes)
            digest.update(b"\0" + str(offset).encode("ascii") + b":" +
                          hashlib.sha256(sample).digest())
    return FastFingerprint(
        algorithm="sha256-sampled-v1",
        value=digest.hexdigest(),
        file_size=size,
        sample_bytes=window_bytes,
    )
