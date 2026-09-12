"""Minimal GGUF metadata reader (pure stdlib).

We only need one number per model: the trained context length, stored in the
GGUF header as `<arch>.context_length` (e.g. `qwen2.context_length`). That lives
in the metadata block at the very start of the file, so we parse just the header
and seek past every value we don't care about - never reading the multi-GB
tensor payload, and never loading giant arrays (like the tokenizer vocab) into
memory. Any malformed/unreadable file degrades to None; this module never raises.
"""
import struct

# Context-size policy tiers (see default_ctx).
CTX_FULL     = 150000   # baseline default for models that support it
CTX_FALLBACK = 100000   # cap for models trained below CTX_FULL

# GGUF metadata value types (ggml gguf spec).
_SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_SCALAR_SZ  = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_INT_TYPES  = (0, 1, 2, 3, 4, 5, 10, 11)   # types context_length may be stored as
_STRING = 8
_ARRAY  = 9


def _rd(f, n):
    b = f.read(n)
    if len(b) < n:
        raise EOFError
    return b

def _u32(f): return struct.unpack("<I", _rd(f, 4))[0]
def _u64(f): return struct.unpack("<Q", _rd(f, 8))[0]

def _read_str(f):
    n = _u64(f)
    if n > (1 << 20):            # 1 MiB key/string ceiling = corrupt/hostile file
        raise ValueError("string too long")
    return _rd(f, n).decode("utf-8", "replace")

def _skip_str(f):
    f.seek(_u64(f), 1)

def _read_scalar(f, t):
    return struct.unpack(_SCALAR_FMT[t], _rd(f, _SCALAR_SZ[t]))[0]

def _skip_value(f, t):
    """Advance the cursor past a value of type t without materializing it."""
    if t in _SCALAR_SZ:
        f.seek(_SCALAR_SZ[t], 1)
    elif t == _STRING:
        _skip_str(f)
    elif t == _ARRAY:
        et, cnt = _u32(f), _u64(f)
        if et == _STRING:
            for _ in range(cnt):
                _skip_str(f)
        elif et in _SCALAR_SZ:
            f.seek(_SCALAR_SZ[et] * cnt, 1)
        else:                    # arrays of arrays aren't used in practice
            raise ValueError("unsupported array element type %d" % et)
    else:
        raise ValueError("unsupported value type %d" % t)


def context_length(path):
    """Trained context length from a GGUF file, or None if unreadable/unknown."""
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF":
                return None
            if _u32(f) < 2:              # v1 used 32-bit counts; too old to bother
                return None
            _u64(f)                       # tensor count (unused)
            n_kv = _u64(f)
            if n_kv > 1_000_000:          # sanity guard against a corrupt count
                return None
            arch, ctx = None, {}
            for _ in range(n_kv):
                key = _read_str(f)
                vt  = _u32(f)
                if key == "general.architecture" and vt == _STRING:
                    arch = _read_str(f)
                elif key.endswith(".context_length") and vt in _INT_TYPES:
                    ctx[key] = int(_read_scalar(f, vt))
                else:
                    _skip_value(f, vt)
            if arch and f"{arch}.context_length" in ctx:
                return ctx[f"{arch}.context_length"]
            return next(iter(ctx.values())) if ctx else None
    except Exception:
        return None


def has_nextn(path):
    """True if this GGUF declares MTP/NextN layers (`*.nextn_predict_layers` > 0).

    llama.cpp gates draft-mtp speculative decoding on NextN layers being present
    in the model, not on the arch name, so this file-intrinsic signal is what
    decides whether an mtp-* sidecar can actually be enabled. The curated
    metadata() drops this key, hence a dedicated raw-KV walk. Never raises.
    """
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF":
                return False
            if _u32(f) < 2:
                return False
            _u64(f)                       # tensor count (unused)
            n_kv = _u64(f)
            if n_kv > 1_000_000:
                return False
            for _ in range(n_kv):
                key = _read_str(f)
                vt  = _u32(f)
                if key.endswith(".nextn_predict_layers") and vt in _INT_TYPES:
                    return int(_read_scalar(f, vt)) > 0
                _skip_value(f, vt)
            return False
    except Exception:
        return False


# general.file_type is an LLAMA_FTYPE enum; map the common quant tiers to labels.
_FTYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

# Strings worth decoding by exact key; every other string (chat_template, the
# tokenizer vocab, etc.) is skipped so we never pull KBs/MBs off disk.
_META_STR_KEYS = {"general.architecture", "general.name", "general.size_label",
                  "general.basename", "general.finetune"}
_META_STR_SUFFIX = (".rope.scaling.type",)   # e.g. llama.rope.scaling.type = "yarn"


def _read_header_kv(f, n_kv):
    """Collect scalar KVs (all cheap) + a small allowlist of strings into a dict.
    Arrays and unlisted strings are seeked past, never materialized."""
    kv = {}
    for _ in range(n_kv):
        key = _read_str(f)
        vt = _u32(f)
        if vt in _SCALAR_SZ:                 # int/float/bool: 1-8 bytes, always keep
            kv[key] = _read_scalar(f, vt)
        elif vt == _STRING:
            if key in _META_STR_KEYS or key.endswith(_META_STR_SUFFIX):
                kv[key] = _read_str(f)
            else:
                _skip_str(f)
        else:
            _skip_value(f, vt)
    return kv


def metadata(path):
    """Human-facing GGUF header facts for the model card, or None if unreadable.
    Only keys actually present are returned (missing fields are omitted)."""
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF":
                return None
            if _u32(f) < 2:
                return None
            _u64(f)                            # tensor count (unused)
            n_kv = _u64(f)
            if n_kv > 1_000_000:
                return None
            kv = _read_header_kv(f, n_kv)
    except Exception:
        return None
    arch = kv.get("general.architecture")
    def a(suffix):
        return kv.get(f"{arch}.{suffix}") if arch else None
    ft = kv.get("general.file_type")
    out = {
        "architecture":     arch,
        "name":             kv.get("general.name"),
        "size_label":       kv.get("general.size_label"),
        "quantization":     _FTYPE.get(int(ft)) if isinstance(ft, (int, float)) else None,
        "context_length":   a("context_length"),
        "embedding_length": a("embedding_length"),
        "block_count":      a("block_count"),
        "head_count":       a("attention.head_count"),
        "vocab_size":       a("vocab_size"),
        "expert_count":     a("expert_count"),
        "expert_used_count": a("expert_used_count"),
        "rope_freq_base":   a("rope.freq_base"),
        "rope_scaling":     a("rope.scaling.type"),
    }
    return {k: v for k, v in out.items() if v is not None}


def default_ctx(path):
    """Per-model ctx-size override for a GGUF at `path`.

    Returns:
        int   - write this exact value as the model's ctx-size
        0     - no override needed; model supports the CTX_FULL global default
        None  - trained length unknown (unreadable/missing); leave the model as-is
    """
    n = context_length(path)
    if not n or n <= 0:
        return None
    if n >= CTX_FULL:
        return 0
    return min(CTX_FALLBACK, n)   # cap at trained length; never over-extend


def inspect_model(path):
    """Static geometry from headers and tensor directories, never tensor payloads.

    Offset spans include alignment padding, making byte counts conservative even
    for new quantization types. All shards must be present for a split GGUF.
    """
    import math
    import os
    import re

    public = metadata(path) or {}
    try:
        split = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", str(path), re.I)
        paths = [path]
        if split:
            count = int(split[3])
            if not 1 <= count <= 1024:
                raise ValueError("unreasonable shard count")
            paths = [f"{split[1]}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
        tensors, file_size, scalars = [], 0, {}
        for shard in paths:
            size = os.path.getsize(shard)
            file_size += size
            with open(shard, "rb") as f:
                if _rd(f, 4) != b"GGUF" or _u32(f) not in (2, 3):
                    raise ValueError("unsupported GGUF")
                nt, nk = _u64(f), _u64(f)
                if nt > 1_000_000 or nk > 1_000_000:
                    raise ValueError("unreasonable directory size")
                kv = _read_header_kv(f, nk)
                scalars.update(kv)
                if kv.get("split.count", 1) > 1 and not split:
                    raise ValueError("unrecognized split GGUF naming")
                entries = []
                for _ in range(nt):
                    name, nd = _read_str(f), _u32(f)
                    if not 1 <= nd <= 4:
                        raise ValueError("invalid tensor rank")
                    dims = [_u64(f) for _ in range(nd)]
                    if not all(0 < d <= 2**40 for d in dims):
                        raise ValueError("invalid tensor dimensions")
                    kind, offset = _u32(f), _u64(f)
                    entries.append((offset, name, math.prod(dims), kind))
                alignment = int(kv.get("general.alignment", 32))
                if alignment <= 0 or alignment > 1024 * 1024:
                    raise ValueError("invalid alignment")
                start = ((f.tell() + alignment - 1) // alignment) * alignment
                entries.sort()
                for i, (offset, name, params, kind) in enumerate(entries):
                    end = entries[i + 1][0] if i + 1 < len(entries) else size - start
                    if offset < 0 or end <= offset or start + end > size:
                        raise ValueError("invalid tensor offset")
                    tensors.append({"name": name, "bytes": end - offset,
                                    "parameters": params, "type": kind})
        arch = public.get("architecture") or scalars.get("general.architecture")
        for key, suffix in (("head_count_kv", "attention.head_count_kv"),
                            ("key_length", "attention.key_length"),
                            ("value_length", "attention.value_length")):
            value = scalars.get(f"{arch}.{suffix}")
            if isinstance(value, (int, float)) and value > 0:
                public[key] = value
        layers = int(public.get("block_count") or 0)
        layer_bytes = [0] * layers if 0 < layers <= 4096 else []
        non_layer = 0
        for tensor in tensors:
            match = re.match(r"^blk\.(\d+)\.", tensor["name"])
            if match and int(match[1]) < len(layer_bytes):
                layer_bytes[int(match[1])] += tensor["bytes"]
            else:
                non_layer += tensor["bytes"]
        public.update(file_size_bytes=file_size, tensors=tensors,
                      weight_bytes=sum(t["bytes"] for t in tensors) or file_size,
                      parameter_count=sum(t["parameters"] for t in tensors),
                      layer_weight_bytes=layer_bytes, non_layer_weight_bytes=non_layer)
    except (OSError, ValueError, EOFError, TypeError, OverflowError, struct.error):
        public["inspection_incomplete"] = True
        # Never size a multi-shard model from only its first shard.
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", str(path), re.I):
            public["incomplete_shards"] = True
        else:
            try:
                public["file_size_bytes"] = os.path.getsize(path)
            except (OSError, TypeError):
                pass
    return public
