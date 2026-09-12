"""Static recommendations. No model execution, benchmark, or persistent state.

Geometry -> memory estimate -> placement -> parameter heuristics -> presets.
All budgets are bytes; hardware *_gb inputs use SI, *_mib inputs use binary MiB.
"""
from dataclasses import dataclass, asdict
import logging
import math

import vram_predict

LOG = logging.getLogger(__name__)
MIB = 1024 ** 2
GIB = 1024 ** 3
PRESETS = {"safe": .18, "balanced": .09, "aggressive": .04}


def number(value, default=0):
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 and not isinstance(value, bool) else default
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass(frozen=True)
class Geometry:
    architecture: str
    file_bytes: float
    weights: float
    layers: int
    layer_bytes: tuple
    non_layer_bytes: float
    embedding: int
    kv_heads: int
    key_dim: int
    value_dim: int
    context_max: int
    total_params: float
    active_params: float
    bpw: float
    experts: int
    active_experts: int
    confidence: str


def model_geometry(meta, size_bytes=None):
    meta = meta or {}
    size = number(size_bytes or meta.get("file_size_bytes"))
    weights = number(meta.get("weight_bytes")) or size
    layers = min(4096, max(1, int(number(meta.get("block_count"), 32))))
    embedding = max(1, int(number(meta.get("embedding_length"), 4096)))
    heads = max(1, int(number(meta.get("head_count"), 32)))
    kv_heads = max(1, int(number(meta.get("head_count_kv"), heads)))
    dims = max(1, math.ceil(embedding / heads))
    key_dim = max(1, int(number(meta.get("key_length"), dims)))
    value_dim = max(1, int(number(meta.get("value_length"), dims)))
    per_layer = meta.get("layer_weight_bytes")
    exact = (isinstance(per_layer, (list, tuple)) and len(per_layer) == layers
             and all(number(v) > 0 for v in per_layer))
    if exact:
        per_layer = tuple(float(v) for v in per_layer)
        weights = max(weights, sum(per_layer) + number(meta.get("non_layer_weight_bytes")))
        non_layer = weights - sum(per_layer)
    else:
        # Embeddings/output tensors still occupy host/device memory. Reserve 5%
        # when their individual sizes are unknown instead of treating them as free.
        non_layer = weights * .05
        per_layer = (weights * .95 / layers,) * layers
    physics_model, _ = vram_predict._model_from_gguf({
        **meta, "block_count": layers,
        "expert_count": number(meta.get("expert_count")),
        "expert_used_count": number(meta.get("expert_used_count")),
    }, weights)
    total = number(meta.get("parameter_count")) or (physics_model.total_params if physics_model else 0)
    active_ratio = physics_model.active_params / physics_model.total_params if physics_model else 1
    confidence = "high" if exact and all(number(meta.get(k)) > 0 for k in
        ("block_count", "embedding_length", "head_count", "head_count_kv")) else "medium"
    if not size or not all(number(meta.get(k)) > 0 for k in ("block_count", "embedding_length", "head_count")):
        confidence = "low"
    return Geometry(str(meta.get("architecture") or "unknown"), size, weights, layers,
                    per_layer, non_layer, embedding, kv_heads, key_dim, value_dim,
                    max(1, int(number(meta.get("context_length"), 8192))),
                    total, total * active_ratio, weights * 8 / total if total else 0,
                    int(number(meta.get("expert_count"))), int(number(meta.get("expert_used_count"))), confidence)


@dataclass(frozen=True)
class MemoryHardware:
    ram_available: float
    gpu_available: float
    uma: bool
    backend: str
    physical_cores: int
    device: int
    gpu_bw: float
    ram_bw: float
    disk_bw: float
    confidence: str


def hardware_budget(hw):
    """Choose one usable physical GPU, never add duplicate backend capacities.

    Explicit split-mode=none/main-gpu settings bind the plan to that GPU. Shared
    heaps are bounded by available host memory AND the backend-visible heap.
    """
    hw = hw or {}
    cpu = hw.get("cpu") or {}
    cores = min(1024, max(1, int(number(cpu.get("cores"), max(1, number(cpu.get("threads"), 2) // 2)))))
    total_ram = number(hw.get("ram_total_bytes")) or number(hw.get("ram_gb")) * 1e9
    available = hw.get("ram_available_bytes")
    ram = number(available) if available is not None else total_ram * .7
    conf = "high" if available is not None and total_ram else "low"
    if not total_ram and available is None:
        ram = 4 * GIB  # explicitly low confidence, never assumed large GPU capacity
    if total_ram:
        ram = min(ram, total_ram)
    choices = []
    requested = hw.get("backend", "auto")
    for gpu in hw.get("gpus") or []:
        backends = gpu.get("backends")
        if backends is None:
            backends = [gpu.get("backend")] if gpu.get("backend") else []
        backends = [b for b in backends if b in ("cuda", "hip", "rocm", "vulkan", "metal")]
        if requested != "auto":
            backends = [b for b in backends if b == requested or {b, requested} == {"hip", "rocm"}]
        if not backends:
            continue
        uma = bool(gpu.get("is_uma"))
        capacity = number(gpu.get("memory_total_mib") or gpu.get("fit_vram_mib") or gpu.get("vram_mib")) * MIB
        free = gpu.get("memory_free_mib")
        if free is not None:
            capacity = min(capacity, number(free) * MIB) if capacity else number(free) * MIB
        else:
            capacity = max(0, capacity - number(gpu.get("memory_used_mib", gpu.get("used"))) * MIB)
        if uma:
            # A small local carveout is not the full addressable UMA heap.
            shared = gpu.get("gtt_total_mib")
            if shared is not None:
                capacity = max(capacity, max(0, number(shared) - number(gpu.get("gtt_used_mib"))) * MIB)
            capacity = min(ram, capacity) if capacity else ram
        choices.append((capacity, gpu, backends[0], uma, free is not None or uma))
    if choices:
        capacity, gpu, backend, uma, known = max(choices, key=lambda item: item[0])
        if not known:
            conf = "medium" if conf == "high" else conf
    else:
        capacity, gpu, backend, uma = 0, {}, "cpu", False
    ov = hw.get("bandwidths") or {}
    return MemoryHardware(ram, capacity, uma, backend, cores,
                          int(number(gpu.get("device_index", gpu.get("index")))),
                          number(ov.get("vram_bw")) or vram_predict._preset_vram_bw(gpu.get("name", "")),
                          number(ov.get("ram_bw")) or 50,
                          number(ov.get("disk_bw")) or 3, conf)


def estimate_memory(g, context, kv_type, batch, ubatch, backend):
    """Conservative KV (including block overhead) and temporary buffer estimates.

    Full KV is budgeted on the GPU even for partial offload; the host estimate
    accounts separately for its CPU layer fraction. No speculative tok/s fitting.
    """
    element_bytes = {"f16": 2, "q8_0": 34 / 32, "q4_0": 18 / 32}[kv_type]
    kv = g.layers * context * g.kv_heads * (g.key_dim + g.value_dim) * element_bytes * 1.12
    # Approximate activation/graph/logit work, independent of total layer count.
    compute = 128 * MIB + g.embedding * ubatch * 64 + batch * g.embedding * 8
    overhead = (384 if backend != "cpu" else 128) * MIB
    return {"kv_bytes": kv, "compute_bytes": compute, "backend_bytes": overhead}


def plan_placement(g, h, memory, headroom):
    gpu_limit = max(0, h.gpu_available * (1 - headroom))
    ram_limit = max(0, h.ram_available * (1 - headroom))
    buffers = sum(memory.values())
    budget = max(0, gpu_limit - buffers)
    ngl, gpu_weights = 0, 0
    if h.backend != "cpu" and g.weights > 0 and g.confidence != "low":
        if g.weights <= budget:
            ngl, gpu_weights = g.layers + 1, g.weights
        else:
            # llama.cpp offloads the last transformer blocks first. Charge all
            # non-layer tensors on GPU too (conservative output/embedding bound).
            resident = g.non_layer_bytes
            for weight in reversed(g.layer_bytes):
                if resident + weight > budget:
                    break
                resident += weight
                ngl += 1
            gpu_weights = resident if ngl else 0
    # Embeddings remain on the CPU. Some llama.cpp versions count output as
    # the first offloaded layer; allow one extra CPU block for portable plans.
    cpu_weights = max(g.non_layer_bytes, g.weights - gpu_weights)
    if 0 < ngl <= g.layers:
        cpu_weights = min(g.weights, cpu_weights + g.non_layer_bytes + max(g.layer_bytes))
    cpu_fraction = cpu_weights / g.weights if g.weights else 1
    gpu_used = gpu_weights + buffers if ngl else 0
    # UMA has one physical budget; splitting weights does not create more RAM.
    if h.uma:
        host_used = g.weights + buffers
    else:
        host_used = cpu_weights + memory["kv_bytes"] * cpu_fraction + memory["compute_bytes"] + 128 * MIB
    feasible = (g.weights > 0 and host_used <= ram_limit and gpu_used <= gpu_limit)
    return {"n_gpu_layers": ngl, "gpu_weight_bytes": gpu_weights,
            "gpu_weight_budget_bytes": budget, "cpu_resident_fraction": cpu_fraction,
            "gpu_used_bytes": gpu_used, "ram_used_bytes": host_used,
            "gpu_available_bytes": h.gpu_available, "ram_available_bytes": h.ram_available,
            "gpu_reserve_bytes": h.gpu_available * headroom,
            "ram_reserve_bytes": h.ram_available * headroom, "headroom": headroom,
            "feasible": feasible, **memory}


def parameter_heuristics(g, h, preset):
    """Small arithmetic policy grid, never runtime trials or benchmark search.

    Keep f16 if it fits fully; compression must improve residency or make the
    host fit. Shrink buffers before sacrificing full offload. Only shrink context
    when the original window cannot yield a feasible placement.
    """
    margin = PRESETS[preset]
    context = min(g.context_max, 8192 if g.confidence == "low" else 32768)
    batches = {"safe": [(1024, 256), (512, 128)],
               "balanced": [(2048, 512), (1024, 256), (512, 128)],
               "aggressive": [(2048, 512), (1024, 256), (512, 128)]}[preset]
    if h.backend == "cpu":
        batches = [(512, 128)]
    kv_types = ["f16", "q8_0"] + (["q4_0"] if preset == "aggressive" else [])
    # Quantized V requires flash attention. Unknown backend/model support keeps
    # the portable f16 default; CPU is also kept on f16.
    if h.backend == "cpu" or g.confidence == "low":
        kv_types = ["f16"]
    while True:
        plans = []
        for kv_type in kv_types:
            for batch, ubatch in batches:
                mem = estimate_memory(g, context, kv_type, batch, ubatch, h.backend)
                plan = plan_placement(g, h, mem, margin)
                plans.append((plan, kv_type, batch, ubatch))
        feasible = [p for p in plans if p[0]["feasible"]]
        if feasible or context <= min(g.context_max, 512):
            pool = feasible or plans
            # Residency first; equal residency retains quality, then large batch.
            chosen = max(pool, key=lambda p: (p[0]["n_gpu_layers"], -kv_types.index(p[1]), p[2]))
            break
        context = max(1, min(g.context_max, context // 2))
    plan, kv_type, batch, ubatch = chosen
    fraction = plan["cpu_resident_fraction"]
    base = min(h.physical_cores, max(2, min(8, math.ceil(h.physical_cores / 4))))
    threads = min(h.physical_cores, max(1, math.ceil(base + fraction * (h.physical_cores - base))))
    knobs = {"n-gpu-layers": str(plan["n_gpu_layers"]), "ctx-size": str(context),
             "batch-size": str(batch), "ubatch-size": str(ubatch),
             "threads": str(threads), "threads-batch": str(threads),
             "flash-attn": "auto", "cache-type-k": kv_type, "cache-type-v": kv_type}
    if h.backend != "cpu":
        knobs.update({"split-mode": "none", "main-gpu": str(h.device)})
    reasons = {
        "n-gpu-layers": f"{plan['gpu_weight_budget_bytes'] / GIB:.2f} GiB weight budget after KV, buffers and {margin:.0%} headroom; {g.layers + 1} includes output offload.",
        "ctx-size": "Practical initial window, capped by trained context and memory fit.",
        "batch-size": "Largest representative batch preserving the selected memory placement.",
        "ubatch-size": "Microbatch included in the compute buffer estimate.",
        "threads": f"{fraction:.1%} CPU resident weights; interpolated using {h.physical_cores} physical cores.",
        "threads-batch": "Physical-core budget adjusted for CPU weight residency.",
        "flash-attn": "Let the backend select supported flash attention automatically.",
        "cache-type-k": "Preserve f16 unless compression improves memory placement.",
        "cache-type-v": "Preserve f16 unless compression improves memory placement.",
        "split-mode": "Use the single GPU whose available memory was budgeted.",
        "main-gpu": "Device index selected from available compute backends.",
    }
    confidence = min((g.confidence, h.confidence), key=lambda c: {"low": 0, "medium": 1, "high": 2}[c])
    warnings = []
    if confidence == "low":
        warnings.append("Incomplete metadata or hardware information; conservative estimates need review.")
    if not plan["feasible"]:
        warnings.append("No estimated memory fit. Free memory or select a smaller model before applying.")
    active_bytes = g.weights * g.active_params / g.total_params if g.total_params else 0
    gpu_frac = 1 - fraction
    seconds = active_bytes * (gpu_frac / max(1, h.gpu_bw * 1e9) + fraction / max(1, h.ram_bw * 1e9))
    if h.uma:
        seconds = active_bytes / max(1, h.ram_bw * 1e9)
    estimate = {"regime": "gpu-resident" if plan["n_gpu_layers"] == g.layers + 1 else "hybrid" if plan["n_gpu_layers"] else "cpu",
                "theoretical_decode_ceiling": 1 / seconds if seconds else None,
                "active_weight_bytes": active_bytes, "disk_bandwidth_gbps": h.disk_bw,
                "note": "Bandwidth-only ceiling, not measured throughput."}
    LOG.debug("Static AutoTune %s model_size=%s weights=%s memory=%s threads=%s batch=%s ubatch=%s",
              preset, g.file_bytes, g.weights, plan, threads, batch, ubatch)
    return {"knobs": knobs, "rationale": {k: reasons[k] for k in knobs},
            "memory": plan, "confidence": confidence, "warnings": warnings,
            "applicable": plan["feasible"], "prediction": estimate}


def recommend(meta, hw, size_bytes=None):
    g, h = model_geometry(meta, size_bytes), hardware_budget(hw)
    return {"default": "balanced", "geometry": asdict(g), "hardware": asdict(h),
            **{name: parameter_heuristics(g, h, name) for name in PRESETS}}
