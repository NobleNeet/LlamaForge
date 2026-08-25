"""Hardware-aware knob recommendations for LlamaForge (pure stdlib).

Turns detected hardware + a GGUF's header facts into a small set of
llama-server knobs, shaped by the user's intent. `recommend` is a pure function
(no I/O) so it is trivially testable; `refine` (Task 4) optionally benchmarks.
Only the ~8 knobs that materially affect fit/throughput are set; everything
else keeps llama.cpp's own defaults.
"""

import time

INTENTS = ("balanced", "speed", "context", "coding")

# Fraction of VRAM the weights may claim, leaving room for the KV cache/activations.
_HEADROOM = {"balanced": 0.90, "speed": 0.92, "context": 0.78, "coding": 0.90}

# Maximum context window for "context" intent.
_CTX_MAX = 150000


def _total_vram_mib(hw):
    total = 0
    for g in hw.get("gpus", []):
        fit = g.get("fit_vram_mib")
        if fit is None and not g.get("is_uma"):
            fit = g.get("vram_mib")
        if fit:
            total += fit
    return total


def _has_gpu(hw):
    return bool(hw.get("gpus"))


def _has_only_uma_gpu(hw):
    gpus = hw.get("gpus") or []
    return bool(gpus) and all(g.get("is_uma") for g in gpus)


def _fit_ngl(layers, weights_mib, budget_mib):
    """How many layers to offload. '99' = all (llama.cpp caps to the real count)."""
    if not layers or not weights_mib:
        return "99"                      # unknown size: try full offload, refine can back off
    if budget_mib >= weights_mib:
        return "99"
    return str(max(0, int(layers * budget_mib / weights_mib)))


def recommend(meta, hw, intent="balanced", size_bytes=None, prediction=None):
    intent = intent if intent in INTENTS else "balanced"
    knobs, why = {}, {}
    cpu = hw.get("cpu") or {}
    threads = cpu.get("threads") or cpu.get("cores")
    layers = meta.get("block_count")
    weights_mib = (size_bytes / (1024 * 1024)) if size_bytes else None

    if not _has_gpu(hw):
        knobs["n-gpu-layers"] = "0"
        why["n-gpu-layers"] = "No GPU detected - running on CPU."
        knobs["flash-attn"] = "off"
        why["flash-attn"] = "Flash-attention needs a supported GPU."
    elif _has_only_uma_gpu(hw):
        knobs["n-gpu-layers"] = "99"
        why["n-gpu-layers"] = ("UMA GPU detected - enabling GPU offload, but not estimating a VRAM budget "
                               "from shared memory.")
        knobs["flash-attn"] = "on"
        why["flash-attn"] = "GPU backend available - flash-attention enabled."
    else:
        total = _total_vram_mib(hw)
        budget = int(total * _HEADROOM[intent])
        knobs["n-gpu-layers"] = _fit_ngl(layers, weights_mib, budget)
        if knobs["n-gpu-layers"] == "99":
            # Distinguish between unknown size and genuine fit
            if not layers or not weights_mib:
                why["n-gpu-layers"] = "Model size/layer count unknown - attempting full GPU offload."
            else:
                why["n-gpu-layers"] = f"Weights fit in {total} MiB VRAM - full GPU offload."
        else:
            why["n-gpu-layers"] = (f"~{int(weights_mib)} MiB weights vs {budget} MiB budget "
                                   f"- offloading {knobs['n-gpu-layers']}/{layers} layers.")
        knobs["flash-attn"] = "on"
        why["flash-attn"] = "GPU present - flash-attention enabled."

    if threads:
        knobs["threads"] = str(threads)
        why["threads"] = f"Matched to this CPU's {threads} hardware threads."

    # Balanced context: the model's trained length, capped to a sane ceiling.
    trained = meta.get("context_length")
    ctx = _ctx_for(intent, trained)
    if ctx:
        knobs["ctx-size"] = str(ctx)
        why["ctx-size"] = _ctx_reason(intent, trained, ctx)

    # Intent-specific shaping (KV type, batch, tensor-split, sampling) — Task 3.
    _apply_intent(knobs, why, hw, intent)
    if prediction and prediction.get("regime"):
        tok = prediction.get("tok_s")
        tail = f" Predicted {prediction['regime']}" + (f" ~{tok} tok/s." if tok is not None else ".")
        if "n-gpu-layers" in why:
            why["n-gpu-layers"] += tail
        return {"knobs": knobs, "rationale": why, "prediction": prediction}
    return {"knobs": knobs, "rationale": why}


def _ctx_for(intent, trained):
    ceil = {"balanced": 65536, "speed": 16384, "context": _CTX_MAX, "coding": 65536}[intent]
    if not trained or trained <= 0:
        return None
    return min(trained, ceil)


def _ctx_reason(intent, trained, ctx):
    if intent == "context":
        return f"Max-context: using the model's trained {trained} tokens (capped {_CTX_MAX})."
    if intent == "speed":
        return f"Max-speed: smaller {ctx}-token window to cut KV-cache overhead."
    return f"Balanced {ctx}-token window (trained {trained})."


def _apply_intent(knobs, why, hw, intent):
    gpus = hw.get("gpus") or []

    if intent == "context":
        knobs["cache-type-k"] = knobs["cache-type-v"] = "q8_0"
        why["cache-type-k"] = why["cache-type-v"] = (
            "Max-context: 8-bit KV cache roughly halves memory per token.")
    elif intent == "speed":
        knobs["cache-type-k"] = knobs["cache-type-v"] = "f16"
        why["cache-type-k"] = why["cache-type-v"] = "Max-speed: full-precision KV cache."
        knobs["batch-size"] = "2048"
        knobs["ubatch-size"] = "512"
        why["batch-size"] = "Larger batch for higher prompt throughput."
        why["ubatch-size"] = "Micro-batch tuned for throughput (refine can adjust)."
    elif intent == "coding":
        knobs["temp"] = "0.2"
        knobs["top-p"] = "0.9"
        why["temp"] = "Coding: low temperature for deterministic output."
        why["top-p"] = "Coding: tightened nucleus sampling."

    if len(gpus) > 1:
        split = ",".join(str(round((g.get("vram_mib") or 0) / 1000)) for g in gpus)
        knobs["tensor-split"] = split
        why["tensor-split"] = f"Split across {len(gpus)} GPUs by VRAM ({split})."


def _candidates(base, intent):
    """Base first, then a few high-impact variants worth benchmarking."""
    out = [dict(base)]
    if intent == "speed":
        for ub in ("1024",):
            c = dict(base); c["ubatch-size"] = ub; out.append(c)
        for bs in ("4096",):
            c = dict(base); c["batch-size"] = bs; out.append(c)
    else:
        for ub in ("1024",):
            c = dict(base); c["ubatch-size"] = ub; out.append(c)
    return out


def refine(base_knobs, intent, load_fn, measure_fn, budget_s=60, clock=time.monotonic):
    start = clock()
    best_knobs, best_tok = dict(base_knobs), -1.0
    cands = []
    for cand in _candidates(base_knobs, intent):
        if clock() - start >= budget_s:
            break
        try:
            load_fn(cand)
            tok = float(measure_fn())
        except Exception:
            continue
        cands.append({"knobs": cand, "tok_s": tok})
        if tok > best_tok:
            best_knobs, best_tok = cand, tok
    return {"knobs": best_knobs,
            "measurements": {"candidates": cands,
                             "chosen_tok_s": best_tok if best_tok >= 0 else 0.0}}
