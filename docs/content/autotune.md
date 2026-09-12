---
title: Static AutoTune
section: guides
order: 2
---

# Static AutoTune

**Static recommendation only. No model execution or benchmark is performed.**

AutoTune provides a **fast initial recommendation**, not a guaranteed optimum or a guarantee against OOM. Open a GGUF model in Models to obtain Safe, Balanced (default), and Aggressive presets. Review GPU Layers, Context, Batch, uBatch, Threads, Flash Attention and KV Cache K/V. Select a preset to fill the existing editor; fields remain editable, and nothing is saved until you use the existing Save action. Recalculate refreshes available memory. Insufficient memory produces an explanation and disables application.

## Data flow

1. `gguf.inspect_model`: read scalar header facts and tensor directories, seeking past vocabulary arrays. Tensor payloads are never read. Aggregate split GGUF shards; missing shards cannot be treated as the complete model.
2. `autotune_hardware.snapshot`: reuse CPU/GPU detection and current available RAM/VRAM. Backend availability is separate from physical GPU identity. Apple Silicon uses the existing Metal memory budget.
3. `autotune.model_geometry`: normalize layers, embeddings, KV heads and head dimensions, context limit, weight bytes, parameter count, quantization/bpw and MoE experts. Per-tensor byte spans include alignment. Missing layer sizes fall back to average block sizes with a non-layer allowance.
4. `estimate_memory` and `plan_placement`: reserve KV cache, graph/activation buffers, backend overhead and headroom; accumulate offloaded blocks from the tail of the model. Full offload uses the actual layer count plus output layer. There is no fixed 99-layer recommendation.
5. `parameter_heuristics`: select representative batch/microbatch values, preserve f16 when possible, interpolate CPU threads from physical cores and CPU residency, and return three presets with reasons and confidence.

## Memory policy

| Preset | Reserve | Policy |
|---|---:|---|
| Safe | 18% | Larger memory reserve; batch up to 1024, microbatch up to 256 |
| Balanced | 9% | Default; batch up to 2048, microbatch up to 512 |
| Aggressive | 4% | Smaller reserve, still requires estimated fit |

Placement is calculated before selecting the largest buffer sizes that preserve that placement. KV uses geometry including separate K/V dimensions and quantization block overhead, with a conservative 12% allowance. f16 is preferred at equal residency; q8 can improve fit, and q4 is only considered for Aggressive. Unknown model geometry keeps portable f16 and CPU placement. Flash Attention is delegated to `auto`.

The initial context is capped at 32768 (8192 when geometry is incomplete), never exceeds the trained window, and shrinks when no feasible memory placement exists. This is a common initial window for all three presets, not a classification by application or model name.

UMA uses one shared physical memory pool, bounded by available RAM and the backend-visible shared heap; it never adds RAM and VRAM together. A small local APU carveout does not override a larger detected shared heap. Dedicated GPUs use current free VRAM where available. The planner selects a single usable GPU and sets `split-mode=none` / `main-gpu`; multi-GPU tensor parallel optimization is outside this initial policy. Memory budgets are never summed across duplicate backend views.

Weights use total model size for memory residency. MoE active parameters reuse the vramwise adapter's active-expert estimate for the informational bandwidth ceiling. The ceiling is not measured tok/s and does not select settings. VRAM/RAM/disk bandwidth overrides in Setup are reused. On UMA, the RAM bandwidth applies to the shared pool.

## Limits and fallback

Buffers and KV are conservative approximations, not llama.cpp allocation guarantees. Architecture-specific recurrent, hybrid, sliding-window, MLA or unusually shaped tensors may require different allocations. Large vocabularies, additional adapters/projectors, speculative models, parallel slots and manually overridden launch flags can add memory beyond this estimate. Recommendations describe the selected GGUF with the displayed settings; review other custom settings before launch. Available memory can change after calculation, particularly if another model remains loaded.

Missing dimensions use conservative geometry defaults and lower confidence. Missing file size or insufficient RAM produces non-applicable fallback presets with reasons. A malformed tensor directory can fall back to file-size estimation; missing split shards disable application. New quantization types can still use tensor offset spans without a hard-coded quant format table. Unavailable telemetry lowers confidence. Existing persisted benchmark logs are left untouched but are no longer read or written.

`POST /api/autotune/recommend` accepts `{ "model": "model-id" }`. It returns `safe`, `balanced`, `aggressive`, `default`, `geometry`, and `hardware`. Presets expose settings, rationale, confidence, memory estimates, warnings and `applicable`. It has no job ID, progress, cancellation or result store. The former start/status/result/runs/cancel/preview/refine APIs have been removed.

Enable Python DEBUG logging for `autotune` to see model/weight bytes, KV, compute buffers, available and reserved memory, GPU weight budget/layers, CPU fraction, threads and batch sizes. Normal logs remain quiet.

## Validation

Run `python -m unittest discover -s tests -v`. Static tests cover large UMA, limited discrete VRAM, headroom ordering, full offload, missing metadata, MoE, CPU-only, telemetry failure, split files, asymmetric KV dimensions and read-only API boundaries. They do not load LLMs.
