"""Pure production strategy construction from prepared domain facts."""
from .planner import BenchmarkStrategy, ParameterSpace, StageDefinition
from .results import BenchmarkWorkload
from .environment import execution_fingerprint


TARGET_CONTEXT = 16384
CONTEXT_SAFETY_HEADROOM = 256


def representative_context(gguf, target=TARGET_CONTEXT, safety_headroom=CONTEXT_SAFETY_HEADROOM):
    """Return a bounded context target without probing the model or filesystem."""
    trained = getattr(gguf, "context_length", None)
    if not isinstance(trained, int) or trained <= 0:
        return target
    return max(1, min(target, max(1, trained - safety_headroom)))


def safe_proxy_depth(representative, requested_depth, prompt_tokens, generation_tokens):
    """Leave room for the native PG workload inside its representative context."""
    available = max(0, int(representative) - int(prompt_tokens) - int(generation_tokens))
    return min(max(0, int(requested_depth)), available)


def build_production_strategy(gguf, hardware, prepared_environments, rules):
    """Return strategy data only; probing, files, config, and threads stay in the service."""
    flags = {execution_fingerprint(item.execution_environment): item.binary_capabilities.flags for item in prepared_environments}

    def supported_by(flag):
        return tuple(sorted(key for key, values in flags.items() if flag in values))

    representative = representative_context(gguf)
    flash_depth = safe_proxy_depth(representative, 4096, 2048, 32)
    kv_depth = safe_proxy_depth(representative, 8192, 512, 32)
    return BenchmarkStrategy((
        # Cost/accuracy tradeoff: all later profiles share one balanced offload/thread baseline.
        StageDefinition("coarse", (
            ParameterSpace("n-gpu-layers", (0, "all"), "--n-gpu-layers", supported_by("--n-gpu-layers")),
            ParameterSpace("threads", (4, 8), "--threads", supported_by("--threads")),
        ), (BenchmarkWorkload("pp", 256, 0, 0), BenchmarkWorkload("tg", 0, 32, 0)),
            1, 4, retention_objectives=("balanced",), pareto_retention=False),
        StageDefinition("batch_probe", (
            ParameterSpace("batch-size", (512, 1024), "--batch-size", supported_by("--batch-size")),
            ParameterSpace("ubatch-size", (128, 256), "--ubatch-size", supported_by("--ubatch-size")),
        ), (BenchmarkWorkload("pp", 2048, 0, 0),),
            1, 4, retention_objectives=("prefill",), pareto_retention=False),
        # pg_native is compared as native throughput, never as request rate.
        StageDefinition("flash_probe", (
            ParameterSpace("flash-attn", ("off", "on"), "--flash-attn", supported_by("--flash-attn")),
        ), (BenchmarkWorkload("pg_native", 2048, 32, flash_depth),),
            1, 2, scoring_intent="throughput", retention_objectives=("balanced",), pareto_retention=False),
        StageDefinition("kv_probe", (
            ParameterSpace("cache-type-k", ("f16", "q8_0"), "--cache-type-k", supported_by("--cache-type-k")),
            ParameterSpace("cache-type-v", ("f16", "q8_0"), "--cache-type-v", supported_by("--cache-type-v")),
        ), (BenchmarkWorkload("pg_native", 512, 32, kv_depth),),
            3, 4, scoring_intent="throughput", retention_objectives=("balanced",), pareto_retention=False),
        StageDefinition("validate", (), (
            BenchmarkWorkload("pp", representative, 0, 0),
            BenchmarkWorkload("tg", 0, 128, representative),
            BenchmarkWorkload("request", representative, 128, representative),
        ), 3, 3, scoring_intent="relative"),
    ))
