"""Pure production strategy construction from prepared domain facts."""
from .planner import BenchmarkStrategy, ParameterSpace, StageDefinition
from .results import BenchmarkWorkload
from .environment import execution_fingerprint


def build_production_strategy(gguf, hardware, prepared_environments, rules):
    """Return data only; probing, files, config, and threads belong to the service."""
    flags = {execution_fingerprint(item.execution_environment): item.binary_capabilities.flags for item in prepared_environments}
    def supported_by(flag): return tuple(sorted(key for key, values in flags.items() if flag in values))
    depths = tuple(sorted({0, min(4096, gguf.context_length or 4096), min(8192, gguf.context_length or 8192)}))
    stage3 = []
    for depth in depths:
        stage3.extend((BenchmarkWorkload("pp", 128, 0, depth), BenchmarkWorkload("tg", 0, 128, depth),
                       BenchmarkWorkload("request", 128, 128, depth)))
    return BenchmarkStrategy((
        StageDefinition("coarse", (ParameterSpace("n-gpu-layers", (0, "all"), "--n-gpu-layers", supported_by("--n-gpu-layers")),
                                   ParameterSpace("threads", (4, 8), "--threads", supported_by("--threads"))),
                        (BenchmarkWorkload("pp", 128, 0, 0), BenchmarkWorkload("tg", 0, 128, 0)), 2, 12),
        StageDefinition("refine", (ParameterSpace("batch-size", (512, 1024), "--batch-size", supported_by("--batch-size")),
                                   ParameterSpace("ubatch-size", (128, 256), "--ubatch-size", supported_by("--ubatch-size")),
                                   ParameterSpace("flash-attn", ("off", "on"), "--flash-attn", supported_by("--flash-attn"))),
                        (BenchmarkWorkload("pp", 128, 0, 0), BenchmarkWorkload("tg", 0, 128, 0)), 2, 16),
        StageDefinition("validate", (), tuple(stage3), 2, 12, scoring_intent="relative"),
    ))
