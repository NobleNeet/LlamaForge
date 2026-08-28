"""Build one unambiguous native llama-bench invocation per logical case."""
from .bench_knobs import encode_bench_settings


def build_bench_argv(target, case, repetitions, binary_capabilities=None):
    if not target.model_path or not target.model_fingerprint:
        raise ValueError("BenchmarkTarget requires model_path and model_fingerprint")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    workload = case.workload
    output_format = "jsonl"
    if binary_capabilities is not None:
        output_format = binary_capabilities.structured_output_format()
        if not binary_capabilities.repetitions:
            raise ValueError("llama-bench does not support --repetitions")
    argv = [case.execution_environment.bench_binary.path, "--model", target.model_path,
            "--repetitions", str(repetitions), "--output", output_format]
    if workload.mode == "pp":
        argv.extend(("--n-prompt", str(workload.prompt_tokens), "--n-gen", "0"))
    elif workload.mode == "tg":
        argv.extend(("--n-prompt", "0", "--n-gen", str(workload.generation_tokens)))
    elif workload.mode == "pg_native":
        argv.extend(("-pg", "%s,%s" % (workload.prompt_tokens, workload.generation_tokens)))
    else:
        raise ValueError("derived request workloads are not executable")
    argv.extend(("--n-depth", str(workload.context_depth)))
    argv.extend(encode_bench_settings(case.settings, binary_capabilities))
    return tuple(argv)
