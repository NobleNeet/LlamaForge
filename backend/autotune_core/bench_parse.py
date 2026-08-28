"""Strict llama-bench JSON/JSONL parsing and native repetition expansion."""
import json

from .results import BenchmarkMeasurement


class BenchParseError(ValueError):
    pass


def parse_structured_output(text):
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BenchParseError("stdout is neither JSON nor JSONL: %s" % error)
            if not isinstance(value, dict):
                raise BenchParseError("JSONL record must be an object")
            records.append(value)
        if not records:
            raise BenchParseError("structured output is empty")
        return records


def _matches(record, workload):
    if workload.mode == "pg_native":
        return record.get("n_prompt") == workload.prompt_tokens and record.get("n_gen") == workload.generation_tokens and record.get("n_depth") == workload.context_depth
    expected_prompt = workload.prompt_tokens if workload.mode == "pp" else 0
    expected_generation = workload.generation_tokens if workload.mode == "tg" else 0
    return record.get("n_prompt") == expected_prompt and record.get("n_gen") == expected_generation and record.get("n_depth") == workload.context_depth


def expand_record(record, case, argv, exit_code, started_at, finished_at, duration_seconds):
    if not _matches(record, case.workload):
        raise BenchParseError("output workload does not match benchmark case")
    rates, durations = record.get("samples_ts"), record.get("samples_ns")
    if not isinstance(rates, list) or not isinstance(durations, list) or not rates or len(rates) != len(durations):
        raise BenchParseError("samples_ts and samples_ns must be non-empty equal-length arrays")
    out = []
    for repetition, (rate, sample_ns) in enumerate(zip(rates, durations)):
        if not isinstance(rate, (int, float)) or not isinstance(sample_ns, (int, float)):
            raise BenchParseError("sample values must be numeric")
        prompt_rate = rate if case.workload.mode == "pp" else None
        generation_rate = rate if case.workload.mode == "tg" else None
        native_rate = rate if case.workload.mode == "pg_native" else None
        out.append(BenchmarkMeasurement(case.case_id, repetition, prompt_rate, generation_rate,
                                        command_argv=tuple(argv), raw_structured_result=dict(record), exit_code=exit_code,
                                        started_at=started_at, finished_at=finished_at,
                                        duration_seconds=float(sample_ns) / 1e9,
                                        native_tokens_per_second=native_rate))
    return tuple(out)
