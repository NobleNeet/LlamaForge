"""Strict llama-bench JSON/JSONL parsing and native repetition expansion."""
import json

from .results import BenchmarkMeasurement


class BenchParseError(ValueError):
    pass


def parse_structured_output(text):
    try:
        value = json.loads(text)
        records = value if isinstance(value, list) else [value]
        if not records or not all(isinstance(record, dict) for record in records):
            raise BenchParseError("JSON output must contain only object records")
        return records
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


def select_record(records, workload):
    """Select the one structured record representing this logical workload.

    ``-pg`` can emit auxiliary PP/TG records in addition to its native combined
    record. They are valid diagnostics, but not measurements for this case.
    """
    matches = [record for record in records if _matches(record, workload)]
    if len(matches) != 1:
        raise BenchParseError("structured output does not contain exactly one matching workload record")
    return matches[0]


def expand_record(record, case, argv, exit_code, started_at, finished_at, duration_seconds, requested_repetitions=None):
    if not _matches(record, case.workload):
        raise BenchParseError("output workload does not match benchmark case")
    rates, durations = record.get("samples_ts"), record.get("samples_ns")
    if not isinstance(rates, list) or not isinstance(durations, list) or not rates or len(rates) != len(durations):
        raise BenchParseError("samples_ts and samples_ns must be non-empty equal-length arrays")
    if requested_repetitions is not None and len(rates) != requested_repetitions:
        raise BenchParseError("sample count does not match requested repetitions")
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
