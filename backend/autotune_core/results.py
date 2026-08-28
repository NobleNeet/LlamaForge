"""Runner-ready benchmark schemas; execution and persistence remain separate."""
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from .models import ExecutionEnvironment


@dataclass(frozen=True)
class BenchmarkWorkload:
    """One llama-bench workload. Modes are prompt, generation, or combined."""
    mode: str  # pp, tg, or pg
    prompt_tokens: int
    generation_tokens: int
    context_depth: int
    weight: float = 1.0


@dataclass(frozen=True)
class BenchmarkCase:
    """One logical candidate/workload pairing; repetitions belong to measurements."""
    case_id: str
    stage_id: str
    candidate_id: str
    backend: str
    settings: Mapping[str, object]
    workload: BenchmarkWorkload
    execution_environment: ExecutionEnvironment


@dataclass(frozen=True)
class BenchmarkMeasurement:
    case_id: str
    repetition: int
    prompt_tokens_per_second: Optional[float]
    generation_tokens_per_second: Optional[float]
    command_argv: Tuple[str, ...] = field(default_factory=tuple)
    raw_stdout: Optional[str] = None
    raw_structured_result: Optional[Mapping[str, object]] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class TuneResult:
    result_id: str
    model_fingerprint: str
    hardware_fingerprint: str
    cases: Tuple[BenchmarkCase, ...] = field(default_factory=tuple)
    measurements: Tuple[BenchmarkMeasurement, ...] = field(default_factory=tuple)


def _workload_dict(workload):
    return {"mode": workload.mode, "prompt_tokens": workload.prompt_tokens,
            "generation_tokens": workload.generation_tokens, "context_depth": workload.context_depth,
            "weight": workload.weight}


def _binary_dict(binary):
    return {"backend": binary.backend, "path": binary.path, "build_id": binary.build_id,
            "file_fingerprint": binary.file_fingerprint, "version_text": binary.version_text,
            "provenance": binary.provenance}


def _environment_dict(environment):
    return {"hardware_fingerprint": environment.hardware_fingerprint, "backend": environment.backend,
            "runtime": dict(environment.runtime), "bench_binary": _binary_dict(environment.bench_binary),
            "captured_at": environment.captured_at}


def validate_result(result):
    errors = []
    case_ids = {case.case_id for case in result.cases}
    if not result.result_id:
        errors.append("result_id is required")
    if not result.model_fingerprint:
        errors.append("model_fingerprint is required")
    if not result.hardware_fingerprint:
        errors.append("hardware_fingerprint is required")
    for case in result.cases:
        workload = case.workload
        if workload.mode not in ("pp", "tg", "pg"):
            errors.append("case %s has invalid workload mode" % case.case_id)
        if min(workload.prompt_tokens, workload.generation_tokens, workload.context_depth) < 0:
            errors.append("case %s has negative workload values" % case.case_id)
        if workload.weight <= 0:
            errors.append("case %s has non-positive workload weight" % case.case_id)
        if case.backend != case.execution_environment.backend:
            errors.append("case %s backend does not match execution environment" % case.case_id)
    for measurement in result.measurements:
        if measurement.case_id not in case_ids:
            errors.append("measurement references unknown case %s" % measurement.case_id)
        if measurement.repetition < 0:
            errors.append("measurement repetition must be non-negative")
    return errors


def serialize_result(result):
    return {
        "result_id": result.result_id,
        "model_fingerprint": result.model_fingerprint,
        "hardware_fingerprint": result.hardware_fingerprint,
        "cases": [{"case_id": case.case_id, "stage_id": case.stage_id,
                   "candidate_id": case.candidate_id, "backend": case.backend,
                   "settings": dict(case.settings), "workload": _workload_dict(case.workload),
                   "execution_environment": _environment_dict(case.execution_environment)} for case in result.cases],
        "measurements": [{"case_id": measurement.case_id, "repetition": measurement.repetition,
                          "prompt_tokens_per_second": measurement.prompt_tokens_per_second,
                          "generation_tokens_per_second": measurement.generation_tokens_per_second,
                          "command_argv": list(measurement.command_argv), "raw_stdout": measurement.raw_stdout,
                          "raw_structured_result": measurement.raw_structured_result, "stderr": measurement.stderr,
                          "exit_code": measurement.exit_code, "started_at": measurement.started_at,
                          "finished_at": measurement.finished_at, "duration_seconds": measurement.duration_seconds,
                          "error": measurement.error} for measurement in result.measurements],
    }


def deserialize_result(data):
    from .models import BenchBinaryIdentity
    cases = []
    for item in data.get("cases") or []:
        environment = item.get("execution_environment") or {}
        binary = environment.get("bench_binary") or {}
        execution = ExecutionEnvironment(
            str(environment.get("hardware_fingerprint") or ""), str(environment.get("backend") or ""),
            dict(environment.get("runtime") or {}),
            BenchBinaryIdentity(str(binary.get("backend") or ""), str(binary.get("path") or ""),
                                binary.get("build_id"), binary.get("file_fingerprint"), binary.get("version_text"),
                                str(binary.get("provenance") or "")), str(environment.get("captured_at") or ""))
        workload = item.get("workload") or {}
        cases.append(BenchmarkCase(str(item.get("case_id") or ""), str(item.get("stage_id") or ""),
                                   str(item.get("candidate_id") or ""), str(item.get("backend") or ""),
                                   dict(item.get("settings") or {}),
                                   BenchmarkWorkload(str(workload.get("mode") or ""), int(workload.get("prompt_tokens") or 0),
                                                     int(workload.get("generation_tokens") or 0), int(workload.get("context_depth") or 0),
                                                     float(workload.get("weight") or 1.0)),
                                   execution))
    return TuneResult(
        result_id=str(data.get("result_id") or ""), model_fingerprint=str(data.get("model_fingerprint") or ""),
        hardware_fingerprint=str(data.get("hardware_fingerprint") or ""), cases=tuple(cases),
        measurements=tuple(BenchmarkMeasurement(str(item.get("case_id") or ""), int(item.get("repetition") or 0),
                                                item.get("prompt_tokens_per_second"), item.get("generation_tokens_per_second"),
                                                tuple(str(arg) for arg in item.get("command_argv") or []), item.get("raw_stdout"),
                                                item.get("raw_structured_result"), item.get("stderr"), item.get("exit_code"),
                                                item.get("started_at"), item.get("finished_at"), item.get("duration_seconds"),
                                                item.get("error")) for item in data.get("measurements") or []),
    )
