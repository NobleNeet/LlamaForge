"""Serializable benchmark result schema; execution and persistence come later."""
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    backend: str
    settings: Mapping[str, object]


@dataclass(frozen=True)
class BenchmarkMeasurement:
    case_id: str
    prompt_tokens_per_second: Optional[float]
    generation_tokens_per_second: Optional[float]
    error: Optional[str] = None


@dataclass(frozen=True)
class TuneResult:
    result_id: str
    model_fingerprint: str
    environment_fingerprint: str
    cases: Tuple[BenchmarkCase, ...] = field(default_factory=tuple)
    measurements: Tuple[BenchmarkMeasurement, ...] = field(default_factory=tuple)


def validate_result(result):
    errors = []
    case_ids = {case.case_id for case in result.cases}
    if not result.result_id:
        errors.append("result_id is required")
    if not result.model_fingerprint:
        errors.append("model_fingerprint is required")
    if not result.environment_fingerprint:
        errors.append("environment_fingerprint is required")
    for measurement in result.measurements:
        if measurement.case_id not in case_ids:
            errors.append("measurement references unknown case %s" % measurement.case_id)
    return errors


def serialize_result(result):
    return {
        "result_id": result.result_id,
        "model_fingerprint": result.model_fingerprint,
        "environment_fingerprint": result.environment_fingerprint,
        "cases": [{"case_id": case.case_id, "backend": case.backend, "settings": dict(case.settings)}
                  for case in result.cases],
        "measurements": [{"case_id": measurement.case_id,
                          "prompt_tokens_per_second": measurement.prompt_tokens_per_second,
                          "generation_tokens_per_second": measurement.generation_tokens_per_second,
                          "error": measurement.error} for measurement in result.measurements],
    }


def deserialize_result(data):
    return TuneResult(
        result_id=str(data.get("result_id") or ""),
        model_fingerprint=str(data.get("model_fingerprint") or ""),
        environment_fingerprint=str(data.get("environment_fingerprint") or ""),
        cases=tuple(BenchmarkCase(str(item.get("case_id") or ""), str(item.get("backend") or ""),
                                  dict(item.get("settings") or {})) for item in data.get("cases") or []),
        measurements=tuple(BenchmarkMeasurement(str(item.get("case_id") or ""),
                                                item.get("prompt_tokens_per_second"),
                                                item.get("generation_tokens_per_second"), item.get("error"))
                           for item in data.get("measurements") or []),
    )
