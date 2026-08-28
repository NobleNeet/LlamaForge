"""Pure aggregation of repeated case measurements for staged planning."""
from dataclasses import dataclass
from statistics import median
from typing import Tuple


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    score: float
    case_scores: Tuple[float, ...]


def _measurement_score(workload, measurement):
    if measurement.error or measurement.exit_code not in (None, 0):
        return None
    if workload.mode == "pp":
        return measurement.prompt_tokens_per_second
    if workload.mode == "tg":
        return measurement.generation_tokens_per_second
    values = [value for value in (measurement.prompt_tokens_per_second,
                                  measurement.generation_tokens_per_second) if value is not None]
    return sum(values) / len(values) if values else None


def aggregate_case_measurements(workload, measurements, trim_fraction=0.0):
    """Median by default; optional symmetric trimming precedes aggregation."""
    values = sorted(value for measurement in measurements
                    for value in [_measurement_score(workload, measurement)] if value is not None)
    if not values:
        return None
    trim = int(len(values) * trim_fraction)
    if trim and len(values) > trim * 2:
        values = values[trim:-trim]
    return float(median(values))


def rank_candidates(stage_plan, measurements, trim_fraction=0.0):
    by_case = {}
    for measurement in measurements:
        by_case.setdefault(measurement.case_id, []).append(measurement)
    scores = []
    for candidate in stage_plan.candidates:
        case_scores = []
        for case in stage_plan.cases:
            if case.candidate_id != candidate.candidate_id:
                continue
            score = aggregate_case_measurements(case.workload, by_case.get(case.case_id, ()), trim_fraction)
            if score is not None:
                case_scores.append(score)
        if case_scores:
            scores.append(CandidateScore(candidate.candidate_id, sum(case_scores) / len(case_scores), tuple(case_scores)))
    return tuple(sorted(scores, key=lambda score: (-score.score, score.candidate_id)))
