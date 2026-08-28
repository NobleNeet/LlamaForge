"""Pure aggregation of repeated case measurements for staged planning."""
from dataclasses import dataclass
from statistics import median
from typing import Tuple


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    score: float
    case_scores: Tuple[float, ...]


@dataclass(frozen=True)
class DerivedRequestLatency:
    candidate_id: str
    request_workload_id: str
    source_pp_case_id: str
    source_tg_case_id: str
    aggregate_method: str
    latency_seconds: float


def _measurement_score(workload, measurement):
    if measurement.error or measurement.exit_code not in (None, 0):
        return None
    if workload.mode == "pp":
        return measurement.prompt_tokens_per_second
    if workload.mode == "tg":
        return measurement.generation_tokens_per_second
    if workload.mode == "pg_native":
        return measurement.native_tokens_per_second
    return None


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


def _signature(case):
    workload = case.workload
    return workload.mode, workload.prompt_tokens, workload.generation_tokens, workload.context_depth


def derive_request_latency(candidate_id, request_workload_id, request_workload, pp_case_id, pp_workload, pp_measurements,
                           tg_case_id, tg_workload, tg_measurements, trim_fraction=0.0):
    """Derive request latency from independently aggregated pp/tg processes."""
    pp_rate = aggregate_case_measurements(pp_workload, pp_measurements, trim_fraction)
    tg_rate = aggregate_case_measurements(tg_workload, tg_measurements, trim_fraction)
    if not pp_rate or not tg_rate:
        return None
    latency = request_workload.prompt_tokens / pp_rate + request_workload.generation_tokens / tg_rate
    return DerivedRequestLatency(candidate_id, request_workload_id, pp_case_id, tg_case_id, "median", latency)


def derive_request_latencies(stage_plan, measurements, trim_fraction=0.0):
    """Derive request latency only from independent pp/tg aggregates of one candidate."""
    by_case = {}
    for measurement in measurements:
        by_case.setdefault(measurement.case_id, []).append(measurement)
    requests = [workload for workload in stage_plan.definition.workloads if workload.mode == "request"]
    derived = []
    for candidate in stage_plan.candidates:
        cases = [case for case in stage_plan.cases if case.candidate_id == candidate.candidate_id]
        for request in requests:
            pp = next((case for case in cases if case.workload.mode == "pp" and case.workload.prompt_tokens == request.prompt_tokens
                       and case.workload.context_depth == request.context_depth), None)
            tg = next((case for case in cases if case.workload.mode == "tg" and case.workload.generation_tokens == request.generation_tokens
                       and case.workload.context_depth == request.context_depth), None)
            if pp is None or tg is None:
                continue
            workload_id = "request:%s:%s:%s" % (request.prompt_tokens, request.generation_tokens, request.context_depth)
            value = derive_request_latency(candidate.candidate_id, workload_id, request, pp.case_id, pp.workload,
                                           by_case.get(pp.case_id, ()), tg.case_id, tg.workload,
                                           by_case.get(tg.case_id, ()), trim_fraction)
            if value is not None:
                derived.append(value)
    return tuple(derived)


def _weighted_mean(values):
    total_weight = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / total_weight if total_weight else None


def _latency_score(case_values):
    """Combine workloads as estimated request latency, then invert for ranking."""
    latency = 0.0
    for case, value in case_values:
        workload = case.workload
        if value <= 0:
            return None
        if workload.mode == "pp":
            latency += workload.weight * workload.prompt_tokens / value
        elif workload.mode == "tg":
            latency += workload.weight * workload.generation_tokens / value
        elif workload.mode == "pg_native":
            # Native combined throughput is neither pp/tg throughput nor requests/sec.
            return None
        else:
            return None
    return 1.0 / latency if latency > 0 else None


def rank_candidates(stage_plan, measurements, trim_fraction=0.0, scoring_intent="auto"):
    """Rank stage candidates without averaging raw rates across workload types.

    ``auto`` uses direct throughput only when all cases share one workload; a
    mixed stage uses per-workload relative normalization.  ``latency`` combines
    each workload's token/rate latency, while ``relative`` always normalizes.
    """
    if scoring_intent not in ("auto", "throughput", "relative", "latency"):
        raise ValueError("unknown scoring intent")
    by_case = {}
    for measurement in measurements:
        by_case.setdefault(measurement.case_id, []).append(measurement)
    raw_by_candidate = {}
    for candidate in stage_plan.candidates:
        case_scores = []
        missing_required = False
        for case in stage_plan.cases:
            if case.candidate_id != candidate.candidate_id:
                continue
            score = aggregate_case_measurements(case.workload, by_case.get(case.case_id, ()), trim_fraction)
            if score is not None:
                case_scores.append((case, score))
            elif case.workload.required:
                missing_required = True
        if case_scores and not missing_required:
            raw_by_candidate[candidate.candidate_id] = case_scores
    signatures = {_signature(case) for values in raw_by_candidate.values() for case, _ in values}
    if scoring_intent == "throughput" and len(signatures) > 1:
        raise ValueError("throughput scoring cannot combine distinct workloads")
    intent = "relative" if scoring_intent == "auto" and len(signatures) > 1 else scoring_intent
    best_by_signature = {}
    for values in raw_by_candidate.values():
        for case, value in values:
            key = _signature(case)
            best_by_signature[key] = max(best_by_signature.get(key, 0), value)
    scores = []
    for candidate_id, case_values in raw_by_candidate.items():
        raw_scores = tuple(value for _, value in case_values)
        if intent == "latency":
            score = _latency_score(case_values)
        elif intent == "relative":
            score = _weighted_mean([(value / best_by_signature[_signature(case)], case.workload.weight)
                                    for case, value in case_values])
        else:
            score = _weighted_mean([(value, case.workload.weight) for case, value in case_values])
        if score is not None:
            scores.append(CandidateScore(candidate_id, score, raw_scores))
    return tuple(sorted(scores, key=lambda score: (-score.score, score.candidate_id)))
