"""Bounded staged candidate planning; this module never starts a process."""
from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping, Tuple

from .backends import canonical_backend_id
from .environment import execution_fingerprint
from .results import BenchmarkCase, BenchmarkWorkload
from .rules import candidate_allowed
from .scoring import CandidateScore, rank_candidates


@dataclass(frozen=True)
class ParameterSpace:
    key: str
    values: Tuple[object, ...]


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    parameters: Tuple[ParameterSpace, ...]
    workloads: Tuple[BenchmarkWorkload, ...]
    top_k: int
    max_candidates: int
    scoring_intent: str = "auto"  # auto, throughput, relative, or latency
    retention_objectives: Tuple[str, ...] = ("balanced", "prefill", "decode")
    pareto_retention: bool = True


@dataclass(frozen=True)
class BenchmarkStrategy:
    stages: Tuple[StageDefinition, ...]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    backend: str
    settings: Mapping[str, object]
    execution_environment: object
    source_stage_id: str = ""


@dataclass(frozen=True)
class StagePlan:
    definition: StageDefinition
    candidates: Tuple[Candidate, ...]
    cases: Tuple[BenchmarkCase, ...]


@dataclass(frozen=True)
class StageOutcome:
    plan: StagePlan
    candidate_scores: Tuple[CandidateScore, ...]
    objective_scores: Mapping[str, Tuple[CandidateScore, ...]] = field(default_factory=dict)


def _identity(prefix, payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "%s-%s" % (prefix, hashlib.sha256(encoded).hexdigest()[:16])


def _candidate(backend, settings, environment, source_stage_id):
    canonical = canonical_backend_id(backend)
    if not canonical or canonical != environment.backend:
        raise ValueError("candidate backend must match its execution environment")
    settings = dict(settings)
    return Candidate(
        candidate_id=_identity("candidate", {"backend": canonical, "settings": settings,
                                               "execution": execution_fingerprint(environment)}),
        backend=canonical, settings=settings, execution_environment=environment, source_stage_id=source_stage_id)


def _candidate_settings(candidate):
    return dict(candidate.settings, backend=candidate.backend)


def _preference_score(candidate, resolved_rules):
    settings = _candidate_settings(candidate)
    return sum(preference.weight for preference in resolved_rules.preferences
               if settings.get(preference.key) == preference.value)


def _allowed_unique(candidates, resolved_rules):
    out, seen = [], set()
    for candidate in candidates:
        if not candidate_allowed(_candidate_settings(candidate), resolved_rules):
            continue
        if candidate.candidate_id not in seen:
            seen.add(candidate.candidate_id)
            out.append(candidate)
    return tuple(out)


def _bounded_by_environment(candidates, resolved_rules, maximum):
    """Bound initial/final pools without letting one backend/build consume all slots."""
    groups = {}
    for candidate in _allowed_unique(candidates, resolved_rules):
        key = execution_fingerprint(candidate.execution_environment)
        groups.setdefault(key, []).append(candidate)
    for group in groups.values():
        group.sort(key=lambda candidate: (-_preference_score(candidate, resolved_rules), candidate.candidate_id))
    out = []
    while groups and len(out) < maximum:
        for key in sorted(tuple(groups)):
            if len(out) >= maximum:
                break
            out.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    return tuple(out)


def _parameter_values(candidate, parameter):
    values = list(parameter.values)
    baseline = candidate.settings.get(parameter.key)
    if baseline is not None:
        values.insert(0, baseline)
    unique = {}
    for value in values:
        unique.setdefault(json.dumps(value, sort_keys=True, default=str), value)
    return [unique[key] for key in sorted(unique)] if baseline is None else [baseline] + [
        unique[key] for key in sorted(unique) if unique[key] != baseline]


def _round_robin_children(parents, parameter, definition, resolved_rules):
    """Preserve one candidate per parent before exploring further child values."""
    groups = {}
    for parent in sorted(parents, key=lambda candidate: candidate.candidate_id):
        children = []
        for value in _parameter_values(parent, parameter):
            settings = dict(parent.settings)
            settings[parameter.key] = value
            children.append(_candidate(parent.backend, settings, parent.execution_environment, definition.stage_id))
        children = list(_allowed_unique(children, resolved_rules))
        baseline = parent.settings.get(parameter.key)
        children.sort(key=lambda candidate: (candidate.settings.get(parameter.key) != baseline,
                                             -_preference_score(candidate, resolved_rules), candidate.candidate_id))
        if children:
            groups[parent.candidate_id] = children
    out, seen = [], set()
    while groups and len(out) < definition.max_candidates:
        for key in sorted(tuple(groups)):
            if len(out) >= definition.max_candidates:
                break
            candidate = groups[key].pop(0)
            if candidate.candidate_id not in seen:
                seen.add(candidate.candidate_id)
                out.append(candidate)
            if not groups[key]:
                del groups[key]
    return tuple(out)


def _expand(definition, bases, resolved_rules):
    """Expand each parameter locally and prune immediately; no global product."""
    candidates = _bounded_by_environment(bases, resolved_rules, definition.max_candidates)
    for parameter in definition.parameters:
        candidates = _round_robin_children(candidates, parameter, definition, resolved_rules)
        if not candidates:
            break
    return _bounded_by_environment(candidates, resolved_rules, definition.max_candidates)


def _cases(definition, candidates):
    cases = []
    for candidate in candidates:
        for workload in definition.workloads:
            if workload.mode == "request":
                continue
            case_id = _identity("case", {"candidate": candidate.candidate_id, "stage": definition.stage_id,
                                           "workload": workload.__dict__})
            cases.append(BenchmarkCase(case_id, definition.stage_id, candidate.candidate_id, candidate.backend,
                                       candidate.settings, workload, candidate.execution_environment))
    return tuple(cases)


def initial_stage_plan(strategy, resolved_rules, execution_environments):
    """Create only the first realized plan, one candidate stream per build/backend."""
    if not strategy.stages:
        raise ValueError("strategy has no stages")
    definition = strategy.stages[0]
    bases = []
    seeds = resolved_rules.candidate_seeds or ()
    for environment in execution_environments:
        for seed in seeds or (None,):
            settings = dict(seed.settings) if seed is not None else {}
            requested = settings.pop("backend", environment.backend)
            if canonical_backend_id(requested) != environment.backend:
                continue
            bases.append(_candidate(environment.backend, settings, environment, ""))
    candidates = _expand(definition, bases, resolved_rules)
    return StagePlan(definition, candidates, _cases(definition, candidates))


def stage_outcome(plan, measurements, trim_fraction=0.0):
    balanced = rank_candidates(plan, measurements, trim_fraction, plan.definition.scoring_intent)
    by_candidate = {}
    by_case = {}
    for measurement in measurements:
        by_case.setdefault(measurement.case_id, []).append(measurement)
    from .scoring import aggregate_case_measurements
    for candidate in plan.candidates:
        values = {case.workload.mode: aggregate_case_measurements(case.workload, by_case.get(case.case_id, ()), trim_fraction)
                  for case in plan.cases if case.candidate_id == candidate.candidate_id}
        by_candidate[candidate.candidate_id] = values
    eligible = {score.candidate_id for score in balanced}
    by_candidate = {candidate_id: values for candidate_id, values in by_candidate.items() if candidate_id in eligible}
    objectives = {"balanced": tuple(score for score in balanced if score.candidate_id in eligible)}
    for name, mode in (("prefill", "pp"), ("decode", "tg")):
        scores = [CandidateScore(candidate_id, values[mode], (values[mode],)) for candidate_id, values in by_candidate.items()
                  if values.get(mode) is not None]
        objectives[name] = tuple(sorted(scores, key=lambda score: (-score.score, score.candidate_id)))
    if plan.definition.pareto_retention and any(values.get("pp") is not None and values.get("tg") is not None
                                               for values in by_candidate.values()):
        frontier = []
        for candidate_id, values in by_candidate.items():
            pp, tg = values.get("pp"), values.get("tg")
            if pp is None and tg is None:
                continue
            dominated = any(other_id != candidate_id and (other.get("pp") or 0) >= (pp or 0)
                            and (other.get("tg") or 0) >= (tg or 0)
                            and ((other.get("pp") or 0) > (pp or 0) or (other.get("tg") or 0) > (tg or 0))
                            for other_id, other in by_candidate.items())
            if not dominated:
                frontier.append(CandidateScore(candidate_id, (pp or 0) + (tg or 0), tuple(value for value in (pp, tg) if value is not None)))
        objectives["pareto"] = tuple(sorted(frontier, key=lambda score: score.candidate_id))
    return StageOutcome(plan, balanced, objectives)


def eligible_candidates(plan, measurements, trim_fraction=0.0):
    """The single eligibility rule for every objective and generated speed profile."""
    eligible_ids = {score.candidate_id for score in rank_candidates(plan, measurements, trim_fraction,
                                                                      plan.definition.scoring_intent)}
    return tuple(candidate for candidate in plan.candidates if candidate.candidate_id in eligible_ids)


def next_stage_plan(strategy, outcome, resolved_rules):
    """Create a later plan only from a completed preceding stage outcome."""
    try:
        index = next(i for i, stage in enumerate(strategy.stages)
                     if stage.stage_id == outcome.plan.definition.stage_id)
    except StopIteration:
        raise ValueError("outcome does not belong to strategy")
    if index + 1 >= len(strategy.stages):
        return None
    definition = strategy.stages[index + 1]
    candidates_by_id = {candidate.candidate_id: candidate for candidate in outcome.plan.candidates}
    winner_ids, seen = [], set()
    for objective in outcome.plan.definition.retention_objectives:
        for score in outcome.objective_scores.get(objective, ())[:definition.top_k]:
            if score.candidate_id not in seen:
                seen.add(score.candidate_id); winner_ids.append(score.candidate_id)
    if outcome.plan.definition.pareto_retention:
        for score in outcome.objective_scores.get("pareto", ()):
            if score.candidate_id not in seen:
                seen.add(score.candidate_id); winner_ids.append(score.candidate_id)
    winners = [candidates_by_id[candidate_id] for candidate_id in winner_ids if candidate_id in candidates_by_id]
    if not winners:
        return StagePlan(definition, (), ())
    candidates = _expand(definition, winners, resolved_rules)
    return StagePlan(definition, candidates, _cases(definition, candidates))
