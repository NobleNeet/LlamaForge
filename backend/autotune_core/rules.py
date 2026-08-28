"""Data-driven rule loading and monotonic constraint resolution."""
from dataclasses import dataclass, field
import json
from typing import Mapping, Optional, Tuple

from .backends import canonical_backend_id


@dataclass(frozen=True)
class CandidateSeed:
    settings: Mapping[str, object]
    source_rule_id: str


@dataclass(frozen=True)
class Preference:
    key: str
    value: object
    weight: float
    source_rule_id: str


@dataclass(frozen=True)
class HardConstraint:
    key: str
    allowed_values: Optional[Tuple[object, ...]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass(frozen=True)
class Exclusion:
    settings: Mapping[str, object]
    source_rule_id: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    when: Mapping[str, Tuple[str, ...]]
    candidate_seeds: Tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    preferences: Tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    hard_constraints: Tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    exclusions: Tuple[Mapping[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuleSet:
    schema_version: int
    rules: Tuple[Rule, ...]


@dataclass(frozen=True)
class ResolvedRules:
    candidate_seeds: Tuple[CandidateSeed, ...]
    preferences: Tuple[Preference, ...]
    hard_constraints: Tuple[HardConstraint, ...]
    exclusions: Tuple[Exclusion, ...]


def _string_tuple(value):
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("rule selector values must be string lists")
    return tuple(value)


def _mapping_list(value, name):
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("%s must be a list of objects" % name)
    return tuple(dict(item) for item in value)


def load_rule_set(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("rules"), list):
        raise ValueError("unsupported Auto Tune rule schema")
    rules = []
    ids = set()
    for raw in data["rules"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
            raise ValueError("each rule requires a non-empty id")
        if raw["id"] in ids:
            raise ValueError("duplicate rule id %s" % raw["id"])
        ids.add(raw["id"])
        when = raw.get("when") or {}
        if not isinstance(when, dict):
            raise ValueError("rule when must be an object")
        unknown = set(when) - {"architectures", "backends", "gpu_vendors"}
        if unknown:
            raise ValueError("unknown rule selector %s" % sorted(unknown)[0])
        rules.append(Rule(
            rule_id=raw["id"],
            when={key: _string_tuple(value) for key, value in when.items()},
            candidate_seeds=_mapping_list(raw.get("candidate_seeds"), "candidate_seeds"),
            preferences=_mapping_list(raw.get("preferences"), "preferences"),
            hard_constraints=_mapping_list(raw.get("hard_constraints"), "hard_constraints"),
            exclusions=_mapping_list(raw.get("exclusions"), "exclusions"),
        ))
    return RuleSet(schema_version=1, rules=tuple(rules))


def _matches(rule, model, environment):
    when = rule.when
    architectures = when.get("architectures", ())
    if architectures and model.architecture not in architectures:
        return False
    backends = tuple(canonical_backend_id(value) for value in when.get("backends", ()))
    if backends and not set(backends).intersection(environment.available_backends):
        return False
    vendors = {gpu.vendor.lower() for gpu in environment.physical_gpus}
    requested_vendors = {value.lower() for value in when.get("gpu_vendors", ())}
    return not requested_vendors or bool(vendors.intersection(requested_vendors))


def _constraint_from(raw):
    key = raw.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("hard constraint requires key")
    allowed = raw.get("allowed_values")
    if allowed is not None and (not isinstance(allowed, list) or not allowed):
        raise ValueError("allowed_values must be a non-empty list")
    minimum, maximum = raw.get("minimum"), raw.get("maximum")
    if minimum is not None and not isinstance(minimum, (int, float)):
        raise ValueError("minimum must be numeric")
    if maximum is not None and not isinstance(maximum, (int, float)):
        raise ValueError("maximum must be numeric")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum exceeds maximum")
    if allowed is None and minimum is None and maximum is None:
        raise ValueError("hard constraint must restrict a value")
    return HardConstraint(key, tuple(allowed) if allowed is not None else None, minimum, maximum)


def _merge_constraint(current, incoming):
    allowed = current.allowed_values
    if incoming.allowed_values is not None:
        allowed = incoming.allowed_values if allowed is None else tuple(value for value in allowed if value in incoming.allowed_values)
    minimum = max(value for value in (current.minimum, incoming.minimum) if value is not None) if (current.minimum is not None or incoming.minimum is not None) else None
    maximum = min(value for value in (current.maximum, incoming.maximum) if value is not None) if (current.maximum is not None or incoming.maximum is not None) else None
    if not allowed and (current.allowed_values is not None or incoming.allowed_values is not None):
        raise ValueError("incompatible hard constraints for %s" % current.key)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("incompatible hard constraints for %s" % current.key)
    return HardConstraint(current.key, allowed, minimum, maximum)


def resolve_rules(rule_sets, model, environment):
    seeds, preferences, exclusions, constraints = [], [], [], {}
    seen_seeds = set()
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            if not _matches(rule, model, environment):
                continue
            for settings in rule.candidate_seeds:
                marker = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
                if marker not in seen_seeds:
                    seen_seeds.add(marker)
                    seeds.append(CandidateSeed(settings, rule.rule_id))
            for raw in rule.preferences:
                if not isinstance(raw.get("key"), str) or "value" not in raw:
                    raise ValueError("preference requires key and value")
                preferences.append(Preference(raw["key"], raw["value"], float(raw.get("weight", 1)), rule.rule_id))
            for raw in rule.hard_constraints:
                incoming = _constraint_from(raw)
                constraints[incoming.key] = _merge_constraint(constraints[incoming.key], incoming) if incoming.key in constraints else incoming
            for settings in rule.exclusions:
                exclusions.append(Exclusion(settings, rule.rule_id))
    return ResolvedRules(tuple(seeds), tuple(preferences), tuple(constraints[key] for key in sorted(constraints)), tuple(exclusions))


def candidate_allowed(settings, resolved):
    """Apply non-overridable constraints and exclusions after all seeds exist."""
    for constraint in resolved.hard_constraints:
        value = settings.get(constraint.key)
        if constraint.allowed_values is not None and value not in constraint.allowed_values:
            return False
        if constraint.minimum is not None and (not isinstance(value, (int, float)) or value < constraint.minimum):
            return False
        if constraint.maximum is not None and (not isinstance(value, (int, float)) or value > constraint.maximum):
            return False
    return not any(all(settings.get(key) == value for key, value in exclusion.settings.items())
                   for exclusion in resolved.exclusions)
