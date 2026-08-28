"""Structured profile provenance and field-wise staleness checks."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProfileIdentity:
    model_fingerprint: str
    hardware_fingerprint: str
    runtime_identity: object
    bench_binary_identity: object
    capabilities_fingerprint: str
    rules_fingerprint: str
    strategy_fingerprint: str
    scoring_schema_version: str
    backend: str

    def as_dict(self):
        return asdict(self)


def stale_reasons(saved, current):
    """Compare only the backend a profile actually used."""
    if saved.backend != current.backend:
        return ("runtime_changed",)
    fields = (("model_fingerprint", "model_changed"), ("hardware_fingerprint", "hardware_changed"),
              ("runtime_identity", "runtime_changed"), ("bench_binary_identity", "bench_build_changed"),
              ("capabilities_fingerprint", "capabilities_changed"),
              ("rules_fingerprint", "tuning_policy_changed"), ("strategy_fingerprint", "tuning_policy_changed"),
              ("scoring_schema_version", "tuning_policy_changed"))
    return tuple(reason for field, reason in fields if getattr(saved, field) != getattr(current, field))
