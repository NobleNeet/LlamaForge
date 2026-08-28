"""Run-level Auto Tune control flow; candidate failures are expected search evidence."""
from dataclasses import dataclass
import hashlib
import json

from .planner import eligible_candidates, initial_stage_plan, next_stage_plan, stage_outcome
from .resource_lease import BenchmarkResourceLease, ResourceBusyError
from .results import TuneResult
from .scoring import derive_request_latencies, rank_candidates
from .environment import execution_fingerprint
from .bench_capabilities import CapabilityProbeError, probe_binary_capabilities, probe_runtime_capabilities
from .staleness import ProfileIdentity


SCORING_SCHEMA_VERSION = "phase4.1-v1"


class FatalAutoTuneError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedProfile:
    name: str
    candidate_id: str
    settings: dict
    evidence: str  # measured or heuristic
    source: str
    provenance: ProfileIdentity


@dataclass(frozen=True)
class AutoTuneOutcome:
    result: TuneResult
    profiles: tuple


def _fingerprint(value):
    return "sha256-policy-v1:" + hashlib.sha256(json.dumps(value, default=str, sort_keys=True).encode()).hexdigest()


def generate_profiles(final_plan, measurements, identity_for_candidate):
    """Generate measured speed profiles; memory remains explicit heuristic evidence."""
    profiles, eligible = [], eligible_candidates(final_plan, measurements)
    for name, intent in (("balanced", "auto"), ("fast_prefill", "relative"), ("fast_decode", "relative")):
        if name == "fast_prefill":
            cases = tuple(case for case in final_plan.cases if case.workload.mode == "pp")
        elif name == "fast_decode":
            cases = tuple(case for case in final_plan.cases if case.workload.mode == "tg")
        else:
            cases = final_plan.cases
        scores = rank_candidates(type("Plan", (), {"candidates": eligible, "cases": cases})(), measurements,
                                  scoring_intent=intent)
        if scores:
            candidate = next(candidate for candidate in eligible if candidate.candidate_id == scores[0].candidate_id)
            profiles.append(GeneratedProfile(name, candidate.candidate_id, dict(candidate.settings), "measured", "benchmark",
                                             identity_for_candidate(candidate)))
    # No peak memory telemetry exists yet, so no low-memory recommendation is emitted.
    return tuple(profiles)


class AutoTuneOrchestrator:
    def __init__(self, store, runner, resource_root=None, resource_id_for_case=None, capability_probe=probe_binary_capabilities,
                 runtime_capability_probe=probe_runtime_capabilities, resource_wait_seconds=0.25,
                 resource_lease_factory=BenchmarkResourceLease):
        self.store, self.runner = store, runner
        self.resource_root = resource_root or store.root_dir
        # One physical GPU can expose HIP and Vulkan.  Until per-device mapping is
        # available, serialize conservatively across all GPU work on this hardware.
        self.resource_id_for_case = resource_id_for_case or (lambda case: "hardware-" + case.execution_environment.hardware_fingerprint)
        self.capability_probe = capability_probe
        self.runtime_capability_probe = runtime_capability_probe
        self.runtime_capabilities = {}
        self.resource_wait_seconds = resource_wait_seconds
        self.resource_lease_factory = resource_lease_factory

    def run(self, run_id, strategy, rules, environments, target, repetitions, timeout_seconds, cancellation=None):
        self.store.acquire(run_id)
        try:
            capabilities = {}
            usable_environments = []
            for environment in environments:
                try:
                    capabilities[execution_fingerprint(environment)] = self.capability_probe(environment.bench_binary.path)
                    usable_environments.append(environment)
                    try:
                        self.runtime_capabilities[execution_fingerprint(environment)] = self.runtime_capability_probe(
                            environment.bench_binary.path, environment.backend)
                    except CapabilityProbeError:
                        pass  # Hardware/runtime detection remains the availability source.
                except CapabilityProbeError:
                    continue
            if not usable_environments:
                raise FatalAutoTuneError("no usable llama-bench binary capabilities")
            all_cases, all_measurements = [], []
            plan = initial_stage_plan(strategy, rules, usable_environments)
            final_plan = plan
            stage_index = 0
            while plan is not None:
                counts = {"succeeded": 0, "failed": 0, "skipped": 0}
                self.store.update_progress(run_id, {"stage_index": stage_index, "stage_id": plan.definition.stage_id,
                                                    "candidates": len(plan.candidates), "cases": len(plan.cases),
                                                    "counts": counts})
                for case in plan.cases:
                    if cancellation is not None and cancellation.cancelled():
                        self.store.finish(run_id, "cancelled")
                        return None
                    lease = self.resource_lease_factory(self.resource_root, self.resource_id_for_case(case),
                                                        instance_id=self.store.instance_id, pid=self.store.pid, hostname=self.store.hostname)
                    while True:
                        try:
                            lease.acquire(self.resource_wait_seconds, cancellation)
                            break
                        except ResourceBusyError:
                            if cancellation is not None and cancellation.cancelled():
                                self.store.finish(run_id, "cancelled")
                                return None
                            self.store.update_progress(run_id, {"stage_index": stage_index, "stage_id": plan.definition.stage_id,
                                                                "status": "waiting_for_resource", "candidates": len(plan.candidates),
                                                                "cases": len(plan.cases), "counts": counts})
                    try:
                        status, measurements, _ = self.runner.run_case(
                            run_id, target, case, repetitions, timeout_seconds, cancellation,
                            capabilities[execution_fingerprint(case.execution_environment)])
                    finally:
                        lease.release()
                    if cancellation is not None and cancellation.cancelled():
                        self.store.finish(run_id, "cancelled")
                        return None
                    if status == "completed":
                        counts["succeeded"] += 1; all_measurements.extend(measurements)
                    else:
                        counts["failed"] += 1
                    self.store.update_progress(run_id, {"stage_index": stage_index, "stage_id": plan.definition.stage_id,
                                                        "candidates": len(plan.candidates), "cases": len(plan.cases), "counts": counts})
                all_cases.extend(plan.cases)
                if cancellation is not None and cancellation.cancelled():
                    self.store.finish(run_id, "cancelled")
                    return None
                outcome = stage_outcome(plan, all_measurements)
                valid = bool(outcome.candidate_scores)
                stage_status = "completed" if valid and not (counts["failed"] or counts["skipped"]) else "partial" if valid else "failed"
                self.store.update_progress(run_id, {"stage_index": stage_index, "stage_id": plan.definition.stage_id,
                                                    "status": stage_status, "candidates": len(plan.candidates),
                                                    "cases": len(plan.cases), "counts": counts})
                if not valid:
                    # Exhausted candidates are normal benchmark evidence, not a run-level failure.
                    break
                final_plan = plan
                plan = next_stage_plan(strategy, outcome, rules)
                stage_index += 1
            derived = derive_request_latencies(final_plan, all_measurements)
            result = TuneResult(run_id, target.model_fingerprint, environments[0].hardware_fingerprint,
                                tuple(all_cases), tuple(all_measurements), derived)
            def identity_for(candidate):
                environment = candidate.execution_environment
                binary = environment.bench_binary
                return ProfileIdentity(target.model_fingerprint, environment.hardware_fingerprint, dict(environment.runtime),
                                       {"backend": binary.backend, "build_id": binary.build_id,
                                        "file_fingerprint": binary.file_fingerprint, "version_text": binary.version_text},
                                       capabilities[execution_fingerprint(environment)].fingerprint, _fingerprint(rules),
                                       _fingerprint(strategy), SCORING_SCHEMA_VERSION, candidate.backend)
            profiles = generate_profiles(final_plan, all_measurements, identity_for) if all_measurements else ()
            if cancellation is not None and cancellation.cancelled():
                self.store.finish(run_id, "cancelled")
                return None
            self.store.record_result(run_id, result, profiles)
            if cancellation is not None and cancellation.cancelled():
                self.store.finish(run_id, "cancelled")
                return None
            self.store.finish(run_id, "completed")
            return AutoTuneOutcome(result, profiles)
        except Exception:
            # A runner candidate failure is returned as data. Anything escaping
            # orchestration is run-fatal, provided this instance still owns it.
            try:
                self.store.finish(run_id, "failed")
            except Exception:
                pass
            raise

    @staticmethod
    def strategy_fingerprint(strategy):
        return _fingerprint(strategy)
