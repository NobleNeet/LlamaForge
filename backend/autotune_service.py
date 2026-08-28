"""HTTP-independent Auto Tune application service and background job lifecycle."""
from dataclasses import dataclass
import hashlib
import json
import os
import threading
import uuid
from statistics import median

import config

from autotune_core.bench_artifacts import identify_bench_binary, resolve_bench_binary
from autotune_core.bench_capabilities import probe_binary_capabilities, probe_runtime_capabilities
from autotune_core.bench_runner import BenchRunner, CancellationToken
from autotune_core.environment import capture_environment, environment_fingerprint, execution_environment
from autotune_core.gguf_normalize import fast_fingerprint, normalize_gguf
from autotune_core.models import BenchmarkTarget, PreparedEnvironment
from autotune_core.orchestrator import AutoTuneOrchestrator
from autotune_core.rules import RuleSet, resolve_rules
from autotune_core.run_store import RunOwnershipError, RunStore
from autotune_core.resource_lease import BenchmarkResourceLease
from autotune_core.strategy_factory import build_production_strategy


class AutoTuneServiceError(RuntimeError):
    status = 400


class DuplicateRunError(AutoTuneServiceError):
    status = 409
    def __init__(self, run_id): self.run_id = run_id; super().__init__("duplicate_active_run")


class NotOwnedError(AutoTuneServiceError): status = 409


@dataclass(frozen=True)
class AutoTuneStartRequest:
    model_path: str


class AutoTuneService:
    """All expensive preparation occurs in its worker, never in an HTTP thread."""
    def __init__(self, root_dir, config_loader=config.load, environment_capture=capture_environment,
                 fingerprint=fast_fingerprint, normalizer=normalize_gguf, runner_factory=None):
        self.store = RunStore(root_dir)
        self.config_loader, self.environment_capture = config_loader, environment_capture
        self.fingerprint, self.normalizer = fingerprint, normalizer
        self.runner_factory = runner_factory or (lambda: BenchRunner(self.store))
        self._registry, self._registry_lock = {}, threading.RLock()
        self._start_lock = threading.RLock()  # process-local fast path; RunStore manifests remain durable truth.
        self._reconciled = False

    @staticmethod
    def _run_id(value):
        try: return str(uuid.UUID(str(value)))
        except (ValueError, TypeError, AttributeError): raise AutoTuneServiceError("invalid run_id")

    @staticmethod
    def _duplicate_key(model_fingerprint, hardware_fingerprint, policy_fingerprint=None):
        payload = {"model_fingerprint": model_fingerprint, "hardware_fingerprint": hardware_fingerprint}
        if policy_fingerprint: payload["policy_fingerprint"] = policy_fingerprint
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def start(self, request):
        self._reconcile_once()
        path = os.path.abspath(os.path.expanduser(request.model_path))
        if not os.path.isfile(path): raise AutoTuneServiceError("model_path is not a file")
        model_fp, snapshot = self.fingerprint(path).value, self.environment_capture()
        hardware_fp = environment_fingerprint(snapshot)
        duplicate_key = self._duplicate_key(model_fp, hardware_fp)
        with self._start_lock, self.store.duplicate_lock(duplicate_key):
            manifest = self.store.find_active_by_duplicate_key(duplicate_key)
            if manifest: raise DuplicateRunError(manifest["run_id"])
            run_id = str(uuid.uuid4())
            self.store.create_run(run_id, {"duplicate_key": duplicate_key}, {"model_name": os.path.basename(path), "model_fingerprint": model_fp})
            token = CancellationToken()
            worker = threading.Thread(target=self._worker, args=(run_id, path, snapshot, hardware_fp, model_fp, token), daemon=True)
            with self._registry_lock: self._registry[run_id] = (token, worker)
            try: worker.start()
            except Exception:
                with self._registry_lock: self._registry.pop(run_id, None)
                try: self.store.acquire(run_id); self.store.finish(run_id, "failed")
                except Exception: pass
                raise
        return {"run_id": run_id, "status": "planned"}

    def _worker(self, run_id, path, snapshot, hardware_fp, expected_model_fp, token):
        try:
            if self.fingerprint(path).value != expected_model_fp: raise RuntimeError("model_changed_after_start")
            gguf, cfg = self.normalizer(path), self.config_loader()
            prepared = []
            for backend in snapshot.available_backends:
                ref = resolve_bench_binary(cfg, backend)
                identity = identify_bench_binary(ref)
                if identity is None: continue
                binary_capabilities = probe_binary_capabilities(identity.path)
                try: runtime_capabilities = probe_runtime_capabilities(identity.path, backend)
                except Exception: runtime_capabilities = None
                runtime = {"backend": backend, "devices": list(getattr(runtime_capabilities, "devices", ())),
                           "source": getattr(runtime_capabilities, "source", "detector")}
                env = execution_environment(hardware_fp, backend, runtime, identity)
                prepared.append(PreparedEnvironment(env, binary_capabilities, runtime_capabilities, identity.file_fingerprint))
            if not prepared: raise RuntimeError("no usable llama-bench artifact")
            rules = resolve_rules((RuleSet(1, ()),), gguf, snapshot)
            strategy = build_production_strategy(gguf, snapshot, tuple(prepared), rules)
            AutoTuneOrchestrator(self.store, self.runner_factory()).run(run_id, strategy, rules,
                tuple(item.execution_environment for item in prepared), BenchmarkTarget(path, expected_model_fp),
                3, 120, token, tuple(prepared))
        except Exception as exc:
            try:
                self.store.acquire(run_id); self.store.fail(run_id, exc.__class__.__name__, str(exc))
            except Exception:
                pass
        finally:
            with self._registry_lock: self._registry.pop(run_id, None)

    def status(self, run_id):
        manifest = self.store.load_manifest(self._run_id(run_id)); return self._summary(manifest)

    def cancel(self, run_id):
        run_id = self._run_id(run_id)
        manifest = self.store.load_manifest(run_id)
        if manifest.get("status") in ("completed", "failed", "cancelled", "interrupted"):
            raise NotOwnedError("terminal_run")
        with self._registry_lock: job = self._registry.get(run_id)
        if job is None: raise NotOwnedError("not_owned_by_this_instance")
        job[0].cancel(); return self.status(run_id)

    def list_runs(self, limit=20):
        return [self._summary(item) for item in self.store.list_manifests(min(max(1, int(limit)), 100))]

    def result(self, run_id):
        manifest = self.store.load_manifest(self._run_id(run_id))
        if manifest.get("status") != "completed": raise AutoTuneServiceError("run is not completed")
        artifact = self.store.load_result(manifest["run_id"])
        result, profiles = artifact["result"], artifact.get("profiles", [])
        candidate_ids = {profile.get("candidate_id") for profile in profiles}
        cases = [case for case in result.get("cases", []) if case.get("candidate_id") in candidate_ids]
        final_stage = cases[-1].get("stage_id") if cases else None
        case_by_id = {case.get("case_id"): case for case in cases if case.get("stage_id") == final_stage}
        grouped = {}
        for measurement in result.get("measurements", []):
            case = case_by_id.get(measurement.get("case_id"))
            if not case: continue
            workload = case.get("workload", {})
            key = (case.get("candidate_id"), workload.get("mode"), workload.get("prompt_tokens"),
                   workload.get("generation_tokens"), workload.get("context_depth"))
            grouped.setdefault(key, []).append(measurement)
        aggregates = []
        for key, values in grouped.items():
            pp = [item["prompt_tokens_per_second"] for item in values if item.get("prompt_tokens_per_second") is not None]
            tg = [item["generation_tokens_per_second"] for item in values if item.get("generation_tokens_per_second") is not None]
            aggregates.append({"candidate_id": key[0], "workload": {"mode": key[1], "prompt_tokens": key[2],
                               "generation_tokens": key[3], "context_depth": key[4]},
                               "prompt_tokens_per_second": median(pp) if pp else None,
                               "generation_tokens_per_second": median(tg) if tg else None})
        return {"run_id": manifest["run_id"], "profiles": artifact.get("profiles", []),
                "derived_request_latencies": result.get("derived_request_latencies", []), "metrics": aggregates,
                "staleness": {"state": "not_evaluated"}}

    def reconcile_startup(self):
        return self._reconcile_once()

    def _reconcile_once(self):
        with self._registry_lock:
            if self._reconciled: return []
            self._reconciled = True
        return self.store.reconcile_interrupted_runs() + BenchmarkResourceLease.reconcile_orphans(self.store.root_dir) + self.store.reconcile_duplicate_locks()

    @staticmethod
    def _summary(manifest):
        return {"run_id": manifest.get("run_id"), "status": manifest.get("status"), "progress": manifest.get("progress"),
                "model": manifest.get("target", {}).get("model_name"), "created_at": manifest.get("created_at"), "error": manifest.get("error"),
                "finished_at": manifest.get("finished_at")}
