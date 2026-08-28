"""References to llama-bench binaries without assuming they match server_bin."""
from dataclasses import dataclass
import os
from typing import Optional

from .backends import canonical_backend_id


@dataclass(frozen=True)
class BenchBinaryRef:
    backend: Optional[str]
    build_id: Optional[str]
    path: str
    provenance: str  # configured, artifact, or sibling_fallback


def _refs_from_config(configured):
    for item in configured.get("autotune_bench_binaries", []) if isinstance(configured, dict) else []:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        yield BenchBinaryRef(canonical_backend_id(item.get("backend")), item.get("build_id"),
                             item["path"], "configured")
    artifacts = configured.get("autotune_build_artifacts", []) if isinstance(configured, dict) else []
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("llama_bench_bin"), str):
            continue
        yield BenchBinaryRef(canonical_backend_id(item.get("backend")), item.get("build_id"),
                             item["llama_bench_bin"], "artifact")


def _sibling_fallback(server_bin):
    if not isinstance(server_bin, str) or not server_bin:
        return None
    directory, name = os.path.dirname(server_bin), os.path.basename(server_bin)
    extension = ".exe" if name.lower().endswith(".exe") else ""
    return os.path.join(directory, "llama-bench" + extension)


def resolve_bench_binary(configured, backend=None, build_id=None, exists=os.path.isfile):
    """Resolve exact backend/build artifacts first; sibling inference is last."""
    backend = canonical_backend_id(backend)
    refs = list(_refs_from_config(configured))
    for ref in refs:
        if ref.backend == backend and ref.build_id == build_id and exists(ref.path):
            return ref
    for ref in refs:
        if ref.backend == backend and ref.build_id is None and exists(ref.path):
            return ref
    fallback = _sibling_fallback((configured or {}).get("server_bin"))
    if fallback and exists(fallback):
        return BenchBinaryRef(backend, None, fallback, "sibling_fallback")
    return None
