"""References to llama-bench binaries without assuming they match server_bin."""
from dataclasses import dataclass
import os
import hashlib
from typing import Optional

from .backends import canonical_backend_id
from .models import BenchBinaryIdentity


@dataclass(frozen=True)
class BenchBinaryRef:
    backend: Optional[str]
    build_id: Optional[str]
    path: str
    provenance: str  # configured, artifact, or sibling_fallback


class BenchArtifactAmbiguityError(ValueError):
    pass


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
    refs = [ref for ref in _refs_from_config(configured) if ref.backend == backend and exists(ref.path)]
    if build_id is not None:
        exact = [ref for ref in refs if ref.build_id == build_id]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise BenchArtifactAmbiguityError("multiple explicit llama-bench artifacts match backend/build")
    elif len(refs) == 1:
        return refs[0]
    elif len(refs) > 1:
        raise BenchArtifactAmbiguityError("multiple explicit llama-bench artifacts match backend without build_id")
    selected = canonical_backend_id((configured or {}).get("llama_backend"))
    fallback = _sibling_fallback((configured or {}).get("server_bin"))
    if fallback and exists(fallback) and (selected is None or selected == backend):
        return BenchBinaryRef(backend, None, fallback, "sibling_fallback")
    return None


def identify_bench_binary(ref, version_text=None, window_bytes=65536):
    """Capture a bounded-cost binary identity without executing the binary."""
    if ref is None or not os.path.isfile(ref.path):
        return None
    size = os.path.getsize(ref.path)
    digest = hashlib.sha256()
    digest.update(b"llamaforge-bench-binary-v1\\0" + str(size).encode("ascii"))
    with open(ref.path, "rb") as handle:
        for offset in sorted({0, max(0, size - window_bytes)}):
            handle.seek(offset)
            digest.update(hashlib.sha256(handle.read(window_bytes)).digest())
    return BenchBinaryIdentity(ref.backend or "cpu", ref.path, ref.build_id,
                               "sha256-sampled-v1:" + digest.hexdigest(), version_text,
                               ref.provenance)
