"""Scan directories (or whole drives) for GGUF models and turn them into
model entries the router understands. Portable: no hardcoded paths.

Rules:
- skip mmproj* (vision projectors; attached to their model instead)
- skip recycle bin and obvious non-model shards handling
- attach an mmproj sibling only to vision-capable models
- treat *embed* models as embedding endpoints
- multi-shard sets (foo-00001-of-00005.gguf) collapse to the first shard
- disambiguate duplicate names by walking up parent folders until unique
"""
import os, re
from collections import defaultdict

import osplat

# Architectures known to use an mmproj sidecar (vision/multimodal).
_VISION_ARCHES = frozenset({
    "clip", "mllama", "qwen2_vl", "llava", "moondream", "nanollava",
    "idefics3", "minicpm_v", "molmo", "pixtral", "granite_vision",
})

def _windows_drives():
    import string, ctypes
    drives = []
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    for i, letter in enumerate(string.ascii_uppercase):
        if bitmask & (1 << i):
            root = f"{letter}:\\"
            # only fixed drives
            if ctypes.windll.kernel32.GetDriveTypeW(root) == 3:
                drives.append(root)
    return drives

def posix_roots(home, extra_candidates, isdir=os.path.isdir):
    """Home dir plus whichever common mount points exist. Pure for tests."""
    roots = [home]
    for c in extra_candidates:
        if isdir(c):
            roots.append(c)
    return roots

def list_drives():
    if osplat.IS_WIN:
        return _windows_drives()
    extra = ["/Volumes"] if osplat.IS_MAC else ["/mnt", "/media", "/srv", "/data"]
    return posix_roots(os.path.expanduser("~"), extra)

def find_ggufs(roots, min_mb=50):
    hits = []
    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            low = dirpath.lower().replace("\\", "/")
            if "$recycle.bin" in low or "/.git" in low or "/.trash" in low or "/proc" == low[:5]:
                dirnames[:] = []
                continue
            for fn in files:
                if fn.lower().endswith(".gguf"):
                    full = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(full) >= min_mb * 1024 * 1024:
                            hits.append(full)
                    except OSError:
                        pass
    return hits

def _slug(s):
    s = re.sub(r"\.gguf$", "", s, flags=re.I).lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"[^a-z0-9.\-]", "", s)
    return re.sub(r"-+", "-", s).strip("-")

def _base(p): return os.path.basename(p)
def _is_mmproj(p): return _base(p).lower().startswith("mmproj")
def _is_mtp(p):    return _base(p).lower().startswith("mtp-")
def _is_embed(p):  return "embed" in _base(p).lower()
def _shard(p):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", _base(p), re.I)
    return (m.group(1), m.group(2)) if m else None

def build_entries(paths):
    """Return list of {id, model, mmproj?, embeddings?, gib, existing_id?}."""
    mmproj_by_dir = {}
    mtp_by_dir = {}
    for p in paths:
        if _is_mmproj(p):
            mmproj_by_dir[os.path.dirname(p)] = p
        elif _is_mtp(p):
            mtp_by_dir[os.path.dirname(p)] = p

    mains = []
    seen_shard_sets = set()
    for p in paths:
        if _is_mmproj(p) or _is_mtp(p):
            continue
        sh = _shard(p)
        if sh:
            key = (os.path.dirname(p), re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", _base(p), flags=re.I))
            if key in seen_shard_sets or sh[0] != "00001":
                continue  # only the first shard represents the set
            seen_shard_sets.add(key)
        mains.append(p)

    def mk_id_map(items):
        groups = defaultdict(list)
        for p in items:
            groups[_slug(_base(p))].append(p)
        out = {}
        for base_id, paths in groups.items():
            if len(paths) == 1:
                out[paths[0]] = base_id
                continue
            parents_by_path = {}
            for p in sorted(paths):
                parents = []
                cur = os.path.dirname(p)
                while cur and cur not in (os.path.sep, ".", ""):
                    part = _slug(_base(cur))
                    if part:
                        parents.append(part)
                    nxt = os.path.dirname(cur)
                    if nxt == cur:
                        break
                    cur = nxt
                parents_by_path[p] = parents
            depths = {p: 1 for p in paths}
            while True:
                cand_to_paths = defaultdict(list)
                for p in paths:
                    parents = parents_by_path[p]
                    depth = min(depths[p], len(parents))
                    if depth <= 0:
                        cand = base_id
                    else:
                        cand = "--".join(list(reversed(parents[:depth])) + [base_id])
                    cand_to_paths[cand].append(p)
                clashes = [ps for ps in cand_to_paths.values() if len(ps) > 1]
                if not clashes:
                    for cand, ps in cand_to_paths.items():
                        out[ps[0]] = cand
                    break
                for ps in clashes:
                    for p in ps:
                        if depths[p] < len(parents_by_path[p]):
                            depths[p] += 1
                        else:
                            # Root-level duplicates with identical ancestry are
                            # practically impossible here, but keep forward
                            # progress deterministic if they ever happen.
                            depths[p] = len(parents_by_path[p])
        return out

    id_map = mk_id_map(mains)

    entries = []
    for p in sorted(mains):
        try:
            gib = round(os.path.getsize(p) / 1024**3, 2)
        except OSError:
            gib = 0
        e = {"id": id_map[p], "model": p.replace("\\", "/"), "gib": gib}
        # Only attach mmproj if the model is vision-capable.
        mm = mmproj_by_dir.get(os.path.dirname(p))
        if mm:
            from gguf import metadata
            arch = (metadata(p) or {}).get("architecture", "")
            if arch in _VISION_ARCHES:
                e["mmproj"] = mm.replace("\\", "/")
        # Attach an mtp-* sibling as a speculative draft model. Attaching alone
        # is inert; only enable spec-type=draft-mtp when the sidecar actually
        # declares NextN layers, the signal llama.cpp itself gates on.
        mt = mtp_by_dir.get(os.path.dirname(p))
        if mt:
            from gguf import has_nextn
            e["draft_model"] = mt.replace("\\", "/")
            if has_nextn(mt):
                e["draft_mtp"] = True
        if _is_embed(p):
            e["embeddings"] = True
        entries.append(e)
    return entries

def scan(roots=None, min_mb=50):
    roots = roots or list_drives()
    paths = find_ggufs(roots, min_mb)
    return build_entries(paths)

if __name__ == "__main__":
    import json, sys
    roots = sys.argv[1:] or None
    e = scan(roots)
    print(f"found {len(e)} models")
    print(json.dumps(e[:5], indent=2))
