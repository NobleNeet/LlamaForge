"""Safe model-level deletion of GGUF models from disk.

The user-facing unit is always "delete this model", never "delete this
directory". A directory may hold several quantizations of the same repo
(Discover's download layout keeps them side by side), so recursive removal
of a model's parent directory is never justified by parenthood alone.

This module turns a models.ini section into a deletion *plan* - the exact
file set that belongs to the selected model - and executes it with
os.remove/os.rmdir only. shutil.rmtree is deliberately absent: the plan is
computed from a directory listing, files are removed one by one, and the
containing directory is removed only when it has actually become empty.
Symlinks are unlinked, never followed, so a link cannot turn a delete into
a recursive removal of somewhere unintended.

Ownership rules (docs/content/models.md "Delete a model"):

- The model payload is its main GGUF plus, when sharded, every shard of
  the same shard set (same base name and same total count).
- An mmproj is deleted only when it is provably exclusive to this model:
  the section references it, it lives beside the model, no other section
  references it, and no other model payload remains in the directory.
  Anything ambiguous is preserved.
- The containing directory is removed if and only if it becomes empty.
"""
import os
import re

_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)

# Section keys that can point at a file on disk. A file referenced by any of
# these in *another* section is shared and must survive.
_PATH_KEYS = ("model", "mmproj", "spec-draft-model")


class DeleteError(Exception):
    """The deletion was refused. The message is user-presentable."""


class NotFound(DeleteError):
    """The model id has no recorded section/file to delete."""


def _is_gguf(name):
    return name.lower().endswith(".gguf")


def _same(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _expand(path):
    return os.path.expanduser((path or "").strip())


def _shard_match(name):
    m = _SHARD_RE.match(name)
    return (m.group("base").lower(), m.group("total")) if m else None


def _referenced_paths(sections):
    """Every file path referenced by any of the given sections."""
    out = set()
    for sect in sections:
        if not isinstance(sect, dict):
            continue
        for key in _PATH_KEYS:
            v = _expand(sect.get(key))
            if v:
                out.add(os.path.normcase(os.path.abspath(v)))
    return out


def plan(section, other_sections=()):
    """Compute the file set that belongs to one model.

    `section` is the model's models.ini section (needs `model`).
    `other_sections` is an iterable of every *other* model's section dict,
    used to detect shared mmproj references.

    Returns {"files": [...], "directory": str, "delete_directory": bool,
             "size_bytes": int, "mmproj": str or None}.
    Raises DeleteError when the request cannot be made safe.
    """
    if not isinstance(section, dict):
        raise DeleteError("no settings recorded for this model")
    mpath = _expand(section.get("model"))
    if not mpath:
        raise DeleteError("this model has no model file recorded")
    if not os.path.isabs(mpath):
        raise DeleteError(f"model path is not absolute: {mpath}")
    if not os.path.exists(mpath):
        raise DeleteError(f"model file not found: {mpath}")
    if os.path.isdir(mpath):
        raise DeleteError(f"model path is a directory, not a file: {mpath}")

    directory = os.path.dirname(mpath)
    if not directory or os.path.dirname(directory) == directory:
        raise DeleteError("refusing to treat a filesystem root as a model directory")
    if os.path.islink(directory):
        raise DeleteError(f"model directory is a symlink: {directory}")

    # Another entry pointing at this same file means it is shared; deleting
    # the file would break that model too.
    if os.path.normcase(os.path.abspath(mpath)) in _referenced_paths(other_sections):
        raise DeleteError("this model file is also referenced by another model entry")

    try:
        entries = sorted(os.listdir(directory))
    except OSError as e:
        raise DeleteError(f"cannot read model directory: {e}")

    files = [mpath]
    shard = _shard_match(os.path.basename(mpath))
    if shard:
        members = [os.path.join(directory, fn) for fn in entries
                   if _shard_match(fn) == shard]
        if members:
            files = sorted(set(files) | set(members))

    # mmproj: delete only when provably exclusive to this model.
    mmproj = None
    proj = _expand(section.get("mmproj"))
    if proj and os.path.exists(proj) and not os.path.isdir(proj) \
            and not os.path.islink(proj) \
            and os.path.normpath(os.path.dirname(proj)) == os.path.normpath(directory):
        referenced_elsewhere = any(
            _same(proj, _expand(s.get(key)))
            for s in other_sections if isinstance(s, dict)
            for key in _PATH_KEYS if _expand(s.get(key)))
        # A projector left beside other model payloads may belong to them.
        remaining_payload = [f for f in entries
                           if _is_gguf(f)
                           and not _same(os.path.join(directory, f), proj)
                           and not any(_same(os.path.join(directory, f), m) for m in files)]
        if not referenced_elsewhere and not remaining_payload:
            mmproj = proj
            files.append(proj)

    covered = {os.path.normcase(os.path.abspath(f)) for f in files}
    leftover = [fn for fn in entries
               if os.path.normcase(os.path.abspath(os.path.join(directory, fn))) not in covered]
    delete_directory = not leftover

    size = 0
    for f in files:
        try:
            size += os.path.getsize(f)
        except OSError:
            pass
    return {"files": files, "directory": directory,
            "delete_directory": delete_directory,
            "size_bytes": size, "mmproj": mmproj}


def execute(p):
    """Run a plan from plan(). Returns (ok, error).

    Files are unlinked individually; the directory is removed only when the
    plan says it becomes empty and os.rmdir confirms it actually is.
    """
    errors = []
    for f in p["files"]:
        try:
            if os.path.isdir(f) and not os.path.islink(f):
                raise DeleteError(f"refusing to remove a directory as a file: {f}")
            os.remove(f)
        except FileNotFoundError:
            pass
        except OSError as e:
            errors.append(f"{f}: {e.strerror or e}")
    if p.get("delete_directory") and not errors:
        d = p["directory"]
        try:
            if os.path.islink(d):
                raise DeleteError(f"model directory became a symlink: {d}")
            os.rmdir(d)          # fails loudly if something is still inside
        except FileNotFoundError:
            pass
        except OSError as e:
            errors.append(f"{d}: {e.strerror or e}")
    if errors:
        return False, "; ".join(errors)
    return True, ""
