"""Introspect every llama.cpp server argument from `llama-server --help`.

Produces a categorized, typed schema that the UI renders as "all knobs".
Because it is generated from the binary, it stays correct across llama.cpp
versions automatically.
"""
import re, subprocess

# args the router owns / that must not be set per-model in the panel
RESERVED = {
    "help", "usage", "version", "completion-bash", "cache-list",
    "host", "port", "api-key", "api-key-file", "alias", "model", "mmproj",
    "hf-repo", "hf-repo-draft", "hf-repo-v", "hf-file", "hf-token",
    "models-dir", "models-preset", "models-max", "models-autoload",
    "no-models-autoload", "ssl-key-file", "ssl-cert-file", "path",
}

SECTION_RE = re.compile(r"^-+\s*(.+?)\s*-+\s*$")

# llama.cpp occasionally changes which long alias it prints first in --help
# (for example --gpu-layers vs --n-gpu-layers). The dashboard persists knobs by
# key name, so the UI must keep a stable canonical key across upstream churn.
PREFERRED_LONGS = {
    "gpu-layers": "n-gpu-layers",
}


def _canonical_long(longs):
    for long_ in longs:
        pref = PREFERRED_LONGS.get(long_)
        if pref and pref in longs:
            return pref
    return longs[0]

def _balance_parens(s):
    """Drop orphan ')' left over after trimming a '(default:/env:...)' tail.
    Upstream --help text (and our own truncation) can leave a dangling ')'
    with no matching '(' - it shows up as a stray ')' in the UI, so strip it."""
    out, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue            # orphan close paren -> drop
            depth -= 1
        out.append(ch)
    return "".join(out).strip()

# curated types/options for common knobs (help text lacks enum values for these)
OVERRIDES = {
    "cache-type-k": ("enum", ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]),
    "cache-type-v": ("enum", ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]),
    "spec-type":    ("enum", ["none", "draft-simple", "draft-eagle3", "draft-mtp",
                               "ngram-simple", "ngram-map-k", "ngram-map-k4v",
                               "ngram-mod", "ngram-cache"]),
    "tensor-split": ("str", None),
    "override-tensor": ("str", None),
    "cpu-range":    ("str", None),
    "cpu-range-batch": ("str", None),
}

def _classify(placeholder, default):
    """Return (type, options) for a value placeholder."""
    p = (placeholder or "").strip()
    if not p:
        return "bool", None
    # enum: bracketed [a|b|c] / <0|1> / {none,mean,cls}  OR bare word list a,b,c
    m = re.search(r"[\[<{]([^\]>}]*[|,][^\]>}]*)[\]>}]", p)
    body = m.group(1) if m else (p if ("," in p or "|" in p) else "")
    if body and "..." not in body:
        opts = [o.strip() for o in re.split(r"[|,]", body) if o.strip()]
        # numeric placeholder list (N0,N1,...) is a free string, not an enum
        if opts and not any(re.match(r"^[NM]\d*$", o) for o in opts):
            return "enum", opts
    if re.fullmatch(r"[NM]", p) or re.search(r"<[\d.\s\-]+\.\.\.?[\d.\s]*>", p):
        return ("float" if re.search(r"\d\.\d", default or "") else "int"), None
    if p in ("FNAME", "PATH", "FILE"):
        return "path", None
    return "str", None


def _preset_supported(placeholder):
    """Whether llama.cpp's models.ini can express this option as one key=value.

    Router presets currently reject options that require multiple positional
    values, e.g. `--control-vector-layer-range START END`.
    """
    p = (placeholder or "").strip()
    if not p:
        return True
    return not re.fullmatch(r"[A-Z][A-Z0-9_-]*(\s+[A-Z][A-Z0-9_-]*)+", p)

def parse_help(text):
    section = "general"
    items, pending = [], None

    def flush():
        nonlocal pending
        if pending:
            items.append(pending); pending = None

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not line.strip():
            continue
        sm = SECTION_RE.match(stripped)
        if sm and "params" in sm.group(1).lower() or (sm and len(sm.group(1)) < 40 and not raw.startswith((" ", "\t"))):
            flush(); section = sm.group(1).strip(); continue

        # help output wraps long descriptions/default notes onto indented lines.
        # Treat any indented line as continuation; some wrapped tails begin with
        # "--foo)" and were previously mistaken for brand new flags.
        if pending and raw.startswith((" ", "\t")):
            extra = stripped
            env = re.search(r"\(env:\s*([A-Z0-9_]+)\)", extra)
            if env: pending["env"] = env.group(1)
            dflt = re.search(r"\(default:\s*(.*?)\)", extra)
            if dflt and not pending.get("default"): pending["default"] = dflt.group(1)
            if extra and not extra.startswith("(env:") and not extra.startswith("(default:"):
                desc = pending.get("desc", "")
                if desc:
                    pending["desc"] = (desc + " " + extra).strip()
                else:
                    pending["desc"] = extra
            continue

        # Real option definitions are left-aligned in llama.cpp --help.
        if not raw.startswith("-"):
            continue
        # column-aligned help: split on runs of 2+ spaces
        parts = re.split(r"\s{2,}", line.strip())
        flag_parts, desc_parts = [], []
        for p in parts:
            if p.startswith("-") and not desc_parts:
                flag_parts.append(p)
            else:
                desc_parts.append(p)
        if not flag_parts:
            continue
        flush()
        flags, placeholder = [], ""
        for fp in flag_parts:
            toks = fp.split()
            j = 0
            while j < len(toks) and toks[j].rstrip(",").startswith("-"):
                flags.append(toks[j].rstrip(",")); j += 1
            if j < len(toks):
                placeholder = " ".join(toks[j:])
        longs = [f[2:] for f in flags if f.startswith("--")]
        if not longs:
            continue
        key = _canonical_long(longs)
        desc = "  ".join(desc_parts)
        env = re.search(r"\(env:\s*([A-Z0-9_]+)\)", desc)
        dflt = re.search(r"\(default:\s*(.*?)\)", desc)
        default_val = dflt.group(1) if dflt else ""
        typ, opts = OVERRIDES.get(key, _classify(placeholder, default_val))
        # canonical default: the value before any ", explanation" tail
        clean_default = re.split(r",\s", default_val)[0].strip() if default_val else ""
        pending = {
            "key": key,
            "flags": flags,
            "aliases": [key] + [a for a in longs if a != key],
            "section": section,
            "type": typ, "options": opts,
            "placeholder": placeholder,
            "desc": _balance_parens(re.sub(r"\s*\((env|default):.*", "", desc)),
            "default": clean_default,
            "env": env.group(1) if env else "",
            "reserved": any(l in RESERVED for l in longs) or not _preset_supported(placeholder),
        }
    flush()
    return items

def build_schema(server_bin):
    """Run the server's --help and return grouped, editable knobs."""
    if not server_bin:
        return {"error": "server_bin is not set in config.json", "groups": []}
    out, err = _help_text(server_bin)
    if err:
        return {"error": err, "groups": []}
    items = [i for i in parse_help(out) if not i["reserved"]]
    if not items:
        return {"error": "could not parse any arguments from --help output "
                         "(unrecognized help format?)", "groups": []}
    groups = {}
    for it in items:
        groups.setdefault(it["section"], []).append(it)
    ordered = [{"name": k, "knobs": v} for k, v in groups.items()]
    return {"groups": ordered, "count": len(items)}


def _help_text(server_bin):
    """Run the server's --help and return (text, error_message)."""
    if not server_bin:
        return "", "server_bin is not set in config.json"
    try:
        # utf-8 explicitly: text=True alone decodes with the locale codepage on
        # Windows (cp1252), which can blow up or mangle upstream help text.
        r = subprocess.run([server_bin, "--help"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=20)
    except Exception as e:
        return "", str(e)
    # some builds/forks route usage through the log system -> stderr
    out = r.stdout if r.stdout.strip() else r.stderr
    if not out.strip():
        return "", (f"`{server_bin} --help` produced no output "
                    f"(exit code {r.returncode}) - missing DLLs or wrong binary?")
    return out, ""


def build_key_aliases(server_bin):
    """Return the current valid keys and alias->canonical map for models.ini."""
    out, err = _help_text(server_bin)
    if err:
        return {"error": err, "keys": set(), "alias_to_key": {}}
    alias_to_key, keys = {}, set()
    for it in parse_help(out):
        if it.get("reserved"):
            continue
        key = it["key"]
        keys.add(key)
        for alias in it.get("aliases", []):
            alias_to_key[alias] = key
            keys.add(alias)
    return {"keys": keys, "alias_to_key": alias_to_key}

if __name__ == "__main__":
    import json, sys
    print(json.dumps(build_schema(sys.argv[1]), indent=2)[:3000])
