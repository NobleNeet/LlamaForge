"""Shared llama-server knob normalization with an Auto Tune-specific adapter."""


def _aliases(knob_schema):
    alias_to_key, key_to_aliases = {}, {}
    for group in (knob_schema or {}).get("groups", ()):
        for knob in group.get("knobs", ()):
            key = knob.get("key")
            if not key:
                continue
            family = [key] + [alias for alias in knob.get("aliases", ()) if alias and alias != key]
            key_to_aliases[key] = family
            alias_to_key.update({alias: key for alias in family})
    return alias_to_key, key_to_aliases


def clean_settings(updates, knob_schema=None):
    """Canonicalize aliases; blank values retain the existing unset semantics."""
    alias_to_key, key_to_aliases = _aliases(knob_schema)
    clean = {}
    for raw_key, value in (updates or {}).items():
        key = alias_to_key.get(raw_key, raw_key)
        value = ("" if value is None else str(value)).strip()
        clean[key] = None if value == "" else value
        for alias in key_to_aliases.get(key, ()):
            if alias != key and alias not in clean:
                clean[alias] = None
    return clean


def force_max_gpu_layers(clean):
    """Legacy save-path compatibility: preserve the historic all-or-nothing rule."""
    out = dict(clean or {})
    if out.get("n-gpu-layers") != "0":
        out["n-gpu-layers"] = "99"
    return out


def materialize_autotune_settings(settings, knob_schema=None):
    """Convert profile values to llama-server editor values without legacy coercion."""
    alias_to_key, known_aliases = _aliases(knob_schema)
    known = set(known_aliases) if known_aliases else None
    values, warnings = {}, []
    for raw_key, raw_value in (settings or {}).items():
        key = alias_to_key.get(raw_key, raw_key)
        if known is not None and key not in known:
            warnings.append({"key": str(raw_key), "code": "unsupported_knob",
                             "message": "This llama-server build does not support the recommended knob."})
            continue
        value = str(raw_value).strip()
        if key == "n-gpu-layers" and value.lower() == "all":
            value = "99"
        values[key] = value
    return {"settings": values, "warnings": warnings, "applicable": not warnings}
