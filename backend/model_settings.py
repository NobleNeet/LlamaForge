"""Shared llama-server knob normalization."""


def _aliases(knob_schema):
    alias_to_key, key_to_aliases, knobs = {}, {}, {}
    for group in (knob_schema or {}).get("groups", ()):
        for knob in group.get("knobs", ()):
            key = knob.get("key")
            if not key:
                continue
            family = [key] + [alias for alias in knob.get("aliases", ()) if alias and alias != key]
            key_to_aliases[key] = family
            alias_to_key.update({alias: key for alias in family})
            knobs[key] = knob
    return alias_to_key, key_to_aliases, knobs


def _normalize_value(value, knob):
    if isinstance(value, (list, tuple)):
        sep = ","
        if isinstance(knob, dict) and knob.get("multiple"):
            sep = knob.get("separator") or ","
        return sep.join(str(part).strip() for part in value if str(part).strip())
    return "" if value is None else str(value)


def clean_settings(updates, knob_schema=None):
    """Canonicalize aliases; blank values retain the existing unset semantics."""
    alias_to_key, key_to_aliases, knobs = _aliases(knob_schema)
    clean = {}
    for raw_key, value in (updates or {}).items():
        key = alias_to_key.get(raw_key, raw_key)
        value = _normalize_value(value, knobs.get(key)).strip()
        clean[key] = None if value == "" else value
        if raw_key != key:
            for alias in key_to_aliases.get(key, ()):
                if alias != key and alias not in clean:
                    clean[alias] = None
    return clean
