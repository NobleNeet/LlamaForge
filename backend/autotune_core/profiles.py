"""Pure settings precedence for future model/profile integration."""


def resolve_effective_settings(defaults, selected_profile=None, explicit_knobs=None):
    """Resolve defaults < selected profile < explicit user knobs without mutation.

    Calling code must keep the selected profile as a reference.  Copying profile
    settings into models.ini is intentionally outside this function and requires
    a separate explicit user action.
    """
    resolved = dict(defaults or {})
    if selected_profile is not None:
        resolved.update(selected_profile.settings)
    resolved.update(explicit_knobs or {})
    return resolved
