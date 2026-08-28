"""Isolated foundations for the optional Auto Tune feature.

This package deliberately has no HTTP, UI, router, or models.ini dependency.
"""

from .models import (
    EnvironmentSnapshot,
    FastFingerprint,
    ModelSource,
    NormalizedGGUF,
    PhysicalGPU,
    TuneProfile,
)

__all__ = [
    "EnvironmentSnapshot",
    "FastFingerprint",
    "ModelSource",
    "NormalizedGGUF",
    "PhysicalGPU",
    "TuneProfile",
]
