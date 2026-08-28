"""Isolated foundations for the optional Auto Tune feature.

This package deliberately has no HTTP, UI, router, or models.ini dependency.
"""

from .models import (
    BenchBinaryIdentity,
    EnvironmentSnapshot,
    ExecutionEnvironment,
    FastFingerprint,
    HardwareIdentity,
    ModelSource,
    NormalizedGGUF,
    PhysicalGPU,
    TuneProfile,
)

__all__ = [
    "EnvironmentSnapshot",
    "HardwareIdentity",
    "ExecutionEnvironment",
    "BenchBinaryIdentity",
    "FastFingerprint",
    "ModelSource",
    "NormalizedGGUF",
    "PhysicalGPU",
    "TuneProfile",
]
