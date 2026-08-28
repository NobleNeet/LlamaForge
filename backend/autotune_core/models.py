"""Data models shared by future Auto Tune planning and benchmark phases."""
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class ModelSource:
    path: str
    source_kind: str  # "registered" or "discovered"
    registered_model_id: Optional[str] = None


@dataclass(frozen=True)
class FastFingerprint:
    """A bounded-cost content fingerprint; strong_sha256 is reserved for opt-in use."""
    algorithm: str
    value: str
    file_size: int
    sample_bytes: int
    strong_sha256: Optional[str] = None


@dataclass(frozen=True)
class NormalizedGGUF:
    path: str
    architecture: Optional[str]
    name: Optional[str]
    quantization: Optional[str]
    context_length: Optional[int]
    embedding_length: Optional[int]
    block_count: Optional[int]
    attention_head_count: Optional[int]
    attention_head_count_kv: Optional[int]
    sliding_window: Optional[int]
    expert_count: Optional[int]
    expert_used_count: Optional[int]
    nextn_predict_layers: Optional[int]
    has_nextn: bool
    tensor_type_summary: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalGPU:
    stable_id: str
    vendor: str
    name: str
    memory_bytes: Optional[int]
    architecture: Optional[str]
    is_uma: bool
    available_backends: Tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentSnapshot:
    platform: str
    cpu: Mapping[str, object]
    physical_gpus: Tuple[PhysicalGPU, ...]
    available_backends: Tuple[str, ...]
    captured_at: str


@dataclass(frozen=True)
class TuneProfile:
    """A result reference, never an implicit replacement for models.ini knobs."""
    profile_id: str
    model_fingerprint: str
    environment_fingerprint: str
    settings: Mapping[str, object]
    result_id: Optional[str] = None
