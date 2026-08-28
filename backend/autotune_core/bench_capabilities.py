"""Capability probes for arbitrary llama-bench binaries, isolated from execution."""
from dataclasses import dataclass, field
import hashlib
import json
import subprocess


class CapabilityProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class BinaryCapabilities:
    flags: frozenset = field(default_factory=frozenset)
    output_formats: frozenset = field(default_factory=frozenset)
    repetitions: bool = False
    context_depth: bool = False
    fingerprint: str = ""

    def structured_output_format(self):
        if "jsonl" in self.output_formats:
            return "jsonl"
        if "json" in self.output_formats:
            return "json"
        raise CapabilityProbeError("llama-bench has no supported structured output format")

    def supports_flag(self, flag):
        return flag in self.flags


@dataclass(frozen=True)
class RuntimeCapabilities:
    backend: str
    devices: tuple = ()
    source: str = "list-devices"


def _fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256-cap-v1:" + hashlib.sha256(encoded).hexdigest()


def parse_binary_capabilities(help_text):
    if not isinstance(help_text, str):
        raise CapabilityProbeError("llama-bench help output is not text")
    flags = frozenset(token for token in help_text.replace("=", " ").split() if token.startswith("-"))
    formats = set()
    lowered = help_text.lower()
    if "jsonl" in lowered:
        formats.add("jsonl")
    if "json" in lowered:
        formats.add("json")
    result = BinaryCapabilities(flags, frozenset(formats), "--repetitions" in flags,
                                "--n-depth" in flags, "")
    return BinaryCapabilities(result.flags, result.output_formats, result.repetitions, result.context_depth,
                              _fingerprint({"flags": sorted(result.flags), "formats": sorted(result.output_formats),
                                            "repetitions": result.repetitions, "depth": result.context_depth}))


def probe_binary_capabilities(binary_path, timeout_seconds=5, runner=subprocess.run):
    try:
        completed = runner([binary_path, "--help"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CapabilityProbeError("llama-bench capability probe failed: %s" % exc) from exc
    if completed.returncode != 0:
        raise CapabilityProbeError("llama-bench --help failed: %s" % (completed.stderr or completed.returncode))
    return parse_binary_capabilities(completed.stdout)


def probe_runtime_capabilities(binary_path, backend, timeout_seconds=5, runner=subprocess.run):
    """Use runtime enumeration; help text must never be treated as device availability."""
    try:
        completed = runner([binary_path, "--list-devices"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CapabilityProbeError("llama-bench device probe failed: %s" % exc) from exc
    if completed.returncode != 0:
        raise CapabilityProbeError("llama-bench --list-devices failed: %s" % (completed.stderr or completed.returncode))
    devices = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    return RuntimeCapabilities(backend, devices)
