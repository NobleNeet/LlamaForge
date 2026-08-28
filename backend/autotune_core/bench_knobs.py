"""The only translation boundary between Auto Tune settings and llama-bench."""
from dataclasses import dataclass


class UnsupportedBenchKnobError(ValueError):
    pass


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError("%s must be an integer" % name)
    try:
        return str(int(value))
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer" % name)


def _choice(value, name, choices):
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError("%s must be one of %s" % (name, ", ".join(sorted(choices))))
    return normalized


def _gpu_layers(value):
    if isinstance(value, str) and value.strip().lower() == "all":
        return "-1"
    return _integer(value, "n-gpu-layers")


def _device(value):
    value = str(value).strip()
    if not value or "\x00" in value:
        raise ValueError("device must be a non-empty device selector")
    return value


def _tensor_split(value):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("tensor-split must not be empty")
        return "/".join(str(float(item)) for item in value)
    value = str(value).strip().replace(",", "/")
    if not value or any(not part.replace(".", "", 1).isdigit() for part in value.split("/")):
        raise ValueError("tensor-split must contain numeric fractions")
    return value


def _boolean(value, name):
    if isinstance(value, bool):
        return "1" if value else "0"
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "on", "yes"):
        return "1"
    if normalized in ("0", "false", "off", "no"):
        return "0"
    raise ValueError("%s must be boolean" % name)


@dataclass(frozen=True)
class BenchKnobSpec:
    setting: str
    flag: str
    encoder: object

    def encode(self, value):
        return self.flag, self.encoder(value)


_SPECS = (
    BenchKnobSpec("n-gpu-layers", "--n-gpu-layers", _gpu_layers),
    BenchKnobSpec("threads", "--threads", lambda value: _integer(value, "threads")),
    BenchKnobSpec("batch-size", "--batch-size", lambda value: _integer(value, "batch-size")),
    BenchKnobSpec("ubatch-size", "--ubatch-size", lambda value: _integer(value, "ubatch-size")),
    BenchKnobSpec("cache-type-k", "--cache-type-k", lambda value: str(value).strip().lower()),
    BenchKnobSpec("cache-type-v", "--cache-type-v", lambda value: str(value).strip().lower()),
    BenchKnobSpec("flash-attn", "--flash-attn", lambda value: _choice(value, "flash-attn", {"on", "off", "auto"})),
    BenchKnobSpec("device", "--device", _device),
    BenchKnobSpec("tensor-split", "--tensor-split", _tensor_split),
    BenchKnobSpec("main-gpu", "--main-gpu", lambda value: _integer(value, "main-gpu")),
    BenchKnobSpec("split-mode", "--split-mode", lambda value: _choice(value, "split-mode", {"none", "layer", "row", "tensor"})),
    BenchKnobSpec("no-kv-offload", "--no-kv-offload", lambda value: _boolean(value, "no-kv-offload")),
    BenchKnobSpec("poll", "--poll", lambda value: _integer(value, "poll")),
)
_BY_SETTING = {spec.setting: spec for spec in _SPECS}


def encode_bench_settings(settings):
    argv = []
    for key in sorted((settings or {}).keys()):
        spec = _BY_SETTING.get(key)
        if spec is None:
            raise UnsupportedBenchKnobError("unsupported llama-bench setting: %s" % key)
        flag, encoded = spec.encode(settings[key])
        if not encoded:
            raise ValueError("%s has no value" % key)
        argv.extend((flag, encoded))
    return tuple(argv)
