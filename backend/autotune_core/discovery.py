"""Model-source enumeration independent of the registered model list."""
import os

from .models import ModelSource


def registered_model_source(model_id, path):
    return ModelSource(path=os.path.abspath(path), source_kind="registered", registered_model_id=model_id)


def discover_gguf_sources(model_dir):
    out = []
    if not os.path.isdir(model_dir):
        return out
    for root, dirs, files in os.walk(model_dir):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith(".gguf"):
                out.append(ModelSource(path=os.path.abspath(os.path.join(root, name)), source_kind="discovered"))
    return out


def merge_model_sources(registered, discovered):
    """Prefer registered identity while exposing each resolved GGUF once."""
    by_path = {}
    for source in list(discovered) + list(registered):
        by_path[os.path.normcase(os.path.abspath(source.path))] = source
    return [by_path[key] for key in sorted(by_path)]
