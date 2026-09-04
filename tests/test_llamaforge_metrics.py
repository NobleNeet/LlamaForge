"""Focused tests for the standalone LlamaForge metrics collector."""

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "llamaforge_metrics.py"
SPEC = importlib.util.spec_from_file_location("llamaforge_metrics", MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def test_normalize_cli_params_keeps_all_flags_and_types_values():
    raw = {
        "--host": "127.0.0.1",
        "--jinja": True,
        "--port": "56323",
        "--model": "/models/example.gguf",
        "--alias": "123",
        "--flash-attn": "on",
        "--new-option": "0.5",
        "--spec-type": "draft-mtp",
        "--repeat": ["3", "false"],
    }

    assert metrics._normalize_cli_params(raw) == {
        "host": "127.0.0.1",
        "jinja": True,
        "port": 56323,
        "model": "/models/example.gguf",
        "alias": "123",
        "flash_attn": True,
        "new_option": 0.5,
        "spec_type": ["draft-mtp"],
        "repeat": [3, False],
    }


def test_session_parameters_types_runtime_kv_cache_boolean(tmp_path):
    collector = metrics.Collector("stdout.log", "stderr.log", tmp_path,
                                  metrics.ZoneInfo("UTC"))
    session = metrics.LoadSession("session", 1, None, None, None,
                                  kv_unified="true")

    assert collector.session_parameters(session) == {"kv_cache": True}
