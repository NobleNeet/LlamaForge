import conftest_paths  # noqa: F401
import os
import struct
import tempfile
import unittest

from autotune_core.gguf_normalize import fast_fingerprint, normalize_gguf


def _string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _u32(key, value):
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _text(key, value):
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _tensor(name, tensor_type):
    return _string(name) + struct.pack("<I", 2) + struct.pack("<QQ", 4, 4) + struct.pack("<IQ", tensor_type, 0)


def _write_gguf(path, kvs, tensors=()):
    with open(path, "wb") as handle:
        handle.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
        handle.write(b"".join(kvs))
        handle.write(b"".join(tensors))


class TestGGUFNormalization(unittest.TestCase):
    def test_normalizes_rule_and_mtp_fields_without_changing_gguf_api(self):
        path = os.path.join(tempfile.mkdtemp(), "model.gguf")
        _write_gguf(path, [
            _text("general.architecture", "llama"),
            _text("general.name", "Test"),
            _u32("general.file_type", 15),
            _u32("llama.context_length", 8192),
            _u32("llama.attention.head_count", 32),
            _u32("llama.attention.head_count_kv", 8),
            _u32("llama.attention.sliding_window", 4096),
            _u32("llama.expert_count", 64),
            _u32("llama.expert_used_count", 6),
            _u32("llama.nextn_predict_layers", 2),
        ], [_tensor("blk.0.attn_q.weight", 1), _tensor("blk.0.ffn.weight", 12), _tensor("blk.1.ffn.weight", 12)])
        model = normalize_gguf(path)
        self.assertEqual(model.attention_head_count, 32)
        self.assertEqual(model.attention_head_count_kv, 8)
        self.assertEqual(model.sliding_window, 4096)
        self.assertEqual(model.expert_used_count, 6)
        self.assertEqual(model.nextn_predict_layers, 2)
        self.assertTrue(model.has_nextn)
        self.assertEqual(model.tensor_type_summary, {"F16": 1, "Q4_K": 2})

    def test_missing_optional_fields_degrade_to_none_or_empty(self):
        path = os.path.join(tempfile.mkdtemp(), "model.gguf")
        _write_gguf(path, [_text("general.architecture", "gemma")])
        model = normalize_gguf(path)
        self.assertIsNone(model.attention_head_count_kv)
        self.assertIsNone(model.sliding_window)
        self.assertFalse(model.has_nextn)
        self.assertEqual(model.tensor_type_summary, {})

    def test_fast_fingerprint_is_path_independent_and_detects_sampled_content_changes(self):
        directory = tempfile.mkdtemp()
        first, second = os.path.join(directory, "one.gguf"), os.path.join(directory, "two.gguf")
        payload = bytearray(b"A" * 300000)
        with open(first, "wb") as handle:
            handle.write(payload)
        with open(second, "wb") as handle:
            handle.write(payload)
        original = fast_fingerprint(first, 4096)
        self.assertEqual(original.value, fast_fingerprint(second, 4096).value)
        with open(second, "r+b") as handle:
            # The v1 fast fingerprint samples deterministic third-points.
            handle.seek(100000)
            handle.write(b"changed")
        changed = fast_fingerprint(second, 4096)
        self.assertNotEqual(original.value, changed.value)
        self.assertEqual(original.algorithm, "sha256-sampled-v1")
        self.assertIsNone(original.strong_sha256)
