import conftest_paths  # noqa: F401
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import gguf
from test_gguf_metadata import _kv_str, _kv_u32, _s


def write_model(path, entries, split_count=1):
    kv = [_kv_str('general.architecture', 'llama'), _kv_u32('llama.block_count', 2),
          _kv_u32('llama.attention.head_count_kv', 4), _kv_u32('llama.attention.key_length', 128),
          _kv_u32('split.count', split_count)]
    header = b'GGUF' + struct.pack('<IQQ', 3, len(entries), len(kv)) + b''.join(kv)
    offset = 0
    for name, size in entries:
        header += _s(name) + struct.pack('<I', 1) + struct.pack('<QIQ', size // 2, 1, offset)
        offset += size
    with open(path, 'wb') as f:
        f.write(header)
        f.write(b'\0' * (-len(header) % 32))
        # Sparse payload: the reader must only walk the directory.
        if offset:
            f.seek(offset - 1, 1); f.write(b'\0')


class GeometryInspectionTests(unittest.TestCase):
    def test_tensor_spans_and_parameter_count_without_payload_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.gguf'
            write_model(path, [('token_embd.weight', 64), ('blk.0.attn.weight', 128), ('blk.1.ffn.weight', 256)])
            real_open = open
            class HeaderOnly:
                def __init__(self, *a, **kw): self.f = real_open(*a, **kw)
                def __enter__(self): return self
                def __exit__(self, *a): self.f.close()
                def read(self, size):
                    if self.f.tell() >= os.path.getsize(path) - 448:
                        raise AssertionError('tensor payload was read')
                    return self.f.read(size)
                def seek(self, *a): return self.f.seek(*a)
                def tell(self): return self.f.tell()
            with mock.patch('builtins.open', HeaderOnly):
                m = gguf.inspect_model(path)
            self.assertEqual(m['layer_weight_bytes'], [128, 256])
            self.assertEqual(m['non_layer_weight_bytes'], 64)
            self.assertEqual(m['parameter_count'], 224)
            self.assertEqual(m['head_count_kv'], 4)
            self.assertEqual(m['key_length'], 128)

    def test_split_files_aggregate_and_missing_shard_is_not_sized_as_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / 'm-00001-of-00002.gguf'
            second = Path(tmp) / 'm-00002-of-00002.gguf'
            write_model(first, [('blk.0.weight', 128)], 2)
            write_model(second, [('blk.1.weight', 256)], 2)
            m = gguf.inspect_model(first)
            self.assertEqual(m['layer_weight_bytes'], [128, 256])
            self.assertEqual(m['file_size_bytes'], first.stat().st_size + second.stat().st_size)
            second.unlink()
            self.assertTrue(gguf.inspect_model(first)['incomplete_shards'])
            self.assertNotIn('file_size_bytes', gguf.inspect_model(first))

    def test_truncated_directory_and_unreadable_file_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'broken.gguf'
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 1, 0))
            self.assertTrue(gguf.inspect_model(path)['inspection_incomplete'])
            self.assertTrue(gguf.inspect_model(str(path) + '.missing')['inspection_incomplete'])
