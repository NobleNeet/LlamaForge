"""Exercise real Build handlers and Models refresh/editor reconciliation in Node."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class RuntimeSchemaFrontendTests(unittest.TestCase):
    def test_switch_refresh_poll_preservation_and_stale_response(self):
        result = subprocess.run(['node', str(Path(__file__).with_name('runtime_schema_frontend_checks.mjs'))],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
