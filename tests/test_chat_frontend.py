"""Run pure protocol and real UI handlers with Node's DOM/network fixtures."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required')
class ChatFrontendTest(unittest.TestCase):
    def test_protocol_and_ui(self):
        run = subprocess.run(['node', str(Path(__file__).with_name('chat_frontend_checks.mjs'))],
                             capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__': unittest.main()
