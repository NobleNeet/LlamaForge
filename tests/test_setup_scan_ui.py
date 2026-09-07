"""Run Setup's real scan handler with a small DOM/API fixture."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node.js required for Setup UI tests")
class SetupScanUiTest(unittest.TestCase):
    def run_scan(self, response, known, apply=False):
        script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(input.source, 'utf8')
  .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '');
const nodes = {}, calls = [], events = [];
const context = vm.createContext({
  $: selector => nodes[selector] ||= {},
  $$: () => [],
  esc: value => String(value).replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
  setHTML: (node, html) => { node.innerHTML = html; },
  models: () => input.known,
  api: async (path, body) => {
    calls.push({path, body});
    if (path === '/api/scan') return input.response;
    if (path === '/api/scan/apply') return {added: body.entries.length};
    throw new Error('Unexpected request: ' + path);
  },
  emit: (...args) => events.push(args),
  toast: () => {},
});
vm.runInContext(source, context);
(async () => {
  await vm.runInContext('scanDrives()', context);
  if (input.apply) await nodes['#btn-apply'].onclick();
  process.stdout.write(JSON.stringify({nodes, calls, events}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
        source = Path(__file__).resolve().parents[1] / "web/js/setup.js"
        run = subprocess.run(
            [shutil.which("node"), "-e", script],
            input=json.dumps({"source": str(source), "response": response,
                              "known": known, "apply": apply}),
            capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        return json.loads(run.stdout)

    def test_existing_only_scan_reports_repair_without_apply_button(self):
        mid = "gemma-4-31b-it-qat-q4-0"
        out = self.run_scan({"entries": [{"id": mid, "mmproj": "/m/mmproj.gguf"}],
                             "updated": [mid]}, [{"id": mid}])
        self.assertEqual([c["path"] for c in out["calls"]], ["/api/scan"])
        self.assertIn("0 new, 1 updated", out["nodes"]["#scan-msg"]["textContent"])
        html = out["nodes"]["#scan-out"]["innerHTML"]
        self.assertIn(mid, html)
        self.assertIn("Unload and reload", html)
        self.assertNotIn('id="btn-apply"', html)
        self.assertIn(["refresh", True], out["events"])

    def test_new_models_still_use_apply_and_keep_projector(self):
        entry = {"id": "new", "model": "/m/model.gguf", "mmproj": "/m/mmproj.gguf"}
        out = self.run_scan({"entries": [entry], "updated": []}, [], apply=True)
        self.assertEqual(out["calls"][1], {"path": "/api/scan/apply", "body": {"entries": [entry]}})

    def test_response_without_updates_remains_supported(self):
        out = self.run_scan({"entries": []}, [])
        self.assertIn("0 updated", out["nodes"]["#scan-msg"]["textContent"])
        self.assertEqual(out["events"], [])
