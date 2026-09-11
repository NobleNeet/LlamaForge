"""Exercise the Build card's real save handler and status polling in Node."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node.js required for Build UI tests")
class BuildScheduleUiTest(unittest.TestCase):
    def render_and_save(self, response, action="schedule"):
        script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(input.source, 'utf8')
  .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '');
const nodes = {}, calls = [], saves = [];
const context = vm.createContext({
  $: selector => nodes[selector] ||= {style:{}},
  esc: value => String(value),
  setHTML: (node, html) => { node.innerHTML = html; },
  localStorage: {getItem: () => 'ikllama', setItem: () => {}},
  setInterval: () => 1, clearInterval: () => {},
  toast: () => {}, agoText: () => '', fmtDur: () => '',
  api: async (path, body) => {
    calls.push({path, body});
    if (path.startsWith('/api/build/info')) return {};
    if (path === '/api/state') return {config: {llama_backend:'vulkan', build_auto_update_enabled:true, build_auto_update_time:'04:15'}};
    if (path.startsWith('/api/vllm/version')) return {error:'unsupported'};
    if (path.startsWith('/api/build/log')) return {schedule: {last_date:'2026-09-09', status:'Skipped: busy', timezone:'JST (UTC+0900)'}};
    if (path === '/api/config') {
      saves.push({buildDisabled:nodes['#btn-build'].disabled, backendDisabled:nodes['#build-backend'].disabled});
      return input.response;
    }
    throw Error('Unexpected request: ' + path);
  },
});
vm.runInContext(source, context);
(async () => {
  await vm.runInContext('loadBuild()', context);
  if (input.action === 'backend') {
    nodes['#build-backend'].value = 'hip';
    await nodes['#build-backend'].onchange();
  } else {
    nodes['#build-auto-time'] = {value: '05:30'};
    nodes['#build-auto-time'].reportValidity = () => true;
    nodes['#build-auto-enabled'] = {checked: false};
    await nodes['#btn-save-build-schedule'].onclick();
  }
  process.stdout.write(JSON.stringify({nodes, calls, saves}));
})().catch(error => {console.error(error); process.exitCode=1;});
"""
        run = subprocess.run([shutil.which("node"), "-e", script],
                             input=json.dumps({"source": str(Path(__file__).resolve().parents[1] / "web/js/build.js"),
                                               "response": response, "action": action}),
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        return json.loads(run.stdout)

    def test_persisted_card_saves_schedule_and_displays_server_timezone(self):
        result = self.render_and_save({"ok": True})
        nodes = result["nodes"]
        html = nodes["#view-build"]["innerHTML"]
        self.assertIn('id="build-auto-enabled" checked', html)
        self.assertIn('value="04:15"', html)
        self.assertIn('Automatic Update · llama.cpp', html)
        self.assertEqual(nodes["#build-schedule-timezone"]["textContent"], 'JST (UTC+0900)')
        self.assertIn('Skipped: busy', nodes["#build-schedule-status"]["textContent"])
        self.assertEqual(result["calls"][-1], {"path": "/api/config", "body": {
            "build_auto_update_enabled": False, "build_auto_update_time": "05:30"}})
        self.assertEqual(nodes["#build-schedule-msg"]["textContent"], 'Schedule saved')
        self.assertFalse(nodes["#btn-save-build-schedule"]["disabled"])

    def test_partial_rejection_is_not_reported_as_saved(self):
        result = self.render_and_save({"ok": True, "rejected": ["build_auto_update_time"]})
        self.assertEqual(result["nodes"]["#build-schedule-msg"]["className"], 'msg err')

    def test_backend_save_blocks_rebuild_until_selection_is_saved(self):
        result = self.render_and_save({"ok": True}, action="backend")
        self.assertEqual(result["saves"], [{"buildDisabled": True, "backendDisabled": True}])
        self.assertIn({"path": "/api/config", "body": {"llama_backend": "hip"}}, result["calls"])
        self.assertEqual(sum(call["path"].startswith("/api/build/info") for call in result["calls"]), 2)
        self.assertFalse(result["nodes"]["#btn-build"]["disabled"])

    def test_rejected_backend_restores_previous_selection(self):
        result = self.render_and_save({"ok": True, "rejected": ["llama_backend"]}, action="backend")
        self.assertEqual(result["nodes"]["#build-backend"]["value"], "vulkan")
        self.assertFalse(result["nodes"]["#btn-build"]["disabled"])
