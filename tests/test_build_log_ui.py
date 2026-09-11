"""Exercise log polling without replacing text that the user is reading."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node.js required for log UI tests")
class BuildLogUiTest(unittest.TestCase):
    def test_log_preserves_selection_scroll_and_resumes_following(self):
        script = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(JSON.parse(fs.readFileSync(0, 'utf8')), 'utf8')
  .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '');
let now = 0, selection = null, active = false, content = 'idle', writes = 0;
const log = {
  scrollTop: 0, clientHeight: 100, scrollHeight: 100,
  matches: () => active,
  get textContent() { return content; },
  set textContent(text) { content = text; writes++; this.scrollHeight = 300; },
};
const context = vm.createContext({
  localStorage: {getItem: () => null}, Date: {now: () => now},
  getSelection: () => selection, log,
});
vm.runInContext(source, context);
const update = text => {context.nextText = text; vm.runInContext('updateBuildLog(log, nextText)', context);};
update('first');
assert.equal(content, 'first');
assert.equal(writes, 1);
log.scrollTop = 200;
update('first');
assert.equal(writes, 1, 'unchanged content must retain the text node');

log.scrollTop = 50;
update('second');
assert.equal(content, 'first', 'reading old lines pauses refresh');
assert.equal(log.scrollTop, 50);
log.scrollTop = 200;
log.onscroll();
update('second');
assert.equal(content, 'first', 'active scrolling pauses even at the bottom');
now = 1001;
update('second');
assert.equal(content, 'second', 'returning to bottom resumes');

selection = {isCollapsed: false, rangeCount: 2, getRangeAt: i => ({intersectsNode: node => i === 1 && node === log})};
update('third');
assert.equal(content, 'second', 'selection crossing log boundary is preserved');
selection = {isCollapsed: true};
active = true;
update('third');
assert.equal(content, 'second', 'pointer down before range forms is preserved');
active = false;
update('third');
assert.equal(content, 'third', 'clearing selection resumes');

for (const action of [() => log.onwheel(), () => log.ontouchmove(), () => log.onkeydown({key:'PageUp'})]) {
  const before = writes;
  action();
  update('new ' + now);
  assert.equal(writes, before);
  now += 1001;
  update('new ' + now);
  assert.equal(writes, before + 1);
}
selection = {isCollapsed: false, rangeCount: 1, getRangeAt: () => ({intersectsNode: () => false})};
update('selection elsewhere');
assert.equal(content, 'selection elsewhere');
"""
        run = subprocess.run([shutil.which("node"), "-e", script],
                             input=json.dumps(str(Path(__file__).resolve().parents[1] / "web/js/build.js")),
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
