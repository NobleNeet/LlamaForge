import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const handlers = {}, requests = [], staged = [], pending = [];
const el = {innerHTML: '', firstElementChild: null, dataset: {autotunePanel: 'm'}};
const fakeApi = (url, body) => {
  requests.push({url, body});
  return new Promise((resolve, reject) => pending.push({resolve, reject}));
};
const ctx = vm.createContext({
  api: fakeApi, esc: s => String(s), toast: () => {},
  setHTML: (node, text) => { node.innerHTML = text; node.firstElementChild = {}; },
  CSS: {escape: s => s},
  document: {querySelector: () => el, addEventListener: (name, fn) => { handlers[name] = fn; }},
});
const source = fs.readFileSync(new URL('../web/js/autotune.js', import.meta.url), 'utf8')
  .replace(/^import .*;\n/gm, '').replace(/export function/g, 'function');
vm.runInContext(source + '\nglobalThis.entry = {initAutoTune, syncAutoTune};', ctx);
ctx.entry.initAutoTune({stage: (id, settings) => staged.push({id, settings})});
const flush = () => new Promise(resolve => setImmediate(resolve));
const result = {};
for (const name of ['safe', 'balanced', 'aggressive']) {
  result[name] = {knobs: {'n-gpu-layers': name === 'safe' ? '12' : '16', 'ctx-size': '8192'},
    rationale: {}, memory: {headroom: .09}, confidence: 'medium', applicable: true, warnings: []};
}
ctx.entry.syncAutoTune({id: 'm', settings: {model: '/m.gguf'}});
assert.equal(requests.length, 1);
assert.equal(requests[0].url, '/api/autotune/recommend');
assert.equal(staged.length, 0); // opening never overwrites user edits
pending.shift().resolve(result); await flush();
assert.match(el.innerHTML, /value="balanced" selected/);
assert.match(el.innerHTML, /GPU Layers/);
ctx.entry.syncAutoTune({id: 'm', settings: {model: '/m.gguf', threads: '7'}});
assert.equal(requests.length, 1); // normal state polling does not recalculate or erase edits
const select = {value: 'safe', closest: () => el};
handlers.change({target: {closest: () => select}});
assert.equal(staged.length, 1);
assert.equal(staged[0].settings['n-gpu-layers'], '12');
const recalc = {closest: () => el, hasAttribute: () => true};
handlers.click({target: {closest: () => recalc}, preventDefault() {}});
assert.equal(requests.length, 2);
pending.shift().resolve({...result, safe: {...result.safe, applicable: false}}); await flush();
handlers.change({target: {closest: () => select}});
assert.equal(staged.length, 1); // impossible placement cannot be applied
assert.match(el.innerHTML, /data-static-apply disabled/);
ctx.entry.syncAutoTune({id: 'm', settings: {model: '/replacement.gguf'}});
assert.equal(requests.length, 3);
pending.shift().reject(new Error('unavailable')); await flush();
assert.match(el.innerHTML, /Presets unavailable/);
assert.equal(staged.length, 1);
assert.ok(requests.every(r => r.url === '/api/autotune/recommend'));
console.log('Static preset UI behavior checks passed');
