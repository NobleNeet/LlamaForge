import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// The Models-tab delete flow: opening the dialog must NOT call the delete
// API; only the "Delete permanently" button inside the modal may.
const nodes = {}, requests = [], toasts = [];
function makeNode(id) {
  return {
    id, innerHTML: '', hidden: false, disabled: false, dataset: {}, style: {},
    classList: {add(){}, remove(){}, toggle(){}, contains: () => false},
    set textContent(v) { this._tc = v; }, get textContent() { return this._tc || ''; },
    closest: () => null,
    remove() {},
    insertAdjacentHTML(pos, html) { this.innerHTML += html; },
    querySelector: () => null,
    get firstElementChild() { return null; },
    get lastElementChild() { return null; },
  };
}
const $ = sel => {
  const id = String(sel).replace(/^#/, '');
  return nodes[id] ||= makeNode(id);
};
const setHTML = (el, html) => {
  el.innerHTML = html;
  for (const m of String(html).matchAll(/id="([^"]+)"/g)) {
    nodes[m[1]] ||= makeNode(m[1]);
  }
};
const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let handler = async (path, body) => {
  requests.push({path, body});
  if (path.startsWith('/api/model/delete_plan')) {
    return {model: 'm', backend: 'llamacpp',
      files: ['/models/repo/model-Q4.gguf'], directory: '/models/repo',
      delete_directory: false, size_bytes: 4402340000, mmproj: null};
  }
  if (path === '/api/models/delete') return {ok: true, error: '', backend: 'llamacpp'};
  if (path === '/api/schema') return {groups: [], count: 0};
  if (path === '/api/state') return {gpus: [], models: [], global: {}};
  throw new Error('unexpected ' + path);
};
const api = (path, body) => handler(path, body);
let rows = [{id: 'm', backend: 'llamacpp', status: 'offline', in_ini: true}];
const store = {};
const ctx = vm.createContext({
  $, $$: () => [], esc, setHTML, api,
  toast: (m, c) => toasts.push([m, c]),
  meter: () => '',
  S: {STATE: null, SCHEMA: {groups: [], count: 0}},
  modelRows: () => rows,
  cfgOf: () => ({}),
  on: () => {}, emit: () => {}, activeTab: () => 'models',
  initAutoTune: () => {}, syncAutoTune: () => {},
  document: {title: '', addEventListener: () => {}, createElement: () => makeNode('created')},
  localStorage: {getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; },
                 removeItem: k => { delete store[k]; }},
  CSS: {escape: s => s},
  setTimeout, clearTimeout, Date, JSON, Set, Map, Object, Array, String, Number, Math, Promise,
});
const source = fs.readFileSync(new URL('../web/js/models.js', import.meta.url), 'utf8')
  .replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
vm.runInContext(source + '\nglobalThis.__fns = {openDeleteDialog, editorButtons};', ctx);
globalThis.__setHandler = h => { handler = h; };

// 1. The Delete button exists in the action row and is styled as danger.
const btns = ctx.__fns.editorButtons({id: 'm', backend: 'llamacpp', status: 'offline', in_ini: true});
assert.match(btns, /data-act="delete"/);
assert.match(btns, /class="ghost danger"/);
// It is the LAST action in the row (far right of the existing action column).
assert.ok(btns.indexOf('data-act="delete"') > btns.indexOf('data-act="client"'));
// vLLM rows get the same unified delete button (no separate vdelete flow).
const vbtns = ctx.__fns.editorButtons({id: 'v', backend: 'vllm', status: 'offline', in_ini: true});
assert.match(vbtns, /data-act="delete"/);
assert.ok(!vbtns.includes('vdelete'));
// Auto-discovered rows (not in models.ini) have no recorded file set -> no delete button.
const autoBtns = ctx.__fns.editorButtons({id: 'a', backend: 'llamacpp', status: 'offline', in_ini: false});
assert.ok(!autoBtns.includes('data-act="delete"'));

// 2. Opening the dialog fetches the plan but never calls the delete API.
await ctx.__fns.openDeleteDialog('m');
const planReqs = requests.filter(r => r.path.startsWith('/api/model/delete_plan'));
assert.equal(planReqs.length, 1, 'plan endpoint queried once');
assert.equal(requests.filter(r => r.path === '/api/models/delete').length, 0,
            'no deletion without explicit confirmation');
const modal = nodes['modal-root'].innerHTML;
assert.match(modal, /Delete permanently/);
assert.match(modal, /model-Q4\.gguf/);          // actual target path shown
assert.match(modal, /cannot be undone/);         // permanence warning
assert.match(modal, /GiB/);                    // size shown
assert.match(modal, /data-mclose/);             // Cancel remains

// 3. "Delete permanently" calls the API, refreshes the list, closes the modal.
const confirmBtn = nodes['delete-perm-confirm'];
assert.ok(confirmBtn, 'confirm button registered');
await confirmBtn.onclick();
assert.equal(requests.filter(r => r.path === '/api/models/delete').length, 1);
assert.match(nodes['modal-root'].innerHTML, /^$/, 'modal closed after success');
assert.ok(toasts.some(([m]) => /Deleted m/.test(m)), 'success toast shown');
assert.ok(requests.some(r => r.path === '/api/state'), 'model list re-fetched after delete');

// 4. A failed delete keeps the modal open and shows the reason.
rows = [{id: 'busy', backend: 'llamacpp', status: 'loaded', in_ini: true}];
__setHandler(async (path, body) => {
  requests.push({path, body});
  if (path.startsWith('/api/model/delete_plan')) {
    return {model: 'busy', backend: 'llamacpp', files: ['/models/repo/b.gguf'],
      directory: '/models/repo', delete_directory: false, size_bytes: 1, mmproj: null};
  }
  if (path === '/api/models/delete') {
    return {ok: false, error: 'model is loaded or loading - unload it first', backend: 'llamacpp'};
  }
  if (path === '/api/state') return {gpus: [], models: [], global: {}};
  throw new Error('unexpected ' + path);
});
await ctx.__fns.openDeleteDialog('busy');
const confirmBtn2 = nodes['delete-perm-confirm'];
const before = requests.length;
await confirmBtn2.onclick();
assert.match(nodes['modal-root'].innerHTML, /Delete permanently/, 'modal stays open on failure');
assert.match(nodes['del-error'].textContent, /loaded or loading/, 'failure reason shown in the dialog');
assert.ok(toasts.some(([, c]) => c === 'err'), 'error toast shown');
assert.ok(!requests.slice(before).some(r => r.path === '/api/state'),
          'no list refresh after a failed delete');

console.log('models delete frontend checks passed');
