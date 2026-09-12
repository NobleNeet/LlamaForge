// Static, automatically generated presets. Staging only changes editable fields.
import { api, esc, setHTML, toast } from "./core.js";

let bridge;
const states = new Map();
const names = ["safe", "balanced", "aggressive"];
const labels = {"n-gpu-layers": "GPU Layers", "ctx-size": "Context",
  "batch-size": "Batch", "ubatch-size": "uBatch", "threads": "Threads",
  "threads-batch": "Threads batch", "flash-attn": "Flash Attention",
  "cache-type-k": "KV Cache K", "cache-type-v": "KV Cache V",
  "split-mode": "GPU split", "main-gpu": "GPU device"};
const panel = id => document.querySelector(`[data-autotune-panel="${CSS.escape(id)}"]`);

function render(id) {
  const el = panel(id), s = states.get(id);
  if (!el || !s) return;
  if (s.pending) { setHTML(el, '<div class="autotune">Calculating static presets…</div>'); return; }
  if (s.error) {
    setHTML(el, `<div class="autotune">${esc(s.error)} <button type="button" class="qbtn" data-preset-recalculate>Recalculate presets</button></div>`);
    return;
  }
  const rec = s.result[s.selected];
  const rows = Object.entries(rec.knobs).map(([k,v]) =>
    `<div title="${esc(rec.rationale[k] || '')}"><span>${esc(labels[k] || k)}</span><b>${esc(v)}</b></div>`).join('');
  setHTML(el, `<div class="autotune"><label>Static presets
    <select data-static-preset>${names.map(n => `<option value="${n}" ${n === s.selected ? 'selected' : ''}>${n[0].toUpperCase() + n.slice(1)}</option>`).join('')}</select></label>
    <button type="button" class="qbtn" data-static-apply ${rec.applicable ? '' : 'disabled'}>Apply preset to editor</button>
    <button type="button" class="qbtn" data-preset-recalculate>Recalculate presets</button>
    <div class="at-profile"><div class="at-settings">${rows}</div>
    <div class="at-meta">${Math.round(rec.memory.headroom * 100)}% memory reserve · ${esc(rec.confidence)} confidence</div>
    ${(rec.warnings || []).map(w => `<div class="note">${esc(w)}</div>`).join('')}
    <div class="at-meta">Static estimate only. No model execution or benchmark. Review and edit before saving.</div></div></div>`);
}

async function calculate(id) {
  const s = states.get(id);
  if (!s || s.pending) return;
  s.pending = true; s.error = null; render(id);
  try {
    const result = await api('/api/autotune/recommend', {model: id});
    if (states.get(id) !== s) return; // discard response after model path changed
    if (result.error) throw new Error(result.error);
    s.result = result;
  } catch (e) { s.error = `Presets unavailable: ${e.message || e}`; }
  s.pending = false; render(id);
}

export function syncAutoTune(model) {
  if (!panel(model.id)) return;
  const path = model.model || model.settings?.model || '';
  const old = states.get(model.id);
  if (!old || old.path !== path) {
    states.set(model.id, {path, selected: 'balanced', pending: false});
    calculate(model.id);
  } else if (!panel(model.id).firstElementChild) render(model.id);
}

function apply(id) {
  const s = states.get(id), rec = s?.result?.[s.selected];
  if (!rec?.applicable) return;
  if (bridge.stage(id, rec.knobs) === false) return;
  toast(`${s.selected[0].toUpperCase() + s.selected.slice(1)} preset loaded into editor`, 'ok');
}

export function initAutoTune(callbacks) {
  bridge = callbacks;
  document.addEventListener('change', event => {
    const select = event.target.closest('[data-static-preset]');
    if (!select) return;
    const id = select.closest('[data-autotune-panel]').dataset.autotunePanel;
    states.get(id).selected = select.value;
    render(id); apply(id);
  });
  document.addEventListener('click', event => {
    const el = event.target.closest('[data-static-apply], [data-preset-recalculate]');
    if (!el) return;
    event.preventDefault();
    const id = el.closest('[data-autotune-panel]').dataset.autotunePanel;
    if (el.hasAttribute('data-preset-recalculate')) calculate(id);
    else apply(id);
  });
}
