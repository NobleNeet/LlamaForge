// Ephemeral inference diagnostics. Model/load state stays owned by the registry/router.
import { $, esc, setHTML, api } from './core.js';
import { S } from './state.js';
import { makeRequest, requestMetrics, streamCompletion, logSince } from './chat-protocol.js';

const LABELS = {llamacpp:'llama.cpp', ikllama:'ik_llama', vllm:'vLLM'};
let mounted = false, polling = false, selected = '', snapshot = {}, history = [];
let controller = null, loading = false, last = null, rawRequest = null;
let logs = {router:'', llama:''}, baseline = null, requestNumber = 0;
let drawPending = false;
const value = (v, suffix = '') => v == null ? 'Unavailable' : `${typeof v === 'number' ? v.toLocaleString(undefined, {maximumFractionDigits:1}) : v}${suffix}`;
const kv = (label, v) => `<div class="kv"><span class="k">${esc(label)}</span><span class="v">${esc(v)}</span></div>`;
const pretty = v => JSON.stringify(v, null, 2);

export async function loadTestChat() {
  if (!mounted) {
    mounted = true;
    setHTML($('#view-testchat'), `
      <div class="chat-layout">
        <section class="chat-main card" aria-label="Test Chat">
          <h3>Test Chat <span class="note">Inference diagnostics</span></h3>
          <div class="toolbar"><label for="chat-model">Model</label><select id="chat-model" aria-label="Model"></select><button id="chat-load">Load model</button></div>
          <div class="note" id="chat-state">Checking model state…</div>
          <div class="chat-messages" id="chat-messages" role="log" aria-label="Conversation" aria-live="polite"><div class="note">Send a message to test the managed inference path.</div></div>
          <details class="chat-options"><summary>System Prompt</summary><textarea id="chat-system" rows="3" placeholder="Optional — applies to this conversation only" aria-label="System Prompt"></textarea></details>
          <details class="chat-options"><summary>Generation Settings</summary>
            <label><input type="radio" name="chat-sampling" id="chat-defaults" checked> Model defaults</label>
            <label><input type="radio" name="chat-sampling" id="chat-override"> Override for this chat</label>
            <div class="chat-settings" id="chat-settings" hidden>
              ${[['temperature','Temperature','0','2','0.1'],['top_p','Top P','0','1','0.05'],['top_k','Top K','0','','1'],['max_tokens','Max tokens','1','','1'],['seed','Seed','-1','','1']].map(([id,label,min,max,step]) => `<label>${label}<input id="chat-${id}" type="number" min="${min}" ${max ? `max="${max}"` : ''} step="${step}" placeholder="Server default"></label>`).join('')}
            </div>
            <div class="note">Model defaults sends no sampling overrides. Blank overrides also use server defaults. Context Wiki injection follows the normal API path.</div>
          </details>
          <form id="chat-form"><label for="chat-input">Message</label><textarea id="chat-input" rows="3" placeholder="Enter to send · Shift+Enter for a new line" required></textarea>
            <div class="actions"><button class="primary" id="chat-send" type="submit">Send</button><button id="chat-stop" type="button" disabled>Stop</button><button id="chat-clear" type="button">Clear Chat</button></div>
          </form><div class="msg err" id="chat-error" role="alert"></div>
        </section>
        <aside class="chat-diagnostics" aria-label="Diagnostics">
          <section class="card"><h3>Overview</h3><div id="chat-overview"></div><div class="note" id="chat-path"></div></section>
          <section class="card"><h3>Last Request</h3><div id="chat-last">No request yet.</div><div class="note">Tokens: response usage. Speeds: server timings, not aggregate Stats rates. TTFT: browser time to first text, reasoning or tool output; includes network and queue time.</div></section>
          <section class="card"><h3>Parameters</h3><div class="note" id="chat-argv-source"></div><pre id="chat-argv">No running instance.</pre></section>
          <section class="card"><h3>Logs</h3><div class="toolbar"><select id="chat-log-source" aria-label="Log source"><option value="router">Router stdout / stderr</option><option value="llama">llama-server stdout</option></select><select id="chat-log-filter" aria-label="Log filter"><option value="all">All</option><option value="request">Request</option><option value="warnings">Warnings</option><option value="errors">Errors</option></select></div><div class="note">Shared log tail. Request shows the window since this chat's last send; other clients may appear in it.</div><pre class="log" id="chat-log">No log yet.</pre></section>
          <section class="card"><h3>API Details</h3><details><summary>Raw Request (browser → panel)</summary><pre id="chat-raw-request">No request yet.</pre></details><details><summary>Effective Request (panel → router)</summary><pre id="chat-effective">Available when the stream begins.</pre></details><details><summary>Response metadata</summary><pre id="chat-raw-response">No response yet.</pre></details></section>
        </aside>
      </div>`);
    $('#chat-form').onsubmit = e => { e.preventDefault(); sendMessage(); };
    $('#chat-input').onkeydown = e => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendMessage(); }
    };
    $('#chat-stop').onclick = () => controller?.abort();
    $('#chat-clear').onclick = clearChat;
    $('#chat-load').onclick = loadModel;
    $('#chat-model').onchange = () => {
      selected = $('#chat-model').value;
      history = []; last = null; rawRequest = null; baseline = null;
      renderMessages(); renderRequest(); refreshTestChat();
    };
    const toggle = () => { $('#chat-settings').hidden = !$('#chat-override').checked; };
    $('#chat-defaults').onchange = toggle; $('#chat-override').onchange = toggle;
    $('#chat-log-filter').onchange = renderLogs; $('#chat-log-source').onchange = renderLogs;
  }
  await refreshTestChat();
}

function setError(error = '') { $('#chat-error').textContent = error; }
function controls() {
  const busy = !!controller || loading;
  $('#chat-model').disabled = busy;
  $('#chat-load').disabled = busy || !selected || !snapshot.available || ['loaded','loading','downloading'].includes(snapshot.status);
  $('#chat-load').textContent = snapshot.router_up === false ? 'Start router & load' : 'Load model';
  $('#chat-send').disabled = busy || !selected || !snapshot.available || snapshot.status !== 'loaded';
  $('#chat-stop').disabled = !controller;
  $('#chat-clear').disabled = busy;
}

export async function refreshTestChat() {
  if (!mounted || polling) return;
  polling = true;
  const requested = selected;
  try {
    const d = await api('/api/chat/diagnostics?model=' + encodeURIComponent(requested));
    if (d.error && !Array.isArray(d.models)) throw new Error(d.error);
    if (requested !== selected) return; // a model change invalidated this poll
    const models = d.models || [];
    if (!selected) selected = (models.find(m => m.status === 'loaded' && m.backend === 'llamacpp') || models.find(m => m.backend === 'llamacpp') || {}).id || '';
    const signature = JSON.stringify(models);
    if ($('#chat-model').dataset.models !== signature) {
      setHTML($('#chat-model'), '<option value="">Select a model</option>' + models.map(m => `<option value="${esc(m.id)}">${esc(m.id)} · ${esc(m.backend === 'llamacpp' ? LABELS[d.engine] || d.engine : LABELS[m.backend] || m.backend)} · ${esc(m.failed ? 'error' : m.status)}</option>`).join(''));
      $('#chat-model').dataset.models = signature;
    }
    $('#chat-model').value = selected;
    snapshot = selected !== requested ? await api('/api/chat/diagnostics?model=' + encodeURIComponent(selected)) : d;
    renderOverview(); controls();
    await refreshLogs();
  } catch (e) {
    snapshot = {...snapshot, status:'offline', available:false};
    setError('Diagnostics: ' + e.message); renderOverview(); controls();
  } finally { polling = false; }
}

async function refreshLogs() {
  const [router, llama] = await Promise.all([api('/api/router/log'), api('/api/llama/log')]);
  logs = {router:router.log || router.error || '', llama:llama.log || llama.error || ''};
  renderLogs();
}
function renderOverview() {
  const d = snapshot, engine = d.backend === 'llamacpp' ? d.engine : d.backend;
  const status = loading ? `Load requested: ${selected}…` : d.status || 'unknown';
  $('#chat-state').textContent = !selected ? 'No model is currently loaded. Select a model to load; add models in Models / Setup if this list is empty.' :
    `${selected} · ${LABELS[engine] || engine || 'Unknown backend'} · ${status}` + (d.note ? ` · ${d.note}` : d.status !== 'loaded' ? ' · Use Load model to start inference.' : '');
  setHTML($('#chat-overview'), kv('Model / alias', d.alias || selected || 'None') + kv('Backend', LABELS[engine] || engine || 'Unavailable') +
    kv('Status', status) + kv('Router port', value(d.router_port == null ? null : String(d.router_port))) + kv('Server instance port', value(d.port == null ? null : String(d.port))) + kv('PID', value(d.pid == null ? null : String(d.pid))) +
    kv('Runtime context capacity', value(d.context_capacity, ' tokens')) +
    kv('Last prompt (incl. context)', last && rawRequest?.model === selected ? value(requestMetrics(last).prompt, ' tokens') : 'Unavailable'));
  $('#chat-path').textContent = selected ? `Test Chat → panel :${d.panel_port ?? '?'} /v1/chat/completions → router :${d.router_port ?? '?'} → ${selected}${d.port ? ' :' + d.port : ''}` : '';
  $('#chat-argv-source').textContent = d.argv?.length ? d.argv_source + '. Authentication values redacted. One argv element per line.' : 'Actual runtime arguments unavailable. Saved model settings are not substituted.';
  $('#chat-argv').textContent = d.argv?.length ? d.argv.map(a => JSON.stringify(a)).join('\n') : 'No runtime argv reported.';
}

async function loadModel() {
  if (loading || controller || !selected || !snapshot.available) return;
  loading = true; controls(); renderOverview(); setError();
  try {
    if (snapshot.router_up === false) {
      $('#chat-state').textContent = 'Starting router and checking health…';
      const started = await api('/api/router/restart', {});
      if (!started.ok) throw new Error(started.error || 'Router could not start');
      const deadline = Date.now() + 30000;
      let ready = false;
      while (Date.now() < deadline) {
        const status = await api('/api/chat/diagnostics?model=' + encodeURIComponent(selected));
        if (status.router_up) { ready = true; break; }
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      if (!ready) throw new Error('Router did not become ready within 30 seconds');
    }
    const r = await api('/api/models/load', {model:selected, backend:snapshot.backend});
    if (!r.ok) throw new Error(typeof r.error === 'string' ? r.error : pretty(r.error));
  } catch (e) { setError('Load failed: ' + e.message); }
  finally { loading = false; await refreshTestChat(); controls(); }
}

function renderMessages() {
  const list = $('#chat-messages'), bottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80;
  setHTML(list, history.length ? history.map(m => `<article class="chat-message" data-role="${esc(m.role)}"><strong>${m.role === 'user' ? 'User' : 'Assistant'}</strong>
    ${m.reasoning ? `<details><summary>Reasoning</summary><div class="chat-text">${esc(m.reasoning)}</div></details>` : ''}
    <div class="chat-text${m.html ? ' chat-markdown' : ''}">${m.html || esc(m.content || (m.role === 'assistant' && controller ? 'Waiting for output…' : ''))}</div>
    ${m.role === 'assistant' ? `<div class="chat-meta">${esc(messageMetrics(m))}</div>` : ''}</article>`).join('') : '<div class="note">Send a message to test the managed inference path.</div>');
  if (bottom || controller) list.scrollTop = list.scrollHeight;
}
function queueDraw() {
  if (drawPending) return;
  drawPending = true;
  const schedule = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : fn => fn();
  schedule(() => { drawPending = false; renderMessages(); renderRequest(); });
}
function messageMetrics(m) {
  const r = m.result;
  if (!r) return m.error || '';
  const v = requestMetrics(r), fields = [];
  if (v.generated != null) fields.push(value(v.generated, ' tokens'));
  if (v.decode != null) fields.push(value(v.decode, ' tok/s'));
  if (v.ttft != null) fields.push('TTFT ' + value(v.ttft, ' ms'));
  if (m.error) fields.push(m.error);
  return fields.join(' · ');
}
function renderRequest() {
  if (!last) {
    $('#chat-last').textContent = 'No request yet.';
    for (const id of ['chat-raw-request','chat-effective','chat-raw-response']) $('#' + id).textContent = 'No request yet.';
    return;
  }
  const m = requestMetrics(last);
  setHTML($('#chat-last'), kv('Requested model', rawRequest?.model || 'Unavailable') + kv('Response model', last.model || 'Unavailable') +
    kv('Response ID', last.id || 'Unavailable') + kv('Prompt tokens', value(m.prompt)) + kv('Generated tokens', value(m.generated)) +
    kv('Prompt evaluation', value(m.prefill, ' tok/s')) + kv('Decode', value(m.decode, ' tok/s')) + kv('TTFT (browser)', value(m.ttft, ' ms')) +
    kv('Total (browser)', value(m.total == null ? null : (m.total / 1000).toFixed(3), ' sec')) + kv('finish_reason', last.finish_reason || 'Unavailable') +
    (m.drafted > 0 ? kv('Drafted tokens', value(m.drafted)) + kv('Accepted tokens', value(m.accepted)) + kv('Acceptance', value(m.acceptance, ' %')) : ''));
  $('#chat-raw-request').textContent = pretty(rawRequest);
  $('#chat-effective').textContent = last.effective_request ? pretty(last.effective_request) : 'Not received yet.';
  const {content, reasoning, effective_request, ...metadata} = last;
  $('#chat-raw-response').textContent = pretty(metadata);
}
function renderLogs() {
  const source = $('#chat-log-source').value || 'router', filter = $('#chat-log-filter').value || 'all';
  let text = logs[source] || 'No log available.';
  if (filter === 'request') text = baseline ? `── Request #${requestNumber} · ${rawRequest?.model || selected} · shared log window ──\n` + logSince(baseline[source], text) : 'Send a message to mark a request window.';
  if (filter === 'warnings') text = text.split('\n').filter(l => /warn(?:ing)?/i.test(l)).join('\n') || 'No warnings in this tail.';
  if (filter === 'errors') text = text.split('\n').filter(l => /error|fail|abort|exception|fatal/i.test(l)).join('\n') || 'No errors in this tail.';
  $('#chat-log').textContent = text;
}

async function sendMessage() {
  if (controller || loading || !selected || !snapshot.available || snapshot.status !== 'loaded') return;
  const input = $('#chat-input'), content = input.value.trim();
  if (!content) return;
  let body;
  try {
    const override = $('#chat-override').checked ? Object.fromEntries(['temperature','top_p','top_k','max_tokens','seed'].map(k => [k, $('#chat-' + k).value])) : null;
    body = makeRequest(selected, [...history, {role:'user', content}], $('#chat-system').value, override);
  } catch (e) { setError(e.message); return; }
  const own = new AbortController(); controller = own; controls(); setError();
  history.push({role:'user', content}); const assistant = {role:'assistant', content:''}; history.push(assistant);
  input.value = ''; last = null; rawRequest = body; requestNumber++;
  renderMessages(); renderRequest();
  try {
    // Capture the shared log tail at send time. Failure must not block inference.
    try { await refreshLogs(); } catch (_) {}
    baseline = {...logs};
    if (own.signal.aborted) throw new DOMException('Stopped', 'AbortError');
    await streamCompletion(body, {signal:own.signal, token:S.STATE?.config?.router_api_key || '', onUpdate:r => {
      last = r; assistant.content = r.content; assistant.reasoning = r.reasoning; assistant.result = r;
      queueDraw();
    }});
  } catch (e) {
    assistant.error = own.signal.aborted ? 'Stopped by user' : 'Request failed: ' + e.message;
    setError(assistant.error);
  } finally {
    controller = null; controls(); renderMessages();
    // The same escaped Markdown renderer as Help; never treat model text as HTML.
    if (assistant.content) api('/api/chat/render', {content:assistant.content}).then(r => {
      if (r.html) { assistant.html = r.html; renderMessages(); }
    }).catch(() => {});
    await refreshTestChat(); input.focus();
  }
}
function clearChat() {
  if (controller || loading) return;
  history = []; last = null; rawRequest = null; baseline = null;
  renderMessages(); renderRequest(); renderLogs(); setError();
}
