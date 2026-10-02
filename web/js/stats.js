// Stats tab: totals, live throughput, a daily activity chart, per-model usage.
import { $, esc, setHTML, api, toast, fmtNum, fmtDur, fmtAgo } from "./core.js";

let statsSort = "tokens", statsRange = 14;
let expandedModel = null, runLimit = 10, selectedConfig = null, commandOpen = false;
let historyData = null, historyError = '', historyRequest = 0;

async function fetchHistory() {
  const model = expandedModel, limit = runLimit, request = ++historyRequest;
  if (model === null) return;
  try {
    const data = await api(`/api/stats/runs?model=${encodeURIComponent(model)}&limit=${limit}`);
    if (request !== historyRequest || model !== expandedModel || limit !== runLimit) return;
    if (data.error || !Array.isArray(data.runs)) throw Error(data.error || 'History unavailable');
    historyData = data;
    historyError = '';
    if (!selectedConfig || !data.configs[selectedConfig]) selectedConfig = data.runs[0]?.load_config_id ?? null;
  } catch (e) {
    if (request !== historyRequest) return;
    historyError = 'Run history unavailable';
  }
}

function localTime(timestamp) {
  const d = new Date(timestamp * 1000), pad = n => String(n).padStart(2, '0');
  const time = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return {time, full:`${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${time}:${pad(d.getSeconds())}`};
}
const metric = (n, decimals = 0) => n == null ? '—' : Number(n).toLocaleString(undefined, {minimumFractionDigits:decimals, maximumFractionDigits:decimals});
const engineLabel = c => `${c.engine_name}${c.engine_commit ? ` (${c.engine_commit.slice(0,7)})` : ''}`;
const kv = (label, value) => `<div class="kv"><span class="k">${esc(label)}</span><span class="v" style="overflow-wrap:anywhere">${esc(value ?? '—')}</span></div>`;
// POSIX quoting preserves argv boundaries, including spaces and literal quotes.
const quoteArg = arg => /^[a-zA-Z0-9_@%+=:,./-]+$/.test(arg) ? arg : "'" + arg.replaceAll("'", "'\"'\"'") + "'";

function loadConfigDetail(c) {
  const groups = c.normalized_options || {};
  const group = (title, values) => values && Object.keys(values).length ? `<h3>${title}</h3>${Object.entries(values).map(([k,v])=>kv(k,/^\d+$/.test(v) ? Number(v).toLocaleString() : v)).join('')}` : '';
  return `<div class="card"><h3>LOAD CONFIG #${c.id}<span style="float:right">Loaded ${esc(localTime(c.loaded_at).full)}</span></h3>
    <h3>ENGINE</h3>${kv('Engine', engineLabel(c))}${kv('Executable', c.engine_executable)}
    ${group('PERFORMANCE', groups.performance)}${group('SPECULATIVE DECODING', groups.speculative)}
    ${c.other_options?.length ? `<h3>ENGINE-SPECIFIC / OTHER OPTIONS</h3><pre class="log">${esc(c.other_options.map(quoteArg).join(' '))}</pre>` : ''}
    ${group('MODELS', groups.models)}
    <button class="qbtn" data-command aria-expanded="${commandOpen}">${commandOpen ? 'Hide' : 'Show'} full launch command</button>
    ${commandOpen ? `<h3>FULL LAUNCH COMMAND</h3>${kv('Engine', engineLabel(c))}${kv('Executable', c.engine_executable)}<pre class="log" style="white-space:pre-wrap;overflow-wrap:anywhere">${esc(c.launch_command || c.argv.map(quoteArg).join(' '))}</pre>` : ''}</div>`;
}

function historyDetail() {
  if (historyError) return `<div class="note">${esc(historyError)}</div>`;
  if (!historyData) return '<div class="note">Loading run history…</div>';
  const runs = historyData.runs, c = historyData.configs[selectedConfig];
  return `<div class="card"><h3>RECENT RUNS <select data-runlimit aria-label="Recent Runs limit" style="float:right">${[10,25,50,100].map(n=>`<option value="${n}" ${n===runLimit?'selected':''}>Latest ${n}</option>`).join('')}</select></h3>
    ${runs.length ? `<div style="overflow-x:auto"><table style="width:100%;text-align:right"><thead><tr>${['Time','Prompt (tok)','PP (tok/s)','Generated (tok)','TG (tok/s)','Total (s)','MTP Accept (%)','Load Config'].map(label=>`<th>${label}</th>`).join('')}</tr></thead><tbody>
    ${runs.map(r=>{const date=localTime(r.timestamp); const mtp=r.mtp_acceptance == null ? '' : `Acceptance: ${r.mtp_acceptance*100}% · ${r.mtp_accepted ?? '—'} accepted / ${r.mtp_generated ?? '—'} generated · mean length: ${r.mtp_mean_len ?? '—'}`;
      return `<tr><td title="${esc(date.full)}">${date.time}</td><td>${metric(r.prompt_tokens)}</td><td>${metric(r.pp_tps,2)}</td><td>${metric(r.generated_tokens)}</td><td>${metric(r.tg_tps,2)}</td><td>${metric(r.total_ms == null ? null : r.total_ms/1000,2)}</td><td title="${esc(mtp)}">${metric(r.mtp_acceptance == null ? null : r.mtp_acceptance*100,1)}</td><td><button class="qbtn" data-loadconfig="${r.load_config_id}" aria-pressed="${r.load_config_id===Number(selectedConfig)}">#${r.load_config_id}</button></td></tr>`;}).join('')}
    </tbody></table></div>` : '<div class="note">No completed runs recorded. vLLM supports aggregate stats only.</div>'}
    ${c ? loadConfigDetail(c) : ''}</div>`;
}
const SORT_COLS = {tokens:"Total", prompt:"Prompt", generated:"Gen",
                   avg_tps:"Tok/s", runs:"Runs", loaded_secs:"Loaded"};

function setStatsRange(n) { statsRange = n; loadStats(true); }
function sortStats(c) { statsSort = c; loadStats(true); }
async function resetStats() {
  if (!confirm("Reset ALL usage statistics? Per-model totals, daily totals, Run and Load Config history will be zeroed. This cannot be undone.")) return;
  await api("/api/stats/reset", {});
  expandedModel = selectedConfig = historyData = null;
  commandOpen = false;
  ++historyRequest;
  toast("Stats reset", "ok");
  loadStats(true);
}

// The stats view is fully re-rendered on each load, so its controls are wired
// once by delegation on the container rather than per-render.
export function initStats() {
  const view = $("#view-stats");
  if (!view) return;
  view.addEventListener("click", e => {
    const range = e.target.closest("[data-range]");
    if (range) { setStatsRange(+range.dataset.range); return; }
    const sort = e.target.closest("[data-sort]");
    if (sort) { sortStats(sort.dataset.sort); return; }
    if (e.target.closest("[data-statsreset]")) { resetStats(); return; }
    const config = e.target.closest('[data-loadconfig]');
    if (config) { selectedConfig = config.dataset.loadconfig; commandOpen = false; loadStats(true); return; }
    if (e.target.closest('[data-command]')) { commandOpen = !commandOpen; loadStats(true); return; }
    const model = e.target.closest('[data-historymodel]');
    if (model) {
      expandedModel = expandedModel === model.dataset.historymodel ? null : model.dataset.historymodel;
      historyData = selectedConfig = null; historyError = ''; commandOpen = false;
      loadStats(true);
    }
  });
  view.addEventListener('change', e => {
    if (e.target.matches('[data-runlimit]')) {
      runLimit = Number(e.target.value); loadStats(true);
    }
  });
  view.addEventListener('keydown', e => {
    if (e.target.matches('[data-historymodel]') && ['Enter',' '].includes(e.key)) {
      e.preventDefault(); e.target.click();
    }
  });
}

function statCard(label, val) {
  return `<div class="gpu"><div class="stats" style="margin:0"><span>${esc(label)}</span></div><div style="font-family:var(--disp);font-weight:600;color:var(--ink-strong);font-size:22px;margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(val)}</div></div>`;
}

export async function loadStats(silent) {
  const v = $("#view-stats");
  if (!silent) setHTML(v, `<div class="skel">LOADING STATS...</div>`);
  let s;
  try { s = await api("/api/stats"); } catch (e) { s = null; }
  // fetch() doesn't reject on HTTP errors, so a 404/500 arrives as a parsed
  // error body, not an exception - guard on shape, not just the catch.
  if (!s || s.error || !Array.isArray(s.per_model)) {
    if (!silent) setHTML(v, `<div class="skel" style="color:var(--red)">BACKEND UNREACHABLE</div>`);
    return;
  }
  if (expandedModel !== null) await fetchHistory();
  const t = s.totals, live = s.live;
  // The router can hold several models loaded at once (`router_models_max`);
  // older backends only reported a single `loaded_model`.
  const loaded = live.loaded_models || (live.loaded_model ? [live.loaded_model] : []);
  const rows = [...s.per_model].sort((a,b) => (b[statsSort]||0) - (a[statsSort]||0));
  const daily = s.daily.slice(-statsRange);
  const maxDaily = Math.max(1, ...daily.map(d => d.prompt + d.generated));
  setHTML(v, `
    <div class="gpus" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
      ${statCard("Tokens processed", fmtNum(t.tokens))}
      ${statCard("Generated", fmtNum(t.generated))}
      ${statCard("Inference time", fmtDur(t.loaded_hours*3600))}
      ${statCard("Models used", t.models_used)}
      ${statCard("Runs (approx)", fmtNum(t.total_runs))}
      ${statCard("Most used", t.most_used||"-")}
    </div>
    <div class="card"><h3>Live Throughput${live.router_up?"":` <span style="color:var(--red);font-size:10px">(router offline)</span>`}</h3>
      <div class="kv"><span class="k">loaded model${loaded.length>1?"s":""}</span><span class="v ${loaded.length?"ok":""}">${esc(loaded.join(", ")||"none")}</span></div>
      <div class="kv"><span class="k">generation</span><span class="v">${(live.gen_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">prompt eval</span><span class="v">${(live.prompt_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">active requests</span><span class="v">${esc(live.requests_processing)}</span></div>
    </div>
    <div class="card"><h3>Activity${daily.length?` (last ${daily.length} days)`:""}
        <span style="float:right">
          <span class="chip ${statsRange===14?"on":""}" data-range="14">14d</span>
          <span class="chip ${statsRange===30?"on":""}" data-range="30">30d</span>
        </span></h3>
      ${daily.length?`<div style="display:flex;align-items:flex-end;gap:4px;height:120px;margin-top:10px">
        ${daily.map(d=>{const hp=Math.round(100*d.prompt/maxDaily),hg=Math.round(100*d.generated/maxDaily);
          return `<div title="${esc(d.date)} &middot; ${fmtNum(d.generated)} generated + ${fmtNum(d.prompt)} prompt" style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%">
            <div style="height:${hg}%;min-height:${d.generated?2:0}px;background:var(--amber);box-shadow:0 0 6px var(--amber-dim)"></div>
            <div style="height:${hp}%;min-height:${d.prompt?2:0}px;background:var(--cyan);opacity:.55"></div></div>`;}).join("")}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;color:var(--dim);font-size:9px">
        <span>${esc(daily[0].date)}</span>
        <span><span style="color:var(--amber)">&#9632;</span> generated &nbsp;<span style="color:var(--cyan)">&#9632;</span> prompt</span>
        <span>${esc(daily[daily.length-1].date)}</span></div>`
      :`<div class="note">No usage recorded yet - load a model and run some inference.</div>`}
    </div>
    <div class="card"><h3>Per-model Usage</h3>
      <div class="note" style="margin:0 0 6px">Usage is scraped from the router's own metrics and totalled per model across all clients. Per-client / per-IP breakdown isn't available: clients hit the llama.cpp router directly, so the dashboard never sees individual request origins.</div>
      ${rows.length?`<div class="toolbar" style="margin:6px 0 0">
        ${Object.keys(SORT_COLS).map(c=>`<span class="chip ${statsSort===c?"on":""}" data-sort="${c}">${SORT_COLS[c]}</span>`).join("")}
        <span class="chip" data-statsreset style="margin-left:auto;color:var(--red);border-color:var(--red)" title="zero all usage statistics">Reset stats</span>
      </div>
      <div class="list" style="margin-top:12px">${rows.map(m=>`
        <div class="row"><div class="rhead" ${m.has_history ? `data-historymodel="${esc(m.id)}" role="button" tabindex="0" aria-expanded="${expandedModel===m.id}"` : ''} style="cursor:${m.has_history ? 'pointer' : 'default'};grid-template-columns:9px 1fr auto auto auto auto auto">
          <span class="led ${loaded.includes(m.id)?"loaded":""}"></span>
          <span class="mid">${esc(m.id)}</span>
          <span class="ctxpill" title="prompt ${fmtNum(m.prompt)} + generated ${fmtNum(m.generated)}">${fmtNum(m.tokens)} tok</span>
          <span class="stat" title="average generation speed while active">${m.avg_tps?m.avg_tps+" tok/s":"-"}</span>
          <span class="stat">${fmtNum(m.runs)} runs</span>
          <span class="stat">${fmtDur(m.loaded_secs)}</span>
          <span class="stat">${fmtAgo(m.last_used)}</span>
        </div>${expandedModel===m.id ? historyDetail() : ''}</div>`).join("")}</div>`
      :`<div class="note">No models have logged usage yet.</div>`}
    </div>`);
}
