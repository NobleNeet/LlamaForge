/* New benchmark-core Auto Tune UI. It deliberately does not import models.js. */
import { api, esc, setHTML, toast } from "./core.js";

const sessions = new Map();
let bridge = null;
const terminal = new Set(["completed", "failed", "cancelled", "interrupted"]);
const stageLabel = {coarse: "Initial search", batch_probe: "Batch sizing", flash_probe: "Flash attention", kv_probe: "KV cache", validate: "Final validation"};
const RUN_STORAGE_KEY = "lf_autotune_runs_v1";

function savedRuns() {
  try { return JSON.parse(localStorage.getItem(RUN_STORAGE_KEY) || "{}"); }
  catch (_) { return {}; }
}
function saveRun(modelId, runId) {
  const runs = savedRuns();
  if (runId) runs[modelId] = runId; else delete runs[modelId];
  try { localStorage.setItem(RUN_STORAGE_KEY, JSON.stringify(runs)); } catch (_) { /* browser storage is optional */ }
}
function stateFor(modelId) {
  if (!sessions.has(modelId)) sessions.set(modelId, {runId: savedRuns()[modelId] || null, restored: false});
  return sessions.get(modelId);
}
function panel(modelId) { return document.querySelector(`[data-autotune-panel="${CSS.escape(modelId)}"]`); }
function profileCard(profile, result) {
  const settings = Object.entries(profile.settings || {}).map(([k,v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
  const latency = (result.derived_request_latencies || []).filter(x => x.candidate_id === profile.candidate_id)
    .map(x => `<span>${esc(x.request_workload_id)}: ${(Number(x.latency_seconds)||0).toFixed(2)} s</span>`).join(" ");
  const provenance = profile.provenance || {};
  const binary = provenance.bench_binary_identity || {};
  const identity = [provenance.model_fingerprint && `model ${String(provenance.model_fingerprint).slice(0, 12)}`,
    provenance.hardware_fingerprint && `hardware ${String(provenance.hardware_fingerprint).slice(0, 12)}`,
    binary.build_id && `build ${binary.build_id}`, provenance.scoring_schema_version].filter(Boolean).join(" · ");
  return `<section class="at-profile"><h4>${esc(profile.name.replace("_", " "))}</h4><div class="at-meta">${esc(provenance.backend || "unknown")} · ${esc(profile.evidence || "measured")}</div><div class="at-settings">${settings}</div>${latency ? `<div class="at-latency">${latency}</div>` : ""}${identity ? `<div class="at-meta">${esc(identity)}</div>` : ""}<button class="qbtn" data-autotune-preview="${esc(profile.name)}">Preview</button></section>`;
}
function stageProgress(item) {
  const counts = item.counts || {};
  const done = Number(counts.succeeded || 0) + Number(counts.failed || 0) + Number(counts.skipped || 0);
  const extras = [counts.failed && `${counts.failed} failed`, counts.skipped && `${counts.skipped} skipped`].filter(Boolean);
  return `${done} / ${item.cases || 0} cases${extras.length ? ` · ${extras.join(" · ")}` : ""}`;
}
function stageHistory(progress) {
  return (progress.stages || []).map(item => {
    const current = item.stage_index === progress.stage_index;
    const marker = current ? "›" : item.status === "completed" ? "✓" : item.status === "partial" ? "!" : "·";
    return `<div class="at-stage${current ? " current" : ""}"><b>${marker}</b><span>${esc(stageLabel[item.stage_id] || item.stage_id)}</span><small>${esc(stageProgress(item))}</small></div>`;
  }).join("");
}
function render(modelId) {
  const el = panel(modelId); if (!el) return;
  const s = stateFor(modelId);
  if (!s.runId) { setHTML(el, `<div class="autotune"><span class="tunebar-label">Auto Tune</span><button class="qbtn" data-autotune-start ${s.starting ? "disabled" : ""}>${s.starting ? "Starting..." : "Run Auto Tune"}</button></div>`); return; }
  const status = s.status || "planned", progress = s.progress || {}, counts = progress.counts || {};
  let body = `<div class="autotune"><span class="tunebar-label">Auto Tune</span><b>${esc(status)}</b>`;
  if (!terminal.has(status)) {
    if (progress.stage_id) {
      const stageNumber = Number(progress.stage_index || 0) + 1, stageCount = progress.stage_count || "?";
      const waiting = progress.status === "waiting_for_resource" ? "Waiting for benchmark resource..." : "";
      body += `<div class="at-stage-title">Stage ${stageNumber} / ${esc(stageCount)} · ${esc(stageLabel[progress.stage_id] || progress.stage_id)}</div><div class="at-progress">${waiting || `${Number(counts.succeeded||0) + Number(counts.failed||0) + Number(counts.skipped||0)} / ${esc(progress.cases || "?")} cases in this stage`}</div>${waiting ? `<div class="at-progress">${Number(counts.succeeded||0) + Number(counts.failed||0) + Number(counts.skipped||0)} / ${esc(progress.cases || "?")} cases in this stage</div>` : ""}<div class="at-stages">${stageHistory(progress)}</div>`;
    } else body += `<div class="at-progress">Preparing benchmark...</div>`;
    body += `<button class="qbtn stop" data-autotune-cancel ${s.cancelling ? "disabled" : ""}>${s.cancelling ? "Cancelling..." : "Cancel"}</button>`;
  }
  if (status === "failed" && s.error) body += `<div class="msg err">${esc(s.error.message || "Auto Tune failed.")}</div>`;
  setHTML(el, body + "</div>");
  if (status === "completed") {
    body += `<button class="qbtn" data-autotune-rerun ${s.starting ? "disabled" : ""}>${s.starting ? "Starting..." : "Run again"}</button>`;
    if (s.result) body += `<div class="at-progress">Completed · ${esc(progress.stage_count || (progress.stages || []).length)} stages</div><div class="at-stages">${stageHistory(progress)}</div><div class="at-profiles">${(s.result.profiles || []).map(profile => profileCard(profile, s.result)).join("")}</div>`;
  }
  setHTML(el, body + "</div>");
}
async function poll(modelId) {
  const s = stateFor(modelId); if (!s.runId || s.polling) return;
  s.polling = true;
  try {
    const response = await api(`/api/autotune/status?run_id=${encodeURIComponent(s.runId)}`);
    if (response.error && !response.status) {
      if (String(response.error).includes("unknown")) { saveRun(modelId); sessions.set(modelId, {restored: true}); }
      else { s.status = "failed"; s.error = {message: "Auto Tune status is unavailable."}; }
      render(modelId); return;
    }
    Object.assign(s, response); render(modelId);
    if (terminal.has(s.status)) {
      if (s.status === "completed") s.result = await api(`/api/autotune/result?run_id=${encodeURIComponent(s.runId)}`);
      render(modelId); return;
    }
    s.timer = setTimeout(() => poll(modelId), 1500);
  } finally { s.polling = false; }
}
async function start(modelId, rerun = false) {
  const model = bridge.model(modelId), path = model?.settings?.model, s = stateFor(modelId);
  if (!path) { toast("This model has no GGUF path", "err"); return; }
  if (s.starting || (s.runId && !rerun)) return;
  s.starting = true; render(modelId);
  const response = await api("/api/autotune/start", {model_path: path}); s.starting = false;
  if (response.run_id) { s.runId = response.run_id; s.status = response.status || "planned"; s.result = null; saveRun(modelId, s.runId); poll(modelId); return; }
  toast(response.error || "Auto Tune could not start", "err"); render(modelId);
}
async function preview(modelId, name) {
  const s = stateFor(modelId), response = await api("/api/autotune/preview", {run_id: s.runId, profile: name, model: modelId});
  if (!response.settings) { toast(response.error || "Profile preview is unavailable", "err"); return; }
  s.preview = response;
  const rows = (response.changes || []).map(change => `<tr class="${String(change.current ?? "") === String(change.recommended) ? "" : "diff"}"><td>${esc(change.key)}</td><td>${esc(change.current ?? "inherit")}</td><td>${esc(change.recommended)}</td></tr>`).join("");
  const warning = (response.warnings || []).map(x => `<div class="msg err">${esc(x.message)}</div>`).join("");
  bridge.modal(`Auto Tune - ${name}`, `<table class="cmptbl"><tr><th>knob</th><th>Current</th><th>Recommended</th></tr>${rows}</table>${warning}<div class="actions"><button class="primary" data-autotune-load="${esc(modelId)}" ${response.applicable ? "" : "disabled"}>Load into editor</button><button class="ghost" data-mclose>Cancel</button></div>`);
}
function requestLoad(modelId) {
  const s = stateFor(modelId); if (!s.preview) return;
  if (bridge.unsaved(modelId)) { bridge.modal("Replace unsaved edits?", `<div class="note">You have unsaved knob changes. Loading this Auto Tune profile will replace edited fields.</div><div class="actions"><button class="primary" data-autotune-confirm="${esc(modelId)}">Replace edits</button><button class="ghost" data-mclose>Cancel</button></div>`); return; }
  bridge.stage(modelId, s.preview.settings); bridge.closeModal();
}
function confirmLoad(modelId) { const s = stateFor(modelId); bridge.stage(modelId, s.preview.settings); bridge.closeModal(); }

export function syncAutoTune(model) {
  const s = stateFor(model.id); render(model.id);
  if (s.runId && !s.restored) { s.restored = true; poll(model.id); }
}
export function initAutoTune(interface_) {
  bridge = interface_;
  document.addEventListener("click", async event => {
    const startButton = event.target.closest("[data-autotune-start]");
    if (startButton) { event.preventDefault(); start(startButton.closest("[data-autotune-panel]").dataset.autotunePanel); return; }
    const rerunButton = event.target.closest("[data-autotune-rerun]");
    if (rerunButton) { event.preventDefault(); start(rerunButton.closest("[data-autotune-panel]").dataset.autotunePanel, true); return; }
    const cancelButton = event.target.closest("[data-autotune-cancel]");
    if (cancelButton) { const modelId = cancelButton.closest("[data-autotune-panel]").dataset.autotunePanel, s = stateFor(modelId); s.cancelling = true; render(modelId); await api("/api/autotune/cancel", {run_id: s.runId}); poll(modelId); return; }
    const previewButton = event.target.closest("[data-autotune-preview]");
    if (previewButton) { preview(previewButton.closest("[data-autotune-panel]").dataset.autotunePanel, previewButton.dataset.autotunePreview); return; }
    const loadButton = event.target.closest("[data-autotune-load]");
    if (loadButton) { requestLoad(loadButton.dataset.autotuneLoad); return; }
    const confirmButton = event.target.closest("[data-autotune-confirm]");
    if (confirmButton) confirmLoad(confirmButton.dataset.autotuneConfirm);
  });
}
