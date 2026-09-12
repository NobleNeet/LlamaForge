// First-run wizard: engine -> hardware -> model -> tune -> load.
// Talks to the model list only through the bus, so neither imports the other.
import { $, esc, setHTML, api, toast } from "./core.js";
import { S, models } from "./state.js";
import { on, emit } from "./bus.js";
import { switchTab, applyMode } from "./ui.js";

const WIZ = {step:0, engine:null, model:null, intent:"balanced", rec:null,
  steps:["engine","hardware","model","tune","load"]};

function wizShow() { $("#wizard").hidden = false; wizRender(); }
function wizHide() { $("#wizard").hidden = true; }

function wizRender() {
  const kind = WIZ.steps[WIZ.step];
  const body = $("#wizard-body");
  ({engine:wizEngine, hardware:wizHardware, model:wizModel,
    tune:wizTune, load:wizLoad}[kind])(body);
  $("#wiz-back").disabled = WIZ.step === 0;
}

function wizEngine(body) {
  setHTML(body, `<div class="wizard-step"><h2>llama.cpp engine</h2>
    <p>Do you already have a llama.cpp build?</p>
    <label><input type="radio" name="eng" value="have" checked> Yes, I have one built</label><br>
    <label><input type="radio" name="eng" value="clone"> No — clone &amp; build it for me</label>
    <div style="margin-top:10px">
      <label>Flavor:
        <select id="eng-flavor">
          <option value="official">official llama.cpp</option>
          <option value="fork">mainline fork</option>
          <option value="ik" disabled>ik_llama (coming soon)</option>
        </select>
      </label>
    </div></div>`);
}

function wizHardware(body) {
  const g = (S.STATE && S.STATE.gpus) || [];
  const rows = (g.length && !g[0].error)
    ? g.map(x => `<li>${esc(x.name||"GPU")} — ${esc(x.total?(x.total/1024).toFixed(1):"?")} GB</li>`).join("")
    : "";
  setHTML(body, `<div class="wizard-step"><h2>Your hardware</h2>
    <ul>${rows||"<li>No GPU detected — CPU mode.</li>"}</ul></div>`);
}

function wizModel(body) {
  const ms = models().filter(m => m.backend !== "vllm");
  if (!ms.length) {
    setHTML(body, `<div class="wizard-step"><h2>Pick a model</h2>
      <p>No models found. <a href="#" id="wiz-discover">Find one in Discover</a>.</p></div>`);
    const d = $("#wiz-discover");
    if (d) d.onclick = e => { e.preventDefault(); wizHide(); switchTab("discover"); };
    return;
  }
  const opts = ms.map(m => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join("");
  setHTML(body, `<div class="wizard-step"><h2>Pick a model</h2>
    <select id="wiz-model">${opts}</select></div>`);
  const sel = $("#wiz-model"); if (sel) WIZ.model = sel.value;
}

async function wizTune(body) {
  const model = WIZ.model;
  WIZ.rec = null;
  setHTML(body, `<div class="wizard-step"><h2>Static presets</h2>
    <p>No model execution or benchmark. Choose memory headroom, then review the settings.</p>
    <select id="wiz-intent" disabled>
      <option value="safe">Safe</option>
      <option value="balanced" selected>Balanced</option>
      <option value="aggressive">Aggressive</option>
    </select><div id="wiz-tune-out">Calculating presets…</div></div>`);
  try {
    const r = await api("/api/autotune/recommend", {model});
    if (WIZ.model !== model || WIZ.steps[WIZ.step] !== "tune") return;
    if (r.error) throw new Error(r.error);
    const sel = $("#wiz-intent");
    sel.disabled = false;
    const choose = () => {
      WIZ.intent = sel.value;
      const rec = r[sel.value];
      WIZ.rec = rec.applicable ? rec : null;
      wizRenderRec(rec);
    };
    sel.onchange = choose; choose();
  } catch (e) {
    if (WIZ.steps[WIZ.step] === "tune") setHTML($("#wiz-tune-out"), `<p>${esc(String(e))}</p>`);
  }
}

function wizRenderRec(r) {
  const rows = Object.entries(r.knobs||{}).map(([k,v]) =>
    `<tr><td>${esc(k)}</td><td>${esc(v)}</td><td class="wizard-rationale">${esc((r.rationale||{})[k]||"")}</td></tr>`).join("");
  setHTML($("#wiz-tune-out"), `${(r.warnings || []).map(w => `<p>${esc(w)}</p>`).join('')}<table>${rows}</table>`);
}

function wizLoad(body) {
  setHTML(body, `<div class="wizard-step"><h2>Ready</h2>
    <p>${WIZ.rec ? "Apply the reviewed preset to" : "Use existing settings for"} <b>${esc(WIZ.model)}</b> and load it now. You can edit individual values in the Models tab.</p></div>`);
}

async function wizNext() {
  const kind = WIZ.steps[WIZ.step];
  if (kind === "engine") {
    const sel = document.querySelector('input[name="eng"]:checked');
    WIZ.engine = sel ? sel.value : "have";
    // "clone" path reuses the Build tab flow; minimal wizard triggers it then continues.
  }
  if (kind === "model") {
    const sel = $("#wiz-model"); if (sel) WIZ.model = sel.value;
    if (!WIZ.model) return;                    // require a model
  }
  if (kind === "load") {
    try {
      if (WIZ.rec) await api("/api/save", {model: WIZ.model, settings: WIZ.rec.knobs});
      // detect both thrown errors and failed responses
      let loadErr = false;
      try {
        const r = await api("/api/load", {model: WIZ.model});
        if (!r.success) loadErr = true;
      } catch (e) { loadErr = true; }
      // always persist onboarding and close, regardless of load outcome
      await api("/api/config", {onboarded: true, ui_mode: "lite"});
      applyMode("lite"); wizHide(); emit("refresh", true);
      toast(loadErr ? "Setup done — model failed to load; load it from the Models tab"
                    : "Setup complete", loadErr ? "err" : "ok");
    } catch (e) { toast("Setup failed", "err"); }
    return;
  }
  WIZ.step = Math.min(WIZ.step + 1, WIZ.steps.length - 1);
  wizRender();
}

export function initWizard() {
  $("#wiz-next").onclick = wizNext;
  $("#wiz-back").onclick = () => { WIZ.step = Math.max(0, WIZ.step-1); wizRender(); };
  $("#wiz-skip").onclick = async () => {
    await api("/api/config", {onboarded: true, ui_mode: "advanced"});
    applyMode("advanced"); wizHide();
  };
  // show it the first time state says onboarding hasn't happened
  on("state", s => {
    const el = $("#wizard");
    if (!(s.onboarding||{}).onboarded && el && el.hidden) wizShow();
  });
}
