// Setup tab: prerequisites, detected hardware, drive scanning, startup options,
// LAN access, the agent-connect panel, and vLLM/WSL installation.
import { $, $$, esc, setHTML, api, toast } from "./core.js";
import { S, models, config as cfgOf } from "./state.js";
import { emit } from "./bus.js";

let vllmSetupPoll = null;

function normalizeScanDirs(values) {
  const seen = new Set();
  const out = [];
  (values || []).forEach(v => {
    const s = String(v == null ? "" : v).trim();
    if (!s || seen.has(s)) return;
    seen.add(s);
    out.push(s);
  });
  return out;
}

function scanDirRows(values) {
  const rows = normalizeScanDirs(values);
  if (!rows.length) {
    return `<div class="row scan-dir-row">
      <input class="scan-dir-input" placeholder="/home/me/.lmstudio/models or /mnt/models" style="flex:1">
      <button type="button" class="ghost" data-scan-dir-add>Add</button>
    </div>`;
  }
  return rows.map((path, i) => `<div class="row scan-dir-row">
      <input class="scan-dir-input" value="${esc(path)}" placeholder="/home/me/.lmstudio/models or /mnt/models" style="flex:1">
      <button type="button" class="ghost" data-scan-dir-remove="${i}">Remove</button>
    </div>`).join("")
    + `<div class="row scan-dir-row">
      <input class="scan-dir-input" placeholder="/another/path" style="flex:1">
      <button type="button" class="ghost" data-scan-dir-add>Add</button>
    </div>`;
}

function currentScanDirs() {
  return normalizeScanDirs($$(".scan-dir-input").map(el => el.value));
}

function renderScanDirEditor(values) {
  const host = $("#scan-dir-list");
  if (!host) return;
  setHTML(host, scanDirRows(values));
  $$("[data-scan-dir-remove]", host).forEach(btn => btn.onclick = () => {
    const next = currentScanDirs();
    next.splice(Number(btn.dataset.scanDirRemove), 1);
    renderScanDirEditor(next);
  });
  $$("[data-scan-dir-add]", host).forEach(btn => btn.onclick = () => {
    const next = currentScanDirs();
    renderScanDirEditor([...next, ""]);
    const inputs = $$(".scan-dir-input", host);
    if (inputs.length) inputs[inputs.length - 1].focus();
  });
}

function pollVllmSetup() {
  clearInterval(vllmSetupPoll);
  const log = $("#vllm-setup-log"); if (log) log.style.display = "";
  const tick = async () => {
    const s = await api("/api/vllm/setup");
    const l = $("#vllm-setup-log");
    if (l) { l.textContent = s.setup_log||"idle"; l.scrollTop = l.scrollHeight; }
    const msg = $("#vllm-inst-msg");
    const job = s.setup_job || {};
    if (job.running) { if (msg) { msg.className = "msg work"; msg.textContent = "installing..."; } }
    else if (job.phase === "done") {
      if (msg) { msg.className = "msg ok"; msg.textContent = "installed"; }
      clearInterval(vllmSetupPoll); toast("vLLM installed", "ok"); setTimeout(loadSetup, 1200);
    } else if (job.phase === "failed") {
      if (msg) { msg.className = "msg err"; msg.textContent = "install failed - see log"; }
      clearInterval(vllmSetupPoll);
    }
  };
  tick();
  vllmSetupPoll = setInterval(tick, 2000);
}

export async function loadSetup() {
  const v = $("#view-setup");
  setHTML(v, `<div class="skel">PROBING SYSTEM...</div>`);
  const [s, net, vs] = await Promise.all([api("/api/setup"), api("/api/network"), api("/api/vllm/setup")]);
  const p = s.prereqs, hw = s.hardware;
  const toolRow = (name, t) => `<div class="kv"><span class="k">${esc(name)}</span>
    <span class="v ${t.present?'ok':'bad'}">${t.present?esc(t.version||"present"):"MISSING"}
    ${!t.present&&t.installable?` <button data-install="${esc(name)}" style="padding:3px 8px;margin-left:8px">Install</button>`:""}
    ${!t.present&&!t.installable&&t.hint?`<div class="note" style="margin-top:4px">${esc(t.hint)}</div>`:""}</span></div>`;
  const accelRow = (label, a) => {
    if (!a || a.applicable === false) return "";
    return `<div class="kv"><span class="k">${esc(label)}</span><span class="v ${a.present?'ok':'bad'}">${a.present?esc(a.version||"present"):"not found"}${!a.present&&a.hint?`<div class="note" style="margin-top:4px">${esc(a.hint)}</div>`:""}</span></div>`;
  };
  const gpuLines = (hw.gpus||[]).map(g => {
    const be = (g.backends||[]).join(", ") || "none";
    const arch = g.architecture ? ` &middot; ${esc(g.architecture)}` : "";
    const total = g.memory_total_mib ?? g.vram_mib;
    const mem = g.is_uma ? ` &middot; ${((total||0)/1024).toFixed(1)} GB unified` : (total ? ` &middot; ${(total/1024).toFixed(1)} GB` : "");
    return `<div class="kv"><span class="k">GPU ${esc(g.index)}</span><span class="v">${esc(g.vendor||"GPU")} &middot; ${esc(g.name)}${arch}${mem} &middot; backends: ${esc(be)}${g.is_uma?" &middot; UMA":""}</span></div>`;
  }).join("");
  const avail = (hw.available_backends||[]).length ? hw.available_backends.join(", ") : "cpu";
  const bw = cfgOf().vram_bandwidths || {};
  setHTML(v, `
    <div class="card"><h3>Prerequisites</h3>
      ${Object.entries(p.tools).map(([n,t])=>toolRow(n,t)).join("")}
      <div class="kv"><span class="k">${esc(p.msvc.label||"C++ compiler")}</span><span class="v ${p.msvc.present?'ok':'bad'}">${p.msvc.present?"present":"MISSING"+(p.msvc.url?" &mdash; "+esc(p.msvc.url):"")}</span></div>
      ${p.cuda.applicable===false?"":`<div class="kv"><span class="k">CUDA toolkit</span><span class="v ${p.cuda.present?'ok':'bad'}">${p.cuda.present?esc(p.cuda.version||"present"):"not found (CPU build only)"}</span></div>`}
      ${accelRow("ROCm / HIP", p.rocm)}
      ${accelRow("Vulkan", p.vulkan)}
      <div class="kv"><span class="k">installers</span><span class="v">${esc(Object.keys(p.installers||{}).filter(k=>p.installers[k]).join(" ")||"none")}</span></div>
      <div class="note">Missing prerequisites can be installed with your permission where a package manager allows it (winget/choco/brew). On Linux the exact install command is shown instead &mdash; the dashboard never runs sudo.</div>
    </div>
    <div class="card"><h3>Detected Hardware</h3>
      <div class="kv"><span class="k">CPU</span><span class="v">${esc(hw.cpu.name||"?")} (${esc(hw.cpu.cores||"?")}c/${esc(hw.cpu.threads||"?")}t)</span></div>
      ${gpuLines}
      <div class="kv"><span class="k">Available acceleration</span><span class="v">${esc(avail)}</span></div>
      <div class="kv"><span class="k">Selected llama.cpp backend</span><span class="v">${esc((cfgOf().llama_backend||"auto"))} &rarr; ${esc(hw.selected_backend||"cpu")}</span></div>
      <div class="flags">${Object.entries(hw.cmake_flags).map(([k,val])=>`<span class="flagpill">${esc(k)}=${esc(val)}</span>`).join("")}</div>
      ${hw.notes.map(n=>`<div class="note">&bull; ${esc(n)}</div>`).join("")}
    </div>
    <div class="card"><h3>Speed Estimates <span style="color:var(--dim);font-weight:normal;font-size:11px">(advanced &mdash; optional)</span></h3>
      <div class="note">The "Will it run?" panel and Discover speed badges estimate tok/s from memory bandwidth. Detected GPU presets are used by default; override here only if you've measured your machine. Blank = use the preset/default.</div>
      <div class="row" style="gap:8px;margin-top:10px;flex-wrap:wrap;align-items:flex-end">
        <div class="fld"><label>VRAM GB/s</label><input id="bw-vram" type="number" min="0" step="any" placeholder="preset" value="${esc(String(bw.vram_bw ?? ""))}" style="width:110px"></div>
        <div class="fld"><label>RAM GB/s</label><input id="bw-ram" type="number" min="0" step="any" placeholder="50" value="${esc(String(bw.ram_bw ?? ""))}" style="width:110px"></div>
        <div class="fld"><label>Disk GB/s</label><input id="bw-disk" type="number" min="0" step="any" placeholder="5.7" value="${esc(String(bw.disk_bw ?? ""))}" style="width:110px"></div>
        <button id="bw-save">Save</button>
        <span class="msg" id="bw-msg"></span>
      </div>
    </div>
    <div class="card"><h3>Scan Drives for Models</h3>
      <div class="note">Limit GGUF discovery to specific directories. Leave this list empty to fall back to the default system roots.</div>
      <div id="scan-dir-list" style="margin-top:10px"></div>
      <div class="actions">
        <button id="btn-scan-save" class="ghost">Save scan directories</button>
        <button id="btn-scan">Scan for GGUF models</button>
        <button class="ghost" id="btn-missing">Check for deleted models</button>
        <span class="msg" id="scan-msg"></span>
      </div>
      <div id="scan-out"></div>
      <div id="missing-out"></div>
    </div>
    <div class="card"><h3>Startup</h3>
      <div class="kv"><span class="k">auto-load a model on launch</span>
        <span class="v"><select id="auto-load" style="background:var(--inset);border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:6px">
          <option value="">none</option>
          ${models().map(m=>`<option value="${esc(m.id)}" ${cfgOf().auto_load_model===m.id?"selected":""}>${esc(m.id)}</option>`).join("")}
        </select></span></div>
      <div class="note">The selected model loads automatically once the router is ready after launch &mdash; handy for always-on setups. An optional tray icon (loaded-model count, quick open) is available if you <b>pip install pystray pillow</b>; without them LlamaForge stays pure-stdlib.</div>
    </div>
    <div class="card"><h3>Network Access</h3>
      <div class="kv"><span class="k">router status</span><span class="v ${net.router_running?'ok':'bad'}">${net.router_running?"running":"not running"}</span></div>
      <div class="kv"><span class="k">currently bound to</span><span class="v">${esc(net.host)}:${esc(net.port)}${net.host!=="127.0.0.1"?" (LAN-accessible)":" (local only)"}</span></div>
      <div class="kv"><span class="k">this machine's LAN IP</span><span class="v">${esc(net.lan_ip||"not detected")}</span></div>
      <div class="note">By default the router only answers on 127.0.0.1 (this machine only). Enabling LAN access lets other devices on your network reach it at <b>http://${esc(net.lan_ip||"<lan-ip>")}:${esc(net.port)}/</b> &mdash; with no key set, anyone on your network can use it unauthenticated. An API key is optional but recommended.</div>
      <div class="actions" style="margin-top:10px">
        <label style="font-size:11px;color:var(--dim)"><input type="checkbox" id="net-lan" ${net.host!=="127.0.0.1"?"checked":""}> allow access from other devices on my network</label>
      </div>
      <div id="net-keyrow" style="display:${net.host!=="127.0.0.1"?"":"none"};margin-top:10px">
        <label style="font-size:11px;color:var(--dim);display:block;margin-bottom:8px"><input type="checkbox" id="net-require-key" checked> require an API key (won't enable LAN access until a key is set)</label>
        <div class="fld"><label>API key (clients send it as Authorization: Bearer &lt;key&gt;)</label>
          <input id="net-apikey" value="" placeholder="${net.has_api_key?"(unchanged - a key is already set)":"leave blank for no key"}">
        </div>
      </div>
      <div class="actions" style="margin-top:10px">
        <button class="primary" id="btn-net-apply">Apply &amp; Restart Router</button>
        <button class="ghost" id="btn-net-genkey" style="display:${net.host!=="127.0.0.1"?"":"none"}">Generate Key</button>
        <span class="msg" id="net-msg"></span>
      </div>
    </div>
    <div id="agent-connect" class="card"></div>`
    + (vs.supported === false ? "" : `<div class="card"><h3>vLLM Backend (WSL2)</h3>
      <div class="kv"><span class="k">WSL2</span><span class="v ${vs.wsl.present?'ok':'bad'}">${vs.wsl.present?"installed":"NOT INSTALLED"}</span></div>
      ${vs.wsl.present?`<div class="kv"><span class="k">distro</span><span class="v">
        <select id="vllm-distro" style="background:var(--inset);border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:6px">
        ${(vs.distros||[]).map(d=>`<option value="${esc(d.name)}" ${d.name===vs.chosen?"selected":""}>${esc(d.name)} (${esc(d.state)})</option>`).join("")}
        </select></span></div>
      <div class="kv"><span class="k">GPU passthrough</span><span class="v ${vs.gpu.present?'ok':'bad'}">${vs.gpu.present?esc((vs.gpu.info||"").split("\n")[0]||"detected"):"NOT DETECTED (check NVIDIA driver)"}</span></div>
      <div class="kv"><span class="k">vLLM</span><span class="v ${vs.vllm.present?'ok':'bad'}">${vs.vllm.present?"v"+esc(vs.vllm.version):"not installed"}</span></div>`
      :`<div class="note">WSL2 is required to run vLLM. Install it (admin PowerShell): <b>wsl --install -d Ubuntu</b>, reboot, then reload this tab.</div>`}
      ${vs.wsl.present&&!vs.vllm.present?`<div class="actions"><button class="primary" id="btn-vllm-install">Install vLLM (uv, no sudo)</button><span class="msg" id="vllm-inst-msg"></span></div>
      <div class="note">Downloads uv + a standalone Python and installs vLLM into ~/.llamaforge/vllm-venv. Several GB; watch the log.</div>`:""}
      <div class="log" id="vllm-setup-log" style="display:${(vs.setup_job&&vs.setup_job.running)?"":"none"}">${esc(vs.setup_log||"idle")}</div>
    </div>`));
  $$("[data-install]", v).forEach(b => b.onclick = async () => {
    b.disabled = true; b.textContent = "installing...";
    const r = await api("/api/setup/install", {tool: b.dataset.install});
    toast(r.ok?"Installed":"Install failed", r.ok?"ok":"err"); loadSetup();
  });
  renderScanDirEditor((cfgOf().model_dirs || []).slice());
  const scanSave = $("#btn-scan-save");
  if (scanSave) scanSave.onclick = async () => {
    const dirs = currentScanDirs();
    await api("/api/config", {model_dirs: dirs});
    const msg = $("#scan-msg");
    msg.className = "msg ok";
    msg.textContent = dirs.length ? `saved ${dirs.length} scan director${dirs.length === 1 ? "y" : "ies"}` : "saved default scan roots";
  };
  $("#btn-scan").onclick = scanDrives;
  $("#btn-missing").onclick = checkMissing;
  const autoSel = $("#auto-load");
  if (autoSel) autoSel.onchange = async () => {
    await api("/api/config", {auto_load_model: autoSel.value});
    toast(autoSel.value?`Auto-load: ${autoSel.value}`:"Auto-load disabled", "ok");
  };
  const bwSave = $("#bw-save");
  if (bwSave) bwSave.onclick = async () => {
    const num = sel => { const val = $(sel).value.trim(); return val === "" ? undefined : Number(val); };
    const ov = {};
    const vram = num("#bw-vram"), ram = num("#bw-ram"), disk = num("#bw-disk");
    if (vram !== undefined && !Number.isNaN(vram)) ov.vram_bw = vram;
    if (ram !== undefined && !Number.isNaN(ram)) ov.ram_bw = ram;
    if (disk !== undefined && !Number.isNaN(disk)) ov.disk_bw = disk;
    await api("/api/config", {vram_bandwidths: ov});
    const m = $("#bw-msg"); m.className = "msg ok"; m.textContent = Object.keys(ov).length ? "saved" : "cleared (using defaults)";
  };
  $("#net-lan").onchange = e => {
    $("#net-keyrow").style.display = e.target.checked ? "" : "none";
    $("#btn-net-genkey").style.display = e.target.checked ? "" : "none";
  };
  $("#btn-net-genkey").onclick = () => {
    $("#net-apikey").value = [...crypto.getRandomValues(new Uint8Array(24))]
      .map(b => b.toString(16).padStart(2,"0")).join("");
  };
  $("#btn-net-apply").onclick = async () => {
    const msg = $("#net-msg"), lan = $("#net-lan").checked;
    const host = lan ? "0.0.0.0" : "127.0.0.1";
    const apiKey = $("#net-apikey").value.trim();
    if (lan && $("#net-require-key").checked && !apiKey && !net.has_api_key) {
      msg.className = "msg err";
      msg.textContent = 'set or generate an API key first (or uncheck "require an API key")';
      return;
    }
    msg.className = "msg work"; msg.textContent = "restarting router...";
    const r = await api("/api/network", {host, api_key: lan?(apiKey||undefined):""});
    if (r.ok) {
      msg.className = "msg ok"; msg.textContent = "applied";
      toast(lan?"LAN access enabled":"LAN access disabled", "ok");
      setTimeout(loadSetup, 1500);
    } else { msg.className = "msg err"; msg.textContent = r.error || "failed"; }
  };
  const distroSel = $("#vllm-distro");
  if (distroSel) distroSel.onchange = () => api("/api/config", {wsl_distro: distroSel.value}).then(() => loadSetup());
  const instBtn = $("#btn-vllm-install");
  if (instBtn) instBtn.onclick = async () => {
    const msg = $("#vllm-inst-msg"); msg.className = "msg work"; msg.textContent = "starting install...";
    const r = await api("/api/vllm/setup/install", {distro: distroSel?distroSel.value:undefined});
    if (r.started) { toast("vLLM install started", "ok"); pollVllmSetup(); }
    else msg.textContent = "already running";
  };
  if (vs.setup_job && vs.setup_job.running) pollVllmSetup();
  renderAgentConnect();
}

/* ---------- connect an agent ---------- */
function agentModelOptions(sel) {
  return models().map(m => `<option value="${esc(m.id)}"${m.id===sel?" selected":""}>${esc(m.id)}</option>`).join("");
}
function renderAgentConnect() {
  const host = $("#agent-connect"); if (!host) return;
  setHTML(host, `<h3>Connect an agent</h3>
    <div class="note">Point a coding agent at your local models. Claude Code uses the
      Anthropic-compatible endpoint; Codex and pi.dev use the OpenAI-compatible router.</div>
    <div style="margin-top:8px">
      <select id="ac-agent">
        <option value="claude-code">Claude Code</option>
        <option value="codex">Codex</option>
        <option value="pi">pi.dev</option>
      </select>
      <select id="ac-model">${agentModelOptions()}</select>
      <select id="ac-small" hidden>${agentModelOptions()}</select>
      <button id="ac-apply" class="primary">Apply</button>
    </div>
    <div id="ac-out" class="agent-out"></div>`);
  const agentSel = $("#ac-agent"), smallSel = $("#ac-small");
  const syncSmall = () => { smallSel.hidden = agentSel.value !== "claude-code"; };
  syncSmall();
  agentSel.onchange = () => { syncSmall(); loadAgentConfig(); };
  $("#ac-model").onchange = loadAgentConfig;
  smallSel.onchange = loadAgentConfig;
  $("#ac-apply").onclick = applyAgentConfig;
  loadAgentConfig();
}
async function loadAgentConfig() {
  const agentSel = $("#ac-agent"), modelSel = $("#ac-model"), smallSel = $("#ac-small");
  if (!agentSel || !modelSel || !smallSel || !modelSel.value) {
    setHTML($("#ac-out"), `<div class="note">No models available yet &mdash; scan for models above first.</div>`);
    return;
  }
  const agent = agentSel.value, model = modelSel.value;
  const small = smallSel.hidden ? "" : smallSel.value;
  let q = `/api/agent/config?agent=${encodeURIComponent(agent)}&model=${encodeURIComponent(model)}`;
  if (small) q += `&small=${encodeURIComponent(small)}`;
  const r = await api(q);
  if (r.error) { setHTML($("#ac-out"), `<div class="note" style="color:var(--red)">${esc(r.error)}</div>`); return; }
  const snip = (label, text) => `<div class="slabel">${esc(label)}</div><div class="snip"><button class="qbtn scopy" data-copytext="${esc(text)}">Copy</button>${esc(text)}</div>`;
  setHTML($("#ac-out"),
    `<div class="note">Target: <b>${esc(r.target_path)}</b> &middot; endpoint <b>${esc(r.endpoint)}</b><br>${esc(r.instructions)}</div>`
    + snip(r.target_path, r.content));
}
async function applyAgentConfig() {
  const agentSel = $("#ac-agent"), modelSel = $("#ac-model"), smallSel = $("#ac-small");
  if (!modelSel || !modelSel.value) { toast("No model selected", "err"); return; }
  const small = smallSel.hidden ? "" : smallSel.value;
  const r = await api("/api/agent/apply", {agent: agentSel.value, model: modelSel.value, small});
  if (r.error) { toast(r.error, "err"); return; }
  toast(`${r.action}: ${r.path}${r.backup?` (backup: ${r.backup})`:""}`, "ok");
}

/* ---------- drive scanning ---------- */
async function scanDrives() {
  const roots = currentScanDirs();
  const msg = $("#scan-msg");
  msg.className = "msg work";
  msg.textContent = roots.length ? "scanning selected directories..." : "scanning default roots...";
  const r = await api("/api/scan", {roots});
  const known = new Set(models().map(m => m.id));
  const fresh = r.entries.filter(e => !known.has(e.id));
  msg.className = "msg ok"; msg.textContent = `${r.entries.length} found, ${fresh.length} new`;
  const scanned = (r.roots || roots).length
    ? `<div class="note">Scanned: ${esc((r.roots || roots).join(", "))}</div>`
    : `<div class="note">Scanned the default system roots.</div>`;
  setHTML($("#scan-out"), `${scanned}<div class="note" style="margin-top:8px">${esc(fresh.length)} new models not yet in your config:</div>
    <div class="list" style="margin-top:10px">${fresh.map(e=>`<div class="row"><div class="rhead" style="cursor:default;grid-template-columns:1fr auto">
      <span class="mid">${esc(e.id)}${e.mmproj?'<span class="tag vis">vision</span>':''}${e.embeddings?'<span class="tag">embed</span>':''}</span>
      <span class="ctxpill">${esc(e.gib)} GiB</span></div></div>`).join("")||'<div class="note">nothing new</div>'}</div>
    ${fresh.length?`<div class="actions"><button class="primary" id="btn-apply">Add ${fresh.length} models to config</button><span class="msg" id="apply-msg"></span></div>`:""}`);
  if (fresh.length) $("#btn-apply").onclick = async () => {
    const am = $("#apply-msg"); am.className = "msg work"; am.textContent = "writing config...";
    const rr = await api("/api/scan/apply", {entries: fresh});
    am.className = "msg ok"; am.textContent = `added ${rr.added}`;
    toast("Models added", "ok"); emit("refresh", true);
  };
}

async function checkMissing() {
  const out = $("#missing-out");
  setHTML(out, `<div class="note">checking configured models against disk...</div>`);
  let r;
  try { r = await api("/api/scan/missing"); }
  catch (e) { setHTML(out, `<div class="note" style="color:var(--red)">backend unreachable</div>`); return; }
  const miss = (r && r.missing) || [];
  if (!miss.length) { setHTML(out, `<div class="note">All configured models still exist on disk.</div>`); return; }
  setHTML(out, `<div class="note">${esc(miss.length)} configured model(s) whose file is gone:</div>
    <div class="list" style="margin-top:10px">${miss.map(m=>`<div class="row"><div class="rhead" style="cursor:default;grid-template-columns:1fr auto">
      <span class="mid">${esc(m.id)}${m.loaded?'<span class="tag">loaded</span>':''}</span>
      <span class="ctxpill" title="${esc(m.model)}" style="color:var(--red);border-color:var(--red)">missing file</span></div></div>`).join("")}</div>
    <div class="actions"><button class="primary" id="btn-prune">Remove ${miss.length} missing</button><span class="msg" id="prune-msg"></span></div>`);
  $("#btn-prune").onclick = async () => {
    const pm = $("#prune-msg"); pm.className = "msg work"; pm.textContent = "removing...";
    const rr = await api("/api/scan/prune", {ids: miss.map(m => m.id)});
    toast(`Removed ${rr.removed.length} missing model(s)`, "ok");
    emit("refresh", true); checkMissing();
  };
}
