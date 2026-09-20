// Build tab: current commit vs upstream, CMake flags, rebuild, log.
// Built-in engines and user-saved custom build recipes.
// Also surfaces the vLLM pip package version, since updating it is a build-ish
// concern rather than a setup one.
import { $, esc, setHTML, api, toast, agoText, fmtDur } from "./core.js";

let buildPoll = null;
let buildViewVersion = 0;
let buildPollVersion = 0;
let _target = localStorage.getItem("build_target") || "llamacpp";

function setTarget(t) {
  _target = t;
  localStorage.setItem("build_target", t);
}

const ENGINE_LABELS = {
  llamacpp: "llama.cpp",
  ikllama: "ik_llama",
};

const ENGINE_REPOS = {
  llamacpp: "github.com/ggml-org/llama.cpp",
  ikllama: "github.com/ikawrakow/ik_llama.cpp",
};
const BACKEND_LABELS = {auto:"Auto", cuda:"CUDA", hip:"ROCm / HIP", vulkan:"Vulkan", cpu:"CPU"};

const logReaders = new WeakMap();

function updateBuildLog(log, text) {
  if (!log) return;
  let reader = logReaders.get(log);
  if (!reader) {
    reader = {scrollingUntil: 0};
    logReaders.set(log, reader);
    // Also pause at the bottom while wheel/touch/keyboard scrolling is active.
    const scrolling = () => { reader.scrollingUntil = Date.now() + 1000; };
    log.onscroll = scrolling;
    log.onwheel = scrolling;
    log.ontouchmove = scrolling;
    log.onkeydown = event => {
      if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) scrolling();
    };
  }
  const selection = globalThis.getSelection?.();
  if (selection && !selection.isCollapsed) {
    for (let i = 0; i < selection.rangeCount; i++) {
      if (selection.getRangeAt(i).intersectsNode(log)) return;
    }
  }
  if (log.matches?.(":active") || Date.now() < reader.scrollingUntil) return;
  if (log.scrollTop + log.clientHeight < log.scrollHeight - 2) return;
  // Replacing an unchanged text node still destroys the browser's selection.
  if (log.textContent === text) return;
  log.textContent = text;
  log.scrollTop = log.scrollHeight;
}

export async function loadBuild(force) {
  const viewVersion = ++buildViewVersion;
  ++buildPollVersion;
  const v = $("#view-build");
  if (force) {
    const s = $("#upstream-status");
    if (s) { s.textContent = "checking github..."; s.className = "v work"; }
  } else setHTML(v, `<div class="skel">QUERYING GIT + GITHUB...</div>`);
  clearInterval(buildPoll);
  const listed = await api("/api/build/targets");
  if (viewVersion !== buildViewVersion) return;
  const targets = listed.targets || [];
  if (!targets.some(t => t.id === _target)) setTarget("llamacpp");
  const selected = targets.find(t => t.id === _target);
  const custom = selected && !selected.builtin;
  const q = (force ? "?force=1&" : "?") + `target=${encodeURIComponent(_target)}`;
  const b = await api("/api/build/info" + q);
  const st = await api("/api/state");
  const activeEngine = st.active_engine || "llamacpp";
  const vver = await api("/api/vllm/version" + (force ? "?force=1" : ""));
  if (viewVersion !== buildViewVersion) return;
  const cur = b.current||{}, up = b.updates||{};
  const flags = b.saved_flags && Object.keys(b.saved_flags).length ? b.saved_flags : b.recommended_flags||{};
  const behind = up.ok ? up.behind : 0;
  const checked = up.cached ? `checked ${agoText(up.checked_secs_ago)}` : "checked just now";
  const remoteUrl = b.remote || ENGINE_REPOS[_target] || "";
  const label = selected?.name || ENGINE_LABELS[_target] || _target;
  const activeBuild = listed.active_build || {id: activeEngine, name: ENGINE_LABELS[activeEngine] || activeEngine};
  const isActive = _target === activeBuild.id;
  const reqBackend = ((st.config||{}).llama_backend) || "auto";
  const availBackends = ["auto"].concat(b.available_backends || []).filter((v, i, a) => a.indexOf(v) === i);
  const schedule = st.config || {};

  setHTML(v, `
    <div class="card buildtarget">
      <span class="k">Build Target</span>
      <select id="build-target" aria-label="Build Target">${targets.map(t => `<option value="${esc(t.id)}" ${t.id===_target?'selected':''}>${esc(t.name)}</option>`).join("")}</select>
      <button class="ghost" id="btn-add-target">+ Add Target</button>
      ${custom ? '<button class="ghost" id="btn-edit-target">Edit</button><button class="ghost" id="btn-remove-target">Remove</button>' : ''}
      <span class="buildtarget-active">
        Active runtime: <strong class="${isActive?'ok':'dim'}">${esc(ENGINE_LABELS[activeEngine]||activeEngine)}</strong>
        <span>Active build: <strong>${esc(activeBuild.name)}</strong></span>
        ${activeBuild.server_bin ? `<div class="note">${esc(activeBuild.server_bin)}</div>` : ""}
        ${isActive ? '<span class="ok">Active</span>' : ''}
        ${custom && !isActive ? '<button class="primary" id="btn-use-build">Use this build</button>' : ''}
        ${!isActive && !custom?`<button class="ghost" id="btn-switch-engine">Switch to ${esc(label)}</button>`:''}
      </span>
    </div>
    <div class="card"><h3>Current Build · ${esc(label)}</h3>
      <div class="kv"><span class="k">commit</span><span class="v">${esc(cur.hash||"?")} &middot; ${esc((cur.subject||"").slice(0,60))}</span></div>
      <div class="kv"><span class="k">branch</span><span class="v">${esc(cur.branch||"?")}</span></div>
      <div class="kv"><span class="k">date</span><span class="v">${esc(cur.date||"?")}</span></div>
    </div>
    <div class="card"><h3>Upstream (${esc(remoteUrl)})</h3>
      <div class="kv"><span class="k">status</span><span class="v ${behind>0?'bad':'ok'}" id="upstream-status">${up.ok?(behind>0?behind+" commits behind":"up to date"):"check failed"}</span></div>
      ${up.error ? `<div class="note">${esc(up.error)}</div>` : ""}
      ${up.latest?`<div class="kv"><span class="k">latest</span><span class="v">${esc(up.latest.hash)} &middot; ${esc((up.latest.subject||"").slice(0,60))}</span></div>`:""}
      <div class="actions" style="margin-top:6px">
        <button class="ghost" id="btn-refresh-upstream">Check Update</button>
        <span class="note" style="margin:0">${esc(checked)} &middot; auto-checks at most every 15 min</span>
      </div>
    </div>
    ${custom ? `<div class="card"><h3>Custom Build Target</h3>
      ${["repository", "branch", "source", "build", "server_binary"].map(k => `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(selected[k])}</span></div>`).join("")}
      <h3>Build Command</h3><pre>${esc(selected.build_command)}</pre>
      <div class="actions"><button class="primary" id="btn-build">Pull &amp; Build</button><span class="msg" id="build-msg"></span></div>
      <div class="note">Runs your saved command in the Build directory. After building, select Use this build to activate it as the llama.cpp runtime.</div>
    </div>` : `<div class="card"><h3>Acceleration Backend</h3>
      <div class="kv"><span class="k">selected</span><span class="v">
        <select id="build-backend" style="background:var(--inset);border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:6px">
          ${["auto","cuda","hip","vulkan","cpu"].map(k => `<option value="${esc(k)}" ${reqBackend===k?"selected":""} ${availBackends.includes(k)||k==="auto"||k==="cpu"?"":"disabled"}>${esc(BACKEND_LABELS[k]||k)}${availBackends.includes(k)||k==="auto"||k==="cpu"?"":" (unavailable)"}</option>`).join("")}
        </select></span></div>
      <div class="kv"><span class="k">effective</span><span class="v">${esc(BACKEND_LABELS[b.selected_backend] || b.selected_backend || "CPU")}</span></div>
      ${(b.backend_notes||[]).map(n => `<div class="note">&bull; ${esc(n)}</div>`).join("")}
    </div>
    <div class="card"><h3>Build Flags · ${esc(label)}</h3>
      <div class="flags">${Object.entries(flags).map(([k,val])=>`<span class="flagpill">${esc(k)}=${esc(val)}</span>`).join("")}</div>
      <div class="actions">
        <button class="primary" id="btn-build">${behind>0?"Pull latest &amp; Rebuild":"Rebuild current"}</button>
        <label style="font-size:11px;color:var(--dim)"><input type="checkbox" id="opt-pull" ${behind>0?"checked":""}> git pull first</label>
        <span class="msg" id="build-msg"></span>
      </div>
      <div class="note">Rebuilds ${esc(label)} with CMake. Prior binaries are backed up first. Takes several minutes; watch the log below.</div>
    </div>
    `}
    <div class="card"><h3>Automatic Update · llama.cpp</h3>
      <div class="actions">
        <label><input type="checkbox" id="build-auto-enabled" ${schedule.build_auto_update_enabled ? "checked" : ""}> Pull latest &amp; rebuild automatically when idle</label>
        <label for="build-auto-time">Daily time</label>
        <input type="time" id="build-auto-time" value="${esc(schedule.build_auto_update_time || "03:00")}" required>
        <button class="primary" id="btn-save-build-schedule">Save schedule</button>
        <span class="msg" id="build-schedule-msg"></span>
      </div>
      <div class="note">Server local time: <span id="build-schedule-timezone"></span>. LlamaForge must be running at this time; the browser may be closed. Busy or unknown state skips that day. Inference is unavailable during the update; the router and loaded models are restored afterward. Applies to llama.cpp only.</div>
      <div class="kv"><span class="k">last check</span><span class="v" id="build-schedule-status">${esc(schedule.build_auto_update_status || "Not run yet")}</span></div>
    </div>
    <div class="card"><h3>Build Log · ${esc(label)}</h3><div class="note">Updates pause while selecting text or reading earlier lines. Clear the selection and scroll to the bottom to resume.</div><div class="log" id="build-log" tabindex="0">idle</div></div>`
    + (vver.error ? "" : `<div class="card"><h3>vLLM (pip package in WSL)</h3>
      <div class="kv"><span class="k">installed</span><span class="v ${vver.installed&&vver.installed.present?'ok':'bad'}">${vver.installed&&vver.installed.present?"v"+esc(vver.installed.version):"not installed (see Setup)"}</span></div>
      <div class="kv"><span class="k">latest on PyPI</span><span class="v">${esc(vver.latest||"?")}</span></div>
      ${vver.installed&&vver.installed.present&&vver.latest&&vver.latest!==vver.installed.version?`<div class="actions"><button class="primary" id="btn-vllm-update">Update vLLM to ${esc(vver.latest)}</button><span class="msg" id="vllm-upd-msg"></span></div>`:`<div class="note">${vver.installed&&vver.installed.present?"vLLM is up to date.":"Install vLLM from the Setup tab first."}</div>`}
      <div class="log" id="vllm-update-log" style="display:none">idle</div>
    </div>`));

  $("#build-target").onchange = event => { setTarget(event.target.value); loadBuild(); };
  $("#btn-add-target").onclick = () => editTarget();
  if (custom) {
    $("#btn-edit-target").onclick = () => editTarget(selected);
    $("#btn-remove-target").onclick = async () => {
      if (!confirm(`Remove ${label} from LlamaForge? Source, Build, binaries and Git repository will remain on disk.`)) return;
      const result = await api("/api/build/targets/remove", {id: selected.id});
      if (!result.ok) return toast(result.error || "Remove failed", "err");
      setTarget("llamacpp");
      loadBuild();
    };
  }

  const useBtn = $("#btn-use-build");
  if (custom && !isActive && useBtn) useBtn.onclick = async () => {
    useBtn.disabled = true;
    useBtn.textContent = "activating...";
    try {
      const result = await api("/api/build/activate", {target: selected.id});
      if (result.ok) toast(`Using ${label}`, "ok");
      else toast([result.error || "Activation failed", result.rollback_error ? `Recovery failed: ${result.rollback_error}` : ""].filter(Boolean).join(" · "), "err");
      await loadBuild();
    } catch (error) { toast(error.message, "err"); }
    finally { useBtn.disabled = false; useBtn.textContent = "Use this build"; }
  };

  // Engine switch
  const switchBtn = $("#btn-switch-engine");
  if (switchBtn) switchBtn.onclick = async () => {
    switchBtn.disabled = true;
    switchBtn.textContent = "switching...";
    const r = await api("/api/engine/switch", {engine: _target});
    if (r.ok) {
      toast(`Switched to ${label}`, "ok");
      setTimeout(loadBuild, 1500);
    } else {
      toast([r.error || "switch failed", r.rollback_error].filter(Boolean).join(" · "), "err");
      switchBtn.disabled = false;
      switchBtn.textContent = `Switch to ${label}`;
    }
  };

  $("#btn-build").onclick = startBuild;
  $("#btn-save-build-schedule").onclick = async () => {
    const input = $("#build-auto-time"), btn = $("#btn-save-build-schedule");
    if (!input.reportValidity()) return;
    btn.disabled = true;
    const msg = $("#build-schedule-msg");
    try {
      const r = await api("/api/config", {
        build_auto_update_enabled: $("#build-auto-enabled").checked,
        build_auto_update_time: input.value,
      });
      const ok = r.ok && !(r.rejected || []).length;
      msg.className = ok ? "msg ok" : "msg err";
      msg.textContent = ok ? "Schedule saved" : (r.error || "Could not save schedule");
    } catch (e) {
      msg.className = "msg err"; msg.textContent = e.message;
    } finally { btn.disabled = false; }
  };
  const beSel = $("#build-backend");
  if (beSel) beSel.onchange = async () => {
    const buildBtn = $("#btn-build");
    buildBtn.disabled = true;
    beSel.disabled = true;
    try {
      const r = await api("/api/config", {llama_backend: beSel.value});
      if (!r.ok || (r.rejected || []).length) {
        throw new Error(r.error || "Could not save acceleration backend");
      }
      await loadBuild();
    } catch (e) {
      beSel.value = reqBackend;
      toast(e.message, "err");
    } finally {
      buildBtn.disabled = false;
      beSel.disabled = false;
    }
  };
  const refBtn = $("#btn-refresh-upstream");
  if (refBtn) refBtn.onclick = () => { refBtn.disabled = true; loadBuild(true); };
  pollBuild();
  const updBtn = $("#btn-vllm-update");
  if (updBtn) updBtn.onclick = async () => {
    const msg = $("#vllm-upd-msg"); msg.className = "msg work"; msg.textContent = "starting update...";
    const r = await api("/api/vllm/update");
    if (r.started) {
      toast("vLLM update started", "ok");
      $("#vllm-update-log").style.display = "";
      const iv = setInterval(async () => {
        const s = await api("/api/vllm/setup");
        const l = $("#vllm-update-log");
        if (l) { l.textContent = s.setup_log||""; l.scrollTop = l.scrollHeight; }
        if (s.setup_job && !s.setup_job.running) {
          clearInterval(iv); msg.className = "msg ok"; msg.textContent = "done";
          setTimeout(loadBuild, 1200);
        }
      }, 2000);
    } else msg.textContent = "a job is already running";
  };
}

async function startBuild() {
  const pull = $("#opt-pull")?.checked ?? true, msg = $("#build-msg");
  msg.className = "msg work"; msg.textContent = "starting build...";
  const r = await api("/api/build/start", {pull, target: _target});
  if (r.started) toast(`Build started (${ENGINE_LABELS[_target] || _target})`, "ok");
  else msg.textContent = r.error || "a build is already running";
  pollBuild();
}

async function pollBuild() {
  clearInterval(buildPoll);
  const pollVersion = ++buildPollVersion;
  const tick = async () => {
    const target = _target;
    const s = await api("/api/build/log?target=" + encodeURIComponent(target));
    if (target !== _target || pollVersion !== buildPollVersion) return;
    const sched = s.schedule || {}, status = $("#build-schedule-status");
    if (status) status.textContent = [sched.last_date, sched.status].filter(Boolean).join(" · ");
    const tz = $("#build-schedule-timezone");
    if (tz) tz.textContent = sched.timezone || "";
    const log = $("#build-log");
    updateBuildLog(log, s.log||"idle");
    const msg = $("#build-msg");
    if (msg && s.running) { msg.className = "msg work"; msg.textContent = "building: " + s.phase; }
    else if (msg && s.phase === "done") {
      msg.className = "msg ok";
      msg.textContent = "build OK" + (s.started&&s.finished?` in ${fmtDur(s.finished-s.started)}`:"");
    } else if (msg && s.phase === "done_warnings") {
      // llama-server built, but a non-essential later target (UI assets) failed.
      msg.className = "msg warn";
      msg.textContent = "built with warnings - " + (s.warning || "see log");
    } else if (msg && s.phase === "failed") {
      msg.className = "msg err"; msg.textContent = "build failed - see log";
    }
  };
  await tick();
  if (pollVersion === buildPollVersion) buildPoll = setInterval(tick, 2000);
}


function editTarget(target = {}) {
  const dialog = document.createElement("dialog");
  const fields = {name:"Name", repository:"Repository", branch:"Branch", source:"Source", build:"Build", server_binary:"Server Binary", build_command:"Build Command"};
  const defaults = {branch:"master", server_binary:"{build}/bin/llama-server"};
  setHTML(dialog, `<form style="min-width:320px;max-width:760px">
    <h3>${target.id ? "Edit" : "Add"} Custom Build Target</h3>
    ${Object.entries(fields).map(([k,label]) => `<label style="display:block;margin:10px 0">${label}
      ${k === "build_command" ? `<textarea name="${k}" rows="8" style="width:100%" required>${esc(target[k] || "")}</textarea>` : `<input name="${k}" style="display:block;width:100%" value="${esc(target[k] || defaults[k] || "")}" required>`}</label>`).join("")}
    <p class="note">Save and Validate do not clone or execute commands. Pull &amp; Build runs this command with your account's permissions. Use only commands you trust. Bash is required (Git Bash on Windows). The working directory is Build; placeholders: {source}, {build}, {jobs}. Quote paths where needed. Existing checkouts must match Repository and Branch.</p>
    <div class="actions"><button type="button" data-action="validate">Validate</button><button type="submit">Save Target</button><button type="button" data-action="cancel">Cancel</button></div>
    <p data-status role="status"></p>
  </form>`);
  document.body.appendChild(dialog);
  dialog.onclose = () => dialog.remove();
  dialog.querySelector('[data-action="cancel"]').onclick = () => dialog.close();
  const form = dialog.querySelector("form"), status = dialog.querySelector("[data-status]");
  const submit = async save => {
    if (!form.reportValidity()) return;
    const body = Object.fromEntries(new FormData(form));
    if (target.id) body.id = target.id;
    const buttons = [...dialog.querySelectorAll("button")];
    buttons.forEach(b => b.disabled = true);
    try {
      const result = await api(`/api/build/targets/${save ? "save" : "validate"}`, body);
      if (!result.ok) { status.textContent = result.error || "Validation failed"; return; }
      if (save) {
        setTarget(result.target.id); dialog.close(); await loadBuild();
      } else status.textContent = `Valid. Server Binary: ${result.server_binary}`;
    } catch (error) { status.textContent = error.message; }
    finally { buttons.forEach(b => b.disabled = false); }
  };
  dialog.querySelector('[data-action="validate"]').onclick = () => submit(false);
  form.onsubmit = event => { event.preventDefault(); submit(true); };
  dialog.showModal();
}
