// Entry point. Owns two things and no view logic:
//   1. which loader runs when a tab is shown
//   2. the polling timers
//
// There are no `window`-assigned globals: every handler is wired with
// addEventListener (toolbar controls by id, dynamic rows by delegation), so the
// HTML and view templates carry no inline on* attributes to keep in sync.
import { $, api, esc, setHTML } from "./core.js";
import { S } from "./state.js";
import { on } from "./bus.js";
import * as ui from "./ui.js";
import * as models from "./models.js";
import * as stats from "./stats.js";
import { loadDiscover } from "./discover.js";
import { loadWillRun } from "./willrun.js";
import { loadTestChat, refreshTestChat } from "./test-chat.js";
import { loadBuild } from "./build.js";
import { loadSetup } from "./setup.js";
import { loadContext } from "./context.js";
import { loadDocs } from "./help.js";
import { initWizard } from "./wizard.js";
import { initOnboarding } from "./onboarding.js";

/* ---------- tab loaders ---------- */
ui.onTabShown("testchat", loadTestChat);
ui.onTabShown("build", loadBuild);
ui.onTabShown("setup", loadSetup);
ui.onTabShown("discover", loadDiscover);
ui.onTabShown("willrun", loadWillRun);
ui.onTabShown("stats", stats.loadStats);
ui.onTabShown("context", loadContext);
ui.onTabShown("help", loadDocs);

/* ---------- boot ---------- */
ui.initTabs();
ui.initModeToggle();
ui.initThemeControls();
ui.initSidebar();
ui.initDrawer();
ui.updatePageTitle();
initWizard();
initOnboarding();
models.initModels();
stats.initStats();

// deep-linkable tabs: #<tab> in the URL activates that tab (docs deep-links +
// tools/shoot.py). Runs after initTabs() has wired the click handlers.
window.addEventListener("hashchange", () => {
  const h = location.hash.slice(1);
  if (h) ui.switchTab(h);
});
if (location.hash) ui.switchTab(location.hash.slice(1));

// Use the backend's active-build resolver, shared with Build / Update.
// State events update immediately after switches; the clock remains a fallback.
const ENGINE_LABEL = { llamacpp: "llama.cpp", ikllama: "ik_llama" };
let shownRuntime = null;

function renderEngineBadge(state = S.STATE) {
  const engine = state?.active_engine || "";
  const build = state?.active_build;
  const c = state?.config || {};
  const label = build?.name || ENGINE_LABEL[engine] || engine;
  const identity = JSON.stringify([engine, build?.id, label, build?.server_bin,
    c.active_llamacpp_build_target, c.server_bin, c.ik_llama_server_bin]);
  if (identity === shownRuntime) return;
  const el = $("#engine-badge");
  if (!el) return;
  shownRuntime = identity;
  setHTML(el, engine
    ? `<span class="tag be-${esc(engine)}">${esc(label)}</span>`
    : "");
}
on("state", renderEngineBadge);

function clock() {
  const el = $("#clock");
  if (el) el.textContent = new Date().toLocaleTimeString("en-GB") + " LOCAL";
  renderEngineBadge();
}
clock();
setInterval(clock, 1000);

(async () => {
  await models.refresh();
  // theme/cvd defaults from config.json, used only when this device hasn't chosen
  try {
    const cfg = (S.STATE && S.STATE.config) || {};
    if (!localStorage.getItem("theme") && cfg.theme) ui.applyTheme(cfg.theme);
    if (localStorage.getItem("cvd") === null && cfg.cvd) ui.applyCvd(true);
    ui.applyMode(((S.STATE||{}).onboarding||{}).ui_mode || "lite");
  } catch (e) {}
})();

/* ---------- polls (idle unless their tab is showing) ---------- */
setInterval(() => { if (ui.activeTab() === "models") models.refresh(true); }, 4000);
setInterval(() => { if (ui.activeTab() === "stats") stats.loadStats(true); }, 4000);
setInterval(() => { if (ui.activeTab() === "models") models.refreshRouterLog(); }, 3000);
setInterval(() => { if (ui.activeTab() === "models") models.refreshLlamaLog(); }, 3000);
setInterval(() => { if (ui.activeTab() === "models") models.refreshVllmLog(); }, 3000);
models.refreshRouterLog();
models.refreshLlamaLog();
models.refreshVllmLog();

setInterval(() => { if (ui.activeTab() === "testchat") refreshTestChat(); }, 3000);
