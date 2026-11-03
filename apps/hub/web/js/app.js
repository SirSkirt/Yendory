"use strict";
// Yendory Hub window. Everything it shows comes from the backend (src-tauri): the models folder
// Studio uses, the Hub's shelves, and the programs found on this computer.

const TAURI = window.__TAURI__;
const invoke = (cmd, args) => TAURI.core.invoke(cmd, args);
const listen = (event, f) => TAURI.event.listen(event, f);
const $ = id => document.getElementById(id);
const el = (tag, attrs = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "class") e.className = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : String(c));
  return e;
};
const store = {
  get(k, d) { try { return localStorage.getItem("hub." + k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("hub." + k, v); } catch { /* not kept */ } },
};

// Studio opens the model the Hub hands it (`--model <id>`, docs/HANDOFF.md) from this version
// on, and a running Studio takes it from a second start.
const STUDIO_OPENS_MODELS_FROM = "0.9.3";

const S = {
  models: [],
  shelves: { shelves: [], places: {} },
  // The suite's components: hub, network, studio, tools (Downloads & updates).
  components: [],
  about: null,
  options: null,
  shelf: store.get("shelf", ""),
  sel: store.get("sel", ""),
  sort: store.get("sort", "updated"),
  search: "",
  view: "shelf",
  job: null,
  networkError: null,
};

// ---- Formatting --------------------------------------------------------------------------------
const nf = new Intl.NumberFormat("en-US");
const fmt = n => nf.format(n);
const trim1 = v => (v >= 10 ? String(Math.round(v)) : String(Math.round(v * 10) / 10));
function short(n) {
  if (n >= 1e9) return trim1(n / 1e9) + "B";
  if (n >= 1e6) return trim1(n / 1e6) + "M";
  if (n >= 1e3) return trim1(n / 1e3) + "K";
  return String(n);
}
function bytes(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(1) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return Math.round(b / 1e3) + " KB";
  return b + " bytes";
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function when(t) {
  if (!t) return "—";
  const d = new Date(t * 1000), now = new Date();
  const hm = d.toTimeString().slice(0, 5);
  if (d.toDateString() === now.toDateString()) return "today " + hm;
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return "yesterday " + hm;
  const s = `${d.getDate()} ${MONTHS[d.getMonth()]}`;
  return d.getFullYear() === now.getFullYear() ? s : `${s} ${d.getFullYear()}`;
}
const archName = a => (a === "neuro" ? "Neuro" : "Bicameral");
const plural = (n, one, many) => `${fmt(n)} ${n === 1 ? one : many}`;
const shortName = p => p.name.replace(/^Yendory /, "");

// ---- Messages and dialogs ------------------------------------------------------------------------
let toastTimer = 0;
function toast(message, error = false) {
  const t = $("toast");
  t.textContent = message;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), error ? 10000 : 5000);
}
/** Runs a backend call; a refusal or failure is shown and resolves undefined. */
async function act(cmd, args) {
  try {
    return await invoke(cmd, args);
  } catch (e) {
    toast(String(e), true);
    return undefined;
  }
}

// In-page dialogs that wait for the answer (the webview's confirm() and prompt() do not reliably
// wait on every platform). `field`: an element whose value is the answer (null: yes/no).
function dialog(message, field, okLabel = "OK", content = null) {
  return new Promise(resolve => {
    const before = document.activeElement;
    const ok = el("button", { type: "button", class: "primary" }, okLabel);
    const cancel = el("button", { type: "button" }, "Cancel");
    const overlay = el("div", { class: "modal" },
      el("div", { class: "modal-box", role: "dialog", "aria-modal": "true", "aria-label": message },
        el("p", {}, message), content || "", field || "", el("div", { class: "actions" }, cancel, ok)));
    const done = yes => {
      overlay.remove();
      document.removeEventListener("keydown", key, true);
      before?.focus?.();
      resolve(field ? (yes ? field.value : null) : yes);
    };
    const key = e => {
      if (document.querySelector(".ydd-list:not([hidden])")) return;
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(false); }
      else if (e.key === "Enter" && document.activeElement !== cancel) { e.preventDefault(); e.stopPropagation(); done(true); }
    };
    ok.addEventListener("click", () => done(true));
    cancel.addEventListener("click", () => done(false));
    overlay.addEventListener("click", e => { if (e.target === overlay) done(false); });
    document.addEventListener("keydown", key, true);
    document.body.append(overlay);
    if (field?.tagName === "INPUT") { field.focus(); field.select(); } else ok.focus();
  });
}
const askYes = (message, okLabel) => dialog(message, null, okLabel);
const askText = (message, initial = "") => dialog(message, el("input", { type: "text", value: initial, maxlength: 60, "aria-label": message }));

// ---- Loading -------------------------------------------------------------------------------------
const networkReady = () => !!S.about?.network;

async function loadAbout() {
  // Connects to the Yendory Network (starting it when it is installed).
  const a = await act("about");
  if (a) { S.about = a; S.networkError = a.networkError; }
}
async function loadLibrary() {
  const shelves = await act("shelves");
  if (shelves) S.shelves = shelves;
  if (!networkReady()) return;
  try {
    S.models = await invoke("list_models");
  } catch (e) {
    S.networkError = String(e);
  }
}
async function loadComponents(check = false) {
  const c = await act("components", { check });
  if (c) S.components = c;
}
async function refresh() {
  await Promise.all([loadLibrary(), loadComponents()]);
  render();
}

// ---- Shelves -------------------------------------------------------------------------------------
const shelfIds = () => S.shelves.shelves.map(s => s.id);
const currentShelf = () => S.shelves.shelves.find(s => s.id === S.shelf) || S.shelves.shelves[0];
function shelfOf(m) {
  const placed = S.shelves.places[m.id];
  return shelfIds().includes(placed) ? placed : S.shelves.shelves[0]?.id;
}
const component = id => S.components.find(c => c.id === id);
const programs = () => S.components.filter(c => c.kind === "program");
const selected = () => S.models.find(m => m.id === S.sel);

function visibleModels() {
  const q = S.search.trim().toLowerCase();
  const ms = q ? S.models.filter(m => m.name.toLowerCase().includes(q)) : S.models.filter(m => shelfOf(m) === currentShelf()?.id);
  const by = {
    updated: (a, b) => b.updated - a.updated,
    created: (a, b) => b.created - a.created,
    name: (a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }),
    size: (a, b) => a.parameters - b.parameters || a.name.localeCompare(b.name),
  }[S.sort] || ((a, b) => b.updated - a.updated);
  return [...ms].sort(by);
}

/** Spine height follows model size: Nova-2's 1.2M is the shortest common book. */
function spineHeight(p) {
  const h = 132 + 24 * Math.log2(Math.max(p, 1) / 1.18e6);
  return Math.round(Math.min(220, Math.max(112, h)));
}

const atLeast = (have, need) => {
  const parts = v => v.split(/[-+]/)[0].split(".").map(Number);
  const [a, b] = [parts(have), parts(need)];
  for (let i = 0; i < 3; i++) if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) > (b[i] || 0);
  return true;
};
const opensModels = p => p?.id === "tools" || !!(STUDIO_OPENS_MODELS_FROM && p?.version && atLeast(p.version, STUDIO_OPENS_MODELS_FROM));
/** Installed on its own (not by the Hub): it can be switched to the Hub. */
const onItsOwn = c => c && c.installed && !c.managed && c.legacy.length > 0;

// ---- Rendering -----------------------------------------------------------------------------------
function render() {
  const visible = visibleModels();
  if (!visible.some(m => m.id === S.sel)) S.sel = visible[0]?.id || "";
  store.set("sel", S.sel);
  const setup = S.view === "shelf" && !networkReady();
  for (const v of ["shelf", "new", "programs", "settings"]) $("view-" + v).hidden = S.view !== v || setup;
  $("view-setup").hidden = !setup;
  renderSidebar();
  if (setup) renderSetup();
  else if (S.view === "shelf") { renderSwitchOffer(); renderShelf(visible); renderCard(); }
  if (S.view === "programs") renderComponents();
  if (S.view === "settings") renderSettings();
}

/** Stage text for an install in progress. */
function jobText(job) {
  const name = component(job.id)?.name || job.id;
  if (job.stage === "downloading") return `Downloading ${name} · ${bytes(job.received)}${job.total ? " of " + bytes(job.total) : ""}`;
  if (job.stage === "checking") return `Checking ${name}'s signature`;
  if (job.stage === "switching") return job.note || `Switching ${name} to the Hub`;
  return `Installing ${name}`;
}
const jobPercent = job => (job.stage === "downloading" ? (job.total ? Math.min(100, (100 * job.received) / job.total) : 0) : 100);

function renderSidebar() {
  $("shelfList").replaceChildren(...S.shelves.shelves.map(s => {
    const n = S.models.filter(m => shelfOf(m) === s.id).length;
    const on = S.view === "shelf" && !S.search && s.id === currentShelf()?.id;
    return el("button", { type: "button", class: "sb-item" + (on ? " on" : ""), title: `${s.name}: ${plural(n, "model", "models")}`, "aria-current": on ? "true" : null,
      onclick: () => { S.shelf = s.id; store.set("shelf", s.id); clearSearch(); show("shelf"); } },
      el("span", { class: "smark", "aria-hidden": "true" }, (s.name.trim()[0] || "?").toUpperCase()),
      el("span", { class: "lbl" }, s.name), el("span", { class: "cnt" }, n));
  }));
  const m = selected();
  $("programList").replaceChildren(...programs().map(p => {
    const status = p.released === false && !p.installed ? "not released" : p.running ? "open"
      : onItsOwn(p) ? "on its own" : p.update ? "update" : p.installed ? "installed" : "install";
    const opens = p.installed && (!p.running || opensModels(p));
    const title = opens ? (m ? `Open ${m.name} in ${shortName(p)}` : `Start ${shortName(p)}`) : `${p.name}: ${status}`;
    return el("button", { type: "button", class: "sb-item", title,
      onclick: () => (opens ? launch(p.id, m?.id) : show("programs")) },
      programMark(p),
      p.running ? el("i", { class: "live", title: "Running" }) : "",
      el("span", { class: "lbl" }, shortName(p)),
      el("span", { class: "st" + (p.update || onItsOwn(p) ? " warn" : p.running ? " ok" : "") }, status));
  }));
  const pending = S.components.filter(c => c.update || c.restartReady || (c.kind === "service" && !c.managed && c.released)).length;
  $("updateCount").textContent = pending || "";
  for (const b of document.querySelectorAll(".sb-foot [data-view]")) b.classList.toggle("on", S.view === b.dataset.view);
  const a = S.about;
  if (a) {
    $("versions").textContent = `Hub ${a.hub} · ` + (a.network ? `Network ${a.network}` : "no Network");
    $("versions").title = a.network ? `Yendory Network ${a.network}, Engine ${a.engine}` : a.networkError || "";
  }
  const job = S.job;
  $("job").hidden = !job;
  if (job) {
    $("jobText").textContent = jobText(job);
    $("jobBar").style.width = jobPercent(job) + "%";
  }
}

// ---- The Yendory Network, first ------------------------------------------------------------------
function renderSetup() {
  const net = component("network");
  const box = $("setupBody");
  const installing = S.job?.id === "network" || net?.installing;
  const parts = [];
  if (installing) {
    parts.push(el("div", { class: "meter" }, el("i", { style: `width:${S.job ? jobPercent(S.job) : 0}%` })),
      el("p", { class: "note" }, S.job ? jobText(S.job) : "Starting…"));
  } else if (net?.managed) {
    parts.push(el("p", {}, "The Yendory Network is installed but isn't answering."),
      el("p", { class: "note mono" }, S.networkError || ""),
      el("div", { class: "actions" },
        el("button", { type: "button", class: "primary", onclick: retryNetwork }, "Try again"),
        net.canInstall ? el("button", { type: "button", onclick: () => install(net) }, "Install it again") : ""));
  } else if (net?.released === false) {
    parts.push(el("p", { class: "note" }, "The Yendory Network hasn't been published on the release page yet, so the Hub can't install it. Models appear here once it is."));
  } else if (net?.released === undefined || net?.released === null) {
    parts.push(el("p", { class: "note" }, "Looking for the newest Yendory Network on the release page…"),
      el("div", { class: "actions" }, el("button", { type: "button", onclick: () => checkUpdates() }, "Look again")));
  } else {
    parts.push(el("div", { class: "actions" },
      el("button", { type: "button", class: "primary", disabled: !net.canInstall, onclick: () => install(net, true) }, `Install the Yendory Network ${net.latest || ""}`),
      net.installNote ? el("span", { class: "note warn" }, net.installNote) : ""));
  }
  box.replaceChildren(...parts);
}
async function retryNetwork() {
  await loadAbout();
  await refresh();
}

// ---- Switching Studio to the Hub -----------------------------------------------------------------
const LEGACY = { installer: "its own installer", system: "a system package (.deb)", "hub-0.1": "Hub 0.1", located: "a file you chose" };
function legacyText(c) {
  return c.legacy.map(l => LEGACY[l.kind] || l.kind).join(" and ");
}
function renderSwitchOffer() {
  const studio = component("studio");
  const box = $("switchOffer");
  if (!onItsOwn(studio) || store.get("switchLater", "") === (studio.version || "?")) { box.hidden = true; return; }
  box.hidden = false;
  const ready = studio.released === true && !studio.running;
  box.replaceChildren(
    el("p", {}, el("b", {}, `Yendory Studio${studio.version ? " " + studio.version : ""} is installed on its own`), ` (from ${legacyText(studio)}).`),
    el("p", { class: "note" }, "Switch it to the Hub to have the Hub keep it updated. The old copy is removed and the Hub's is installed; your models, datasets, chats and settings stay where they are."),
    el("div", { class: "actions" },
      el("button", { type: "button", class: "primary", disabled: !ready, onclick: switchStudio }, "Switch to the Hub…"),
      el("button", { type: "button", class: "quiet", onclick: () => { store.set("switchLater", studio.version || "?"); render(); } }, "Not now"),
      !ready ? el("span", { class: "note" }, studio.running ? "Close Studio first." : studio.released === false
        ? "Available once Studio publishes its Hub package." : "Looking for Studio's Hub package…") : ""));
}
async function switchStudio() {
  const studio = component("studio");
  const steps = [
    `The Hub downloads Studio ${studio.latest || ""} and checks its signature.`,
    "Your Studio data folder is set aside, so nothing can touch it.",
    studio.legacy.some(l => l.kind === "system") ? "The old Studio package is removed (the system asks for your password)." : "The old Studio is removed.",
    "Your data folder goes back, and the Hub's Studio is installed with its menu shortcut.",
  ];
  const list = el("ol", { class: "steps" }, ...steps.map(s => el("li", {}, s)));
  if (!(await dialog("Switch Yendory Studio to the Hub?", null, "Switch", list))) return;
  S.job = { id: "studio", stage: "downloading", received: 0, total: null };
  render();
  const c = await act("switch_studio");
  S.job = null;
  if (c) { S.components = c; toast("Yendory Studio is now installed and updated by the Hub"); }
  render();
}

function renderShelf(visible) {
  const q = S.search.trim();
  const shelf = currentShelf();
  $("shelfTitle").textContent = q ? `Search: “${q}”` : shelf?.name || "Models";
  $("shelfCount").textContent = q ? `${plural(visible.length, "model", "models")} found` : plural(visible.length, "model", "models");
  $("renameShelf").hidden = $("deleteShelf").hidden = !!q;
  $("deleteShelf").disabled = S.shelves.shelves.length < 2;
  $("deleteShelf").title = S.shelves.shelves.length < 2 ? "The last shelf can't be removed" : "";
  const spines = visible.map(m => {
    const pulled = m.id === S.sel;
    const label = `${m.name}, ${archName(m.architecture)} ${short(m.parameters)}` + (m.plan?.inProgress ? ", partway through a training plan" : "");
    return el("button", { type: "button", class: "spine" + (pulled ? " pulled" : "") + (q ? " hit" : ""), style: `height:${spineHeight(m.parameters)}px`,
      "aria-label": label, "aria-pressed": pulled ? "true" : "false", title: label, "data-id": m.id,
      onclick: () => { S.sel = m.id; render(); } },
      el("span", { class: `band ${m.architecture}`, "aria-hidden": "true" }),
      m.plan?.inProgress ? el("i", { class: "rib", "aria-hidden": "true" }) : "",
      el("span", { class: "name" + (m.name.length > 14 ? " longer" : m.name.length > 9 ? " long" : "") }, m.name),
      el("span", { class: "size" }, short(m.parameters)),
      el("span", { class: "foot", "aria-hidden": "true" }));
  });
  const ghosts = q ? [] : [
    el("button", { type: "button", class: "spine ghost", "aria-label": "New model", title: "Make a new model", onclick: openNew },
      el("span", { class: "name" }, "New model"), el("span", { class: "size" }, "+")),
    el("button", { type: "button", class: "spine ghost", "aria-label": "Import a checkpoint", title: "Import a .nnova2 or .nneuro checkpoint", onclick: importModel },
      el("span", { class: "name" }, "Import…"), el("span", { class: "size" }, "↓")),
  ];
  $("shelf").replaceChildren(...spines, ...ghosts);
  $("sortChoice").value = S.sort;
}

function planText(p) {
  const at = `course ${Math.min(p.position + 1, p.courses)} of ${p.courses}`;
  switch (p.status) {
    case "running": return `${at}, being taught`;
    case "stopped": return `stopped at ${at}`;
    case "failed": return `a course failed (${at})`;
    case "needs_attention": return `needs a decision in Studio (${at})`;
    case "finished": return `finished (${plural(p.courses, "course", "courses")})`;
    default: return p.inProgress ? `paused at ${at}` : "not started";
  }
}

function renderCard() {
  const card = $("card"), recent = $("recent");
  const m = selected();
  if (!m) {
    const any = S.models.length > 0;
    card.replaceChildren(el("h2", {}, S.search ? "Nothing found" : any ? "An empty shelf" : "No models yet"),
      el("p", { class: "empty" }, S.search ? "No model's name contains that." : any
        ? "Make a new model, import a checkpoint, or move a model here from another shelf."
        : "Make your first model with New model, or import a checkpoint such as Nova's."));
    recent.replaceChildren();
    return;
  }
  const row = (label, value, opts = {}) => el("div", { class: "row" + (opts.cls ? " " + opts.cls : "") }, el("span", {}, label), el("span", { class: opts.mono ? "m" : "" }, value));
  const words = m.wordLimit != null ? `${fmt(m.words)} of up to ${fmt(m.wordLimit)}` : `${fmt(m.words)} of ${fmt(m.slots)} slots`;
  const studio = component("studio"), tools = component("tools");
  const studioRunning = !!studio?.running;
  let studioButton, why = "";
  if (!studio) {
    studioButton = el("button", { type: "button", class: "primary", disabled: true }, "Open in Studio");
  } else if (!studio.installed) {
    studioButton = el("button", { type: "button", class: "primary", onclick: () => show("programs") }, "Install Studio…");
  } else if (studioRunning && !opensModels(studio)) {
    studioButton = el("button", { type: "button", class: "primary", disabled: true }, "Open in Studio");
    why = `Studio ${studio.version || ""} is open. Switch to it and open ${m.name} from its Models page: this Studio can't be handed a model while it runs (Studio ${STUDIO_OPENS_MODELS_FROM} and later can).`;
  } else {
    studioButton = el("button", { type: "button", class: "primary", onclick: () => launch("studio", m.id) }, "Open in Studio");
    if (studioRunning) why = `Studio is open: ${m.name} opens there.`;
    else if (!opensModels(studio)) why = `Studio${studio.version ? " " + studio.version : ""} starts on its Models page: choose ${m.name} there. Studio ${STUDIO_OPENS_MODELS_FROM} and later open it directly.`;
  }
  // Tools hands a second start to the running copy, so it can be given a model while open.
  const toolsButton = tools?.installed
    ? el("button", { type: "button", onclick: () => launch("tools", m.id) }, "Open in Tools")
    : tools && tools.released !== false
      ? el("button", { type: "button", onclick: () => show("programs") }, "Install Tools…")
      : el("button", { type: "button", disabled: true }, "Tools · not released yet");
  if (tools?.running && !why) why = `Tools is open: ${m.name} opens there.`;
  const skills = (m.skills || []).map(k => `${k.name} (generation ${k.generation}, ${fmt(k.steps)} steps)`);
  const locked = studioRunning || m.openBy.length > 0;
  const lockedTitle = locked ? `Close ${m.openBy.length ? m.openBy.join(" and ") : "Studio"} first: it may have this model open` : null;
  card.replaceChildren(
    el("h2", {}, m.name, el("span", { class: `tag ${m.architecture}` }, `${archName(m.architecture)} ${short(m.parameters)}`)),
    m.plan ? row("Training plan", `${m.plan.name}: ${planText(m.plan)}`, { cls: m.plan.inProgress ? "plan" : "" }) : "",
    m.openBy.length ? row("Open in", m.openBy.join(", ")) : "",
    row("Training steps", fmt(m.step), { mono: true }),
    row("Parameters", fmt(m.parameters), { mono: true }),
    row("Words learned", words),
    row("Context", `${fmt(m.context)} tokens`),
    row("Read", m.read.length ? m.read.join(", ") : "Nothing yet"),
    row("Lessons taught", m.lessons ? fmt(m.lessons) : "None yet"),
    m.architecture === "neuro" ? row("Game skills", skills.length ? skills.join(", ") : "None yet: open it in Tools to teach it a game") : "",
    row("Changed", when(m.updated)),
    row("Origin", m.origin === "new" || !m.origin ? `Created ${when(m.created)}` : `Imported from ${m.origin}`),
    row("File", bytes(m.fileBytes), { mono: true }),
    el("div", { class: "take" },
      el("p", {}, `Take ${m.name} to…`),
      el("div", { class: "btns" }, studioButton, toolsButton),
      why ? el("p", { class: "why" }, why) : "",
      el("div", { class: "more" },
        el("button", { type: "button", class: "quiet", onclick: () => moveModel(m) }, "Move to shelf…"),
        el("button", { type: "button", class: "quiet", onclick: () => renameModel(m), disabled: locked, title: lockedTitle }, "Rename…"),
        el("button", { type: "button", class: "quiet", onclick: () => exportModel(m) }, "Export…"),
        el("button", { type: "button", class: "quiet", onclick: () => deleteModel(m), disabled: locked, title: lockedTitle }, "Delete…"))));
  recent.replaceChildren(el("h3", {}, "Recently"),
    ...(m.recent.length
      ? m.recent.map(r => el("div", { class: "rec" }, el("span", { class: "when" }, when(r.time)), el("span", { class: "who" }, r.program), r.text))
      : [el("p", { class: "empty note" }, "Nothing yet. Lessons, reading and what the Hub does with this model appear here.")]));
}

/** A component's mark: the program's own logo, or its initial while it has none. */
function programMark(p) {
  const off = p.installed ? "" : " off";
  return LOGOS[p.id]
    ? el("img", { class: "pmark logo" + off, src: LOGOS[p.id], alt: "", "aria-hidden": "true" })
    : el("span", { class: `pmark ${p.id}${off}`, "aria-hidden": "true" }, shortName(p)[0]);
}
const LOGOS = { studio: "img/studio.png" };

// ---- Model actions -------------------------------------------------------------------------------
async function launch(id, model) {
  const p = component(id);
  const ok = await act("launch", { id, model: model || null });
  if (ok === undefined) return;
  toast(model ? `Opening ${S.models.find(m => m.id === model)?.name || "the model"} in ${p?.name || id}…` : `Starting ${p?.name || id}…`);
  setTimeout(refresh, 2500);
}
async function importModel() {
  const id = await act("import_model", { shelf: currentShelf()?.id || null });
  if (!id) return;
  S.sel = id;
  clearSearch();
  await refresh();
  toast(`Imported ${selected()?.name || "the model"}`);
}
async function moveModel(m) {
  const pick = el("select", { "aria-label": "Shelf" }, ...S.shelves.shelves.map(s => el("option", { value: s.id, selected: s.id === shelfOf(m) }, s.name)));
  const shelf = await dialog(`Move ${m.name} to which shelf?`, pick, "Move");
  if (!shelf || shelf === shelfOf(m)) return;
  const s = await act("move_model", { model: m.id, shelf });
  if (!s) return;
  S.shelves = s;
  S.shelf = shelf; store.set("shelf", shelf);
  S.sel = m.id;
  clearSearch();
  render();
}
async function renameModel(m) {
  const name = await askText(`A new name for ${m.name}:`, m.name);
  if (name == null || name.trim() === m.name) return;
  if ((await act("rename_model", { id: m.id, name })) === undefined) return;
  await refresh();
}
async function exportModel(m) {
  const file = await act("export_model", { id: m.id });
  if (file) { toast(`Saved ${file}`); await refresh(); }
}
async function deleteModel(m) {
  const yes = await askYes(`Delete ${m.name}? Its checkpoint, lessons archive, history and recovery points are removed from the models folder, for every program. Export it first if you may want it back.`, "Delete");
  if (!yes) return;
  if ((await act("delete_model", { id: m.id })) === undefined) return;
  toast(`Deleted ${m.name}`);
  await refresh();
}

// ---- Shelf actions -------------------------------------------------------------------------------
async function newShelf() {
  const name = await askText("Name the new shelf:");
  if (!name) return;
  const s = await act("add_shelf", { name });
  if (!s) return;
  S.shelves = s;
  S.shelf = s.shelves[s.shelves.length - 1].id; store.set("shelf", S.shelf);
  clearSearch();
  show("shelf");
}
async function renameShelf() {
  const shelf = currentShelf();
  const name = await askText(`A new name for the shelf ${shelf.name}:`, shelf.name);
  if (!name || name.trim() === shelf.name) return;
  const s = await act("rename_shelf", { shelf: shelf.id, name });
  if (s) { S.shelves = s; render(); }
}
async function deleteShelf() {
  const shelf = currentShelf();
  const others = S.shelves.shelves.filter(s => s.id !== shelf.id);
  const n = S.models.filter(m => shelfOf(m) === shelf.id).length;
  const msg = n
    ? `Remove the shelf ${shelf.name}? Its ${plural(n, "model moves", "models move")} to ${others[0].name}. No model is deleted.`
    : `Remove the empty shelf ${shelf.name}?`;
  if (!(await askYes(msg, "Remove"))) return;
  const s = await act("delete_shelf", { shelf: shelf.id });
  if (!s) return;
  S.shelves = s;
  S.shelf = s.shelves[0].id; store.set("shelf", S.shelf);
  render();
}

// ---- Views ---------------------------------------------------------------------------------------
function show(view) {
  S.view = view;
  render();
  $("main").scrollTop = 0;
}
function clearSearch() {
  S.search = "";
  $("search").value = "";
}

// ---- New model -----------------------------------------------------------------------------------
async function openNew() {
  if (!S.options) S.options = await act("model_options");
  if (!S.options) return;
  $("newName").value = "";
  $("newShelfChoice").replaceChildren(...S.shelves.shelves.map(s => el("option", { value: s.id, selected: s.id === currentShelf()?.id }, s.name)));
  renderDesign();
  show("new");
  $("newName").focus();
}
function vocabLimit() {
  const v = $("vocabChoice").value;
  if (v === "fixed") return null;
  if (v === "custom") return Number($("vocabCustom").value);
  return Number(v);
}
function contextChoice() {
  const v = $("contextChoice").value;
  return v === "custom" ? Number($("contextCustom").value) : Number(v);
}
// ---- New model: a Nova Neuro brain chosen network by network ------------------------------------
// The Network sizes each network as the Engine builds it (models.plan) and gives the exact counts.
const NETWORKS = [
  { id: "transformer", name: "Transformer", note: "Reads the words in order with attention: language, and finding what matters earlier in the context.", on: true, size: 2_000_000 },
  { id: "memory", name: "Working memory", note: "Keeps a running state from step to step (a gated recurrent network): what is happening now.", on: true, size: 500_000 },
  { id: "associative", name: "Associative memory", note: "Learned patterns recalled by similarity: facts and habits it has seen many times.", on: true, size: 500_000 },
  { id: "relational", name: "Relational network", note: "Slots for the things being talked about, and how they connect (who has what, where).", on: false, size: 250_000 },
  { id: "experts", name: "Experts", note: "Several small networks, and a router that learns which to use when: specialisation.", on: false, size: 1_000_000 },
  { id: "consolidation", name: "Consolidation", note: "Squeezes what the bus holds into a short code: a compact summary to remember by.", on: false, size: 250_000 },
];
const BRAIN_SIZES = [50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 4e6, 6e6, 8e6, 12e6, 16e6, 24e6, 32e6, 48e6, 64e6, 100e6];
const CONTROLLER_SIZES = [0, 10e3, 25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6];
const B = { nets: Object.fromEntries(NETWORKS.map(n => [n.id, { on: n.on, size: n.size }])), controller: 0, plan: null, asked: 0 };
const sizeName = n => (n >= 1e6 ? `${n / 1e6}M` : `${n / 1e3}K`) + " weights";

function brainChoice() {
  const c = { width: Number($("brainWidth").value) };
  for (const n of NETWORKS) if (B.nets[n.id].on) c[n.id] = B.nets[n.id].size;
  if (B.controller) c.controller = B.controller;
  return c;
}
const modulesOn = () => NETWORKS.some(n => n.id !== "transformer" && B.nets[n.id].on);

function buildingBrain() {
  return $("archChoice").value === "neuro" && document.querySelector("input[name=designWay]:checked")?.value === "build";
}

function renderDesign() {
  const neuro = $("archChoice").value === "neuro";
  $("designWays").hidden = !neuro;
  $("brainBuilder").hidden = !buildingBrain();
  $("presetSizes").hidden = buildingBrain();
  if (buildingBrain()) planBrain(); else renderSizes();
}

function renderBrain() {
  const plan = B.plan;
  // Numbers from the last plan are only shown for networks still chosen.
  const got = id => plan?.networks?.find(x => x.name === id)?.parameters;
  const cell = n => (n == null ? "—" : fmt(n));
  const rows = NETWORKS.map(n => {
    const st = B.nets[n.id];
    const box = el("input", { type: "checkbox", checked: st.on, "aria-label": n.name });
    box.addEventListener("change", () => { st.on = box.checked; planBrain(); });
    const pick = el("select", { "aria-label": `${n.name}: size`, disabled: !st.on },
      ...BRAIN_SIZES.map(v => el("option", { value: String(v), selected: v === st.size }, sizeName(v))));
    pick.addEventListener("change", () => { st.size = Number(pick.value); planBrain(); });
    return el("tr", { class: st.on ? "" : "off" },
      el("td", {}, el("label", { class: "net-name" }, box, el("span", {}, el("b", {}, n.name), el("p", { class: "note" }, n.note)))),
      el("td", {}, pick),
      el("td", { class: "num" }, st.on ? cell(got(n.id)) : "—"));
  });
  // The Executive Controller exists when there is a network besides the transformer to weigh.
  const ctl = el("select", { "aria-label": "Executive Controller: size", disabled: !modulesOn() },
    ...CONTROLLER_SIZES.map(v => el("option", { value: String(v), selected: v === B.controller }, v ? sizeName(v) : "automatic (small)")));
  ctl.addEventListener("change", () => { B.controller = Number(ctl.value); planBrain(); });
  rows.push(el("tr", { class: modulesOn() ? "" : "off" },
    el("td", {}, el("span", { class: "net-name" }, el("span", {}, el("b", {}, "Executive Controller"),
      el("p", { class: "note" }, modulesOn() ? "Reads the bus and decides how much each network contributes, step by step." : "Only with a network besides the transformer: it weighs those networks.")))),
    el("td", {}, ctl),
    el("td", { class: "num" }, modulesOn() ? cell(got("controller")) : "—")));
  for (const [id, name, note] of [["bus", "Communication bus", "How the networks exchange what they hold (made to fit the networks chosen)."], ["words", "Words", "The vectors of the characters and words it reads and writes (from the vocabulary)."]]) {
    if (got(id) == null) continue;
    rows.push(el("tr", { class: "fixed" }, el("td", {}, el("span", { class: "net-name" }, el("span", {}, el("b", {}, name), el("p", { class: "note" }, note)))), el("td", {}, ""), el("td", { class: "num" }, cell(got(id)))));
  }
  if (plan) rows.push(el("tr", { class: "total" }, el("td", {}, "In all"), el("td", {}, bytes(plan.fileBytes)), el("td", { class: "num" }, fmt(plan.parameters))));
  $("brainTable").tBodies[0].replaceChildren(...rows);
  $("brainNote").textContent = B.error
    ? B.error
    : plan && !B.pending ? `The file holds the weights and the optimizer's state (12 bytes per weight).${vocabLimit() != null ? " With a vocabulary limit the words grow as it learns them; this is the size once every slot is used." : ""}` : "Working out the sizes…";
  $("brainNote").classList.toggle("warn", !!B.error);
  $("createModel").disabled = !!B.error || !plan || !!B.pending;
}

let planTimer = 0;
function planBrain() {
  // The last plan stays shown (the page keeps its place) until the new one arrives.
  B.pending = true;
  renderBrain();
  clearTimeout(planTimer);
  planTimer = setTimeout(async () => {
    const n = ++B.asked;
    try {
      const plan = await invoke("plan_model", { choice: { name: "plan", architecture: "neuro", preset: "custom", brain: brainChoice(), wordLimit: vocabLimit(), context: contextChoice() || 256 } });
      if (n !== B.asked) return;
      B.plan = plan; B.error = null; B.pending = false;
    } catch (e) {
      if (n !== B.asked) return;
      B.plan = null; B.pending = false;
      B.error = /no method/.test(String(e))
        ? "Choosing the networks needs the Yendory Network 0.1.3 or later: update it on Downloads & updates. The ready-made designs work now."
        : String(e);
    }
    if (buildingBrain()) renderBrain();
  }, 150);
}

function renderSizes() {
  if (buildingBrain()) { planBrain(); return; }
  $("createModel").disabled = false;
  const arch = $("archChoice").value;
  const sizes = S.options[arch];
  const perParam = S.options.bytesPerParameter || 12;
  const body = $("sizeTable").tBodies[0];
  const chosen = body.querySelector("input:checked")?.value;
  const keep = sizes.some(s => s.id === chosen) ? chosen : sizes[0].id;
  const limit = vocabLimit();
  body.replaceChildren(...sizes.map(s => {
    // A vocabulary limit starts with no word slots: 256 fewer rows than Nova-2's layout.
    const start = limit == null ? s.parameters : s.parameters - 256 * s.perWord;
    const radio = el("input", { type: "radio", name: "newSize", value: s.id, checked: s.id === keep, "aria-label": s.name });
    const tr = el("tr", { class: "pick" + (s.id === keep ? " current-row" : "") },
      el("td", {}, radio), el("td", {}, s.name), el("td", { class: "num" }, fmt(start)), el("td", { class: "num" }, bytes(start * perParam)));
    tr.addEventListener("click", () => { radio.checked = true; renderSizes(); });
    return tr;
  }));
  const s = sizes.find(x => x.id === keep);
  $("sizeNote").textContent = limit == null
    ? `Nova-2's layout: 256 word slots, all reserved from the start. The file holds the weights and the optimizer's state (${perParam} bytes per parameter).`
    : `With a vocabulary limit the model starts with no word slots and adds ${fmt(s.perWord)} parameters for each word it learns` +
      (Number.isFinite(limit) && limit > 0 ? `: ${fmt(s.parameters - 256 * s.perWord + limit * s.perWord)} once it knows ${fmt(limit)} words.` : ".");
}
async function createModel() {
  const name = $("newName").value.trim();
  if (!name) { toast("Give the model a name", true); $("newName").focus(); return; }
  const wordLimit = vocabLimit();
  if (wordLimit != null && !(Number.isInteger(wordLimit) && wordLimit > 0)) { toast("Enter the vocabulary limit as a whole number of words", true); return; }
  const context = contextChoice();
  if (!(Number.isInteger(context) && context > 0)) { toast("Enter the context as a whole number of tokens", true); return; }
  const building = buildingBrain();
  const preset = building ? "custom" : $("sizeTable").querySelector("input:checked")?.value;
  const shelf = $("newShelfChoice").value;
  const button = $("createModel");
  button.disabled = true; button.textContent = "Creating…";
  const choice = { name, architecture: $("archChoice").value, preset, wordLimit, context };
  if (building) choice.brain = brainChoice();
  const id = await act("create_model", { choice, shelf });
  button.disabled = false; button.textContent = "Create untrained model";
  if (!id) return;
  S.sel = id; S.shelf = shelf; store.set("shelf", shelf);
  clearSearch();
  await loadLibrary();
  show("shelf");
  toast(`Created ${name}. Teach it in Studio or Yendory Tools.`);
}

// ---- Downloads & updates -------------------------------------------------------------------------
function statusCell(c) {
  if (!c.installed) return el("td", {}, c.released === false ? el("span", { class: "note" }, "Not released yet") : "Not installed");
  const how = c.kind === "hub" ? "running now" : c.managed ? "installed by the Hub" : `on its own, from ${legacyText(c)}`;
  return el("td", {},
    el("span", { class: "mono" }, c.version || "version unknown"),
    c.running && c.kind !== "hub" ? el("span", { class: "ok" }, c.kind === "service" ? " · running" : " · open now") : "",
    el("div", { class: "note" + (c.managed || c.kind === "hub" ? "" : " warn") }, how));
}
function actionCell(c) {
  const job = S.job && S.job.id === c.id ? S.job : null;
  if (job || c.installing) {
    return el("td", { class: "act" },
      el("div", { class: "meter" }, el("i", { style: `width:${job ? jobPercent(job) : 0}%` })),
      el("div", { class: "note" }, job ? jobText(job) : "Starting…"),
      job?.stage === "downloading" ? el("button", { type: "button", class: "quiet", onclick: () => invoke("cancel_install") }, "Cancel") : "");
  }
  const b = [];
  const busy = !!S.job;
  if (c.kind === "hub") {
    if (c.restartReady) b.push(el("button", { type: "button", class: "primary", onclick: () => act("restart_hub") }, `Restart to finish (${c.restartReady})`));
    else if (c.update) b.push(el("button", { type: "button", class: "primary", disabled: busy || !c.canInstall, onclick: () => install(c) }, `Update to ${c.latest}`));
  } else if (onItsOwn(c)) {
    b.push(el("button", { type: "button", class: "primary", disabled: busy || c.running || c.released !== true, onclick: switchStudio,
      title: c.running ? `Close ${c.name} first` : c.released !== true ? "Studio hasn't published its Hub package yet" : null }, "Switch to the Hub…"));
    b.push(el("button", { type: "button", disabled: c.running && !opensModels(c), onclick: () => launch(c.id) }, "Start"));
  } else if (!c.installed) {
    if (c.released === true) b.push(el("button", { type: "button", class: "primary", disabled: busy || !c.canInstall, onclick: () => install(c) }, "Install"));
  } else {
    if (c.update) b.push(el("button", { type: "button", class: "primary", disabled: busy || !c.canInstall || (c.running && c.kind === "program"), title: c.running && c.kind === "program" ? `Close ${c.name} first` : null, onclick: () => install(c) }, `Update to ${c.latest}`));
    if (c.kind === "program") {
      b.push(el("button", { type: "button", disabled: c.running && !opensModels(c), onclick: () => launch(c.id) }, c.running ? "Open now" : "Start"));
      b.push(el("button", { type: "button", class: "quiet", disabled: busy || c.running, onclick: () => removeComponent(c) }, "Remove…"));
    }
  }
  return el("td", { class: "act" }, el("div", { class: "act-btns" }, ...b));
}
function renderComponents() {
  $("componentTable").tBodies[0].replaceChildren(...S.components.map(c => el("tr", {},
    el("td", {}, el("div", { class: "prog-name" }, programMark(c), el("b", {}, c.name)), el("div", { class: "note" }, c.does),
      c.installNote && !onItsOwn(c) && c.released !== false ? el("div", { class: "note warn" }, c.installNote) : ""),
    statusCell(c),
    el("td", { class: "mono" }, c.latest || "—"),
    actionCell(c))));
}
async function install(c, quiet = false) {
  if (!quiet) {
    const text = c.kind === "hub"
      ? `Update the Hub to ${c.latest}? The Hub downloads its new version and checks its signature; restart it to finish. Your shelves and settings are kept.`
      : `${c.installed ? "Update" : "Install"} ${c.name} ${c.latest || ""}? The Hub downloads it from the release page, checks its signature, and keeps it in its own folder.` +
        (c.kind === "service" ? " Programs using the Network must be closed first." : "");
    if (!(await askYes(text, c.installed ? "Update" : "Install"))) return;
  }
  S.job = { id: c.id, stage: "downloading", received: 0, total: null };
  render();
  const done = await act("install_component", { id: c.id });
  S.job = null;
  if (done) {
    S.components = done;
    const now = component(c.id);
    toast(c.kind === "hub" ? `Hub ${now?.restartReady} is ready: restart to finish` : `${c.name} ${now?.version || ""} is installed`);
    if (c.kind === "service") { await loadAbout(); await loadLibrary(); }
  }
  render();
}
async function removeComponent(c) {
  if (!(await askYes(`Remove ${c.name} from this computer? Your models and its data stay; you can install it again here.`, "Remove"))) return;
  const done = await act("remove_component", { id: c.id });
  if (done) { S.components = done; toast(`${c.name} was removed`); }
  render();
}
async function checkUpdates() {
  $("checkNote").textContent = "Checking…";
  await loadComponents(true);
  const reached = S.components.some(c => c.released !== null && c.released !== undefined);
  $("checkNote").textContent = reached ? `Checked ${when(Date.now() / 1000)}` : "The release page could not be reached. Check the internet connection.";
  render();
}

// ---- Settings ------------------------------------------------------------------------------------
function renderSettings() {
  const a = S.about;
  if (a) {
    $("modelsFolder").textContent = a.modelsFolder;
    $("programsFolder").textContent = a.programsFolder;
    $("hubFolder").textContent = a.hubFolder;
    $("networkInfo").textContent = a.network ? `Network ${a.network}, Engine ${a.engine}` : a.networkError || "Not installed";
  }
  $("locationTable").tBodies[0].replaceChildren(...programs().filter(p => p.released !== false || p.installed).map(p => {
    const located = p.legacy.some(l => l.kind === "located");
    return el("tr", {},
      el("td", {}, el("b", {}, p.name)),
      el("td", {}, p.installed
        ? [el("div", { class: "mono" }, p.path), el("div", { class: "note" }, p.managed ? "Installed by the Hub" : `On its own, from ${legacyText(p)}`)]
        : el("span", { class: "note" }, "Not found")),
      el("td", { class: "act" }, p.managed ? "" : el("div", { class: "act-btns" },
        el("button", { type: "button", onclick: async () => { if (await act("locate_program", { id: p.id })) { await loadComponents(); render(); } } }, "Choose its file…"),
        located ? el("button", { type: "button", class: "quiet", onclick: async () => { await act("forget_location", { id: p.id }); await loadComponents(); render(); } }, "Forget") : "")));
  }));
}

// ---- Wiring --------------------------------------------------------------------------------------
function setFolded(folded) {
  $("hub").classList.toggle("folded", folded);
  $("sbToggle").setAttribute("aria-expanded", String(!folded));
  $("sbToggle").setAttribute("aria-label", folded ? "Unfold the sidebar" : "Fold the sidebar");
  store.set("folded", folded ? "1" : "0");
}

function wire() {
  $("sbToggle").addEventListener("click", () => setFolded(!$("hub").classList.contains("folded")));
  $("search").closest("label").addEventListener("click", () => {
    if ($("hub").classList.contains("folded")) { setFolded(false); setTimeout(() => $("search").focus(), 200); }
  });
  $("search").addEventListener("input", () => { S.search = $("search").value; if (S.view !== "shelf") S.view = "shelf"; render(); });
  $("search").addEventListener("keydown", e => { if (e.key === "Escape") { clearSearch(); render(); } });
  $("newShelf").addEventListener("click", newShelf);
  for (const b of document.querySelectorAll(".sb-foot [data-view]")) b.addEventListener("click", () => show(b.dataset.view));
  $("renameShelf").addEventListener("click", renameShelf);
  $("deleteShelf").addEventListener("click", deleteShelf);
  $("sortChoice").addEventListener("change", () => { S.sort = $("sortChoice").value; store.set("sort", S.sort); render(); });
  // Left and right arrows move along the shelf.
  $("shelf").addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const spines = [...$("shelf").querySelectorAll(".spine:not(.ghost)")];
    const i = spines.indexOf(document.activeElement);
    if (i < 0) return;
    const next = spines[i + (e.key === "ArrowRight" ? 1 : -1)];
    if (next) { e.preventDefault(); S.sel = next.dataset.id; render(); $("shelf").querySelector(`[data-id="${CSS.escape(S.sel)}"]`)?.focus(); }
  });
  $("archChoice").addEventListener("change", renderDesign);
  for (const r of document.querySelectorAll("input[name=designWay]")) r.addEventListener("change", renderDesign);
  $("brainWidth").addEventListener("change", planBrain);
  $("contextChoice").addEventListener("change", () => { if (buildingBrain()) planBrain(); });
  $("vocabChoice").addEventListener("change", () => { $("vocabCustom").hidden = $("vocabChoice").value !== "custom"; renderSizes(); });
  $("vocabCustom").addEventListener("input", renderSizes);
  $("contextChoice").addEventListener("change", () => { $("contextCustom").hidden = $("contextChoice").value !== "custom"; });
  $("createModel").addEventListener("click", createModel);
  $("cancelNew").addEventListener("click", () => show("shelf"));
  $("newName").addEventListener("keydown", e => { if (e.key === "Enter") createModel(); });
  $("checkUpdates").addEventListener("click", checkUpdates);
  listen("install-progress", e => {
    S.job = { ...(S.job || {}), ...e.payload };
    renderSidebar();
    if (S.view === "programs") renderComponents();
    if (!$("view-setup").hidden) renderSetup();
  });
  listen("network-event", e => {
    const { name } = e.payload;
    if (name === "models.changed") loadLibrary().then(render);
    else if (name === "network.disconnected" && !S.job) {
      // The Network stopped (or was replaced): connect again in a moment.
      setTimeout(async () => { await loadAbout(); await refresh(); }, 1500);
    }
  });
  // Studio may have changed the library or started or stopped while the Hub was in the background.
  window.addEventListener("focus", () => { if (!S.job) refresh(); });
  setInterval(async () => {
    if (!document.hasFocus() || S.job) return;
    const before = JSON.stringify(S.components);
    await loadComponents();
    if (JSON.stringify(S.components) !== before) render();
  }, 10000);
}

async function start() {
  setFolded(store.get("folded", "0") === "1");
  wire();
  await loadComponents();
  render();
  await loadAbout();
  await refresh();
  if (S.about?.updatedFrom) toast(`The Hub was updated from ${S.about.updatedFrom} to ${S.about.hub}`);
  // The newest packages, in the background (needs the internet; quiet when offline).
  loadComponents(true).then(render);
}
start();
