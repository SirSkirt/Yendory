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
// on. null: no released Studio does yet, so the card says Studio opens on its Models page.
const STUDIO_OPENS_MODELS_FROM = null;

const S = {
  models: [],
  shelves: { shelves: [], places: {} },
  programs: [],
  about: null,
  options: null,
  shelf: store.get("shelf", ""),
  sel: store.get("sel", ""),
  sort: store.get("sort", "updated"),
  search: "",
  view: "shelf",
  job: null,
  hubUpdate: null,
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
function dialog(message, field, okLabel = "OK") {
  return new Promise(resolve => {
    const before = document.activeElement;
    const ok = el("button", { type: "button", class: "primary" }, okLabel);
    const cancel = el("button", { type: "button" }, "Cancel");
    const overlay = el("div", { class: "modal" },
      el("div", { class: "modal-box", role: "dialog", "aria-modal": "true", "aria-label": message },
        el("p", {}, message), field || "", el("div", { class: "actions" }, cancel, ok)));
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
async function loadLibrary() {
  const [models, shelves] = await Promise.all([act("list_models"), act("shelves")]);
  if (models) S.models = models;
  if (shelves) S.shelves = shelves;
}
async function loadPrograms(check = false) {
  const p = await act("programs", { check });
  if (p) S.programs = p;
}
async function refresh() {
  await Promise.all([loadLibrary(), loadPrograms()]);
  render();
}

// ---- Shelves -------------------------------------------------------------------------------------
const shelfIds = () => S.shelves.shelves.map(s => s.id);
const currentShelf = () => S.shelves.shelves.find(s => s.id === S.shelf) || S.shelves.shelves[0];
function shelfOf(m) {
  const placed = S.shelves.places[m.id];
  return shelfIds().includes(placed) ? placed : S.shelves.shelves[0]?.id;
}
const program = id => S.programs.find(p => p.id === id);
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

// ---- Rendering -----------------------------------------------------------------------------------
function render() {
  const visible = visibleModels();
  if (!visible.some(m => m.id === S.sel)) S.sel = visible[0]?.id || "";
  store.set("sel", S.sel);
  for (const v of ["shelf", "new", "programs", "settings"]) $("view-" + v).hidden = S.view !== v;
  renderSidebar();
  if (S.view === "shelf") { renderShelf(visible); renderCard(); }
  if (S.view === "programs") renderPrograms();
  if (S.view === "settings") renderSettings();
}

function renderSidebar() {
  const list = $("shelfList");
  list.replaceChildren(...S.shelves.shelves.map(s => {
    const n = S.models.filter(m => shelfOf(m) === s.id).length;
    const on = S.view === "shelf" && !S.search && s.id === currentShelf()?.id;
    return el("button", { type: "button", class: "sb-item" + (on ? " on" : ""), title: `${s.name}: ${plural(n, "model", "models")}`, "aria-current": on ? "true" : null,
      onclick: () => { S.shelf = s.id; store.set("shelf", s.id); clearSearch(); show("shelf"); } },
      el("span", { class: "smark", "aria-hidden": "true" }, (s.name.trim()[0] || "?").toUpperCase()),
      el("span", { class: "lbl" }, s.name), el("span", { class: "cnt" }, n));
  }));
  const m = selected();
  $("programList").replaceChildren(...S.programs.map(p => {
    const status = !p.released ? "not released" : p.running ? "open" : p.installed ? (p.update ? "update" : "installed") : "install";
    const opens = p.installed && !p.running;
    const title = opens ? (m ? `Open ${m.name} in ${shortName(p)}` : `Start ${shortName(p)}`) : `${p.name}: ${status}`;
    return el("button", { type: "button", class: "sb-item", title,
      onclick: () => (opens ? launch(p.id, m?.id) : show("programs")) },
      programMark(p),
      p.running ? el("i", { class: "live", title: "Running" }) : "",
      el("span", { class: "lbl" }, shortName(p)),
      el("span", { class: "st" + (p.update ? " warn" : p.running ? " ok" : "") }, status));
  }));
  const updates = S.programs.filter(p => p.update).length + (S.hubUpdate ? 1 : 0);
  $("updateCount").textContent = updates || "";
  for (const b of document.querySelectorAll(".sb-foot [data-view]")) b.classList.toggle("on", S.view === b.dataset.view);
  if (S.about) $("versions").textContent = `Hub ${S.about.hub} · Engine ${S.about.engine}`;
  const job = S.job;
  $("job").hidden = !job;
  if (job) {
    const name = program(job.id)?.name || job.id;
    const pct = job.total ? Math.min(100, (100 * job.received) / job.total) : 0;
    $("jobText").textContent = job.stage === "downloading"
      ? `Downloading ${name} · ${bytes(job.received)}${job.total ? " of " + bytes(job.total) : ""}`
      : job.stage === "checking" ? `Checking ${name}'s signature` : `Installing ${name}`;
    $("jobBar").style.width = (job.stage === "downloading" ? pct : 100) + "%";
  }
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
  const studio = program("studio"), tools = program("tools");
  const studioRunning = !!studio?.running;
  let studioButton, why = "";
  if (!studio) {
    studioButton = el("button", { type: "button", class: "primary", disabled: true }, "Open in Studio");
  } else if (!studio.installed) {
    studioButton = el("button", { type: "button", class: "primary", onclick: () => show("programs") }, "Install Studio…");
  } else if (studioRunning) {
    studioButton = el("button", { type: "button", class: "primary", disabled: true }, "Open in Studio");
    why = `Studio is open. Switch to it and open ${m.name} from § 1 Models: a program that is already running can't be handed a model yet.`;
  } else {
    studioButton = el("button", { type: "button", class: "primary", onclick: () => launch("studio", m.id) }, "Open in Studio");
    if (!opensModels(studio)) why = `Studio${studio.version ? " " + studio.version : ""} starts on its Models page: choose ${m.name} there. It will open the model directly once Studio supports the Hub's hand-off.`;
  }
  const toolsButton = tools?.installed && !tools.running
    ? el("button", { type: "button", onclick: () => launch("tools", m.id) }, "Open in Tools")
    : el("button", { type: "button", disabled: true }, tools && !tools.released ? "Tools · not released yet" : "Tools");
  const lockedTitle = studioRunning ? "Close Studio first: it may have this model open" : null;
  card.replaceChildren(
    el("h2", {}, m.name, el("span", { class: `tag ${m.architecture}` }, `${archName(m.architecture)} ${short(m.parameters)}`)),
    m.plan ? row("Training plan", `${m.plan.name}: ${planText(m.plan)}`, { cls: m.plan.inProgress ? "plan" : "" }) : "",
    row("Training steps", fmt(m.step), { mono: true }),
    row("Parameters", fmt(m.parameters), { mono: true }),
    row("Words learned", words),
    row("Context", `${fmt(m.context)} tokens`),
    row("Read", m.read.length ? m.read.join(", ") : "Nothing yet"),
    row("Lessons taught", m.lessons ? fmt(m.lessons) : "None yet"),
    row("Changed", when(m.updated)),
    row("Origin", m.origin === "new" || !m.origin ? `Created ${when(m.created)}` : `Imported from ${m.origin}`),
    row("File", bytes(m.fileBytes), { mono: true }),
    el("div", { class: "take" },
      el("p", {}, `Take ${m.name} to…`),
      el("div", { class: "btns" }, studioButton, toolsButton),
      why ? el("p", { class: "why" }, why) : "",
      el("div", { class: "more" },
        el("button", { type: "button", class: "quiet", onclick: () => moveModel(m) }, "Move to shelf…"),
        el("button", { type: "button", class: "quiet", onclick: () => renameModel(m), disabled: studioRunning, title: lockedTitle }, "Rename…"),
        el("button", { type: "button", class: "quiet", onclick: () => exportModel(m) }, "Export…"),
        el("button", { type: "button", class: "quiet", onclick: () => deleteModel(m), disabled: studioRunning, title: lockedTitle }, "Delete…"))));
  recent.replaceChildren(el("h3", {}, "Recently"),
    ...(m.recent.length
      ? m.recent.map(r => el("div", { class: "rec" }, el("span", { class: "when" }, when(r.time)), el("span", { class: "who" }, r.program), r.text))
      : [el("p", { class: "empty note" }, "Nothing yet. Lessons, reading and what the Hub does with this model appear here.")]));
}

function opensModels(studio) {
  if (!STUDIO_OPENS_MODELS_FROM || !studio.version) return false;
  const parts = v => v.split(/[-+]/)[0].split(".").map(Number);
  const [a, b] = [parts(studio.version), parts(STUDIO_OPENS_MODELS_FROM)];
  for (let i = 0; i < 3; i++) if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) > (b[i] || 0);
  return true;
}

/** A program's logo (Studio's own icon), or its initial while it has none. */
function programMark(p) {
  const off = p.installed ? "" : " off";
  return LOGOS[p.id]
    ? el("img", { class: "pmark logo" + off, src: LOGOS[p.id], alt: "", "aria-hidden": "true" })
    : el("span", { class: `pmark ${p.id}${off}`, "aria-hidden": "true" }, shortName(p)[0]);
}
const LOGOS = { studio: "img/studio.png" };

// ---- Model actions -------------------------------------------------------------------------------
async function launch(id, model) {
  const p = program(id);
  const ok = await act("launch", { id, model: model || null });
  if (ok === undefined) return;
  toast(model ? `Starting ${p?.name || id} with ${S.models.find(m => m.id === model)?.name || "the model"}…` : `Starting ${p?.name || id}…`);
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
  renderSizes();
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
function renderSizes() {
  const arch = $("archChoice").value;
  const sizes = S.options[arch];
  const body = $("sizeTable").tBodies[0];
  const chosen = body.querySelector("input:checked")?.value;
  const keep = sizes.some(s => s.id === chosen) ? chosen : sizes[0].id;
  const limit = vocabLimit();
  body.replaceChildren(...sizes.map(s => {
    // A vocabulary limit starts with no word slots: 256 fewer rows than Nova-2's layout.
    const start = limit == null ? s.parameters : s.parameters - 256 * s.perWord;
    const radio = el("input", { type: "radio", name: "newSize", value: s.id, checked: s.id === keep, "aria-label": s.name });
    const tr = el("tr", { class: "pick" + (s.id === keep ? " current-row" : "") },
      el("td", {}, radio), el("td", {}, s.name), el("td", { class: "num" }, fmt(start)), el("td", { class: "num" }, bytes(start * 12)));
    tr.addEventListener("click", () => { radio.checked = true; renderSizes(); });
    return tr;
  }));
  const s = sizes.find(x => x.id === keep);
  $("sizeNote").textContent = limit == null
    ? "Nova-2's layout: 256 word slots, all reserved from the start. The file holds the weights and the optimizer's state (12 bytes per parameter)."
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
  const preset = $("sizeTable").querySelector("input:checked")?.value;
  const shelf = $("newShelfChoice").value;
  const button = $("createModel");
  button.disabled = true; button.textContent = "Creating…";
  const id = await act("create_model", { choice: { name, architecture: $("archChoice").value, preset, wordLimit, context, shelf } });
  button.disabled = false; button.textContent = "Create untrained model";
  if (!id) return;
  S.sel = id; S.shelf = shelf; store.set("shelf", shelf);
  clearSearch();
  await loadLibrary();
  show("shelf");
  toast(`Created ${name}. Open it in Studio to teach it.`);
}

// ---- Downloads & updates -------------------------------------------------------------------------
function renderPrograms() {
  const body = $("programTable").tBodies[0];
  body.replaceChildren(...S.programs.map(p => {
    const installedCell = !p.released ? el("td", { class: "note" }, "Not released yet")
      : p.installed
        ? el("td", {}, el("span", { class: "mono" }, p.version || "version unknown"), p.running ? el("span", { class: "ok" }, " · open now") : "",
          el("div", { class: "note mono" }, p.path))
        : el("td", {}, "Not installed");
    let action;
    if (p.installing || (S.job && S.job.id === p.id)) {
      const j = S.job;
      const pct = j && j.total ? (100 * j.received) / j.total : 0;
      action = el("td", { class: "act" },
        el("div", { class: "meter" }, el("i", { style: `width:${j?.stage === "downloading" ? pct : 100}%` })),
        el("div", { class: "note" }, !j ? "Starting…" : j.stage === "downloading" ? `${bytes(j.received)}${j.total ? " of " + bytes(j.total) : ""}` : j.stage === "checking" ? "Checking the signature…" : "Installing…"),
        el("button", { type: "button", class: "quiet", onclick: () => invoke("cancel_install") }, "Cancel"));
    } else {
      const buttons = [];
      if (p.released && p.canInstall && (!p.installed || p.update)) {
        buttons.push(el("button", { type: "button", class: "primary", disabled: p.running, title: p.running ? `Close ${p.name} first` : null, onclick: () => install(p) },
          p.installed ? `Update to ${p.latest}` : "Install"));
      }
      if (p.installed) buttons.push(el("button", { type: "button", disabled: p.running, onclick: () => launch(p.id) }, p.running ? "Open now" : "Start"));
      if (p.released && !p.canInstall && p.installNote) buttons.push(el("button", { type: "button", class: "quiet", onclick: () => act("open_releases") }, "Release page"));
      action = el("td", { class: "act" }, el("div", { class: "act-btns" }, ...buttons));
    }
    return el("tr", {},
      el("td", {}, el("div", { class: "prog-name" }, programMark(p), el("b", {}, p.name)), el("div", { class: "note" }, p.does), p.released && p.installNote ? el("div", { class: "note warn" }, p.installNote) : ""),
      installedCell,
      el("td", { class: "mono" }, p.latest || "—"),
      action);
  }));
  if (S.about) $("hubVersion").textContent = `Hub ${S.about.hub}`;
}
async function install(p) {
  const verb = p.installed ? "Update" : "Install";
  const where = navigator.userAgent.includes("Windows")
    ? "its installer opens with a progress bar and installs over any earlier version, keeping your models and settings"
    : "the Hub keeps it in its own folder and starts it from there";
  if (!(await askYes(`${verb} ${p.name} ${p.latest || ""}? The Hub downloads it from the release page and checks its signature; then ${where}.`, verb))) return;
  S.job = { id: p.id, stage: "downloading", received: 0, total: null };
  render();
  const programs = await act("install_program", { id: p.id });
  S.job = null;
  if (programs) {
    S.programs = programs;
    toast(navigator.userAgent.includes("Windows") ? `${p.name}'s installer is running` : `${p.name} is installed`);
  }
  render();
  if (programs && navigator.userAgent.includes("Windows")) setTimeout(() => loadPrograms().then(render), 15000);
}
async function checkUpdates() {
  $("checkNote").textContent = "Checking…";
  await loadPrograms(true);
  const reached = S.programs.some(p => p.released && p.latest);
  $("checkNote").textContent = reached ? `Checked ${when(Date.now() / 1000)}` : "The release page could not be reached. Check the internet connection.";
  render();
}
async function checkHub() {
  $("hubNote").textContent = "Checking…";
  try {
    S.hubUpdate = await invoke("check_hub_update");
    $("installHub").hidden = !S.hubUpdate;
    $("hubNote").textContent = S.hubUpdate
      ? `Hub ${S.hubUpdate.version} is available. Updating installs over this version and keeps your shelves and settings.`
      : "This is the newest Hub.";
  } catch (e) {
    $("hubNote").textContent = String(e);
  }
  renderSidebar();
}
async function installHub() {
  $("installHub").disabled = true;
  $("hubNote").textContent = "Downloading…";
  try {
    await invoke("install_hub_update");
  } catch (e) {
    $("hubNote").textContent = String(e);
    $("installHub").disabled = false;
  }
}

// ---- Settings ------------------------------------------------------------------------------------
const SOURCES = { located: "the file you chose", hub: "installed by the Hub", installer: "its installer", system: "a system package" };
function renderSettings() {
  if (S.about) {
    $("modelsFolder").textContent = S.about.modelsFolder;
    $("hubFolder").textContent = S.about.hubFolder;
  }
  $("locationTable").tBodies[0].replaceChildren(...S.programs.filter(p => p.released).map(p => el("tr", {},
    el("td", {}, el("b", {}, p.name)),
    el("td", {}, p.installed ? [el("div", { class: "mono" }, p.path), el("div", { class: "note" }, `Found from ${SOURCES[p.source] || p.source}`)] : el("span", { class: "note" }, "Not found")),
    el("td", { class: "act" },
      el("button", { type: "button", onclick: async () => { if (await act("locate_program", { id: p.id })) { await loadPrograms(); render(); } } }, "Choose its file…"),
      p.source === "located" ? el("button", { type: "button", class: "quiet", onclick: async () => { await act("forget_location", { id: p.id }); await loadPrograms(); render(); } }, "Forget") : ""))));
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
  $("archChoice").addEventListener("change", renderSizes);
  $("vocabChoice").addEventListener("change", () => { $("vocabCustom").hidden = $("vocabChoice").value !== "custom"; renderSizes(); });
  $("vocabCustom").addEventListener("input", renderSizes);
  $("contextChoice").addEventListener("change", () => { $("contextCustom").hidden = $("contextChoice").value !== "custom"; });
  $("createModel").addEventListener("click", createModel);
  $("cancelNew").addEventListener("click", () => show("shelf"));
  $("newName").addEventListener("keydown", e => { if (e.key === "Enter") createModel(); });
  $("checkUpdates").addEventListener("click", checkUpdates);
  $("checkHub").addEventListener("click", checkHub);
  $("installHub").addEventListener("click", installHub);
  listen("install-progress", e => { S.job = e.payload; renderSidebar(); if (S.view === "programs") renderPrograms(); });
  listen("hub-update-progress", e => {
    const { received, total } = e.payload;
    $("hubNote").textContent = `Downloading the new Hub · ${bytes(received)}${total ? " of " + bytes(total) : ""}`;
  });
  // Studio may have changed the library or started or stopped while the Hub was in the background.
  window.addEventListener("focus", () => { if (!S.job) refresh(); });
  setInterval(async () => {
    if (!document.hasFocus() || S.job) return;
    const before = JSON.stringify(S.programs);
    await loadPrograms();
    if (JSON.stringify(S.programs) !== before) render();
  }, 10000);
}

async function start() {
  setFolded(store.get("folded", "0") === "1");
  wire();
  S.about = await act("about");
  await refresh();
  // The newest releases, in the background (needs the internet; quiet when offline).
  loadPrograms(true).then(render);
  invoke("check_hub_update").then(u => { S.hubUpdate = u; if (u) { $("installHub").hidden = false; $("hubNote").textContent = `Hub ${u.version} is available.`; } renderSidebar(); }).catch(() => {});
}
start();
