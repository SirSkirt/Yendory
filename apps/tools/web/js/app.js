"use strict";
// Yendory Tools window. The games, matches, lessons and scoreboard are computed by the backend
// (src-tauri); every line the model writes comes from the Yendory Network.

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
  get(k, d) { try { return localStorage.getItem("tools." + k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("tools." + k, v); } catch { /* not kept */ } },
};

const S = { route: store.get("route", "games"), about: null, games: [], model: null, match: null, reports: [], busy: false, measuring: false };
const ROUTES = {
  games: ["Games", "Games"], play: ["Games", "Play"], lessons: ["Games", "Lessons"], scoreboard: ["Games", "Scoreboard"],
  profiles: ["Real games", "Game profiles"], controls: ["Real games", "Controls check"],
  apps: ["Apps", "Apps"], settings: ["Tools", "Settings"],
};

// ---- Formatting and messages --------------------------------------------------------------------
const nf = new Intl.NumberFormat("en-US");
const fmt = n => nf.format(n);
const trim1 = v => (v >= 10 ? String(Math.round(v)) : String(Math.round(v * 10) / 10));
const short = n => (n >= 1e9 ? trim1(n / 1e9) + "B" : n >= 1e6 ? trim1(n / 1e6) + "M" : n >= 1e3 ? trim1(n / 1e3) + "K" : String(n));
const pct = (a, b) => (b ? `${Math.round((100 * a) / b)}%` : "—");
const when = t => new Date(t * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
const gameInfo = id => S.games.find(g => g.id === id);

let toastTimer = 0;
function toast(message, error = false) {
  let t = $("toast");
  if (!t) { t = el("div", { id: "toast", class: "toast", role: "status" }); document.body.append(t); }
  t.textContent = message;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), error ? 9000 : 5000);
}
async function act(cmd, args) {
  try { return await invoke(cmd, args); } catch (e) { toast(String(e), true); return undefined; }
}
function notice(text, warn = false, button = null) {
  const n = $("notice");
  n.hidden = !text;
  n.className = "handoff-notice" + (warn ? " warn" : "");
  n.replaceChildren(el("span", {}, text || ""), button || "");
}

// ---- Routing ------------------------------------------------------------------------------------
function show(route) {
  if (!ROUTES[route]) route = "games";
  if (route === "profiles" && S.route !== "profiles") refreshWindows();
  S.route = route;
  store.set("route", route);
  for (const p of document.querySelectorAll("main > .page")) p.hidden = p.id !== "page-" + route;
  for (const b of document.querySelectorAll("[data-route]")) b.classList.toggle("active", b.dataset.route === route);
  $("pageCrumb").textContent = ROUTES[route][0];
  $("pageTitle").textContent = ROUTES[route][1];
  render();
  window.scrollTo(0, 0);
}

function render() {
  renderModel();
  if (S.route === "games") renderGames();
  if (S.route === "play") renderPlay();
  if (S.route === "scoreboard") renderScores();
  if (S.route === "settings") renderSettings();
  if (S.route === "profiles") renderProfiles();
  if (S.route === "controls") renderCheck();
  if (S.route !== "profiles") L.watching = false;
  renderWatch();
}

// ---- The open model -------------------------------------------------------------------------------
function renderModel() {
  const m = S.model;
  $("currentModel").replaceChildren(m
    ? el("span", {}, "Open: ", el("b", {}, m.name), ` (${m.architecture === "neuro" ? "Neuro" : "Bicameral"} ${short(m.parameters)} · step ${fmt(m.step)})`)
    : "No model open");
}
async function chooseModel() {
  const list = await act("models");
  if (!list) return;
  if (!list.length) { toast("There are no models yet: make one in Yendory Hub or Studio", true); return; }
  const pick = el("div", { class: "model-pick" });
  const overlay = el("div", { class: "modal" }, el("div", { class: "modal-box", role: "dialog", "aria-modal": "true", "aria-label": "Open a model" },
    el("p", {}, "Open which model?"), pick,
    el("div", { class: "actions" }, el("button", { type: "button", onclick: () => overlay.remove() }, "Cancel"))));
  for (const m of list) {
    pick.append(el("button", { type: "button", onclick: async () => { overlay.remove(); await openModel(m.id); } },
      el("span", {}, el("b", {}, m.name), m.openBy?.length ? ` · open in ${m.openBy.join(", ")}` : ""),
      el("small", {}, `${short(m.parameters)} · step ${fmt(m.step)}`)));
  }
  overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });
  document.body.append(overlay);
  pick.querySelector("button")?.focus();
}
async function openModel(id) {
  const m = await act("open_model", { id });
  if (!m) return;
  S.model = m;
  S.match = null;
  await loadReports();
  toast(`Opened ${m.name}`);
  render();
}
async function loadReports() {
  S.reports = S.model ? (await act("reports", { model: S.model.id })) || [] : [];
}

// ---- 1.1 Games ----------------------------------------------------------------------------------
function latest(game) {
  return [...S.reports].reverse().find(r => r.game === game);
}
function resultText(r) {
  const t = Object.fromEntries(r.tallies);
  if (t.perfect || t.random) {
    return [["perfect", "vs perfect"], ["random", "vs random"]].filter(([k]) => t[k]?.games)
      .map(([k, label]) => `${label} ${t[k].wins}–${t[k].draws}–${t[k].losses}`);
  }
  return [["taught", "taught mazes"], ["held", "new mazes"]].filter(([k]) => t[k]?.games)
    .map(([k, label]) => `${label}: found ${t[k].wins} of ${t[k].games}`);
}
function renderGames() {
  $("gameList").replaceChildren(...S.games.map(g => {
    const r = latest(g.id);
    return el("div", { class: "entry game-entry" },
      el("h3", {}, g.name),
      el("p", {}, el("b", {}, g.summary), " ", g.rules),
      el("p", { class: "note" }, g.perfectNote),
      el("p", { class: "note" }, "What the model sees at the start of every turn:"),
      el("p", { class: "announce" }, g.announcement),
      el("p", { class: "record" }, S.model
        ? (r ? [`${S.model.name}, measured ${when(r.time)}: `, resultText(r).join("; "), `; best moves ${pct(r.moves.best, r.moves.turns)}`] : `${S.model.name} hasn't been measured on ${g.name} yet.`)
        : el("span", { class: "note" }, "Open a model to see how it plays.")),
      el("div", { class: "actions" },
        el("button", { type: "button", class: "primary", onclick: () => { $("playGame").value = g.id; syncFirst(); show("play"); } }, "Play"),
        el("button", { type: "button", onclick: () => { $("lessonGame").value = g.id; show("lessons"); } }, "Lessons"),
        el("button", { type: "button", onclick: () => { $("scoreGame").value = g.id; show("scoreboard"); } }, "Scoreboard")));
  }));
}

// ---- 1.2 Play -----------------------------------------------------------------------------------
function syncFirst() {
  const g = gameInfo($("playGame").value);
  // A puzzle has no opponent and no first player.
  $("firstWrap").hidden = !g?.twoPlayer;
  $("opponentLabel").hidden = !g?.twoPlayer;
  $("playOpponent").hidden = !g?.twoPlayer;
}
async function newGame() {
  const game = $("playGame").value;
  const g = gameInfo(game);
  const m = await act("new_match", { game, opponent: g.twoPlayer ? $("playOpponent").value : "perfect", modelFirst: $("modelFirst").checked });
  if (!m) return;
  S.match = m;
  renderPlay();
}
async function step() {
  if (!S.match || S.busy) return false;
  S.busy = true;
  renderPlay();
  const m = await act("match_step", { id: S.match.id });
  S.busy = false;
  if (m) S.match = m;
  renderPlay();
  return !!m;
}
async function playOut() {
  while (S.match && !S.match.finished && !S.match.needsPerson) {
    if (!(await step())) break;
    await new Promise(r => setTimeout(r, 120));
  }
}
async function personMove(input) {
  const m = await act("person_move", { id: S.match.id, input });
  if (m) { S.match = m; renderPlay(); }
}
function outcomeText(m) {
  if (m.forfeit) return "The model lost: its turn ended without a move.";
  return { win: "The model won.", draw: "A draw.", loss: "The model lost." }[m.outcome] || "";
}
function renderBoard(m) {
  const b = m.board;
  const person = m.needsPerson && !S.busy;
  if (m.game === "tictactoe") {
    return el("div", { class: "ttt", role: "grid", "aria-label": "Board" }, ...b.cells.map((c, i) => {
      const sq = String(i + 1);
      const can = person && m.legal.includes(sq);
      return el("button", { type: "button", class: c === "." ? "" : c, disabled: !can, "aria-label": `Square ${sq}: ${c === "." ? "free" : c}`,
        onclick: () => personMove(sq) }, el("span", { class: "sq" }, sq), c === "." ? "" : c);
    }));
  }
  if (m.game === "nim") {
    return el("div", { class: "nim" },
      el("div", { class: "stones", "aria-hidden": "true" }, ...Array.from({ length: b.stones }, () => el("i", { class: "stone" }))),
      el("p", { class: "count" }, `${b.stones} of ${b.start} stones left`),
      person ? el("div", { class: "take" }, ...m.legal.map(k => el("button", { type: "button", onclick: () => personMove(k) }, `Take ${k}`))) : "");
  }
  if (m.game === "maze") {
    const cells = [];
    for (let i = 0; i < b.size * b.size; i++) {
      const o = b.open[i];
      const cls = ["cell", !o[0] && "wn", !o[1] && "ws", !o[2] && "we", !o[3] && "ww"].filter(Boolean).join(" ");
      cells.push(el("div", { class: cls }, i === b.at ? el("i", { class: "you", title: "The model" }) : i === b.treasure ? el("i", { class: "gold", title: "Treasure" }) : ""));
    }
    return el("div", {}, el("div", { class: "maze", role: "img", "aria-label": `Maze: ${b.movesLeft} moves left` }, ...cells),
      el("p", { class: "note" }, `${b.movesLeft} of ${b.budget} moves left`));
  }
  return el("p", {}, "—");
}
function gradeTags(g) {
  if (!g) return "";
  const tags = [];
  if (!g.legal) tags.push(el("span", { class: "tag nomove" }, "no move"));
  else if (g.best) tags.push(el("span", { class: "tag best" }, "best move"));
  else tags.push(el("span", { class: "tag legal" }, "legal, not best"));
  if (g.heldOut) tags.push(el("span", { class: "tag held", title: "A position no lesson contains" }, "held out"));
  return el("span", { class: "grade" }, ...tags);
}
function renderTurns(m) {
  const lastModel = m.log.map(e => e.who).lastIndexOf("model");
  const out = [];
  m.log.forEach((e, i) => {
    // The opening line is part of the model's first turn once it has played one.
    if (e.who === "game") { if (lastModel < 0) out.push(el("p", { class: "game" }, e.text)); }
    else if (e.who === "model" && e.turn) {
      out.push(el("p", { class: "turn-head" }, el("span", { class: "who" }, "U:"), e.turn.prompt));
      for (const l of e.turn.lines) {
        if (l.kind === "action") out.push(el("p", { class: "act" }, el("span", { class: "who" }, "Action:"), l.text));
        else if (l.kind === "result") out.push(el("p", { class: "res" }, el("span", { class: "who" }, "Result:"), l.text));
        else out.push(el("p", { class: "model" }, el("span", { class: "who" }, "N:"), l.text || el("i", {}, "(nothing)"), gradeTags(e.grade)));
      }
      if (!e.turn.lines.some(l => l.kind === "reply")) out.push(el("p", { class: "model" }, el("span", { class: "who" }, "N:"), el("i", {}, e.turn.ending === "actions" ? "(the action limit ended the turn)" : "(the turn ended)"), gradeTags(e.grade)));
    } else if (i > lastModel) {
      out.push(el("p", { class: "game" }, `${e.who === "person" ? "You" : "Opponent"}: ${e.text}`));
    }
  });
  if (m.finished) out.push(el("p", { class: "game" }, el("b", {}, outcomeText(m))));
  return out;
}
function renderPlay() {
  const m = S.match;
  $("playArea").hidden = !m;
  if (!m) return;
  $("board").replaceChildren(renderBoard(m));
  $("turns").replaceChildren(...renderTurns(m));
  $("turns").scrollTop = $("turns").scrollHeight;
  $("playStatus").textContent = m.finished ? outcomeText(m)
    : S.busy ? (m.modelToMove ? "The model is thinking…" : "…")
    : m.needsPerson ? "Your move." : m.modelToMove ? "The model's turn." : "The opponent's turn.";
  const can = !m.finished && !m.needsPerson && !S.busy;
  $("stepTurn").disabled = !can || (m.modelToMove && !S.model);
  $("playOut").disabled = !can || !S.model;
  $("playNote").textContent = !S.model && m.modelToMove ? "Open a model to let it play." : "";
}

// ---- 1.3 Lessons --------------------------------------------------------------------------------
function lessonArgs() {
  return { game: $("lessonGame").value, count: Number($("lessonCount").value) || 600, seed: Number($("lessonSeed").value) || 0 };
}
async function previewLessons() {
  const a = lessonArgs();
  $("lessonSummary").replaceChildren(el("p", { class: "note" }, "Writing…"));
  const set = await act("lessons_preview", a);
  if (!set) { $("lessonSummary").replaceChildren(); return; }
  const g = gameInfo(a.game);
  $("lessonSummary").replaceChildren(
    el("table", { class: "figures" },
      el("tr", {}, el("th", {}, "Sessions"), el("td", {}, fmt(set.sessions) + (set.sessions < a.count ? `  (${g.name} has no more new turns to teach)` : ""))),
      el("tr", {}, el("th", {}, "Held-out turns left out"), el("td", {}, fmt(set.heldOutSkipped))),
      el("tr", {}, el("th", {}, "Games played to find them"), el("td", {}, fmt(set.games)))),
    el("p", { class: "note" }, "The first sessions:"),
    el("div", { class: "sessions" }, set.sample.join("\n")));
}
async function saveLessons() {
  const path = await act("save_lessons", lessonArgs());
  if (path) toast(`Saved ${path}`);
}

// ---- 1.4 Scoreboard -----------------------------------------------------------------------------
function renderScores() {
  $("measure").disabled = !S.model || S.measuring;
  $("stopMeasure").hidden = !S.measuring;
  $("measureMeter").hidden = !S.measuring;
  if (!S.measuring) $("measureNote").textContent = S.model ? "" : "Open a model to measure it.";
  const rows = [...S.reports].reverse();
  $("scoreTable").tBodies[0].replaceChildren(...(rows.length ? rows.map(r => el("tr", {},
    el("td", {}, when(r.time)),
    el("td", {}, gameInfo(r.game)?.name || r.game),
    el("td", { class: "num" }, fmt(r.step)),
    el("td", { class: "result-cell" }, ...resultText(r).map(t => el("span", {}, t))),
    el("td", { class: "num" }, `${pct(r.moves.legal, r.moves.turns)} of ${fmt(r.moves.turns)}`),
    el("td", { class: "num" }, pct(r.moves.best, r.moves.turns)),
    el("td", { class: "num" }, r.heldOut.turns ? `${pct(r.heldOut.best, r.heldOut.turns)} of ${fmt(r.heldOut.turns)}` : "—")))
    : [el("tr", {}, el("td", { colspan: 7, class: "empty" }, S.model ? `${S.model.name} hasn't been measured yet.` : "Open a model to see its scores."))]));
}
async function measure() {
  const games = Math.max(2, Math.min(200, Number($("scoreGames").value) || 20));
  const ok = await act("measure", { game: $("scoreGame").value, games });
  if (ok === undefined) return;
  S.measuring = true;
  $("measureBar").style.width = "0%";
  $("measureNote").textContent = "Starting…";
  $("sideRun").hidden = false;
  renderScores();
}
function measured() {
  S.measuring = false;
  $("sideRun").hidden = true;
}

// ---- Real games: profiles -----------------------------------------------------------------------
// The game's window, the controls a network may use, what it sees and the safety limits. Input and
// pictures come from the backend (src-tauri/src/live), through its safety layer.
const L = { info: null, profiles: [], windows: [], editing: null, watching: false, running: false, before: null, profile: null };
const RANGE_ID = "yendory-range";
const NEW = "";

function blankProfile() {
  return {
    id: "", name: "", windowTitle: "", windowProcess: "", keys: ["w", "a", "s", "d"], buttons: ["left"], mouse: true,
    mouseScale: 1, capture: { width: 128, height: 72, gray: false, fps: 20 },
    limits: { maxHoldMs: 5000, maxActionsPerSecond: 60, maxMouseStep: 400 }, storageMb: 4096, notes: "",
  };
}

async function loadLive() {
  L.info = (await act("live_info")) || L.info;
  L.profiles = (await act("live_profiles")) || [];
}

function profileOptions(select, chosen, withNew) {
  const opts = L.profiles.map(p => el("option", { value: p.id }, p.name));
  if (withNew) opts.push(el("option", { value: NEW, "data-note": "Choose its window below" }, "New game…"));
  select.replaceChildren(...opts);
  select.value = L.profiles.some(p => p.id === chosen) ? chosen : withNew ? NEW : L.profiles[0]?.id ?? "";
}

function renderStopLine() {
  const i = L.info;
  if (!i) return;
  const line = $("stopLine");
  line.replaceChildren(el("b", {}, "Emergency stop: "), el("span", { class: "mono" }, i.stopKeys),
    i.stopKeysProblem ? ` is not available (${i.stopKeysProblem}); use the Stop button in the page bar.` : " on the keyboard, in any program, or the Stop button: every control is released and everything running ends.");
  line.className = "stop-line" + (i.stopKeysProblem ? " warn" : "");
}

function renderProfiles() {
  if (!L.info) return;
  renderStopLine();
  if (!L.editing) L.editing = L.profiles[0] ? structuredClone(L.profiles[0]) : blankProfile();
  profileOptions($("profilePick"), L.editing.id, true);
  $("profileDelete").disabled = !L.editing.id;
  if (!$("pfKeys").childElementCount) buildKeyChips();
  fillForm(L.editing);
  renderWindowPick();
}

function buildKeyChips() {
  const groups = {};
  for (const k of L.info.keys) (groups[k.group] ||= []).push(k);
  $("pfKeys").replaceChildren(...Object.entries(groups).map(([g, keys]) => el("div", { class: "key-group" },
    el("span", { class: "field-label" }, g),
    el("div", { class: "key-chips" }, keys.map(k => el("label", { class: "key-chip", title: k.name },
      el("input", { type: "checkbox", "data-key": k.name }), el("span", {}, k.label)))))));
}

function fillForm(p) {
  $("pfName").value = p.name;
  $("pfTitle").value = p.windowTitle;
  $("pfProcess").value = p.windowProcess;
  for (const c of document.querySelectorAll("#pfKeys input[data-key]")) c.checked = p.keys.includes(c.dataset.key);
  for (const c of document.querySelectorAll("#profileForm input[data-button]")) c.checked = p.buttons.includes(c.dataset.button);
  $("pfMouse").checked = p.mouse;
  $("pfScale").value = p.mouseScale;
  $("pfWidth").value = p.capture.width;
  $("pfHeight").value = p.capture.height;
  for (const r of document.querySelectorAll("input[name=pfColour]")) r.checked = (r.value === "grey") === p.capture.gray;
  $("pfFps").value = p.capture.fps;
  $("pfHold").value = p.limits.maxHoldMs;
  $("pfRate").value = p.limits.maxActionsPerSecond;
  $("pfStep").value = p.limits.maxMouseStep;
  $("pfSaved").textContent = "";
  sizePreview();
}

function readForm() {
  const p = structuredClone(L.editing);
  const num = id => Number($(id).value);
  p.name = $("pfName").value.trim();
  p.windowTitle = $("pfTitle").value.trim();
  p.windowProcess = $("pfProcess").value.trim();
  p.keys = [...document.querySelectorAll("#pfKeys input[data-key]:checked")].map(c => c.dataset.key);
  p.buttons = [...document.querySelectorAll("#profileForm input[data-button]:checked")].map(c => c.dataset.button);
  p.mouse = $("pfMouse").checked;
  p.mouseScale = num("pfScale");
  p.capture = { width: num("pfWidth"), height: num("pfHeight"), gray: document.querySelector("input[name=pfColour]:checked")?.value === "grey", fps: num("pfFps") };
  p.limits = { maxHoldMs: num("pfHold"), maxActionsPerSecond: num("pfRate"), maxMouseStep: num("pfStep") };
  return p;
}

async function refreshWindows() {
  L.windows = (await act("live_windows")) || [];
  renderWindowPick();
}

/** The open window that matches the form's title and program (the first one). */
function formWindow() {
  const title = $("pfTitle").value.trim().toLowerCase(), proc = $("pfProcess").value.trim().toLowerCase();
  if (!title && !proc) return null;
  return L.windows.find(w => (!title || w.title.toLowerCase().includes(title)) && (!proc || w.process.toLowerCase() === proc)) || null;
}

function renderWindowPick() {
  const pick = $("pfWindowPick");
  const match = formWindow();
  pick.replaceChildren(el("option", { value: "" }, L.windows.length ? "Choose an open window…" : "No windows found"),
    ...L.windows.map(w => el("option", { value: String(w.id), "data-note": `${w.process || "?"} · ${w.width}×${w.height}` }, w.title)));
  pick.value = match ? String(match.id) : "";
  $("previewNote").textContent = match ? `${match.title} (${match.process || "program unknown"}), ${match.width}×${match.height}` : "The game's window isn't open (or the title and program don't match it).";
  $("bringFront").disabled = !match;
  $("previewToggle").disabled = !match;
}

function pickWindow() {
  const w = L.windows.find(x => String(x.id) === $("pfWindowPick").value);
  if (!w) return;
  $("pfTitle").value = w.title;
  $("pfProcess").value = w.process;
  if (!$("pfName").value.trim()) $("pfName").value = w.title.slice(0, 60);
  renderWindowPick();
}

async function saveProfile(e) {
  e.preventDefault();
  const saved = await act("live_save_profile", { profile: readForm() });
  if (!saved) return;
  L.editing = saved;
  await loadLive();
  renderProfiles();
  $("pfSaved").textContent = "Saved.";
}

async function deleteProfile() {
  if (!L.editing.id || !confirm(`Delete the profile “${L.editing.name}”? Its recordings and players (later) are kept.`)) return;
  if ((await act("live_delete_profile", { id: L.editing.id })) === undefined) return;
  L.editing = null;
  await loadLive();
  renderProfiles();
}

async function startRange() {
  const r = await act("range_start", {});
  if (!r) return;
  await loadLive();
  L.editing = structuredClone(L.profiles.find(p => p.id === RANGE_ID) || blankProfile());
  for (let i = 0; i < 40; i++) {
    await refreshWindows();
    if (formWindow() || (fillForm(L.editing), formWindow())) break;
    await new Promise(res => setTimeout(res, 250));
  }
  renderProfiles();
  toast(r.already ? "The test range is already open." : "The test range is open: W A S D, Shift, the mouse and the left button.");
}

// What the network sees: the window's picture at the profile's size and rate.
function sizePreview() {
  const c = $("preview"), w = Number($("pfWidth").value) || 128, h = Number($("pfHeight").value) || 72;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
}
function renderWatch() {
  $("previewToggle").textContent = L.watching ? "Stop watching" : "Watch";
}
async function watchLoop() {
  let frames = 0, since = performance.now(), last = 0;
  while (L.watching && S.route === "profiles") {
    const w = formWindow();
    if (!w) { L.watching = false; break; }
    const fps = Math.max(1, Math.min(60, Number($("pfFps").value) || 10));
    const gray = document.querySelector("input[name=pfColour]:checked")?.value === "grey";
    sizePreview();
    const t0 = performance.now();
    try {
      const bytes = asBytes(await invoke("live_preview", { window: w.id, width: $("preview").width, height: $("preview").height, gray }));
      drawPreview(bytes);
      last = bytes[5];
    } catch (e) { $("previewHelp").textContent = String(e); L.watching = false; break; }
    frames++;
    const now = performance.now();
    if (now - since >= 1000) {
      $("previewHelp").textContent = `${trim1((frames * 1000) / (now - since))} pictures a second of ${fps} asked for; a picture takes ${last} ms to capture.`;
      frames = 0; since = now;
    }
    await new Promise(res => setTimeout(res, Math.max(0, 1000 / fps - (performance.now() - t0))));
  }
  renderWatch();
}
// Raw bytes from the backend arrive as an ArrayBuffer (or, over some IPC paths, as an array).
const asBytes = b => (b instanceof ArrayBuffer ? new Uint8Array(b) : b instanceof Uint8Array ? b : Uint8Array.from(b));
function drawPreview(bytes) {
  const w = bytes[0] | (bytes[1] << 8), h = bytes[2] | (bytes[3] << 8), ch = bytes[4];
  const px = bytes.subarray(8);
  if (px.length < w * h * ch) throw new Error("The picture arrived incomplete");
  const c = $("preview");
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const img = new ImageData(w, h);
  for (let i = 0, j = 0; i < w * h; i++, j += ch) {
    const r = px[j], g = ch === 3 ? px[j + 1] : r, b = ch === 3 ? px[j + 2] : r;
    img.data.set([r, g, b, 255], i * 4);
  }
  c.getContext("2d").putImageData(img, 0, 0);
}
function toggleWatch() {
  L.watching = !L.watching;
  renderWatch();
  if (L.watching) {
    $("previewHelp").textContent = L.info?.platform === "windows" ? "Tools copies what the screen shows where the game's window is, so keep the game visible (borderless or windowed; exclusive full screen may show black)." : "";
    watchLoop();
  }
}

// ---- Real games: controls check -----------------------------------------------------------------
let checkWindow = null;
async function renderCheck() {
  if (!L.info) return;
  profileOptions($("checkProfile"), L.profile ?? L.editing?.id, false);
  const id = $("checkProfile").value;
  L.profile = id || null;
  $("checkRun").disabled = !id || L.running;
  if (!id) { $("checkWindow").textContent = "Make a game profile first (2.1)."; return; }
  checkWindow = await act("live_find_window", { id });
  $("checkWindow").textContent = checkWindow ? `Window: ${checkWindow.title} (${checkWindow.process || "?"})` : "The game's window isn't open.";
  $("checkRun").disabled = !checkWindow || L.running;
}

const rangeNum = (s, k) => Number(s?.[k] ?? 0);
async function runCheck() {
  const id = $("checkProfile").value;
  if (!checkWindow) return;
  $("checkLog").tBodies[0].replaceChildren();
  $("rangeReport").hidden = true;
  L.before = id === RANGE_ID ? await invoke("range_state").catch(() => null) : null;
  const ok = await act("live_run", { id, window: checkWindow.id, actions: $("checkActions").value });
  if (ok === undefined) return;
  L.running = true;
  $("checkStatus").textContent = "Running…";
  $("liveStop").hidden = false;
  $("checkRun").disabled = true;
}

function logRows(entries) {
  const body = $("checkLog").tBodies[0];
  for (const e of entries) {
    body.append(el("tr", { class: e.sent ? "" : "refused" }, el("td", { class: "num" }, fmt(e.ms)), el("td", { class: "mono" }, e.action),
      el("td", {}, e.sent ? (e.note ? `sent (${e.note})` : "sent") : e.note)));
  }
}

async function checkDone(stopped) {
  L.running = false;
  $("liveStop").hidden = true;
  $("checkStatus").textContent = stopped ? "Stopped: everything was released." : "Done: everything was released.";
  $("checkRun").disabled = !checkWindow;
  if (!L.before) return;
  await new Promise(res => setTimeout(res, 150));
  const after = await invoke("range_state").catch(() => null);
  if (!after) return;
  const b = L.before, moved = Math.hypot(rangeNum(after, "x") - rangeNum(b, "x"), rangeNum(after, "y") - rangeNum(b, "y"));
  let turn = ((rangeNum(after, "angle") - rangeNum(b, "angle")) * 180) / Math.PI;
  turn = ((turn + 540) % 360) - 180;
  const r = $("rangeReport");
  r.hidden = false;
  r.replaceChildren(el("h3", {}, "What the range received"), el("table", { class: "figures" },
    el("tr", {}, el("th", {}, "Presses (keys and buttons)"), el("td", {}, fmt(rangeNum(after, "presses") - rangeNum(b, "presses")))),
    el("tr", {}, el("th", {}, "Mouse movement"), el("td", {}, `${fmt(rangeNum(after, "mouseX") - rangeNum(b, "mouseX"))} across, ${fmt(rangeNum(after, "mouseY") - rangeNum(b, "mouseY"))} down`)),
    el("tr", {}, el("th", {}, "Turned"), el("td", {}, `${trim1(turn)}°`)),
    el("tr", {}, el("th", {}, "Moved"), el("td", {}, `${trim1(moved)} cells`)),
    el("tr", {}, el("th", {}, "Shots"), el("td", {}, `${fmt(rangeNum(after, "shots") - rangeNum(b, "shots"))} (${fmt(rangeNum(after, "hits") - rangeNum(b, "hits"))} hit)`)),
    el("tr", {}, el("th", {}, "Still held in the game"), el("td", {}, Object.entries(after.held || {}).filter(([, v]) => v).map(([k]) => k).join(", ") || "nothing"))));
}

// ---- Settings -----------------------------------------------------------------------------------
function renderSettings() {
  const a = S.about;
  if (!a) return;
  $("networkStatus").textContent = a.network ? `Yendory Network ${a.network}, Engine ${a.engine}: connected.` : `Not connected: ${a.networkError}`;
  $("networkStatus").className = "network-status " + (a.network ? "ok" : "warn");
  $("modelsFolder").textContent = a.modelsFolder;
  $("dataFolder").textContent = a.dataFolder;
  $("versions").textContent = `Tools ${a.tools}` + (a.network ? ` · Network ${a.network} · Engine ${a.engine}` : "");
}

// ---- Start --------------------------------------------------------------------------------------
async function handed(h) {
  if (!h) return;
  if (h.problem) notice(h.problem, true);
  else if (h.id) { notice(""); await openModel(h.id); }
}

function wire() {
  for (const b of document.querySelectorAll("[data-route]")) b.addEventListener("click", () => show(b.dataset.route));
  $("sidebarToggle").addEventListener("click", () => {
    const c = document.body.classList.toggle("sidebar-collapsed");
    $("sidebarToggle").setAttribute("aria-expanded", String(!c));
    store.set("collapsed", c ? "1" : "0");
  });
  $("openModel").addEventListener("click", chooseModel);
  $("playGame").addEventListener("change", syncFirst);
  $("newGame").addEventListener("click", newGame);
  $("stepTurn").addEventListener("click", step);
  $("playOut").addEventListener("click", playOut);
  $("previewLessons").addEventListener("click", previewLessons);
  $("saveLessons").addEventListener("click", saveLessons);
  $("measure").addEventListener("click", measure);
  $("stopMeasure").addEventListener("click", () => invoke("stop_measure"));
  listen("measure-progress", e => {
    const { done, total } = e.payload;
    const w = `${(100 * done) / total}%`;
    $("measureBar").style.width = w;
    $("sideRunBar").style.width = w;
    $("measureNote").textContent = `${done} of ${total} games played`;
  });
  listen("measure-done", async () => { measured(); await loadReports(); $("measureNote").textContent = "Done."; render(); });
  listen("measure-failed", e => { measured(); $("measureNote").textContent = String(e.payload); render(); });
  listen("model-handed", e => handed(e.payload));
  $("profilePick").addEventListener("change", () => {
    const id = $("profilePick").value;
    L.editing = id ? structuredClone(L.profiles.find(p => p.id === id)) : blankProfile();
    L.watching = false;
    renderProfiles();
  });
  $("profileForm").addEventListener("submit", saveProfile);
  $("profileDelete").addEventListener("click", deleteProfile);
  $("rangeStart").addEventListener("click", startRange);
  $("pfWindowsRefresh").addEventListener("click", refreshWindows);
  $("pfWindowPick").addEventListener("change", pickWindow);
  for (const id of ["pfTitle", "pfProcess"]) $(id).addEventListener("input", renderWindowPick);
  for (const id of ["pfWidth", "pfHeight"]) $(id).addEventListener("change", sizePreview);
  $("previewToggle").addEventListener("click", toggleWatch);
  $("bringFront").addEventListener("click", () => { const w = formWindow(); if (w) act("live_front", { window: w.id }); });
  $("checkProfile").addEventListener("change", () => { L.profile = $("checkProfile").value; renderCheck(); });
  for (const b of document.querySelectorAll("#presets button")) b.addEventListener("click", () => ($("checkActions").value = b.dataset.actions));
  $("checkRun").addEventListener("click", runCheck);
  $("liveStop").addEventListener("click", () => invoke("live_stop"));
  listen("live-log", e => logRows(e.payload));
  listen("live-done", e => checkDone(e.payload.stopped));
  listen("live-stopped", () => toast("Emergency stop: every control was released."));
}

async function start() {
  if (store.get("collapsed", "0") === "1") document.body.classList.add("sidebar-collapsed");
  wire();
  await loadLive();
  if (S.route === "profiles") await refreshWindows();
  S.games = (await act("games")) || [];
  for (const id of ["playGame", "lessonGame", "scoreGame"]) $(id).replaceChildren(...S.games.map(g => el("option", { value: g.id }, g.name)));
  syncFirst();
  show(S.route);
  S.about = await act("about");
  if (S.about && !S.about.network) notice(`The Yendory Network isn't answering, so no model can play: ${S.about.networkError}. Yendory Hub installs it.`, true);
  S.model = await act("current_model");
  await handed(await act("handed_model"));
  await loadReports();
  render();
}
start();
