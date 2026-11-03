"use strict";
// Yendory Tools window. The games, matches, lessons and scoreboard are computed by the backend
// (src-tauri); every line the model writes comes from the Yendory Network.

const TAURI = window.__TAURI__;

// ---- Debug log ----------------------------------------------------------------------------------
// Every command the page sends (and how it ended), the backend's own lines (Network calls, failures,
// panics: src-tauri/src/debug.rs), events that end something, and errors in the page itself.
const D = { lines: [], open: false, unseen: 0 };
const DEBUG_KEEP = 2000;
// Asked many times a second (pictures, turns): logged only when they fail or are slow.
const QUIET = new Set(["live_preview", "recording_frame", "match_step", "range_state", "debug_lines"]);
const PROBLEMS = new Set(["warn", "error", "wait"]);
const brief = v => {
  let text;
  try { text = JSON.stringify(v); } catch { text = String(v); }
  return text && text.length > 300 ? text.slice(0, 300) + "…" : text;
};
const secs = ms => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const clockOf = t => {
  const d = new Date(t);
  const two = n => String(n).padStart(2, "0");
  return `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}.${String(d.getMilliseconds()).padStart(3, "0")}`;
};

function dbg(level, from, text, t = Date.now()) {
  const line = { t, level, from, text: String(text) };
  D.lines.push(line);
  if (D.lines.length > DEBUG_KEEP) D.lines.splice(0, D.lines.length - DEBUG_KEEP);
  if (PROBLEMS.has(level) && !D.open) D.unseen++;
  debugAppend(line);
  debugBadge();
}

function debugLine(l) {
  const li = document.createElement("li");
  li.className = l.level;
  const when = document.createElement("span");
  when.className = "when";
  when.textContent = clockOf(l.t);
  const from = document.createElement("span");
  from.className = "from";
  from.textContent = l.from === "tools" ? "backend" : "page   ";
  li.append(when, from, l.text);
  return li;
}

function debugShown(l) {
  return !document.getElementById("debugProblems")?.checked || PROBLEMS.has(l.level);
}

function debugAppend(l) {
  const list = document.getElementById("debugLines");
  if (!list || !D.open || !debugShown(l)) return;
  const body = document.getElementById("debugBody");
  const atEnd = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
  list.querySelector(".empty")?.remove();
  list.append(debugLine(l));
  while (list.children.length > DEBUG_KEEP) list.firstChild.remove();
  if (atEnd) body.scrollTop = body.scrollHeight;
}

function debugRender() {
  const list = document.getElementById("debugLines");
  if (!list) return;
  const shown = D.lines.filter(debugShown);
  list.replaceChildren(...(shown.length ? shown.map(debugLine) : [Object.assign(document.createElement("li"), { className: "empty", textContent: "Nothing logged yet." })]));
  const body = document.getElementById("debugBody");
  body.scrollTop = body.scrollHeight;
}

function debugBadge() {
  const c = document.getElementById("debugCount");
  if (!c) return;
  c.hidden = D.open || !D.unseen;
  c.textContent = String(D.unseen);
  c.title = `${D.unseen} problem${D.unseen === 1 ? "" : "s"} since you last looked`;
}

function debugSetOpen(open) {
  D.open = open;
  if (open) D.unseen = 0;
  store.set("debugOpen", open ? "1" : "0");
  document.getElementById("debugBody").hidden = !open;
  document.getElementById("debugTools").hidden = !open;
  document.getElementById("debugToggle").setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("debug-open", open);
  if (open) debugRender();
  debugBadge();
}

function debugText() {
  return D.lines.map(l => `${clockOf(l.t)} ${l.from === "tools" ? "backend" : "page   "} ${l.level.padEnd(5)} ${l.text}`).join("\n");
}

async function debugCopy() {
  const text = debugText();
  try { await navigator.clipboard.writeText(text); }
  catch {
    const area = Object.assign(document.createElement("textarea"), { value: text });
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast(`Copied ${D.lines.length} lines of the debug log.`);
}

async function debugStart() {
  $("debugToggle").addEventListener("click", () => debugSetOpen(!D.open));
  $("debugCopy").addEventListener("click", debugCopy);
  $("debugClear").addEventListener("click", () => { D.lines = []; D.unseen = 0; debugRender(); debugBadge(); });
  $("debugProblems").addEventListener("change", debugRender);
  document.addEventListener("keydown", e => { if (e.ctrlKey && e.key === "`") { e.preventDefault(); debugSetOpen(!D.open); } });
  // The backend's lines: from now on as they happen, and those from before the page opened.
  const seen = new Set();
  const fromTools = l => {
    const key = `${l.t}|${l.text}`;
    if (seen.has(key)) return;
    seen.add(key);
    dbg(l.level, "tools", l.text, l.t);
  };
  await TAURI.event.listen("debug-log", e => fromTools(e.payload));
  try {
    const before = await TAURI.core.invoke("debug_lines");
    const earlier = D.lines;
    D.lines = [];
    for (const l of before) fromTools(l);
    D.lines.push(...earlier.filter(l => l.from !== "tools"));
    D.lines.sort((a, b) => a.t - b.t);
  } catch (e) { dbg("error", "page", `The backend's log could not be read: ${e}`); }
  debugSetOpen(store.get("debugOpen", "0") === "1");
}

window.addEventListener("error", e => dbg("error", "page", `${e.message}${e.filename ? ` (${e.filename.split("/").pop()}:${e.lineno})` : ""}`));
window.addEventListener("unhandledrejection", e => dbg("error", "page", `Unhandled: ${e.reason?.stack || e.reason}`));
{
  const original = console.error.bind(console);
  console.error = (...args) => { dbg("error", "page", args.map(a => (a instanceof Error ? a.stack || a.message : typeof a === "string" ? a : brief(a))).join(" ")); original(...args); };
}

let callNo = 0;
// Every command, logged: what was asked, how long it took, how it ended, and a note when an
// answer is slow to come (a command that never answers shows as asked and waiting).
const invoke = async (cmd, args) => {
  const n = ++callNo;
  const t0 = performance.now();
  const quiet = QUIET.has(cmd);
  if (!quiet) dbg("info", "page", `→ ${cmd} #${n}${args && Object.keys(args).length ? " " + brief(args) : ""}`);
  const waiting = setTimeout(() => dbg("wait", "page", `… ${cmd} #${n} has not answered after 10 s (still waiting)`), 10000);
  try {
    const result = await TAURI.core.invoke(cmd, args);
    const ms = performance.now() - t0;
    if (!quiet || ms > 3000) dbg("info", "page", `← ${cmd} #${n} answered in ${secs(ms)}`);
    return result;
  } catch (e) {
    dbg("error", "page", `← ${cmd} #${n} failed after ${secs(performance.now() - t0)}: ${e}`);
    throw e;
  } finally {
    clearTimeout(waiting);
  }
};
// Events that end something are logged (progress and status are too frequent).
const listen = (event, f) => TAURI.event.listen(event, e => {
  if (/-(failed|done)$|^live-stopped$|^model-handed$/.test(event)) {
    dbg(event.endsWith("-failed") ? "error" : "info", "page", `event ${event}${e.payload != null ? " " + brief(e.payload) : ""}`);
  }
  return f(e);
});
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
  record: ["Real games", "Record"], recordings: ["Real games", "Recordings"],
  train: ["Real games", "Train"], autoplay: ["Real games", "Play"], checks: ["Real games", "Checks"],
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
  if (S.route === "record") renderRecord();
  if (S.route === "recordings") renderRecordings();
  if (S.route !== "recordings") stopReplay();
  if (S.route === "train") renderTrain();
  if (S.route === "autoplay") renderAutoplay();
  if (S.route === "checks") renderChecks();
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
      el("small", {}, `Nova ${m.architecture === "neuro" ? "Neuro" : "Bicameral"} · ${short(m.parameters)} · step ${fmt(m.step)}`)));
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
  T.selected = null;
  T.history = [];
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
    limits: { maxHoldMs: 5000, maxActionsPerSecond: 60, maxMouseStep: 400 }, notes: "",
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
  if (!L.editing.id) return;
  // Tools asks in its own dialog.
  if (!(await act("live_delete_profile", { id: L.editing.id }))) return;
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
function drawPreview(bytes) { drawPreviewTo($("preview"), bytes); }
function drawPreviewTo(c, bytes) {
  const w = bytes[0] | (bytes[1] << 8), h = bytes[2] | (bytes[3] << 8), ch = bytes[4];
  const px = bytes.subarray(8);
  if (px.length < w * h * ch) throw new Error("The picture arrived incomplete");
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

// ---- Real games: recording ----------------------------------------------------------------------
// The person plays; the backend records pictures and controls in step (src-tauri/src/live/record.rs).
const R = { recording: false, profile: null, list: [], view: null, index: 0, playing: false, fetching: false, playFrom: 0, playAt: 0 };
const mb = b => `${trim1(b / 1048576)} MB`;
const clock = ms => { const s = Math.floor(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const keyLabel = name => L.info?.keys.find(k => k.name === name)?.label ?? name;
const chips = (names, on) => names.map(n => el("span", { class: "chip" + (on.includes(n) ? " on" : "") }, keyLabel(n)));

async function renderRecord() {
  if (!L.info) return;
  profileOptions($("recProfile"), R.profile ?? L.editing?.id, false);
  const id = $("recProfile").value;
  R.profile = id || null;
  if (!id) { $("recWindow").textContent = "Make a game profile first (2.1)."; $("recStart").disabled = true; return; }
  const [w, list] = await Promise.all([act("live_find_window", { id }), act("recordings", { id })]);
  $("recWindow").textContent = w ? `Window: ${w.title} (${w.process || "?"})` : "The game's window isn't open.";
  if (list) {
    $("recStorage").textContent = `Recordings of this game so far: ${mb(list.usedBytes)}. A recording has no length limit: it runs until you stop it (or the disk is full).`;
  }
  $("recStart").disabled = !w || R.recording || L.running;
  $("recStart").hidden = R.recording;
  $("recStopBtn").hidden = !R.recording;
}

async function startRecording() {
  const split = document.querySelector("input[name=recSplit]:checked")?.value || "train";
  const ok = await act("record_start", { id: $("recProfile").value, split, notes: $("recNotes").value.trim() });
  if (ok === undefined) return;
  R.recording = true;
  $("recDone").hidden = true;
  $("recLive").hidden = false;
  $("recState").textContent = "Starting…";
  for (const id of ["recTime", "recTicks", "recFps", "recBytes"]) $(id).textContent = "—";
  $("recHeld").replaceChildren();
  renderRecord();
}

function recordStatus(st) {
  if (!R.recording) return;
  $("recLive").classList.toggle("paused", !st.inFront);
  $("recState").textContent = st.inFront ? "Recording" : "Paused: the game is not in front";
  $("recTime").textContent = clock(st.ms);
  $("recTicks").textContent = fmt(st.ticks);
  $("recFps").textContent = st.inFront ? trim1(st.fps) : "—";
  $("recBytes").textContent = mb(st.bytes);
  const p = L.profiles.find(x => x.id === R.profile);
  if (p) $("recHeld").replaceChildren(...chips([...p.keys, ...p.buttons], st.held));
}

function recordEnded(info, error) {
  R.recording = false;
  $("recLive").hidden = true;
  const d = $("recDone");
  d.hidden = false;
  if (error) {
    d.replaceChildren(el("p", { class: "warn" }, `The recording failed: ${error}`));
  } else {
    d.replaceChildren(el("h3", {}, "Recorded"),
      el("p", {}, `${clock(info.durationMs - info.pausedMs)} of play, ${fmt(info.ticks)} pictures, ${mb(info.bytes)}. It ${info.ended === "stopped" ? "was stopped" : info.ended.startsWith("ended") ? info.ended : `ended because ${info.ended}`}.` +
        (info.droppedTicks ? ` ${fmt(info.droppedTicks)} pictures were missed (the computer was busy).` : "")),
      el("button", { type: "button", onclick: () => { R.profile = info.profile; show("recordings"); openRecording(info.id); } }, "Watch it"));
  }
  if (S.route === "record") renderRecord();
  if (S.route === "recordings") renderRecordings();
}

async function renderRecordings() {
  if (!L.info) return;
  profileOptions($("recsProfile"), R.profile ?? L.editing?.id, false);
  const id = $("recsProfile").value;
  R.profile = id || null;
  const body = $("recsTable").tBodies[0];
  if (!id) { body.replaceChildren(el("tr", {}, el("td", { colspan: 8, class: "empty" }, "Make a game profile first (2.1)."))); return; }
  const list = await act("recordings", { id });
  if (!list) return;
  R.list = list.recordings;
  $("recsUsage").textContent = `${R.list.length} recording${R.list.length === 1 ? "" : "s"}, ${mb(list.usedBytes)}`;
  body.replaceChildren(...(R.list.length ? R.list.map(r => el("tr", { class: R.view?.info.id === r.id ? "open-row" : "" },
    el("td", {}, when(r.started)),
    el("td", {}, r.player === "person" ? "you" : r.player),
    el("td", {}, r.split === "test" ? "testing" : "training"),
    el("td", { class: "num" }, clock(r.durationMs - r.pausedMs)),
    el("td", { class: "num" }, fmt(r.ticks)),
    el("td", { class: "num" }, mb(r.bytes)),
    el("td", {}, r.notes + (r.complete ? "" : ` (${r.ended})`)),
    el("td", {}, el("button", { type: "button", onclick: () => openRecording(r.id) }, "Open"))))
    : [el("tr", {}, el("td", { colspan: 8, class: "empty" }, "No recordings of this game yet (2.3 Record)."))]));
}

async function openRecording(rec) {
  stopReplay();
  const t = await act("recording_ticks", { id: R.profile, recording: rec });
  if (!t) return;
  R.view = t;
  R.index = 0;
  $("viewer").hidden = false;
  $("validation").replaceChildren();
  $("viewerTitle").textContent = `${t.info.game}, ${when(t.info.started)}`;
  $("replayPos").max = String(Math.max(0, t.ms.length - 1));
  for (const r of document.querySelectorAll("input[name=viewSplit]")) r.checked = r.value === t.info.split;
  $("viewNotes").value = t.info.notes;
  const c = $("replay");
  c.width = t.info.width; c.height = t.info.height;
  await renderRecordings();
  await showTick(0);
  $("viewer").scrollIntoView({ behavior: "smooth", block: "start" });
}

function tickControls(i) {
  const v = R.view, on = [];
  v.info.keys.forEach((k, b) => { if ((b < 32 ? v.keysLo[i] >>> b : v.keysHi[i] >>> (b - 32)) & 1) on.push(k); });
  v.info.buttons.forEach((k, b) => { if ((v.buttons[i] >> b) & 1) on.push(k); });
  return on;
}

async function showTick(i) {
  const v = R.view;
  if (!v || !v.ms.length) return;
  R.index = Math.max(0, Math.min(v.ms.length - 1, i));
  $("replayPos").value = String(R.index);
  const first = v.ms[0], ms = v.ms[R.index];
  $("replayTime").textContent = `${clock(ms - first)} · picture ${fmt(R.index + 1)} of ${fmt(v.ms.length)}`;
  $("replayControls").replaceChildren(...chips([...v.info.keys, ...v.info.buttons], tickControls(R.index)));
  $("replayMouse").textContent = v.info.mouse ? `Mouse: ${trim1(v.dx[R.index])} across, ${trim1(v.dy[R.index])} down` + (v.gap[R.index] ? " · starts after a pause" : "") : "";
  drawTimeline();
  if (R.fetching) return;
  R.fetching = true;
  try {
    const bytes = asBytes(await invoke("recording_frame", { id: R.profile, recording: v.info.id, index: R.index }));
    drawPreviewTo($("replay"), bytes);
  } catch (e) { toast(String(e), true); } finally { R.fetching = false; }
}

function drawTimeline() {
  const v = R.view, c = $("timeline");
  if (!v) return;
  const names = [...v.info.keys, ...v.info.buttons];
  const rows = names.length + (v.info.mouse ? 1 : 0);
  const w = c.clientWidth || 600, rowH = 12, h = Math.max(24, rows * rowH + 4);
  if (c.width !== w) c.width = w;
  if (c.height !== h) c.height = h;
  const g = c.getContext("2d"), n = v.ms.length, css = getComputedStyle(document.body);
  g.clearRect(0, 0, w, h);
  g.font = "10px monospace";
  const accent = css.getPropertyValue("--accent").trim() || "#284a7e";
  const soft = css.getPropertyValue("--ink-soft").trim() || "#666";
  const label = 46, plot = w - label;
  names.forEach((name, r) => {
    g.fillStyle = soft; g.fillText(keyLabel(name).slice(0, 7), 2, r * rowH + 10);
    g.fillStyle = accent;
    let start = -1;
    for (let i = 0; i <= n; i++) {
      const on = i < n && tickControls(i).includes(name);
      if (on && start < 0) start = i;
      if (!on && start >= 0) { g.fillRect(label + (start / n) * plot, r * rowH + 2, Math.max(1, ((i - start) / n) * plot), rowH - 4); start = -1; }
    }
  });
  if (v.info.mouse) {
    const r = names.length, mid = r * rowH + rowH / 2;
    g.fillStyle = soft; g.fillText("mouse", 2, r * rowH + 10);
    const most = Math.max(1, ...v.dx.map(Math.abs));
    g.fillStyle = accent;
    for (let i = 0; i < n; i++) {
      const hgt = (v.dx[i] / most) * (rowH / 2 - 1);
      if (hgt) g.fillRect(label + (i / n) * plot, Math.min(mid, mid - hgt), Math.max(1, plot / n), Math.abs(hgt));
    }
  }
  g.fillStyle = css.getPropertyValue("--bad").trim() || "#9a2e28";
  g.fillRect(label + (R.index / Math.max(1, n)) * plot, 0, 2, h);
}

function stopReplay() {
  R.playing = false;
  if ($("replayPlay")) $("replayPlay").textContent = "Play";
}
function togglePlay() {
  if (!R.view) return;
  R.playing = !R.playing;
  $("replayPlay").textContent = R.playing ? "Pause" : "Play";
  if (!R.playing) return;
  if (R.index >= R.view.ms.length - 1) R.index = 0;
  R.playFrom = R.view.ms[R.index];
  R.playAt = performance.now();
  requestAnimationFrame(playStep);
}
function playStep() {
  if (!R.playing || !R.view) return;
  const v = R.view, speed = Number($("replaySpeed").value) || 1;
  const target = R.playFrom + (performance.now() - R.playAt) * speed;
  let i = R.index;
  while (i < v.ms.length - 1 && v.ms[i + 1] <= target) i++;
  if (i !== R.index && !R.fetching) showTick(i);
  if (i >= v.ms.length - 1) { stopReplay(); return; }
  requestAnimationFrame(playStep);
}

async function saveRecording() {
  if (!R.view) return;
  const split = document.querySelector("input[name=viewSplit]:checked")?.value || "train";
  const info = await act("recording_update", { id: R.profile, recording: R.view.info.id, split, notes: $("viewNotes").value.trim() });
  if (!info) return;
  R.view.info = info;
  toast("Saved.");
  renderRecordings();
}

async function checkRecording() {
  if (!R.view) return;
  $("validation").replaceChildren(el("p", { class: "note" }, "Reading every picture and rebuilding every step's controls from the raw input…"));
  const v = await act("recording_validate", { id: R.profile, recording: R.view.info.id });
  if (!v) { $("validation").replaceChildren(); return; }
  const row = (k, val) => el("tr", {}, el("th", {}, k), el("td", {}, val));
  $("validation").replaceChildren(
    el("p", { class: v.ok ? "ok" : "bad" }, v.ok ? "The recording checks out: every picture reads, and every step's controls match the raw input." : "The recording has problems:"),
    el("table", { class: "figures narrow" },
      row("Pictures", fmt(v.ticks)),
      row("Played", `${trim1(v.seconds)} s at ${trim1(v.fps)} pictures a second`),
      row("Longest gap between pictures", `${trim1(v.longestGapMs)} ms`),
      row("Pictures that can't be read", fmt(v.picturesBad)),
      row("Steps whose controls don't match the raw input", fmt(v.ticksDisagreeing) + (v.firstDisagreement ? ` (${v.firstDisagreement})` : "")),
      row("Times out of order", fmt(v.timesOutOfOrder)),
      row("Mouse movement in all", `${fmt(Math.round(v.mouseX))} across, ${fmt(Math.round(v.mouseY))} down`),
      ...v.activeTicks.map(([name, n]) => row(`${keyLabel(name)} active`, `${fmt(n)} steps (${pct(n, v.ticks)})`))));
}

async function deleteRecording() {
  if (!R.view) return;
  if (!(await act("recording_delete", { id: R.profile, recording: R.view.info.id }))) return;
  R.view = null;
  $("viewer").hidden = true;
  renderRecordings();
}

// ---- Real games: training (generation 0) --------------------------------------------------------
// Game skills belong to the open model (Nova Neuro or Bicameral, from Yendory Hub); the backend reads
// and saves them through the Network (src-tauri/src/live/skills.rs).
const T = { profile: null, data: null, selected: null, training: false, history: [] };
const f3 = v => (v == null || !isFinite(v) ? "—" : v.toFixed(3));
const runsOf = k => (k?.notes?.runs || []);
const bestOf = k => runsOf(k).map(r => r.best).filter(Boolean).at(-1);

async function renderTrain() {
  if (!L.info) return;
  profileOptions($("trainProfile"), T.profile ?? R.profile ?? L.editing?.id, false);
  const id = $("trainProfile").value;
  T.profile = id || null;
  const body = $("skillsTable").tBodies[0];
  if (!id) { body.replaceChildren(el("tr", {}, el("td", { colspan: 7, class: "empty" }, "Make a game profile first (2.1)."))); $("trainPanel").hidden = true; return; }
  const d = await act("skills_view", { id });
  if (!d) return;
  T.data = d;
  G.running = d.generationsNow || G.running;
  T.training = (d.trainingNow && !G.running) || T.training;
  const rec = r => `${r.count} recording${r.count === 1 ? "" : "s"}, ${clock(r.seconds * 1000)}`;
  $("trainRecs").textContent = `Your play: ${rec(d.training)} for training, ${rec(d.testing)} held out for testing.` + (d.testing.count ? "" : " Mark one for testing (2.4) to measure what is learned.");
  const m = d.model;
  $("trainModel").replaceChildren(m ? el("span", {}, "Model: ", el("b", {}, m.name), ` (Nova ${m.architecture === "neuro" ? "Neuro" : "Bicameral"})`) : el("span", {}, "No model open."),
    d.problem && m ? el("span", { class: "warn" }, ` ${d.problem}`) : "");
  $("trainOpenModel").disabled = T.training;
  const game = gameName(id);
  body.replaceChildren(...(d.skills.length ? d.skills.map(k => {
    const best = bestOf(k);
    return el("tr", { class: T.selected === k.game ? "open-row" : "" },
      el("td", {}, k.name),
      el("td", {}, k.size),
      el("td", {}, `generation ${k.lineage.generation}, ${k.lineage.method}`),
      el("td", { class: "num" }, fmt(k.steps)),
      el("td", { class: "num" + (best && best.loss < best.baselineLoss ? " better" : "") }, best ? f3(best.loss) : "—"),
      el("td", { class: "num" }, best ? f3(best.baselineLoss) : "—"),
      el("td", {}, k.game === id ? el("button", { type: "button", onclick: () => selectSkill(k.game) }, "Open") : el("span", { class: "note" }, "choose its game above")));
  }) : [el("tr", {}, el("td", { colspan: 7, class: "empty" }, m ? (d.problem ? d.problem : `${m.name} has no game skills yet.`) : "Open a model to see its game skills."))]));
  const has = d.skills.some(k => k.game === id);
  $("newSkillForm").hidden = !m || !!d.problem || has;
  $("newSkillNote").hidden = $("newSkillForm").hidden;
  $("newSkill").textContent = m ? `Teach ${m.name} ${game}` : "Add the skill";
  // Each size's weights for this game's picture, and whether the Network can carry it.
  for (const sz of d.sizes || []) {
    const o = $("newSkillSize").querySelector(`option[value=${sz.size}]`);
    if (!o) continue;
    const weights = `${short(sz.weights)} weights`;
    o.dataset.note = sz.problem ? `${weights}: too large for the Network` : `${weights}; ${sz.size === "small" ? "quicker" : "slower"}`;
    o.disabled = !!sz.problem;
  }
  const chosen = (d.sizes || []).find(sz => sz.size === $("newSkillSize").value);
  if (chosen?.problem) {
    const ok = (d.sizes || []).find(sz => !sz.problem);
    if (ok) $("newSkillSize").value = ok.size;
  }
  const blocked = (d.sizes || []).length && (d.sizes || []).every(sz => sz.problem);
  $("newSkill").disabled = !!blocked;
  $("newSkillNote").textContent = blocked
    ? (d.sizes[0].problem + ".")
    : "A new skill starts untrained (random weights). It takes the game's controls and picture as they are now, and its mouse steps from the sizes of your mouse movements in your training recordings.";
  // Players kept in Tools by 0.1.4.
  $("oldPlayers").hidden = !d.oldPlayers.length;
  $("oldPlayersBody").replaceChildren(...d.oldPlayers.map(p => el("tr", {},
    el("td", {}, p.name), el("td", { class: "num" }, `${fmt(p.steps)} steps`),
    el("td", {}, el("button", { type: "button", disabled: !m || !!d.problem || has, title: has ? "The model already has a skill for this game" : null, onclick: () => movePlayer(p.id) }, m ? `Move into ${m.name}` : "Move into a model"),
      " ", el("button", { type: "button", class: "quiet", onclick: () => deleteOldPlayer(p.id) }, "Delete")))));
  if (has && !T.selected) T.selected = id;
  if (T.selected && T.selected !== id) T.selected = has ? id : null;
  if (T.selected && !has) T.selected = null;
  if (T.selected && !T.history.length) loadHistory();
  renderTrainPanel();
}

const gameName = id => L.profiles.find(p => p.id === id)?.name ?? id;
const skillNow = () => T.data?.skills.find(k => k.game === T.selected);

function loadHistory() {
  T.history = [];
  const k = skillNow();
  for (const r of runsOf(k)) {
    if (r.first) T.history.push({ step: r.first.step, held: r.first.loss, base: r.first.baselineLoss });
    if (r.best) T.history.push({ step: r.best.step, held: r.best.loss, base: r.best.baselineLoss });
  }
}

function selectSkill(game) {
  T.selected = game;
  const last = runsOf(skillNow()).at(-1)?.settings;
  $("trSteps").value = last?.steps ?? 1500;
  $("trRate").value = last?.learningRate ?? 0.0005;
  $("trOffset").value = last?.offset ?? 1;
  $("trBatch").value = last?.batch ?? 16;
  $("trLength").value = last?.length ?? 8;
  loadHistory();
  renderTrain();
}

function renderTrainPanel() {
  const k = skillNow();
  $("trainPanel").hidden = !k;
  if (!k) return;
  if (!$("trSteps").value) selectSkill(k.game);
  $("trainTitle").textContent = `${T.data.model?.name ?? "The model"} plays ${k.name}: ${fmt(k.steps)} steps so far, accepted at step ${fmt(k.acceptedStep)}`;
  $("trStart").hidden = T.training || G.running;
  $("trStop").hidden = !T.training || G.running;
  $("trDelete").disabled = T.training || G.running;
  $("trDelete").textContent = "Remove the skill";
  $("trStart").disabled = !T.data.training.count;
  if (!T.data.training.count) $("trNote").textContent = "Record some of your play on 2.3 (marked for training) first.";
  const best = bestOf(k);
  if (best && !T.training) showMeasure(best);
  drawTrainChart();
  renderGenerations(k);
}

async function makeSkill() {
  const k = await act("skill_create", { id: T.profile, size: $("newSkillSize").value });
  if (!k) return;
  T.selected = k.game;
  T.history = [];
  await renderTrain();
  selectSkill(k.game);
}

async function movePlayer(player) {
  if (await act("player_move", { id: T.profile, player })) { T.selected = T.profile; T.history = []; toast("Moved into the model."); }
  renderTrain();
}

async function deleteOldPlayer(player) {
  if (await act("player_delete", { id: T.profile, player })) renderTrain();
}

async function startTraining() {
  const settings = {
    steps: Number($("trSteps").value), batch: Number($("trBatch").value), length: Number($("trLength").value),
    learningRate: Number($("trRate").value), weightDecay: 0.0001, offset: Number($("trOffset").value), measureEvery: 100, seed: Date.now() % 100000,
  };
  const fresh = document.querySelector("input[name=trFrom]:checked")?.value === "fresh";
  const ok = await act("train_start", { id: T.profile, settings, fresh });
  if (ok === undefined) return;
  T.training = true;
  $("trMeter").hidden = false;
  $("trBar").style.width = "0%";
  $("trNote").textContent = "Loading your recordings…";
  $("sideRun").hidden = false;
  $("sideRunTitle").textContent = "Training";
  renderTrainPanel();
}

function trainProgress(pl) {
  if (pl.game !== T.selected) return;
  const g = pl.progress;
  const w = `${(100 * g.step) / g.steps}%`;
  $("trBar").style.width = w;
  $("sideRunBar").style.width = w;
  $("trNote").textContent = `Step ${fmt(g.step)} of ${fmt(g.steps)}; loss on your training play ${f3(g.trainLoss)}` +
    (g.measure ? `; held out ${f3(g.measure.loss)} (baseline ${f3(g.measure.baselineLoss)})${g.accepted ? ", accepted" : ""}` : "");
  T.history.push({ step: g.step, train: g.step ? g.trainLoss : null, held: g.measure?.loss, base: g.measure?.baselineLoss });
  if (g.measure) showMeasure(g.measure);
  drawTrainChart();
}

function trainEnded(spec, error) {
  T.training = false;
  $("trMeter").hidden = true;
  $("sideRun").hidden = true;
  $("sideRunTitle").textContent = "Measuring";
  if (error) $("trNote").textContent = `Training stopped: ${error}`;
  else {
    const r = runsOf(spec).at(-1);
    $("trNote").textContent = `${r.stopped ? "Stopped" : "Done"} after ${fmt(r.stepsDone)} steps; saved in the model.` + (r.best ? ` Accepted: held-out loss ${f3(r.best.loss)} against the baseline's ${f3(r.best.baselineLoss)}.` : r.heldOut.length ? " No measurement beat the accepted weights; they are kept." : " Nothing was held out, so the latest weights were accepted.");
  }
  if (S.route === "train") renderTrain();
}

// ---- Real games: generations (learning from its own play) ---------------------------------------
// The backend: src-tauri/src/live/generations.rs. Only the test range has a score for now.
const G = { running: false };
const gensOf = k => (k?.notes?.generations || []);
const genSettings = () => ({
  generations: Number($("gnCount").value), candidates: Number($("gnPlayers").value),
  episodes: Number($("gnEpisodes").value), seconds: Number($("gnSeconds").value),
  noise: Number($("gnNoise").value) / 100, elite: Number($("gnElite").value) / 100,
  steps: Number($("gnSteps").value), learningRate: Number($("gnRate").value),
  demonstrations: Number($("gnDemos").value) / 100, testEpisodes: Number($("gnTest").value),
});

function renderGenerations(k) {
  const range = k.game === "yendory-range";
  $("genWhere").textContent = range
    ? "On the test range, scored by its count of hits. Start the range on 2.1 first; it must stay open while generations run."
    : "Generations need the game's score. The test range reports its own; a real game's score will be read from the screen (phase 5), so for now they run only on the test range.";
  $("genForm").hidden = !range;
  const s = genSettings();
  const playMin = ((s.candidates * s.episodes + 2 * s.testEpisodes) * (s.seconds + 0.5)) / 60;
  $("gnTime").textContent = `Each generation: about ${playMin.toFixed(0)} minutes of play, then ${fmt(s.steps)} learning steps.` + (T.data?.training.count ? "" : " You have no recordings for training, so it learns from its own play alone.");
  $("gnStart").hidden = G.running;
  $("gnStart").disabled = T.training;
  $("gnStop").hidden = !G.running;
  $("gnMeter").hidden = !G.running;
  const rows = gensOf(k);
  $("genTable").hidden = !rows.length;
  $("genTable").tBodies[0].replaceChildren(...rows.slice().reverse().map(a => el("tr", {},
    el("td", { class: "num" }, a.attempt + 1),
    el("td", {}, String(a.generation)),
    el("td", {}, a.candidates.map(v => v.toFixed(1)).join(" · ") || "—"),
    el("td", {}, a.elite.length ? `${a.elite.length} episode${a.elite.length === 1 ? "" : "s"} (${a.elite.map(e => e[1]).join(", ")} hits), ${fmt(a.learnedSteps)} steps` : "—"),
    el("td", { class: "num" }, a.note ? "—" : a.championScore.toFixed(1)),
    el("td", { class: "num" + (a.note ? "" : a.accepted ? " better" : " worse") }, a.note ? "—" : a.childScore.toFixed(1)),
    el("td", {}, a.note ? a.note : a.accepted ? "new champion" : "the champion stays"))));
}

async function startGenerations() {
  const ok = await act("generations_start", { settings: genSettings() });
  if (!ok) return;
  G.running = true;
  $("gnBar").style.width = "0%";
  $("gnNote").textContent = "Starting…";
  $("sideRun").hidden = false;
  $("sideRunTitle").textContent = "Generations";
  renderTrainPanel();
}

function genProgress(p) {
  G.running = true;
  const phases = { playing: 0, learning: 1, testing: 2, done: 3 };
  const w = `${(100 * (p.run - 1 + phases[p.phase] / 3)) / p.of}%`;
  $("gnBar").style.width = w;
  $("sideRunBar").style.width = w;
  $("gnNote").textContent = `Generation ${p.run} of ${p.of}, ${p.phase}: ${p.detail}`;
  if (p.attempt && S.route === "train") renderTrain();
}

function genEnded(spec, error) {
  G.running = false;
  $("sideRun").hidden = true;
  $("sideRunTitle").textContent = "Measuring";
  if (error) $("gnNote").textContent = `Generations stopped: ${error}`;
  else $("gnNote").textContent = `Done. The champion is generation ${spec.lineage.generation}; it is saved in the model.`;
  if (S.route === "train") renderTrain();
}

function showMeasure(m) {
  const row = (k, a, b, lowerBetter = true) => el("tr", {}, el("th", {}, k), el("td", { class: (lowerBetter ? a < b : a > b) ? "better" : "worse" }, a), el("td", {}, b));
  $("trMeasure").replaceChildren(el("h3", {}, `Held out, at step ${fmt(m.step)} (${fmt(m.ticks)} ticks)`),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", {}, "The network"), el("th", {}, "Frequency baseline"))),
      el("tbody", {},
        row("Loss a tick (all controls)", f3(m.loss), f3(m.baselineLoss)),
        row("Keys and buttons", f3(m.keysLoss), f3(m.baselineKeysLoss)),
        row("Mouse", f3(m.mouseLoss), f3(m.baselineMouseLoss)),
        row("Mouse step right", pct(Math.round(m.mouseAccuracy * 1000), 1000), pct(Math.round(m.baselineMouseAccuracy * 1000), 1000), false))),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Control"), el("th", { class: "num" }, "You used it"), el("th", { class: "num" }, "It would"), el("th", { class: "num" }, "F1 (0 for the baseline)"))),
      el("tbody", {}, m.controls.map(c => el("tr", {}, el("td", {}, keyLabel(c.name)), el("td", { class: "num" }, pct(Math.round(c.used * 1000), 1000)), el("td", { class: "num" }, pct(Math.round(c.predicted * 1000), 1000)), el("td", { class: "num" }, c.f1.toFixed(2)))))));
}

function drawTrainChart() {
  const c = $("trChart"), w = c.clientWidth || 600, h = 160;
  if (c.width !== w) c.width = w;
  const g = c.getContext("2d"), css = getComputedStyle(document.body);
  g.clearRect(0, 0, w, h);
  const pts = T.history;
  if (!pts.length) return;
  const vals = pts.flatMap(p => [p.train, p.held, p.base]).filter(v => v != null && isFinite(v));
  if (!vals.length) return;
  const maxStep = Math.max(1, ...pts.map(p => p.step)), top = Math.max(...vals) * 1.05, low = Math.min(...vals) * 0.95;
  const X = s => 8 + (s / maxStep) * (w - 16), Y = v => 8 + (1 - (v - low) / Math.max(1e-9, top - low)) * (h - 16);
  const line = (key, color, dash) => {
    const p = pts.filter(q => q[key] != null && isFinite(q[key]));
    if (!p.length) return;
    g.strokeStyle = color; g.setLineDash(dash); g.lineWidth = 1.5; g.beginPath();
    p.forEach((q, i) => (i ? g.lineTo(X(q.step), Y(q[key])) : g.moveTo(X(q.step), Y(q[key]))));
    g.stroke(); g.setLineDash([]);
  };
  line("train", css.getPropertyValue("--accent").trim() || "#284a7e", []);
  line("held", css.getPropertyValue("--ink").trim() || "#222", []);
  line("base", css.getPropertyValue("--ink-soft").trim() || "#777", [5, 4]);
}

async function deleteSkill() {
  if (!skillNow()) return;
  if (!(await act("skill_delete", { id: T.profile }))) return;
  T.selected = null;
  T.history = [];
  renderTrain();
}

// ---- Real games: play and measure ---------------------------------------------------------------
const P = { playing: false, measuring: false };

async function renderAutoplay() {
  if (!L.info) return;
  profileOptions($("apProfile"), $("apProfile").value || T.profile || L.editing?.id, false);
  const id = $("apProfile").value;
  const d = id ? await act("skills_view", { id }) : null;
  const k = d?.skills.find(x => x.game === id);
  const m = d?.model;
  $("apSkill").textContent = !m ? "Open a model first." : d.problem ? d.problem : k ? `${m.name} plays with its skill for this game (${fmt(k.steps)} steps).` : `${m.name} has no skill for this game: add one on 2.5.`;
  const can = !!(m && !d.problem && k);
  $("apStart").disabled = !can || P.playing || P.measuring;
  $("apStart").hidden = P.playing;
  $("apStopBtn").hidden = !P.playing;
  const range = id === RANGE_ID;
  $("evStart").disabled = !range || !can || P.playing || P.measuring;
  $("evStart").hidden = P.measuring;
  $("evStop").hidden = !P.measuring;
  if (!range) $("evNote").textContent = "Choose the Yendory Range game to measure on it.";
}

async function startPlay() {
  // Tools asks for permission in its own dialog, every time.
  const decide = document.querySelector("input[name=apDecide]:checked")?.value || "likely";
  const ok = await act("play_start", { id: $("apProfile").value, decide, record: $("apRecord").checked });
  if (!ok) return;
  P.playing = true;
  $("apLive").hidden = false;
  $("apDone").textContent = "";
  $("liveStop").hidden = false;
  renderAutoplay();
}

function playStatus(st) {
  const p = L.profiles.find(x => x.id === $("apProfile").value);
  $("apState").textContent = `${clock(st.ms)}, ${trim1(st.fps)} decisions a second` + (st.refused ? `, ${fmt(st.refused)} refused by the safety layer` : "");
  if (p) $("apHeld").replaceChildren(...chips([...p.keys, ...p.buttons], st.held));
  $("apMouse").textContent = `Mouse: ${trim1(st.mouse[0])} across, ${trim1(st.mouse[1])} down`;
}

function playEnded(r, error) {
  P.playing = false;
  $("apLive").hidden = true;
  $("liveStop").hidden = true;
  $("apDone").textContent = error ? `Play stopped: ${error}` : `Played ${clock(r.ms)} (${fmt(r.ticks)} decisions); ${r.ended}.` + (r.recording ? " Its play is on 2.4." : "");
  renderAutoplay();
}

async function startMeasure() {
  const decide = document.querySelector("input[name=apDecide]:checked")?.value || "likely";
  const ok = await act("evaluate_start", { episodes: Number($("evEpisodes").value), seconds: Number($("evSeconds").value), decide });
  if (!ok) return;
  P.measuring = true;
  $("evTable").hidden = true;
  $("evNote").textContent = "Starting…";
  $("liveStop").hidden = false;
  renderAutoplay();
}

function measureEnded(rows, error) {
  P.measuring = false;
  $("liveStop").hidden = true;
  if (error) { $("evNote").textContent = `Measuring stopped: ${error}`; renderAutoplay(); return; }
  const best = Math.max(...rows.map(r => r.hitsPerMinute));
  $("evNote").textContent = rows[0] && rows[0].hitsPerMinute > Math.max(...rows.slice(1).map(r => r.hitsPerMinute))
    ? `${rows[0].who} beat both baselines on layouts it never saw.` : `${rows[0]?.who ?? "The model"} did not beat both baselines.`;
  $("evTable").hidden = false;
  $("evTable").tBodies[0].replaceChildren(...rows.map(r => el("tr", {},
    el("td", {}, r.who), el("td", { class: "num" + (r.hitsPerMinute === best && best > 0 ? " better" : "") }, trim1(r.hitsPerMinute)),
    el("td", { class: "num" }, trim1(r.shotsPerMinute)), el("td", { class: "num" }, r.accuracy.toFixed(2)),
    el("td", { class: "mono" }, r.episodes.map(e => e.hits).join(" · ")))));
  renderAutoplay();
}

// ---- Real games: checks -------------------------------------------------------------------------
async function renderChecks() {
  if (!L.info) return;
  profileOptions($("check3Game"), $("check3Game").value || RANGE_ID, false);
  profileOptions($("check4Game"), $("check4Game").value || RANGE_ID, false);
}

function showChecks(list, results) {
  list.replaceChildren(...results.map(c => el("li", { class: c.ok ? "pass" : "fail" }, el("b", {}, c.ok ? "✓" : "✗"), c.name, el("span", { class: "note" }, c.detail))));
}

async function runChecks(phase) {
  const list = $("checks" + phase);
  list.replaceChildren(el("li", { class: "note" }, "Running… leave the keyboard and mouse alone."));
  const args = { phase, game: phase === 3 ? $("check3Game").value : phase === 4 ? $("check4Game").value : null };
  try { showChecks(list, await invoke("checks_run", args)); }
  catch (e) { list.replaceChildren(el("li", { class: "fail" }, el("b", {}, "✗"), "The checks could not run", el("span", { class: "note" }, String(e)))); }
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
  else if (h.id) {
    notice("");
    await openModel(h.id);
    if (S.model) notice(`${S.model.name} came from Yendory Hub. To teach it a game, record yourself playing (2.3), then add and train its game skill on 2.5 Train.`, false, el("button", { type: "button", onclick: () => { notice(""); show("train"); } }, "Go to 2.5 Train"));
  }
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
  $("liveStop").addEventListener("click", () => { invoke("live_stop"); invoke("play_stop"); });
  listen("live-log", e => logRows(e.payload));
  listen("live-done", e => checkDone(e.payload.stopped));
  listen("live-stopped", () => toast(R.recording ? "Recording ended with the emergency stop keys; it is kept." : "Emergency stop: every control was released."));
  $("recProfile").addEventListener("change", () => { R.profile = $("recProfile").value; renderRecord(); });
  $("recStart").addEventListener("click", startRecording);
  $("recStopBtn").addEventListener("click", () => invoke("record_stop"));
  listen("record-status", e => recordStatus(e.payload));
  listen("record-done", e => recordEnded(e.payload, null));
  listen("record-failed", e => recordEnded(null, String(e.payload)));
  $("recsProfile").addEventListener("change", () => { R.profile = $("recsProfile").value; R.view = null; $("viewer").hidden = true; stopReplay(); renderRecordings(); });
  $("replayPos").addEventListener("input", () => { stopReplay(); showTick(Number($("replayPos").value)); });
  $("replayPlay").addEventListener("click", togglePlay);
  $("replayBack").addEventListener("click", () => { stopReplay(); showTick(R.index - 1); });
  $("replayFwd").addEventListener("click", () => { stopReplay(); showTick(R.index + 1); });
  $("timeline").addEventListener("click", e => {
    if (!R.view) return;
    const c = $("timeline"), x = e.offsetX * (c.width / c.clientWidth), label = 46;
    stopReplay();
    showTick(Math.round(((x - label) / (c.width - label)) * R.view.ms.length));
  });
  $("viewSave").addEventListener("click", saveRecording);
  $("viewCheck").addEventListener("click", checkRecording);
  $("viewDelete").addEventListener("click", deleteRecording);
  window.addEventListener("resize", () => { if (S.route === "recordings") drawTimeline(); if (S.route === "train") drawTrainChart(); });
  $("trainProfile").addEventListener("change", () => { T.profile = $("trainProfile").value; T.selected = null; T.history = []; renderTrain(); });
  $("newSkill").addEventListener("click", makeSkill);
  $("trainOpenModel").addEventListener("click", chooseModel);
  $("trStart").addEventListener("click", startTraining);
  $("trStop").addEventListener("click", () => invoke("train_stop"));
  $("trDelete").addEventListener("click", deleteSkill);
  listen("train-progress", e => trainProgress(e.payload));
  listen("train-done", e => trainEnded(e.payload, null));
  listen("train-failed", e => trainEnded(null, String(e.payload)));
  $("gnStart").addEventListener("click", startGenerations);
  $("gnStop").addEventListener("click", () => invoke("generations_stop"));
  for (const id of ["gnCount", "gnPlayers", "gnEpisodes", "gnSeconds", "gnTest", "gnSteps"]) $(id).addEventListener("input", () => { const k = skillNow(); if (k) renderGenerations(k); });
  listen("gen-progress", e => genProgress(e.payload));
  listen("gen-done", e => genEnded(e.payload, null));
  listen("gen-failed", e => genEnded(null, String(e.payload)));
  $("apProfile").addEventListener("change", renderAutoplay);
  $("apStart").addEventListener("click", startPlay);
  $("apStopBtn").addEventListener("click", () => invoke("play_stop"));
  listen("play-status", e => playStatus(e.payload));
  listen("play-done", e => playEnded(e.payload, null));
  listen("play-failed", e => playEnded(null, String(e.payload)));
  $("evStart").addEventListener("click", startMeasure);
  $("evStop").addEventListener("click", () => invoke("play_stop"));
  listen("eval-progress", e => { const p = e.payload; $("evNote").textContent = `Episode ${p.episode} of ${p.episodes}: ${p.who} hit ${p.hits} of ${p.shots} shots.`; });
  listen("eval-done", e => measureEnded(e.payload, null));
  listen("eval-failed", e => measureEnded(null, String(e.payload)));
  for (const b of document.querySelectorAll("[data-check]")) b.addEventListener("click", () => runChecks(Number(b.dataset.check)));
  $("check4Game").addEventListener("change", renderChecks);
  $("checkStopKeys").addEventListener("click", async () => {
    $("checks2").replaceChildren(el("li", { class: "note" }, "Press Ctrl+Alt+End now (15 seconds)…"));
    showChecks($("checks2"), [await invoke("checks_stop_keys")]);
  });
}

async function start() {
  if (store.get("collapsed", "0") === "1") document.body.classList.add("sidebar-collapsed");
  await debugStart();
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
