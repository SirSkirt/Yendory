"use strict";
// Yendory Studio frontend. The page only displays; every neural computation runs in the Rust
// Yendory Engine behind these commands.
const $ = id => document.getElementById(id);
const tauri = window.__TAURI__;
const invoke = (cmd, args = {}) => {
  if (!tauri?.core?.invoke) throw new Error("Yendory Studio must run as the installed app.");
  return tauri.core.invoke(cmd, args);
};
const fmt = n => (n ?? 0).toLocaleString("en-US");
const pct = (c, t) => (t ? (100 * c / t).toFixed(1) + "%" : "—");
const el = (tag, attrs = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
};
const when = secs => secs ? new Date(secs * 1000).toLocaleString() : "—";

// ---- diagnostics ---------------------------------------------------------------------------
const diag = [];
function log(level, msg) {
  const text = typeof msg === "string" ? msg : (msg?.message || JSON.stringify(msg));
  diag.push(`[${new Date().toLocaleTimeString()}] ${level}: ${text}`);
  if (diag.length > 200) diag.shift();
  $("diagLog").textContent = diag.join("\n");
  $("diagCount").textContent = String(diag.length);
  if (level === "ERROR") { $("diagnostics").classList.add("has-errors"); $("statusLine").textContent = text; }
}
const safe = (fn, label) => async (...a) => {
  try { return await fn(...a); } catch (e) { log("ERROR", `${label}: ${typeof e === "string" ? e : e?.message || e}`); }
};
window.addEventListener("error", e => log("ERROR", e.message));
window.addEventListener("unhandledrejection", e => log("ERROR", e.reason));
const say = text => { $("statusLine").textContent = text; };

// ---- navigation ----------------------------------------------------------------------------
function go(page) {
  document.querySelectorAll(".contents button").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".page").forEach(p => { p.hidden = p.id !== "page-" + page; });
  if (page === "library") loadHistory();
}
document.querySelectorAll(".contents button").forEach(b => b.addEventListener("click", () => go(b.dataset.page)));

// ---- models --------------------------------------------------------------------------------
let active = null;
let teaching = false;

function showActive(meta) {
  active = meta;
  $("currentModel").replaceChildren(meta
    ? el("span", {}, "Open: ", el("b", {}, meta.name), ` · step ${fmt(meta.step)}`)
    : "No model open");
  $("activeEntry").hidden = !meta;
  if (!meta) return;
  $("activeName").textContent = meta.name;
  $("activeStep").textContent = fmt(meta.step);
  $("activeWords").textContent = fmt(meta.words);
  $("activeLeft").textContent = fmt(meta.groups[0]);
  $("activeRight").textContent = fmt(meta.groups[1]);
  $("activeInter").textContent = fmt(meta.groups[2]);
  $("activeTotal").textContent = fmt(meta.parameters);
  $("activeLessons").textContent = fmt(meta.taught.filter(t => t.outcome === "learned").length);
  $("activeOrigin").textContent = meta.origin === "new" ? "created in Studio" : meta.origin || "—";
}

async function refreshModels() {
  const models = await invoke("list_models");
  const body = $("modelTable").tBodies[0];
  body.replaceChildren();
  if (!models.length) {
    body.append(el("tr", {}, el("td", { colspan: 5, class: "empty" }, "No models yet. Create one or import a checkpoint.")));
    return;
  }
  for (const m of models) {
    const isOpen = active?.id === m.id;
    const row = el("tr", { class: isOpen ? "current-row" : "" },
      el("td", {}, m.name),
      el("td", { class: "num" }, fmt(m.step)),
      el("td", { class: "num" }, fmt(m.words)),
      el("td", {}, when(m.updated)),
      el("td", {},
        isOpen ? el("span", { class: "tag ok" }, "open") : el("button", { onclick: safe(() => openModel(m.id), "Open model") }, "Open"),
        " ",
        el("button", { class: "quiet", onclick: safe(() => deleteModel(m), "Delete model") }, "Delete")));
    body.append(row);
  }
}
async function openModel(id) {
  showActive(await invoke("open_model", { id }));
  await refreshModels();
  await refreshLessons();
  say(`Opened ${active.name}.`);
}
async function deleteModel(m) {
  if (!confirm(`Delete "${m.name}" and its teaching record? Export it first if you want to keep it. This cannot be undone.`)) return;
  await invoke("delete_model", { id: m.id });
  if (active?.id === m.id) showActive(null);
  await refreshModels();
  say(`Deleted ${m.name}.`);
}
$("createModel").addEventListener("click", safe(async () => {
  const name = $("newName").value.trim();
  if (!name) { $("newName").focus(); throw new Error("Give the new model a name"); }
  showActive(await invoke("create_model", { name }));
  $("newName").value = "";
  await refreshModels();
  say(`Created ${active.name}: an untrained Bicameral model.`);
}, "Create model"));
$("importModel").addEventListener("click", safe(async () => {
  const meta = await invoke("import_model");
  if (!meta) return;
  showActive(meta);
  await refreshModels();
  say(`Imported ${meta.name} (step ${fmt(meta.step)}).`);
}, "Import"));
$("exportModel").addEventListener("click", safe(async () => {
  const where = await invoke("export_model");
  if (where) say(`Exported and verified: ${where}`);
}, "Export"));
$("renameModel").addEventListener("click", safe(async () => {
  if (!active) return;
  const name = prompt("New name for this model", active.name);
  if (!name) return;
  showActive({ ...active, ...(await invoke("rename_model", { id: active.id, name })) });
  await refreshModels();
}, "Rename"));

// ---- classroom -----------------------------------------------------------------------------
let lessonsList = [];
const fileState = {};  // lesson index -> outcome text during/after teaching

function lessonStatus(l, i) {
  if (fileState[i]) return fileState[i];
  if (l.error) return el("span", { class: "bad" }, l.error);
  if (!l.heldOut) return el("span", { class: "warn" }, "no held-out units: cannot be judged");
  if (l.alreadyTaught) return el("span", { class: "warn" }, "already learned by this model");
  return el("span", { class: "ok" }, "ready");
}
function renderLessons() {
  const body = $("lessonTable").tBodies[0];
  body.replaceChildren();
  if (!lessonsList.length) {
    body.append(el("tr", {}, el("td", { colspan: 7, class: "empty" }, "No lessons added.")));
    return;
  }
  lessonsList.forEach((l, i) => body.append(el("tr", { id: "lesson-" + i },
    el("td", { class: "num" }, i + 1),
    el("td", {}, l.name),
    el("td", { class: "num" }, fmt(l.train)),
    el("td", { class: "num" }, fmt(l.heldOut)),
    el("td", {}, l.kinds.map(([k, n]) => `${n} ${k}`).join(", ") || "—"),
    el("td", {}, lessonStatus(l, i)),
    el("td", {}, el("button", { class: "quiet", disabled: teaching ? "" : null, onclick: safe(async () => {
      lessonsList = await invoke("remove_lesson", { index: i }); renderLessons();
    }, "Remove lesson") }, "remove")))));
}
async function refreshLessons() { lessonsList = await invoke("lessons"); renderLessons(); }
$("addLessons").addEventListener("click", safe(async () => {
  lessonsList = await invoke("add_lessons");
  for (const k of Object.keys(fileState)) delete fileState[k];
  renderLessons();
}, "Add lessons"));
$("clearLessons").addEventListener("click", safe(async () => {
  lessonsList = await invoke("remove_lesson", { index: null });
  renderLessons();
}, "Clear lessons"));

function setTeaching(on) {
  teaching = on;
  $("teach").disabled = on;
  $("stopTeach").disabled = !on;
  for (const id of ["addLessons", "clearLessons", "createModel", "importModel"]) $(id).disabled = on;
  renderLessons();
}

let progress = { files: 1, index: 0, epochs: 1, candidates: 1, candidate: 0, epoch: 1 };
const FOCI = ["dense", "transformer", "interconnect", "joint"];
$("teach").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open or create a model first");
  const options = {
    epochs: Number($("optEpochs").value),
    batchSize: Number($("optBatch").value),
    learningRate: Number($("optRate").value),
    foundation: $("optFoundation").checked,
    rehearsal: $("optRehearsal").checked,
    protectAnswers: $("optProtect").checked,
  };
  progress = { files: lessonsList.length, index: 0, epochs: options.epochs, candidates: options.foundation ? 4 : 1, candidate: 0, epoch: 1 };
  for (const k of Object.keys(fileState)) delete fileState[k];
  $("candidateTable").tBodies[0].replaceChildren();
  $("courseTable").tBodies[0].replaceChildren();
  $("progressEntry").hidden = false;
  $("progressTitle").textContent = `Teaching ${active.name}`;
  await invoke("teach", { options });
  setTeaching(true);
  say("Teaching started.");
}, "Teach"));
$("stopTeach").addEventListener("click", safe(async () => {
  await invoke("stop_teaching");
  $("stopTeach").disabled = true;
  say("Stopping at the next batch; the epoch in progress will be discarded.");
}, "Stop"));

function scoreCell(s) { return s ? `${s.exact_correct}/${s.exact_total}` : "—"; }
function onTeachEvent(e) {
  if (e.type === "file_started") {
    progress.index = e.index;
    fileState[e.index] = el("span", { class: "tag" }, "teaching…");
    renderLessons();
    $("progressTitle").textContent = `Teaching ${active?.name ?? ""} · lesson ${e.index + 1} of ${e.files}: ${e.name}`;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7 }, el("b", {}, `Lesson ${e.index + 1}: ${e.name}`))));
  } else if (e.type === "session") {
    const s = e.event;
    if (s.type === "started") {
      $("progressLine").textContent = `Baseline: held-out loss ${s.baseline.heldout_loss.toFixed(4)}, exact ${scoreCell(s.baseline)}; ${fmt(s.train_units)} units to learn, ${fmt(s.heldout_units)} held out, ${fmt(s.new_words)} new word units.`;
    } else if (s.type === "batch") {
      progress.epoch = s.epoch;
      progress.candidate = Math.max(0, FOCI.indexOf(s.focus));
      if (progress.candidates === 1) progress.candidate = 0;
      const inFile = (((s.epoch - 1) * progress.candidates + progress.candidate) * s.batches + s.batch) / (progress.epochs * progress.candidates * s.batches);
      $("progressBar").style.width = (100 * (progress.index + Math.min(1, inFile)) / progress.files).toFixed(1) + "%";
      $("progressLine").textContent = `epoch ${s.epoch}/${progress.epochs} · ${s.focus} candidate · batch ${s.batch}/${s.batches} · loss ${s.loss.toFixed(4)} · learning rate ${s.learning_rate.toExponential(2)}`;
    } else if (s.type === "candidate") {
      const decision = s.eligible
        ? (s.reasons.length ? el("span", { class: "warn" }, s.reasons.join("; ")) : el("span", { class: "ok" }, "eligible"))
        : el("span", { class: "bad" }, s.reasons.join("; "));
      $("candidateTable").tBodies[0].append(el("tr", { id: `cand-${e.index}-${s.epoch}-${s.focus}` },
        el("td", { class: "num" }, s.epoch), el("td", {}, s.focus),
        el("td", { class: "num" }, s.train_loss.toFixed(4)), el("td", { class: "num" }, s.score.heldout_loss.toFixed(4)),
        el("td", { class: "num" }, scoreCell(s.score)),
        el("td", { class: "num" }, s.score.retention_loss == null ? "—" : s.score.retention_loss.toFixed(4)),
        el("td", {}, decision)));
    } else if (s.type === "epoch") {
      for (const c of s.candidates) {
        const row = $(`cand-${e.index}-${s.epoch}-${c.focus}`);
        if (!row) continue;
        if (c.selected) { row.classList.add("current-row"); row.lastChild.replaceChildren(el("b", { class: "ok" }, "selected")); }
        else if (c.eligible && !c.reasons.length) row.lastChild.replaceChildren(el("span", {}, "eligible; another candidate was better"));
      }
      log("INFO", `Epoch ${s.epoch}: ${s.selected ? "selected " + s.selected : "no candidate selected"}`);
    }
  } else if (e.type === "file_finished") {
    const cls = { learned: "ok", kept_previous: "warn", duplicate: "warn", failed: "bad", stopped: "warn" }[e.outcome] || "";
    const label = { learned: "learned", kept_previous: "kept previous model", duplicate: "duplicate", failed: "not taught", stopped: "stopped" }[e.outcome] || e.outcome;
    fileState[e.index] = el("span", { class: cls }, label);
    renderLessons();
    const body = $("courseTable").tBodies[0];
    body.append(el("tr", {},
      el("td", {}, e.name),
      el("td", {}, el("span", { class: cls }, label), e.error ? el("div", { class: "note" }, e.error) : ""),
      el("td", { class: "num" }, `${fmt(e.start_step)} → ${fmt(e.end_step)}`),
      el("td", { class: "num" }, `${scoreCell(e.baseline)} → ${scoreCell(e.final_score)}`),
      el("td", { class: "mono", title: e.sha256 }, e.sha256.slice(0, 12))));
  }
}
function onTeachFinished(f) {
  setTeaching(false);
  if (f.model) showActive(f.model);
  $("progressBar").style.width = "100%";
  const learned = f.files.filter(x => x.outcome === "learned").length;
  const text = f.error ? `Teaching ended with an error: ${f.error}`
    : `${f.stopped ? "Stopped" : "Finished"}: ${learned} of ${f.files.length} lesson file(s) learned. Progress is saved after every learned file.`;
  $("progressTitle").textContent = text;
  if (f.error) log("ERROR", f.error); else say(text);
  refreshModels();
}

// ---- exams ---------------------------------------------------------------------------------
let conversation = [];
function addLine(who, text, cls) {
  const t = $("transcript");
  t.querySelector(".empty")?.remove();
  t.append(el("p", { class: cls }, el("span", { class: "who" }, who), text));
  t.scrollTop = t.scrollHeight;
}
$("send").addEventListener("click", safe(async () => {
  const message = $("message").value.trim();
  if (!message || !active) return;
  $("send").disabled = true;
  try {
    addLine("you", message, "user");
    $("message").value = "";
    const answer = await invoke("ask", { history: conversation, message });
    conversation.push([message, answer]);
    addLine(active.name, answer || "(empty reply)", "model");
  } finally { $("send").disabled = false; }
}, "Ask"));
$("message").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); $("send").click(); } });
$("newChat").addEventListener("click", () => {
  conversation = [];
  $("transcript").replaceChildren(el("p", { class: "empty" }, "New conversation."));
});
$("runExam").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open a model first");
  say("Grading… every question is answered by the model.");
  const r = await invoke("exam");
  if (!r) return;
  const out = $("examResult");
  out.replaceChildren();
  const overall = Object.values(r.report.exact).reduce((a, s) => ({ c: a.c + s.correct, t: a.t + s.total }), { c: 0, t: 0 });
  out.append(el("div", { class: "entry" },
    el("h3", {}, `${r.name}: ${pct(overall.c, overall.t)} exact (${overall.c}/${overall.t})`),
    r.fileWasTaught ? el("p", { class: "bad" }, "This exact file was taught to this model, so this is not an independent exam.") : "",
    r.seenBefore ? el("p", { class: "warn" }, `${r.seenBefore} question(s) also appear in material this model was taught.`) : el("p", { class: "ok" }, "No question appears in material this model was taught."),
    el("table", { class: "figures" }, ...Object.entries(r.report.exact).map(([k, s]) => el("tr", {}, el("th", {}, k), el("td", {}, `${s.correct}/${s.total} (${pct(s.correct, s.total)})`))),
      r.report.prefers_supported.total ? el("tr", {}, el("th", {}, "prefers the supported answer"), el("td", {}, pct(r.report.prefers_supported.correct, r.report.prefers_supported.total))) : "")));
  const table = el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", {}, "Question"), el("th", {}, "Expected"), el("th", {}, "Reply"))));
  const body = el("tbody");
  for (const o of r.report.outcomes) {
    body.append(el("tr", {}, el("td", { class: o.correct ? "ok" : "bad" }, o.correct ? "✓" : "✗"),
      el("td", {}, o.prompt.replace(/^U: /, "").replace(/\nN: $/, "").replace(/\n/g, " ⏎ ")), el("td", {}, o.expected), el("td", { class: "mono" }, o.reply)));
  }
  table.append(body);
  out.append(table);
  say(`Graded ${overall.t} question(s).`);
}, "Exam"));

// ---- brain ---------------------------------------------------------------------------------
let lastMeasure = null;
function renderMeasure() {
  const m = lastMeasure;
  if (!m) return;
  const idx = Number($("stageSelect").value || m.trace.stages.length - 1);
  const { attention } = Brain.show(m, idx);
  $("gateMeans").textContent = m.gate_means.map(g => g.toFixed(2)).join(" · ") || "—";
  $("effectRight").textContent = m.ablation.without_right_to_left.toFixed(3);
  $("effectLeft").textContent = m.ablation.without_left_to_right.toFixed(3);
  $("nextTokens").replaceChildren(...m.next_tokens.map(([t, p]) => el("span", { class: "tok" }, `${t} ${(p * 100).toFixed(1)}%`)));
  const top = Math.max(1e-9, ...attention);
  $("attentionTiles").replaceChildren(...attention.map((w, i) => {
    const s = el("span", { class: "tok", title: `token ${i} · ${(w * 100).toFixed(2)}%` }, m.tokens[i]);
    s.style.background = `color-mix(in srgb, var(--brain-left) ${Math.round(8 + 60 * w / top)}%, transparent)`;
    return s;
  }));
}
$("inspect").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open a model first");
  lastMeasure = await invoke("measure", { prompt: $("probe").value });
  const sel = $("stageSelect");
  sel.replaceChildren(...lastMeasure.trace.stages.map((s, i) => el("option", { value: i }, Brain.stageLabel(s))));
  sel.value = String(lastMeasure.trace.stages.length - 1);
  renderMeasure();
  $("brainDetail").textContent = `Measured at step ${fmt(lastMeasure.step)} while reading the last token of "U: ${$("probe").value}". Click a dot for its value.`;
}, "Inspect"));
$("stageSelect").addEventListener("change", renderMeasure);
$("probe").addEventListener("keydown", e => { if (e.key === "Enter") $("inspect").click(); });

// ---- library -------------------------------------------------------------------------------
async function loadHistory() {
  const box = $("historyList");
  if (!active) { box.replaceChildren(el("p", { class: "empty" }, "Open a model to see its record.")); return; }
  const rows = await invoke("history", { id: active.id });
  if (!rows.length) { box.replaceChildren(el("p", { class: "empty" }, "Nothing has been taught to this model in Studio yet.")); return; }
  // Candidates are shown inside their epoch; start markers add nothing to the record.
  const shown = rows.filter(r => r.type || r.event?.type === "file_finished" || (r.event?.type === "session" && r.event.event.type === "epoch"));
  box.replaceChildren(...shown.reverse().map(r => {
    const e = r.event;
    if (r.type === "course_started") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), `Course started: ${r.files} file(s), ${r.options.epochs} epoch(s), ${r.options.foundation ? "Foundation" : "Joint only"}.`);
    if (r.type === "course_finished") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), r.error ? el("span", { class: "bad" }, "Course ended with an error: " + r.error) : `Course ${r.stopped ? "stopped" : "finished"}.`);
    if (e?.type === "file_finished") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)),
      el("b", {}, e.name), ` — ${e.outcome.replace("_", " ")}, steps ${fmt(e.start_step)} → ${fmt(e.end_step)}, exact ${scoreCell(e.baseline)} → ${scoreCell(e.final_score)}`, e.error ? ` (${e.error})` : "",
      el("div", { class: "note mono" }, "sha256 " + e.sha256));
    if (e?.type === "session" && e.event.type === "epoch") {
      const ep = e.event;
      const d = el("details", { class: "record" }, el("summary", {}, el("span", { class: "when" }, when(r.time)),
        `Epoch ${ep.epoch}: ${ep.selected ? "selected " + ep.selected : "no candidate selected"} · exact ${scoreCell(ep.accepted_score)} · held-out loss ${ep.accepted_score.heldout_loss.toFixed(4)}`));
      const t = el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, "Candidate"), el("th", { class: "num" }, "Held-out loss"), el("th", { class: "num" }, "Exact"), el("th", {}, "Decision"))));
      const b = el("tbody");
      for (const c of ep.candidates) b.append(el("tr", {}, el("td", {}, c.focus), el("td", { class: "num" }, c.score.heldout_loss.toFixed(4)), el("td", { class: "num" }, scoreCell(c.score)), el("td", {}, c.selected ? "selected" : (c.reasons.join("; ") || "eligible; another candidate was better"))));
      t.append(b);
      d.append(t);
      for (const [p, exp, got] of ep.sample_replies) d.append(el("p", { class: "note" }, `${p.replace(/\n/g, " ⏎ ")} → expected "${exp}", replied "${got}"`));
      return d;
    }
    return el("div", { class: "record note" }, el("span", { class: "when" }, when(r.time)), e?.type === "session" ? `${e.event.type} (lesson ${e.index + 1})` : (r.type || e?.type || "record"));
  }));
}
$("refreshHistory").addEventListener("click", safe(loadHistory, "Library"));

// ---- start ---------------------------------------------------------------------------------
async function boot() {
  if (!tauri?.core) { say("Open this page inside the Yendory Studio app."); return; }
  await tauri.event.listen("teach-event", ev => { try { onTeachEvent(ev.payload); } catch (err) { log("ERROR", err); } });
  await tauri.event.listen("teach-finished", ev => onTeachFinished(ev.payload));
  const s = await invoke("status");
  showActive(s.active);
  setTeaching(s.teaching);
  await refreshModels();
  await refreshLessons();
  say(`Yendory Studio ${s.engine} · every reply and measurement comes from the model itself.`);
}
document.addEventListener("DOMContentLoaded", safe(boot, "Startup"));
