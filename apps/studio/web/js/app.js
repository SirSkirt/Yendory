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
  window.onPage?.(page);
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
  const latest = new Map(meta.taught.map(t => [t.sha256, t.outcome]));
  const outcomes = [...latest.values()];
  const older = outcomes.filter(o => o === "learned").length;
  $("activeLessons").textContent = `${fmt(outcomes.filter(o => o === "passed").length)} passed · ${fmt(outcomes.filter(o => ["unresolved", "progressed", "kept_previous"].includes(o)).length)} unresolved` +
    (older ? ` · ${fmt(older)} taught before pass marks` : "");
  $("activeOrigin").textContent = meta.origin === "new" ? "created in Studio" : meta.origin || "—";
  const limit = meta.wordLimit;
  $("activeVocab").textContent = `${fmt(meta.words)} words learned · ${fmt(meta.slots)} slots · ` +
    (limit == null ? "fixed (no slots are added)" : `limit ${fmt(limit)} (a slot is added for each word learned)`);
  $("readMaxWords").value = Math.floor(0.75 * (limit ?? meta.slots));
  $("activeContext").textContent = `${fmt(meta.context)} tokens (replies up to ${fmt(Math.min(512, Math.max(48, Math.floor(meta.context / 4))))} tokens; conversations keep as much history as fits)`;
  $("readWindow").max = meta.context;
  if (Number($("readWindow").value) > meta.context) $("readWindow").value = meta.context;
  const read = (meta.read || []).filter(r => r.accepted);
  $("activeReading").textContent = read.length
    ? read.map(r => `${r.corpus}: ${fmt(r.tokens)} tokens, held-out loss ${r.baselineLoss.toFixed(3)} → ${r.bestLoss.toFixed(3)}`).join(" · ")
    : "nothing read yet";
}

async function refreshModels() {
  const models = await invoke("list_models");
  const body = $("modelTable").tBodies[0];
  body.replaceChildren();
  if (!models.length) {
    body.append(el("tr", {}, el("td", { colspan: 6, class: "empty" }, "No models yet. Create one or import a checkpoint.")));
    return;
  }
  for (const m of models) {
    const isOpen = active?.id === m.id;
    const row = el("tr", { class: isOpen ? "current-row" : "" },
      el("td", {}, m.name),
      el("td", { class: "num" }, millions(m.parameters)),
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
// ---- model sizes ---------------------------------------------------------------------------
const millions = n => (n / 1e6).toFixed(n < 1e7 ? 2 : 1) + "M";
const bytes = n => n >= 1 << 30 ? (n / (1 << 30)).toFixed(1) + " GB" : Math.round(n / (1 << 20)) + " MB";
let sizes = [];
let chosenSize = "nova2";
function speedCell(s) {
  const m = s.measured;
  const button = el("button", { class: "quiet", onclick: safe(async () => {
    button.disabled = true;
    button.textContent = "measuring…";
    try {
      s.measured = await invoke("measure_speed", { size: s.id });
      renderSizes();
      say(`Measured ${s.name} on this device.`);
    } finally { button.disabled = false; }
  }, "Measure speed") }, m ? "measure again" : "Measure");
  if (!m) return el("td", {}, el("span", { class: "note" }, "not measured "), button);
  const perMillion = 1e6 / m.training / 60;
  const reading = perMillion < 90 ? `${Math.ceil(perMillion)} min` : `${(perMillion / 60).toFixed(1)} h`;
  return el("td", {},
    el("div", {}, `training ${fmt(Math.round(m.training))} tok/s · replies ${fmt(Math.round(m.generation))} tok/s `, button),
    el("div", { class: "note", title: `Measured with ${m.threads} threads. Foundation lessons train four candidates, so an epoch takes about 4× as long as Joint-only.` },
      `reading 1M tokens ≈ ${reading}`));
}
function renderSizes() {
  const body = $("sizeTable").tBodies[0];
  body.replaceChildren(...sizes.map(s => {
    const radio = el("input", { type: "radio", name: "modelSize", value: s.id, "aria-label": s.name });
    radio.checked = s.id === chosenSize;
    radio.addEventListener("change", () => { chosenSize = s.id; });
    const share = g => (100 * g / s.parameters).toFixed(1) + "%";
    return el("tr", { class: s.id === chosenSize ? "current-row" : "" },
      el("td", {}, radio),
      el("td", {}, s.name),
      el("td", { class: "num" }, fmt(s.parameters)),
      el("td", { class: "num" }, s.groups.map(share).join(" · ")),
      el("td", { class: "num" }, bytes(s.fileBytes)),
      el("td", { class: "num" }, [s.chatBytes, s.readingBytes, s.teachingBytes].map(bytes).join(" · ")),
      speedCell(s));
  }));
  body.querySelectorAll("input[name=modelSize]").forEach(r => r.addEventListener("change", () => { renderSizes(); safe(updateVocabNote, "Vocabulary")(); }));
}
function chosenContext() {
  const v = $("contextChoice").value;
  const n = Number(v === "custom" ? $("contextCustom").value : v);
  if (!(Number.isInteger(n) && n >= 16)) throw new Error("Enter a context of at least 16 tokens");
  return n;
}
$("contextChoice").addEventListener("change", safe(updateVocabNote, "Context"));
$("contextCustom").addEventListener("input", safe(updateVocabNote, "Context"));
$("measureContext").addEventListener("click", safe(async () => {
  const context = chosenContext();
  const b = $("measureContext");
  b.disabled = true;
  b.textContent = "measuring… (long contexts take a while)";
  try {
    const rate = await invoke("measure_context", { size: chosenSize, context });
    const perMillion = 1e6 / rate / 60;
    const s = sizes.find(x => x.id === chosenSize);
    $("contextNote").textContent = `Measured on this device: ${s ? s.name : chosenSize} at ${fmt(context)} tokens trains at ${fmt(Math.round(rate))} tokens/s (reading 1M tokens ≈ ${perMillion < 90 ? Math.ceil(perMillion) + " min" : (perMillion / 60).toFixed(1) + " h"}). A longer context adds no parameters; attention compares every token with every earlier one, so the cost per token grows with the context.`;
  } finally { b.disabled = false; b.textContent = "Measure training at this context"; }
}, "Measure context"));
function chosenLimit() {
  const v = $("vocabChoice").value;
  if (v === "fixed") return null;
  const n = Number(v === "custom" ? $("vocabCustom").value : v);
  if (!(Number.isInteger(n) && n >= 1)) throw new Error("Enter a vocabulary limit of at least 1 word");
  return n;
}
async function updateVocabNote() {
  $("vocabCustom").hidden = $("vocabChoice").value !== "custom";
  $("contextCustom").hidden = $("contextChoice").value !== "custom";
  let limit, context;
  try { limit = chosenLimit(); context = chosenContext(); } catch (e) { $("costNote").textContent = e.message; return; }
  const c = await invoke("model_cost", { size: chosenSize, limit, context });
  $("vocabNote").replaceChildren(limit === null
    ? "Nova-2's layout: 256 word slots reserved from the start. Reading and lessons fill them; the limit can be raised later."
    : el("span", {}, limit > c.maxWords ? el("span", { class: "bad" }, `This size allows up to ${fmt(c.maxWords)} words. `) : "",
      `Slots are added only as words are learned; each word adds ${fmt(c.perWord)} parameters. Training slows as words are learned: the output layer scores every learned word for every token.`));
  $("costNote").replaceChildren(el("b", {}, "With these choices"),
    `${limit === null ? "" : ` and every word up to the limit learned`}: ${fmt(c.parameters)} parameters, file ${bytes(c.fileBytes)}; memory ${[c.chatBytes, c.readingBytes, c.teachingBytes].map(bytes).join(" · ")} (talking · reading · teaching), counting ${c.threads} training rows of ${fmt(context)} tokens (${bytes(c.rowBytes)} each).`);
}
$("vocabChoice").addEventListener("change", safe(updateVocabNote, "Vocabulary"));
$("vocabCustom").addEventListener("input", safe(updateVocabNote, "Vocabulary"));
$("createModel").addEventListener("click", safe(async () => {
  const name = $("newName").value.trim();
  if (!name) { $("newName").focus(); throw new Error("Give the new model a name"); }
  showActive(await invoke("create_model", { name, size: chosenSize, wordLimit: chosenLimit(), context: chosenContext() }));
  $("newName").value = "";
  await refreshModels();
  say(`Created ${active.name}: an untrained Bicameral model with ${fmt(active.parameters)} parameters.`);
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
$("vocabLimit").addEventListener("click", safe(async () => {
  if (!active) return;
  const answer = prompt(`Vocabulary limit for ${active.name}: the most words it may learn. It knows ${fmt(active.words)} now and has ${fmt(active.slots)} slots. Leave empty to keep only the slots it has.`, active.wordLimit ?? "");
  if (answer === null) return;
  const limit = answer.trim() === "" ? null : Number(answer.replace(/[ ,_]/g, ""));
  if (limit !== null && !(Number.isInteger(limit) && limit >= 0)) throw new Error("Enter a whole number of words");
  showActive(await invoke("set_word_limit", { id: active.id, limit }));
  say(limit === null ? "Vocabulary fixed at the slots the model has." : `Vocabulary limit set to ${fmt(limit)} words.`);
}, "Vocabulary limit"));
$("contextLimit").addEventListener("click", safe(async () => {
  if (!active) return;
  const answer = prompt(`Context for ${active.name} in tokens (now ${fmt(active.context)}). No weights change; the model uses a longer context well once it has read or been taught text that long.`, active.context);
  if (answer === null || answer.trim() === "") return;
  const context = Number(answer.replace(/[ ,_]/g, ""));
  if (!(Number.isInteger(context) && context >= 16)) throw new Error("Enter a context of at least 16 tokens");
  showActive(await invoke("set_context", { id: active.id, context }));
  say(`Context set to ${fmt(context)} tokens.`);
}, "Context"));
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
  if (l.alreadyTaught) return el("span", { class: "warn" }, "already passed by this model");
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
  $("read").disabled = on;
  $("stopRead").disabled = !on;
  for (const id of ["addLessons", "clearLessons", "createModel", "importModel"]) $(id).disabled = on;
  renderLessons();
  renderCorpora();
  window.onTeachingChanged?.(on);
}

let progress = { files: 1, index: 0, epochs: 1, candidates: 1, candidate: 0, epoch: 1 };
const FOCI = ["dense", "transformer", "interconnect", "joint"];
// Candidates of each bicameral education stage, in the Engine's order (with Foundation, each
// part the stage may train is also tried on its own).
function stageFoci(stage, foundation) {
  if (!stage || stage === "off") return foundation ? FOCI : ["joint"];
  if (stage === "transformer") return ["transformer"];
  if (stage === "dense") return foundation ? ["dense", "interconnect", "dense+interconnect"] : ["dense+interconnect"];
  return foundation ? ["dense", "transformer", "interconnect", "dense+interconnect", "joint"] : ["dense+interconnect", "joint"];
}
$("teach").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open or create a model first");
  const options = {
    epochs: Number($("optEpochs").value),
    batchSize: Number($("optBatch").value),
    learningRate: Number($("optRate").value),
    foundation: $("optFoundation").checked,
    rehearsal: $("optRehearsal").checked,
    protectAnswers: $("optProtect").checked,
    passPercent: Number($("optPass").value),
    retries: $("optRetries").checked,
    requeue: $("optRequeue").checked,
    education: $("optEducation").value,
  };
  progress = { files: lessonsList.length, done: 0, index: 0, epochs: options.epochs, candidates: stageFoci(options.education, options.foundation).length, foci: stageFoci(options.education, options.foundation), candidate: 0, epoch: 1, attempt: "" };
  for (const k of Object.keys(fileState)) delete fileState[k];
  $("candidateTable").tBodies[0].replaceChildren();
  $("courseTable").tBodies[0].replaceChildren();
  $("progressEntry").hidden = false;
  $("progressTitle").textContent = `Teaching ${active.name}`;
  await invoke("teach", { options });
  setTeaching(true);
  say("Teaching started.");
}, "Teach"));
$("optEducation").addEventListener("change", () => {
  const f = stageFoci($("optEducation").value, true).join(", ");
  $("optFoundation").parentElement.title = $("optEducation").value === "off" ? "" : `In this stage Foundation tries: ${f}`;
});
$("stopTeach").addEventListener("click", safe(async () => {
  await invoke("stop_teaching");
  $("stopTeach").disabled = true;
  say("Stopping at the next batch; the epoch in progress will be discarded.");
}, "Stop"));

const OUTCOME_CLASS = { passed: "ok", learned: "ok", progressed: "warn", deferred: "warn", kept_previous: "bad", unresolved: "bad", duplicate: "warn", failed: "bad", stopped: "warn" };
const OUTCOME_LABEL = { passed: "passed", learned: "learned", progressed: "progressed, not passed", deferred: "not passed yet: moved to the end of the queue", kept_previous: "not learned; previous model kept", unresolved: "unresolved: attempts used up", duplicate: "duplicate", failed: "not taught", stopped: "stopped" };
function scoreCell(s) { return s ? `${s.exact_correct}/${s.exact_total}` : "—"; }
function onTeachEvent(e) {
  if (e.type === "file_started") {
    progress.index = e.index;
    fileState[e.index] = el("span", { class: "tag" }, e.pass > 1 ? "trying again…" : "teaching…");
    renderLessons();
    $("progressTitle").textContent = `Teaching ${active?.name ?? ""} · lesson ${e.index + 1} of ${e.files}: ${e.name}${e.pass > 1 ? " (second try, after other lessons)" : ""}`;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7 }, el("b", {}, `Lesson ${e.index + 1}: ${e.name}`))));
  } else if (e.type === "attempt_started") {
    progress.epochs = e.epochs;
    const what = e.attempt === 0 ? "first attempt" : `retry ${e.attempt} of ${e.attempts - 1}${e.remedy ? ": " + e.remedy : ""}`;
    progress.attempt = `${what} · rate ${e.learning_rate.toExponential(1)} · ${e.strategy} study · ${e.epochs} epochs`;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7, class: "note" }, progress.attempt)));
  } else if (e.type === "diagnosed") {
    const d = e.diagnosis;
    const cause = { unstable: "unstable training", forgetting: "harming earlier lessons", interference: "new learning overwrites known answers", not_learned: "not learned yet", not_generalising: "memorised, not generalised" }[d.cause] || d.cause;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7 }, el("span", { class: "warn" },
      `Not passed (held-out ${(100 * d.heldout_rate).toFixed(0)}%, training questions ${(100 * d.train_rate).toFixed(0)}%) → diagnosis: ${cause}. Next: ${d.remedy}`))));
  } else if (e.type === "session") {
    const s = e.event;
    if (s.type === "started") {
      $("progressLine").textContent = `Baseline: held-out loss ${s.baseline.heldout_loss.toFixed(4)}, exact ${scoreCell(s.baseline)}; ${fmt(s.train_units)} units to learn, ${fmt(s.heldout_units)} held out, ${fmt(s.new_words)} new word units.`;
    } else if (s.type === "batch") {
      progress.epoch = s.epoch;
      progress.candidate = Math.max(0, progress.foci.indexOf(s.focus));
      if (progress.candidates === 1) progress.candidate = 0;
      const inFile = (((s.epoch - 1) * progress.candidates + progress.candidate) * s.batches + s.batch) / (progress.epochs * progress.candidates * s.batches);
      $("progressBar").style.width = (100 * Math.min(1, (progress.done + Math.min(1, inFile)) / progress.files)).toFixed(1) + "%";
      $("progressLine").textContent = `${progress.attempt} — epoch ${s.epoch}/${progress.epochs} · ${s.focus} candidate · batch ${s.batch}/${s.batches} · loss ${s.loss.toFixed(4)} · learning rate ${s.learning_rate.toExponential(2)}`;
    } else if (s.type === "candidate") {
      const decision = s.eligible
        ? (s.reasons.length ? el("span", { class: "warn" }, s.reasons.join("; ")) : el("span", { class: "ok" }, "eligible"))
        : el("span", { class: "bad" }, s.reasons.join("; "));
      $("candidateTable").tBodies[0].append(el("tr", { id: `cand-${e.index}-${s.epoch}-${s.focus}` },
        el("td", { class: "num" }, s.epoch), el("td", {}, s.focus),
        el("td", { class: "num" }, s.train_loss.toFixed(4)), el("td", { class: "num", title: s.score.dense_reliance == null ? "" : "Whole brain; in brackets the transformer alone (Dense's input cut) and Dense's contribution (the difference)" }, s.score.heldout_loss.toFixed(4), s.score.dense_reliance == null ? "" : el("div", { class: "note" }, `without Dense ${s.score.without_dense_loss.toFixed(4)} · Dense +${s.score.dense_reliance.toFixed(3)}`)),
        el("td", { class: "num" }, scoreCell(s.score)),
        el("td", { class: "num" }, s.score.retention_loss == null ? "—" : s.score.retention_loss.toFixed(4)),
        el("td", {}, decision)));
    } else if (s.type === "epoch") {
      for (const c of s.candidates) {
        const row = [...document.querySelectorAll(`[id="cand-${e.index}-${s.epoch}-${c.focus}"]`)].pop();
        if (!row) continue;
        if (c.selected) { row.classList.add("current-row"); row.lastChild.replaceChildren(el("b", { class: "ok" }, "selected")); }
        else if (c.eligible && !c.reasons.length) row.lastChild.replaceChildren(el("span", {}, "eligible; another candidate was better"));
      }
      // The whole brain after this epoch (the accepted model): progress at a glance.
      const a = s.accepted_score;
      $("candidateTable").tBodies[0].append(el("tr", { class: "epoch-summary" }, el("td", { colspan: 7 },
        el("b", {}, `Whole brain after epoch ${s.epoch}: `),
        `exact ${scoreCell(a)} · held-out loss ${a.heldout_loss.toFixed(4)}`,
        a.dense_reliance == null ? "" : ` · transformer alone ${a.without_dense_loss.toFixed(4)} · Dense contribution ${a.dense_reliance.toFixed(3)}`,
        ` · ${s.selected ? "kept " + s.selected : "no candidate kept"}`)));
      log("INFO", `Epoch ${s.epoch}: ${s.selected ? "selected " + s.selected : "no candidate selected"}`);
    }
  } else if (e.type === "file_finished") {
    const cls = OUTCOME_CLASS[e.outcome] || "";
    const label = e.outcome === "unresolved" ? (e.attempts?.some(a => a.learned) ? "unresolved: attempts used up, progress kept" : "unresolved: attempts used up, nothing learned")
      : OUTCOME_LABEL[e.outcome] || e.outcome;
    if (e.outcome !== "deferred") progress.done += 1;
    fileState[e.index] = el("span", { class: cls }, label);
    renderLessons();
    const body = $("courseTable").tBodies[0];
    body.append(el("tr", {},
      el("td", {}, e.name),
      el("td", {}, el("span", { class: cls }, label),
        e.attempts?.length ? el("div", { class: "note" }, `Pass mark ${Math.round(100 * (e.pass_mark ?? 0))}% · ${e.attempts.length} attempt(s)${e.pass > 1 ? ", second pass" : ""}: ` + e.attempts.map(a => `${a.passed ? "passed" : a.learned ? "improved" : "no change"}${a.because ? " after " + a.because.cause.replace("_", " ") : ""} (${(100 * a.final_score.exact_correct / Math.max(1, a.final_score.exact_total)).toFixed(0)}%)`).join(" → ")) : "",
        e.error ? el("div", { class: "note" }, e.error) : ""),
      el("td", { class: "num" }, `${fmt(e.start_step)} → ${fmt(e.end_step)}`),
      el("td", { class: "num" }, `${scoreCell(e.baseline)} → ${scoreCell(e.final_score)}`),
      el("td", { class: "mono", title: e.sha256 }, e.sha256.slice(0, 12))));
  }
}
function onTeachFinished(f) {
  setTeaching(false);
  if (f.model) showActive(f.model);
  $("progressBar").style.width = "100%";
  const passed = f.files.filter(x => x.outcome === "passed").length;
  const unresolved = f.files.filter(x => x.outcome === "unresolved" || x.outcome === "progressed" || x.outcome === "kept_previous").length;
  const text = f.error ? `Teaching ended with an error: ${f.error}`
    : `${f.stopped ? "Stopped" : "Finished"}: ${passed} of ${f.files.length} lesson file(s) passed, ${unresolved} unresolved (their attempts are used up). Every accepted improvement is saved.`;
  $("progressTitle").textContent = text;
  if (f.error) log("ERROR", f.error); else say(text);
  refreshModels();
}

// ---- reading (corpus pre-training) ---------------------------------------------------------
let sourcesList = [];
let corporaList = [];
let familiesList = [];
const mb = bytes => (bytes / 1048576).toFixed(1) + " MB";
const CATEGORY = { general: "Stories and prose", knowledge: "Knowledge", conversation: "Conversation", reasoning: "Reasoning" };
const BUDGET = 1024 * 1048576;
// Reading choices per corpus (kept while the list is refreshed).
const mix = {};  // id -> { read: bool, share: number }

function renderSources() {
  const groups = ["knowledge", "conversation", "reasoning", "general"];
  $("sourceList").replaceChildren(...groups.map(g => el("div", { class: "source-group" },
    el("h5", {}, CATEGORY[g]),
    ...sourcesList.filter(s => s.category === g).map(s => {
      const size = el("input", { type: "number", min: 1, max: Math.min(400, s.sizeMb), value: s.defaultMb, title: "Megabytes to download" });
      return el("div", { class: "source" },
        el("b", {}, s.name),
        s.sizeMb > s.defaultMb ? el("span", {}, "first ", size, ` MB of about ${fmt(s.sizeMb)} MB`) : el("span", {}, `about ${s.sizeMb} MB`),
        el("button", { onclick: safe(async () => {
          const want = s.sizeMb > s.defaultMb ? Number(size.value) || s.defaultMb : s.sizeMb;
          const used = corporaList.reduce((a, c) => a + c.bytes, 0);
          if (used + want * 1048576 > BUDGET && !confirm(`This would bring the corpora on this device to about ${mb(used + want * 1048576)}, over the 1 GB budget. Download anyway?`)) return;
          await invoke("download_corpus", { source: s.id, maxMb: want });
          $("downloadEntry").hidden = false;
          $("downloadTitle").textContent = `Downloading ${s.name}`;
          $("downloadBar").style.width = "0%";
          $("downloadLine").textContent = "Connecting…";
          $("cancelDownload").hidden = false;
        }, "Download") }, "Download"),
        el("p", { class: "note" }, `${s.description} `, el("b", {}, "Licence: "), `${s.license}. `, el("b", {}, "Credit: "), `${s.attribution}. Source: ${s.page}`));
    }))));
}
function renderFamilies() {
  $("familyList").replaceChildren(el("span", { class: "note" }, "Kinds: "), ...familiesList.map(f =>
    el("label", { class: "family" }, el("input", { type: "checkbox", value: f.id, checked: "" }), f.name)));
}
function renderCorpora() {
  const body = $("corpusTable").tBodies[0];
  body.replaceChildren();
  if (!corporaList.length) body.append(el("tr", {}, el("td", { colspan: 8, class: "empty" }, "No corpus yet. Download one, generate practice or use a text file.")));
  for (const c of corporaList) {
    const m = mix[c.id] ??= { read: false, share: 25 };
    const read = el("input", { type: "checkbox", checked: m.read ? "" : null, onchange: e => { m.read = e.target.checked; updateReadNote(); } });
    const share = el("input", { type: "number", class: "share", min: 1, max: 100, value: m.share, onchange: e => { m.share = Number(e.target.value) || 1; updateReadNote(); } });
    body.append(el("tr", {},
      el("td", {}, read),
      el("td", {}, c.name, c.droppedChars ? el("div", { class: "note" }, `${fmt(c.droppedChars)} non-ASCII characters dropped`) : "",
        c.attribution ? el("div", { class: "note" }, "Credit: " + c.attribution) : ""),
      el("td", {}, CATEGORY[c.category] || c.category),
      el("td", { class: "num" }, share),
      el("td", { class: "num" }, mb(c.bytes)),
      el("td", { class: "num" }, `${fmt(c.documents)} (${fmt(c.heldOut)} held out)`),
      el("td", {}, c.license),
      el("td", {}, el("button", { class: "quiet", disabled: teaching ? "" : null, onclick: safe(async () => {
        if (!confirm(`Delete the corpus "${c.name}" from this device?`)) return;
        corporaList = await invoke("delete_corpus", { id: c.id }); renderCorpora();
      }, "Delete corpus") }, "delete"))));
  }
  const used = corporaList.reduce((a, c) => a + c.bytes, 0);
  $("corpusBudget").textContent = `Corpora on this device: ${mb(used)} of the 1 GB budget.`;
  $("corpusBudget").className = used > BUDGET ? "warn" : "note";
  updateReadNote();
}
const chosen = () => corporaList.filter(c => mix[c.id]?.read);
function updateReadNote() {
  const steps = Math.ceil(Number($("readTokens").value) / (Number($("readBatch").value) * Number($("readWindow").value)));
  const parts = chosen();
  const total = parts.reduce((a, c) => a + mix[c.id].share, 0);
  const shares = parts.map(c => `${c.name} ${Math.round(100 * mix[c.id].share / total)}%`).join(", ");
  $("readNote").textContent = !parts.length ? "Tick the corpora to read in the table above."
    : Number.isFinite(steps) && steps > 0
      ? `${fmt(steps)} training steps over ${shares}, with about 20 held-out checks. Only checkpoints that lower the held-out loss are kept.`
      : "—";
}
for (const id of ["readTokens", "readBatch", "readWindow"]) $(id).addEventListener("input", updateReadNote);
// Bicameral reading: the stage each kind of text is read in.
const STAGE_OPTIONS = [["dense", "Stage A: Dense"], ["transformer", "Stage B: Transformer"], ["cooperative", "Stage C: Co-op"], ["off", "Whole brain"]];
const STAGE_DEFAULT = { knowledge: "dense", conversation: "transformer", reasoning: "cooperative", general: "transformer" };
for (const sel of document.querySelectorAll("#stageMap select")) {
  sel.replaceChildren(...STAGE_OPTIONS.map(([v, t]) => el("option", { value: v }, t)));
  sel.value = STAGE_DEFAULT[sel.dataset.cat];
}
$("readMode").addEventListener("change", () => { $("stageMap").hidden = $("readMode").value !== "bicameral"; });
async function refreshCorpora() { corporaList = await invoke("corpora"); renderCorpora(); }
function pick(id) { (mix[id] ??= { read: true, share: 25 }).read = true; }
$("importCorpus").addEventListener("click", safe(async () => {
  say("Reading the file…");
  const c = await invoke("import_corpus", { category: $("importCategory").value });
  if (!c) return;
  pick(c.id);
  await refreshCorpora();
  say(`Added ${c.name}: ${fmt(c.documents)} documents, ${fmt(c.heldOut)} held out.`);
}, "Use text file"));
$("generate").addEventListener("click", safe(async () => {
  const families = [...document.querySelectorAll("#familyList input:checked")].map(i => i.value);
  if (!families.length) throw new Error("Tick at least one kind of practice");
  $("generate").disabled = true;
  try {
    say("Generating practice…");
    const c = await invoke("generate_corpus", { families, difficulty: Number($("genDifficulty").value), mb: Number($("genMb").value) });
    pick(c.id);
    await refreshCorpora();
    say(`Generated ${c.name}: ${fmt(c.documents)} examples (${fmt(c.heldOut)} held out).`);
  } finally { $("generate").disabled = false; }
}, "Generate practice"));
$("cancelDownload").addEventListener("click", safe(() => invoke("cancel_download"), "Cancel download"));
function onDownload(e) {
  if (!e.done) {
    $("downloadBar").style.width = e.total ? (100 * e.received / e.total).toFixed(1) + "%" : "50%";
    $("downloadLine").textContent = `${mb(e.received)}${e.total ? " of " + mb(e.total) : ""} received`;
    return;
  }
  $("cancelDownload").hidden = true;
  if (e.error) { $("downloadLine").textContent = e.error; log("ERROR", e.error); return; }
  $("downloadBar").style.width = "100%";
  const c = e.corpus;
  $("downloadLine").textContent = `Saved ${c.name}: ${mb(c.bytes)}, ${fmt(c.documents)} documents (${fmt(c.heldOut)} held out).`;
  pick(c.id);
  refreshCorpora();
  say(`Downloaded ${c.name}.`);
}

let readStart = 0;
let readParts = [];
$("read").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open or create a model first");
  const parts = chosen();
  if (!parts.length) throw new Error("Tick the corpora to read (download one, generate practice or use a text file first)");
  const stages = {};
  for (const sel of document.querySelectorAll("#stageMap select")) stages[sel.dataset.cat] = sel.value;
  const options = {
    parts: parts.map(c => ({ corpusId: c.id, share: mix[c.id].share })),
    tokens: Number($("readTokens").value),
    batchSize: Number($("readBatch").value),
    learningRate: Number($("readRate").value),
    window: Number($("readWindow").value),
    growVocabulary: $("readWords").checked,
    maxWords: Number($("readMaxWords").value),
    mode: $("readMode").value,
    stages,
  };
  readParts = parts;
  $("evalTable").tBodies[0].replaceChildren();
  $("readEntry").hidden = false;
  $("readBar").style.width = "0%";
  $("readTitle").textContent = `${active.name} is reading ${parts.map(c => c.name).join(" + ")}${options.mode === "bicameral" ? " (bicameral)" : ""}`;
  $("readLine").textContent = "Preparing the corpora…";
  await invoke("read_corpus", { options });
  readStart = 0;
  setTeaching(true);
  say("Reading started.");
}, "Read"));
$("stopRead").addEventListener("click", safe(async () => {
  await invoke("stop_teaching");
  $("stopRead").disabled = true;
  say("Stopping at the next batch; the best checkpoint so far is kept.");
}, "Stop reading"));

function onReadEvent(e) {
  if (e.type === "started") {
    readStart = performance.now();
    readParts = e.parts || [];
    const stageName = { dense_knowledge: "Stage A", transformer_language: "Stage B", cooperative: "Stage C" };
    const parts = readParts.map(p => `${p.name} ${Math.round(100 * p.share)}%${p.stage ? " in " + stageName[p.stage] : ""}`).join(", ");
    $("readLine").textContent = `${fmt(e.train_documents)} documents to read (${fmt(e.train_tokens)} tokens), ${fmt(e.held_out_documents)} held out; ${fmt(e.new_words)} new word units; ${fmt(e.steps)} steps${parts ? " · " + parts : ""}. Starting held-out loss ${e.baseline_loss.toFixed(4)}.`;
    $("evalTable").tBodies[0].append(el("tr", {}, el("td", { class: "num" }, "0"), el("td", {}, ""), el("td", { class: "num" }, e.baseline_loss.toFixed(4)), el("td", { class: "num" }, (e.baseline_dense_reliance ?? 0).toFixed(3)), el("td", {}, "starting point"), el("td", {}, "")));
  } else if (e.type === "batch") {
    $("readBar").style.width = (100 * e.step / e.steps).toFixed(1) + "%";
    const secs = (performance.now() - readStart) / 1000;
    const rate = e.tokens / Math.max(secs, 1e-3);
    const left = (e.steps - e.step) * (e.tokens / e.step) / rate;
    $("readLine").textContent = `step ${fmt(e.step)}/${fmt(e.steps)}${readParts.length > 1 ? " · " + (readParts[e.part]?.name ?? "") : ""} · ${fmt(e.tokens)} tokens · loss ${e.loss.toFixed(4)} · rate ${e.learning_rate.toExponential(2)} · ${fmt(Math.round(rate))} tokens/s · about ${Math.ceil(left / 60)} min left`;
  } else if (e.type === "evaluated") {
    $("evalTable").tBodies[0].append(el("tr", { class: e.accepted ? "current-row" : "" },
      el("td", { class: "num" }, fmt(e.tokens)),
      el("td", { class: "num" }, e.train_loss.toFixed(4)),
      el("td", { class: "num" }, e.held_out_loss.toFixed(4)),
      el("td", { class: "num" }, (e.dense_reliance ?? 0).toFixed(3)),
      el("td", {}, e.accepted ? el("b", { class: "ok" }, "kept (new best)") : el("span", { class: "warn" }, e.reason || `not better than ${e.best_loss.toFixed(4)}`)),
      el("td", { class: "sample" }, ...(e.parts?.length ? e.parts : [{ name: "", held_out_loss: e.held_out_loss, sample_prompt: e.sample_prompt, sample: e.sample }]).map(p =>
        el("div", { class: "part-line" }, p.name ? el("b", {}, `${CATEGORY[p.category] || ""}${p.name ? " · " + p.name : ""}: ${p.held_out_loss.toFixed(4)} `) : "",
          el("span", { class: "note" }, p.sample_prompt.replace(/\n/g, " ⏎ ")), " ", p.sample)))));
  } else if (e.type === "finished" && e.early_stop) {
    log("INFO", "Reading stopped early: the held-out loss stopped improving.");
  }
}
function onReadFinished(f) {
  setTeaching(false);
  if (f.model) showActive(f.model);
  $("readBar").style.width = "100%";
  const text = f.error ? `Reading ended with an error: ${f.error}`
    : f.accepted ? `${f.stopped ? "Stopped" : f.earlyStop ? "Finished early (no further improvement)" : "Finished"} after ${fmt(f.tokens)} tokens: held-out loss ${f.baselineLoss.toFixed(4)} → ${f.bestLoss.toFixed(4)}. The best checkpoint is saved.`
    : `${f.stopped ? "Stopped" : "Finished"}: no checkpoint lowered the held-out loss, so the model is unchanged.`;
  $("readTitle").textContent = text;
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
    (() => {
      const cut = Object.values(r.reportWithoutDense.exact).reduce((a, s) => ({ c: a.c + s.correct, t: a.t + s.total }), { c: 0, t: 0 });
      return el("p", { class: "note" }, `With Dense's input to the transformer cut: ${pct(cut.c, cut.t)} exact (${cut.c}/${cut.t}). The difference is what the answers owe to the dense hemisphere.`);
    })(),
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
function showMeasure(m, caption) {
  lastMeasure = m;
  const sel = $("stageSelect");
  sel.replaceChildren(...m.trace.stages.map((s, i) => el("option", { value: i }, Brain.stageLabel(s))));
  sel.value = String(m.trace.stages.length - 1);
  renderMeasure();
  $("brainDetail").textContent = caption;
}
$("inspect").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open a model first");
  const m = await invoke("measure", { prompt: $("probe").value });
  showMeasure(m, `Measured at step ${fmt(m.step)} while reading the last token of "U: ${$("probe").value}". Click a dot for its value.`);
}, "Inspect"));

// ---- thinking tree -------------------------------------------------------------------------
const bar = (v, cls = "") => el("span", { class: "tt-bar " + cls, title: (100 * v).toFixed(1) + "%" }, (() => { const i = el("i"); i.style.width = (100 * Math.max(0, Math.min(1, v))).toFixed(1) + "%"; return i; })());
const p100 = v => (100 * v).toFixed(v >= 0.995 || v < 0.0005 ? 0 : 1) + "%";
const tokText = t => t.token;

function renderThinking(tree, prompt) {
  const box = $("thinkTree");
  box.replaceChildren();
  box.classList.toggle("no-branches", !$("thinkBranches").checked);
  box.append(el("div", { class: "tt-root" }, `U: ${prompt}`),
    el("p", { class: "note" }, `Model at training step ${fmt(tree.step)} · ${tree.steps.length} token(s) · ${tree.rounds} thinking round(s) per token · ${tree.branches.length} path(s) not taken shown. Click a step for its full record.`));
  const trunk = el("div", { class: "tt-trunk" });
  const branchesAt = new Map();
  for (const b of tree.branches) branchesAt.set(b.at_step, [...(branchesAt.get(b.at_step) || []), b]);
  for (const s of tree.steps) {
    const unsure = s.alternatives[0] && s.alternatives[0].probability >= 0.05;
    const rounds = el("span", { class: "tt-rounds", title: "What the model would have output after the lower stage and after each thinking round" },
      ...s.readouts.flatMap((r, i) => [i ? " → " : "", `${r.stage.replace(" (output)", "")} `, el("b", {}, tokText(r.top[0])), ` ${p100(r.top[0].probability)}`]));
    const detail = el("div", { class: "tt-detail" },
      el("table", {},
        el("tr", {}, el("th", {}, "Chosen"), el("td", {}, `${s.chosen.token}  ${p100(s.chosen.probability)}  (uncertainty ${s.entropy.toFixed(2)} bits)`)),
        el("tr", {}, el("th", {}, "Also weighed"), el("td", {}, s.alternatives.map(a => `${a.token} ${p100(a.probability)}`).join(" · ") || "—")),
        ...s.readouts.map(r => el("tr", {}, el("th", {}, `After ${r.stage}`), el("td", {}, r.top.map(c => `${c.token} ${p100(c.probability)}`).join(" · ")))),
        el("tr", {}, el("th", {}, "Gate openings (exchange, rounds)"), el("td", {}, s.gate_means.map(g => g.toFixed(2)).join(" · "))),
        el("tr", {}, el("th", {}, "Dense hemisphere's influence"), el("td", {}, `${s.right_influence.toFixed(3)} (change of this decision when its input to the left is cut)`)),
        el("tr", {}, el("th", {}, "Attended most (last block)"), el("td", {}, s.attended.map(([t, w]) => `${t} ${p100(w)}`).join(" · ") || "—"))),
      el("div", { class: "actions" }, el("button", { onclick: safe(async () => {
        const m = await invoke("measure_step", { prompt, prefix: tree.path_ids.slice(0, s.index) });
        showMeasure(m, `Step ${s.index + 1} of the traced reply: the forward pass that chose "${s.chosen.token}" (${p100(s.chosen.probability)}). Click a dot for its value.`);
        $("hemisphereBrain").scrollIntoView({ behavior: "smooth", block: "center" });
      }, "Show step in brain scan") }, "Show this step in the brain scan")));
    const node = el("div", { class: "tt-step" },
      el("details", {},
        el("summary", {},
          el("span", { class: "tt-idx" }, s.index + 1),
          el("span", { class: "tt-tok" + (unsure ? " unsure" : ""), title: "chosen token" }, s.chosen.token),
          el("span", { class: "tt-p" }, p100(s.chosen.probability)), bar(s.chosen.probability),
          rounds,
          el("span", { class: "tt-p", title: "How much the dense hemisphere changed this decision" }, "dense "), bar(s.right_influence, "right")),
        detail));
    trunk.append(node);
    for (const b of branchesAt.get(s.index) || []) {
      trunk.append(el("div", { class: "tt-branch" }, "not taken: ",
        el("span", { class: "tt-tok" }, b.first.token), ` ${p100(b.first.probability)} → the model would have continued `,
        el("q", {}, b.text), b.ended ? "" : " …"));
    }
  }
  box.append(trunk, el("div", { class: "tt-leaf" }, `N: ${tree.reply || "(empty reply)"}`));
}
$("thinkButton").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open a model first");
  const prompt = $("probe").value.trim();
  $("thinkButton").disabled = true;
  try {
    say("Tracing the reply… every value is measured from the model.");
    const tree = await invoke("think", { prompt });
    renderThinking(tree, prompt);
    say(`Traced ${tree.steps.length} decision(s).`);
  } finally { $("thinkButton").disabled = false; }
}, "Thinking tree"));
$("thinkBranches").addEventListener("change", () => $("thinkTree").classList.toggle("no-branches", !$("thinkBranches").checked));
$("stageSelect").addEventListener("change", renderMeasure);
$("probe").addEventListener("keydown", e => { if (e.key === "Enter") $("inspect").click(); });

// ---- library -------------------------------------------------------------------------------
async function loadHistory() {
  const box = $("historyList");
  if (!active) { box.replaceChildren(el("p", { class: "empty" }, "Open a model to see its record.")); return; }
  const rows = await invoke("history", { id: active.id });
  if (!rows.length) { box.replaceChildren(el("p", { class: "empty" }, "Nothing has been taught to this model in Studio yet.")); return; }
  // Candidates are shown inside their epoch; start markers add nothing to the record.
  const shown = rows.filter(r => r.type || r.reading?.type === "evaluated" || r.event?.type === "file_finished" || (r.event?.type === "session" && r.event.event.type === "epoch"));
  box.replaceChildren(...shown.reverse().map(r => {
    const e = r.event;
    if (r.type === "reading_started") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)),
      `Reading started${r.mode === "bicameral" ? " (bicameral)" : ""}: ${(r.corpora || [r.corpus]).map(c => `${c.name} (sha256 ${c.sha256.slice(0, 12)}; ${c.license})`).join(" + ")}, up to ${fmt(r.options.tokens)} tokens.`);
    if (r.type === "reading_finished") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), r.error ? el("span", { class: "bad" }, "Reading ended with an error: " + r.error) : `Reading ${r.stopped ? "stopped" : "finished"} after ${fmt(r.tokens)} tokens; ${r.accepted ? "best checkpoint kept" : "model unchanged"}.`);
    if (r.reading?.type === "evaluated") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), `Read ${fmt(r.reading.tokens)} tokens: held-out loss ${r.reading.held_out_loss.toFixed(4)} — ${r.reading.accepted ? "kept" : "not better"}.`, el("div", { class: "note mono" }, `${r.reading.sample_prompt} → ${r.reading.sample}`));
    if (r.type === "course_started") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), `Course started: ${r.files} file(s), ${r.options.epochs} epoch(s), ${r.options.foundation ? "Foundation" : "Joint only"}.`);
    if (r.type === "course_finished") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), r.error ? el("span", { class: "bad" }, "Course ended with an error: " + r.error) : `Course ${r.stopped ? "stopped" : "finished"}.`);
    if (e?.type === "file_finished") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)),
      el("b", {}, e.name), ` — ${e.outcome.replace("_", " ")}${e.pass > 1 ? " (second pass)" : ""}, steps ${fmt(e.start_step)} → ${fmt(e.end_step)}, exact ${scoreCell(e.baseline)} → ${scoreCell(e.final_score)}, ${e.attempts?.length ?? 0} attempt(s)`, e.error ? ` (${e.error})` : "",
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
  await tauri.event.listen("read-event", ev => { try { onReadEvent(ev.payload); } catch (err) { log("ERROR", err); } });
  await tauri.event.listen("read-finished", ev => onReadFinished(ev.payload));
  await tauri.event.listen("corpus-download", ev => onDownload(ev.payload));
  sourcesList = await invoke("corpus_sources");
  renderSources();
  familiesList = await invoke("practice_families");
  renderFamilies();
  await refreshCorpora();
  const s = await invoke("status");
  showActive(s.active);
  setTeaching(s.teaching);
  sizes = await invoke("model_sizes");
  renderSizes();
  await updateVocabNote();
  await refreshModels();
  await refreshLessons();
  say(`Yendory Studio ${s.engine} · every reply and measurement comes from the model itself.`);
}
document.addEventListener("DOMContentLoaded", safe(boot, "Startup"));
