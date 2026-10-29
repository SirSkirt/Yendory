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
  window.showChat?.(meta);
  window.showConversation?.(meta);
  if (!meta) return;
  $("activeName").textContent = meta.name;
  refreshSaved();
  refreshCheckpoints();
  showResume(meta.resume);
  $("activeStep").textContent = fmt(meta.step);
  $("activeWords").textContent = fmt(meta.words);
  const neuroModel = meta.architecture === "neuro";
  // The brain page draws the open model's architecture; a measurement of another model is cleared.
  if (lastMeasure && !!lastMeasure.trace.neuro !== neuroModel) { lastMeasure = null; Brain.clear(); }
  Brain.setArchitecture(neuroModel, meta.design && { ...meta.design, blocks: meta.blocks });
  const label = (id, text) => { const th = $(id).previousElementSibling; if (th) th.textContent = text; };
  label("activeLeft", neuroModel ? "Language core (transformer)" : "Transformer (left)");
  label("activeRight", neuroModel ? "Modules (Nova Neuro)" : "Dense (right)");
  label("activeInter", neuroModel ? "Communication (bus, controller)" : "Interconnect");
  $("exportModel").textContent = neuroModel ? "Export .nneuro…" : "Export .nnova2…";
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
  for (const id of ["readFirstTeach", "readFirstPlan"]) $(id).hidden = read.length > 0;
  $("activeReading").textContent = read.length
    ? read.map(r => `${r.corpus}: ${fmt(r.tokens)} tokens, held-out loss ${r.baselineLoss.toFixed(3)} → ${r.bestLoss.toFixed(3)}`).join(" · ")
    : "nothing read yet";  window.showGrowth?.();
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
      el("td", {}, m.name, m.architecture === "neuro"
        ? el("span", { class: "tag arch arch-neuro", title: "Nova Neuro (experimental)" }, "Neuro")
        : el("span", { class: "tag arch arch-bica", title: "Nova Bicameral" }, "Bica")),
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
  if (!await askYes(`Delete "${m.name}" and its teaching record? Export it first if you want to keep it. This cannot be undone.`)) return;
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
  const onGpu = deviceInfo?.trainingOn === "gpu" && m.gpuTraining;
  const perMillion = 1e6 / (onGpu ? m.gpuTraining : m.training) / 60;
  const reading = perMillion < 90 ? `${Math.ceil(perMillion)} min` : `${(perMillion / 60).toFixed(1)} h`;
  return el("td", {},
    el("div", {}, `training: CPU ${fmt(Math.round(m.training))} tok/s`, m.gpuTraining ? ` · GPU ${fmt(Math.round(m.gpuTraining))} tok/s` : "", ` · replies ${fmt(Math.round(m.generation))} tok/s `, button),
    el("div", { class: "note", title: `Measured with ${m.threads} threads. Foundation lessons train four candidates, so an epoch takes about 4× as long as Joint-only.` },
      `reading 1M tokens ≈ ${reading}${onGpu ? " on the GPU" : ""}`));
}
// ---- training device -----------------------------------------------------------------------
let deviceInfo = null;
function showDevice(d) {
  deviceInfo = d;
  $("deviceChoice").value = d.choice;
  const threads = $("threadChoice");
  if (threads.options.length !== d.cpuTotal) {
    threads.replaceChildren(...[0, ...Array.from({ length: d.cpuTotal - 1 }, (_, i) => i + 1)].map(n => {
      const o = document.createElement("option");
      o.value = n;
      o.textContent = n === 0 ? `All ${d.cpuTotal}` : String(n);
      return o;
    }));
  }
  threads.value = String(d.cpuThreads >= d.cpuTotal ? 0 : d.cpuThreads);
  $("gpuChecks").checked = d.gpuChecks ?? true;
  $("fittedReplies").checked = d.fittedReplies ?? true;
  if (d.options) {
    const fastest = d.benchmark?.result?.fastest;
    const sel = $("deviceChoice");
    sel.replaceChildren(el("option", { value: "auto" }, fastest ? "Auto: the fastest measured" : "Auto: the GPU when there is one"),
      ...d.options.map(o => el("option", { value: o.id, disabled: o.available ? null : "", title: o.detail }, o.available ? o.label : `${o.label} — not available`)));
    sel.value = d.choice;
    const runs = d.active === "cpu" ? "the CPU" : d.activeName;
    const parallel = d.active.startsWith("gpu") ? (d.gpuChecks ?? true ? "; checks run on the GPU, and at the same time the CPU writes out the answers the GPU cannot confirm and reading samples" : "; at the same time the CPU scores lesson candidates and checks reading") : d.active.startsWith("hybrid") ? "; each batch is split between them by their measured speed" : "";
    $("deviceNote").textContent = `Training runs on ${runs}${parallel}.` + (d.note ? " " + d.note : "") + " Talking and exams run on the CPU.";
    showBenchmark(d.benchmark);
  } else {
    $("deviceChoice").value = d.choice;
    $("deviceNote").textContent = d.trainingOn === "gpu" ? `Training runs on ${d.gpu}.` : "Training runs on the CPU.";
  }
  renderSizes();
}
// Set when the Engine reports that the running job could not use the GPU and trains on the CPU.
let cpuFallback = false;
const deviceLabel = () => cpuFallback || !deviceInfo || deviceInfo.active === "cpu" || deviceInfo.trainingOn === "cpu" ? " · on the CPU" : ` · on ${deviceInfo.activeName || deviceInfo.gpu}`;
// The running job's titles name the device it really trains on.
function onDeviceNotice(text) {
  log("WARN", text);
  say(text);
  if (!String(text).includes("on the CPU")) return;
  cpuFallback = true;
  for (const id of ["readTitle", "progressTitle", "planNow"]) {
    const t = $(id);
    if (t && t.textContent.includes(" · on ")) t.textContent = t.textContent.replace(/ · on .*$/, deviceLabel());
  }
}
// The last training benchmark: every device's speed on identical batches, and where the GPU's
// time goes.
function showBenchmark(b) {
  const box = $("benchResult");
  box.hidden = !b;
  if (!b) return;
  const r = b.result;
  const label = d => deviceInfo?.options?.find(o => JSON.stringify(o.id) === JSON.stringify(devId(d)))?.label || devId(d);
  const best = r.fastest ? devId(r.fastest) : null;
  const rows = r.results.map(x => el("tr", { class: devId(x.device) === best ? "current-row" : "" },
    el("td", {}, el("div", {}, label(x.device)), el("div", { class: "note" }, x.name)),
    el("td", { class: "num" }, x.error ? "—" : fmt(Math.round(x.tokens_per_second))),
    el("td", { class: "num" }, x.error ? "—" : `${fmt(Math.round(x.ms_per_step))} ms`),
    el("td", { class: "num" }, x.gpu_memory_bytes ? mb(x.gpu_memory_bytes) : "—"),
    el("td", { class: "num" }, x.gpu_share == null ? "—" : `${Math.round(100 * x.gpu_share)}% GPU`),
    el("td", {}, x.error ? el("span", { class: "bad" }, x.error) : x.loss_difference == null ? "reference" : Math.abs(x.loss_difference) < 1e-3 ? el("span", { class: "ok" }, `same (Δ ${x.loss_difference.toExponential(1)})`) : el("span", { class: "bad" }, `differs by ${x.loss_difference.toFixed(4)}`))));
  const p = r.profile;
  box.replaceChildren(
    el("p", { class: "note" }, `Measured ${new Date(b.time * 1000).toLocaleString()} with ${b.model} (${fmt(b.parameters)} parameters): batches of ${b.rows} rows × ${fmt(b.tokens)} tokens, ${r.steps} steps after warm-up. ${best ? "Fastest: " + label(r.fastest) + (deviceInfo?.choice === "auto" ? " (Auto uses it)." : ".") : ""}`),
    el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, "Device"), el("th", { class: "num" }, "Tokens/s"), el("th", { class: "num" }, "Per step"), el("th", { class: "num" }, "GPU memory"), el("th", { class: "num" }, "Split"), el("th", {}, "Loss vs CPU"))), el("tbody", {}, ...rows)),
    p ? el("details", {}, el("summary", {}, "Where the fastest GPU's time goes"),
      el("p", { class: "note" }, `CPU preparing the step: ${p.cpu_prepare_ms.toFixed(0)} ms · GPU kernels: ${p.gpu_ms.toFixed(0)} ms over ${fmt(p.dispatches)} dispatches${p.timestamps ? " (GPU timestamps)" : " (timed one by one, includes submission cost)"} · GPU memory ${mb(p.memory_bytes)}`),
      el("table", { class: "ledger compact" }, el("tbody", {}, ...p.kernels.slice(0, 12).map(k => el("tr", {}, el("td", { class: "mono" }, k.kernel), el("td", { class: "num" }, `${k.count}×`), el("td", { class: "num" }, `${k.ms.toFixed(1)} ms`), el("td", { class: "num" }, `${(100 * k.ms / Math.max(p.gpu_ms, 1e-9)).toFixed(1)}%`)))))) : "");
}
function devId(d) {
  if (!d || d.kind === "cpu") return "cpu";
  const api = { auto: "", dx12: "-dx12", vulkan: "-vulkan", metal: "-metal" }[d.api] ?? "";
  return d.kind + api;
}
$("benchTraining").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open a model first: the benchmark trains a copy of it");
  const rows = Number($("readBatch").value) || 8, tokens = Math.min(Number($("readWindow").value) || 128, active.context);
  $("benchTraining").disabled = true;
  $("benchNote").textContent = "Benchmarking… (each device trains the same batches; this takes a few minutes)";
  try { showDevice(await invoke("benchmark_training", { rows, tokens, steps: 3 })); say("Benchmark finished."); }
  finally { $("benchTraining").disabled = false; $("benchNote").textContent = "Trains the open model's copy on every device with identical batches (the Reading settings' batch size and window) and measures speed, GPU memory and that every device computes the same loss. Auto then uses the fastest."; }
}, "Benchmark"));
$("deviceChoice").addEventListener("change", safe(async () => {
  try { showDevice(await invoke("set_training_device", { choice: $("deviceChoice").value })); }
  catch (e) { if (deviceInfo) $("deviceChoice").value = deviceInfo.choice; throw e; }
}, "Training device"));
$("threadChoice").addEventListener("change", safe(async () => {
  try { showDevice(await invoke("set_cpu_threads", { threads: Number($("threadChoice").value) })); }
  catch (e) { if (deviceInfo) $("threadChoice").value = String(deviceInfo.cpuThreads); throw e; }
}, "CPU threads"));
$("fittedReplies").addEventListener("change", safe(async () => {
  try { showDevice(await invoke("set_fitted_replies", { on: $("fittedReplies").checked })); }
  catch (e) { if (deviceInfo) $("fittedReplies").checked = deviceInfo.fittedReplies ?? true; throw e; }
}, "Reply limits"));
$("gpuChecks").addEventListener("change", safe(async () => {
  try { showDevice(await invoke("set_gpu_checks", { on: $("gpuChecks").checked })); }
  catch (e) { if (deviceInfo) $("gpuChecks").checked = deviceInfo.gpuChecks ?? true; throw e; }
}, "GPU checks"));
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
  const growthMode = $("newGrowth").value;
  if ($("archChoice").value === "neuro") {
    showActive(await invoke("create_neuro_model", { name, input: neuroInput(), wordLimit: chosenLimit(), context: chosenContext() }));
    if (growthMode !== "off") { await invoke("set_growth_mode", { mode: growthMode }); window.showGrowth?.(); }
    $("newName").value = "";
    await refreshModels();
    say(`Created ${active.name}: an untrained Nova Neuro model with ${fmt(active.parameters)} parameters.`);
    return;
  }
  showActive(await invoke("create_model", { name, size: chosenSize, wordLimit: chosenLimit(), context: chosenContext() }));
  if (growthMode !== "off") { await invoke("set_growth_mode", { mode: growthMode }); window.showGrowth?.(); }
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
  const answer = await askText(`Vocabulary limit for ${active.name}: the most words it may learn. It knows ${fmt(active.words)} now and has ${fmt(active.slots)} slots. Leave empty to keep only the slots it has.`, active.wordLimit ?? "");
  if (answer === null) return;
  const limit = answer.trim() === "" ? null : Number(answer.replace(/[ ,_]/g, ""));
  if (limit !== null && !(Number.isInteger(limit) && limit >= 0)) throw new Error("Enter a whole number of words");
  showActive(await invoke("set_word_limit", { id: active.id, limit }));
  say(limit === null ? "Vocabulary fixed at the slots the model has." : `Vocabulary limit set to ${fmt(limit)} words.`);
}, "Vocabulary limit"));
$("contextLimit").addEventListener("click", safe(async () => {
  if (!active) return;
  const answer = await askText(`Context for ${active.name} in tokens (now ${fmt(active.context)}). No weights change; the model uses a longer context well once it has read or been taught text that long.`, active.context);
  if (answer === null || answer.trim() === "") return;
  const context = Number(answer.replace(/[ ,_]/g, ""));
  if (!(Number.isInteger(context) && context >= 16)) throw new Error("Enter a context of at least 16 tokens");
  showActive(await invoke("set_context", { id: active.id, context }));
  say(`Context set to ${fmt(context)} tokens.`);
}, "Context"));
$("renameModel").addEventListener("click", safe(async () => {
  if (!active) return;
  const name = await askText("New name for this model", active.name);
  if (!name) return;
  showActive({ ...active, ...(await invoke("rename_model", { id: active.id, name })) });
  await refreshModels();
}, "Rename"));

// ---- classroom -----------------------------------------------------------------------------
let lessonsList = [];
const fileState = {};  // lesson index -> outcome text during/after teaching
// Lessons taught together, by file name (every lesson is, unless unticked).
const lessonTogether = {};
const isTogether = l => lessonTogether[l.name] !== false;

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
    body.append(el("tr", {}, el("td", { colspan: 8, class: "empty" }, "No lessons added.")));
    return;
  }
  lessonsList.forEach((l, i) => body.append(el("tr", { id: "lesson-" + i },
    el("td", {}, el("input", { type: "checkbox", checked: isTogether(l) ? "" : null, disabled: teaching ? "" : null, "aria-label": `Teach ${l.name} together with the others`,
      onchange: e => { lessonTogether[l.name] = e.target.checked; } })),
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
// A restored checkpoint's interrupted run, which the next Classroom run continues (its epochs,
// learning rate and schedule) unless unticked.
function showResume(r) {
  $("resumeBox").hidden = !r;
  if (!r) return;
  const rate = r.learning_rate_now.toPrecision(2);
  $("resumeText").textContent = `Continue the interrupted run from epoch ${r.epoch + 1} of ${r.epochs}, at its eased-off learning rate (about ${rate}, set ${r.learning_rate}) instead of starting again at full speed. The run's epochs, learning rate and batch size are used for this.`;
}

// Accepted checkpoints of a training session that did not finish (power cut, crash): restore
// the newest, or discard them. Shown only when there are some.
function renderCheckpoints(list) {
  const box = $("checkpointPanel");
  if (!list.length) { box.replaceChildren(); box.hidden = true; return; }
  box.hidden = false;
  const when = t => t ? new Date(t * 1000).toLocaleString() : "—";
  const restore = c => safe(async () => {
    if (!await askYes(`Restore the checkpoint of epoch ${c.epoch} (step ${fmt(c.step)}) as ${active.name}? The current model file is kept as model-before-restore.nnova2 in the model's folder.`)) return;
    showActive(await invoke("restore_checkpoint", { name: c.name }));
    log("INFO", `Restored the checkpoint of epoch ${c.epoch} (step ${fmt(c.step)}).`);
  }, "Restore checkpoint");
  const discard = el("button", { type: "button", class: "quiet" }, "Discard checkpoints");
  discard.addEventListener("click", safe(async () => {
    if (!await askYes("Discard these checkpoints? The model keeps its last saved state; the progress in them is lost.")) return;
    await invoke("discard_checkpoints");
    refreshCheckpoints();
  }, "Discard checkpoints"));
  box.replaceChildren(
    el("h4", {}, "Unfinished training: checkpoints waiting"),
    el("p", { class: "note warn" }, `A training session of this model did not finish (Studio was closed or the power went out), but it saved accepted epochs. The model itself is still at step ${fmt(active.step)}. Restore the newest checkpoint to continue from there: the next Classroom run then offers to continue that run where it left off (its epoch and eased-off learning rate). Teaching stays locked until you restore or discard them.`),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Checkpoint"), el("th", { class: "num" }, "Step"), el("th", { class: "num" }, "Epoch"), el("th", {}, "Learning rate"), el("th", {}, "Saved"), el("th", { class: "num" }, "Size"), el("th", {}, ""))),
      el("tbody", {}, ...list.map((c, i) => {
        const b = el("button", { type: "button", class: i === 0 ? "primary" : "" }, i === 0 ? "Restore (newest)" : "Restore");
        b.addEventListener("click", restore(c));
        const lr = c.info ? `${c.info.learning_rate_now.toPrecision(2)} (set ${c.info.learning_rate})` : "—";
        return el("tr", {}, el("td", { class: "mono" }, c.name), el("td", { class: "num" }, fmt(c.step)), el("td", { class: "num" }, c.info ? `${c.epoch} of ${c.info.epochs}` : c.epoch), el("td", { class: "num" }, lr),
          el("td", {}, when(c.saved)), el("td", { class: "num" }, bytes(c.bytes)), el("td", {}, b));
      }))),
    el("div", { class: "actions" }, discard));
}
function refreshCheckpoints() { invoke("model_checkpoints").then(renderCheckpoints).catch(() => renderCheckpoints([])); }

// The open model's saved earlier lessons (its rehearsal archive).
function renderSaved(list) {
  $("savedCount").textContent = list.length ? `${list.length} lesson${list.length === 1 ? "" : "s"}` : "none";
  const body = $("savedTable").tBodies[0];
  if (!list.length) { body.replaceChildren(el("tr", {}, el("td", { colspan: 4, class: "empty" }, "No saved lessons."))); return; }
  body.replaceChildren(...list.map(l => {
    const remove = el("button", { type: "button", class: "quiet" }, "Remove");
    remove.addEventListener("click", safe(async () => {
      if (!await askYes(`Remove "${l.name}" from this model's saved lessons? It will no longer be rehearsed, trained alongside or guarded. The model itself is not changed.`)) return;
      renderSaved(await invoke("forget_saved_lesson", { name: l.name }));
    }, "Remove saved lesson"));
    return el("tr", {}, el("td", {}, l.name), el("td", { class: "num" }, fmt(l.units)), el("td", { class: "num" }, fmt(l.held_out)), el("td", {}, remove));
  }));
}
function refreshSaved() { invoke("saved_lessons").then(renderSaved).catch(() => renderSaved([])); }
async function refreshLessons() { lessonsList = await invoke("lessons"); renderLessons(); }
$("addLessons").addEventListener("click", safe(async () => {
  lessonsList = await invoke("add_lessons");
  for (const k of Object.keys(fileState)) delete fileState[k];
  renderLessons();
}, "Add lessons"));
for (const [id, on] of [["togetherAll", true], ["togetherNone", false]]) {
  $(id).addEventListener("click", () => { for (const l of lessonsList) lessonTogether[l.name] = on; renderLessons(); });
}
$("clearLessons").addEventListener("click", safe(async () => {
  lessonsList = await invoke("remove_lesson", { index: null });
  renderLessons();
}, "Clear lessons"));

function setTeaching(on) {
  teaching = on;
  $("teach").disabled = on;
  $("compareSettings").disabled = on;
  $("stopTeach").disabled = !on;
  $("read").disabled = on;
  $("stopRead").disabled = !on;
  for (const id of ["addLessons", "clearLessons", "createModel", "importModel"]) $(id).disabled = on;
  renderLessons();
  renderCorpora();
  window.onTeachingChanged?.(on);
  if (!on) window.refreshUnfinished?.(false).catch(e => log("ERROR", e));
}

let progress = { files: 1, index: 0, epochs: 1, candidates: 1, candidate: 0, epoch: 1 };

// A learning rate the way it is entered: 0.0001, 0.00025, 0.001 (never 1.0e-4).
function rateText(x) {
  if (!x || !Number.isFinite(x)) return String(x);
  const r = Number(x.toPrecision(3));
  const decimals = Math.min(20, Math.max(0, 2 - Math.floor(Math.log10(Math.abs(r)))));
  return r.toFixed(decimals).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}
// The rate in use, and the rate you set when it differs: training starts below it (warm-up)
// and ends below it (the rate eases off towards the end of a run).
function rateLine(now, set, warming) {
  if (!set || Math.abs(now - set) <= set * 0.005) return `learning rate ${rateText(now)}`;
  return `learning rate ${rateText(now)} (${warming ? "warming up to" : "easing off from"} the ${rateText(set)} you set)`;
}
let readRateSet = 0;
// A reading sample as shown: no line breaks before the text, later ones marked ⏎.
const sampleText = s => (s || "").replace(/^\n+/, "").replace(/\n/g, " ⏎ ");
const FOCI = ["dense", "transformer", "interconnect", "joint"];
// Candidates of each bicameral education stage, in the Engine's order (with Foundation, each
// part the stage may train is also tried on its own).
function stageFoci(stage, foundation) {
  if (!stage || stage === "off") return foundation ? FOCI : ["joint"];
  if (stage === "transformer") return ["transformer"];
  if (stage === "dense") return foundation ? ["dense", "interconnect", "dense+interconnect"] : ["dense+interconnect"];
  return foundation ? ["dense", "transformer", "interconnect", "dense+interconnect", "joint"] : ["dense+interconnect", "joint"];
}
function teachOptions() {
  return {
    epochs: Number($("optEpochs").value),
    batchSize: Number($("optBatch").value),
    learningRate: Number($("optRate").value),
    foundation: $("optFoundation").checked,
    rehearsal: $("optRehearsal").checked,
    protectAnswers: $("optProtect").checked,
    passPercent: Number($("optPass").value),
    flatEpochs: Number($("optFlat").value),
    retries: $("optRetries").checked,
    requeue: $("optRequeue").checked,
    education: $("optEducation").value,
    together: lessonsList.map(isTogether),
    trainEarlier: $("optTrainEarlier").checked,
    recoverEarlier: $("optRecoverEarlier").checked,
    backoff: $("optBackoff").checked,
    overrideFirstEpoch: $("optOverrideFirst").checked,
    probationEpochs: Math.max(0, Math.min(10, Number($("optProbation").value) || 0)),
    wordsFromNewLessons: $("optWordsNew").checked,
    guardBeforeGrowth: $("optGuardGrowth").checked,
    earlierEveryLesson: $("optEarlierEvery").checked,
    earlierLossRise: Math.max(0, Math.min(1, Number($("optEarlierRise").value) || 0.02)),
    earlierByAnswers: $("optEarlierAnswers").checked,
    continueRun: $("optContinue").checked,
  };
}
$("teach").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open or create a model first");
  const options = teachOptions();
  if (!await window.beforeNewRun()) return;
  progress = { rateSet: options.learningRate, files: lessonsList.length, done: 0, index: 0, epochs: options.epochs, candidates: stageFoci(options.education, options.foundation).length, foci: stageFoci(options.education, options.foundation), candidate: 0, epoch: 1, attempt: "" };
  for (const k of Object.keys(fileState)) delete fileState[k];
  $("candidateTable").tBodies[0].replaceChildren();
  $("courseTable").tBodies[0].replaceChildren();
  $("progressEntry").hidden = false;
  cpuFallback = false;
  $("progressTitle").textContent = `Teaching ${active.name}${deviceLabel()}`;
  await invoke("teach", { options });
  setTeaching(true);
  say("Teaching started.");
}, "Teach"));
// Comparing settings (an experiment on the open model; it is not changed).
const compareRows = [];
function testCell(t) { return `${t.new_right}/${t.new_total}`; }
// Context-test pairs passed after an epoch, when the comparison asked them.
const ctxText = c => c ? `, context ${c[0]}/${c[1]}` : "";
let compareNames = [];
function onCompareEvent(e) {
  if (e.type === "started") {
    compareNames = e.variants.map(v => v.split(":")[0]);
    compareRows.length = 0;
    $("compareTable").tBodies[0].replaceChildren(el("tr", {}, el("td", {}, "Start (before teaching)"), el("td", {}, ""), el("td", {}, ""),
      el("td", { class: "num" }, testCell(e.start)), el("td", { class: "num" }, `${e.start.earlier_right}/${e.start.earlier_total}`), el("td", {}, ""),
      el("td", { class: "num" }, `${e.start.new_loss.toFixed(3)} / ${e.start.earlier_loss.toFixed(3)}`),
      el("td", { class: "num" }, e.start_context ? `${e.start_context.passed}/${e.start_context.total}` : "—"), el("td", {}, "")));
    $("compareTrajectory").replaceChildren();
    $("compareLine").textContent = `Kept back for testing: ${fmt(e.new_test)} questions of the lessons taught, ${fmt(e.earlier_test)} of the earlier lessons.`;
  } else if (e.type === "variant_started") {
    $("compareLine").textContent = `Teaching ${e.name}…`;
    $("compareTrajectory").append(el("h4", {}, e.name));
  } else if (e.type === "session" && e.event.type === "batch") {
    const s = e.event;
    $("compareLine").textContent = `${compareNames[e.index] ?? e.index} — epoch ${s.epoch} · batch ${s.batch}/${s.batches} · loss ${s.loss.toFixed(4)}`;
  } else if (e.type === "tested") {
    const t = e.test;
    $("compareTrajectory").append(el("div", { class: "note mono" },
      `epoch ${t.epoch}: accepted ${testCell(t.accepted)} new, ${t.accepted.earlier_right}/${t.accepted.earlier_total} earlier${ctxText(t.accepted_context)} · continues from ${testCell(t.working)} new, ${t.working.earlier_right}/${t.working.earlier_total} earlier${ctxText(t.working_context)} (loss ${t.working.new_loss.toFixed(3)} / ${t.working.earlier_loss.toFixed(3)})`));
  } else if (e.type === "variant_finished") {
    const r = e.result;
    const g = r.growth;
    $("compareTable").tBodies[0].append(el("tr", {},
      el("td", {}, r.name),
      el("td", { class: "num" }, `${r.epochs} (${r.accepted_epochs})`),
      el("td", { class: "num" }, `${r.promoted} / ${r.dropped}`),
      el("td", { class: "num" }, testCell(r.end)),
      el("td", { class: "num" }, `${r.end.earlier_right}/${r.end.earlier_total}`),
      el("td", { class: "num" }, `${r.gained} / ${r.lost}`),
      el("td", { class: "num" }, `${r.end.new_loss.toFixed(3)} / ${r.end.earlier_loss.toFixed(3)}`),
      el("td", { class: "num" }, r.context ? `${r.context.passed}/${r.context.total}` : "—"),
      el("td", {}, g ? `${g.words}: earlier answers ${g.earlier_before[0]} → ${g.earlier_after[0]} of ${g.earlier_after[1]} just from adding them` : "none")));
  } else if (e.type === "finished") {
    $("compareLine").textContent = e.stopped ? "Stopped." : "Finished. Compare the rows: the test questions were never trained on or used to choose a checkpoint.";
  }
}
$("compareSettings").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open or create a model first");
  const options = teachOptions();
  const variants = ["A", "B", "C", "D", "E"].filter(v => $(`cmp${v}`).checked);
  if (!variants.length) throw new Error("Tick at least one set of settings to compare");
  $("compareEntry").hidden = false;
  $("compareTable").tBodies[0].replaceChildren();
  $("compareLine").textContent = "Starting…";
  await invoke("compare_settings", { options, variants });
  setTeaching(true);
  say(`Comparing settings: ${variants.join(", ")} are taught one after another.`);
}, "Compare settings"));
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
// What a lesson changed, apart from whether it passed: held-out loss and answers, and the
// share of its own (taught) questions answered right. Small held-out sets are pointed out.
function learningText(r) {
  if (!r || !r.baseline || !r.final_score) return "";
  const b = r.baseline, f = r.final_score, n = f.exact_total;
  const parts = [`held-out loss ${b.heldout_loss.toFixed(3)} → ${f.heldout_loss.toFixed(3)}`,
    `held-out answers ${b.exact_correct} → ${f.exact_correct} of ${n}`];
  if (r.own_before != null && r.own_after != null)
    parts.push(`own questions ${(100 * r.own_before).toFixed(0)}% → ${(100 * r.own_after).toFixed(0)}%`);
  let s = "Learning: " + parts.join(" · ");
  if (n > 0 && n < 20) s += ` · only ${n} held-out questions, each worth ${(100 / n).toFixed(0)}% of the score`;
  return s;
}
// What an epoch decided: accepted, kept training through a flat epoch, or restored.
function epochDecision(ep) {
  if (ep.selected && ep.overridden) return `accepted ${ep.selected} by the first-epoch override (the safeguards had rejected it)`;
  const p = ep.probation;
  if (ep.selected && p?.state === "promoted") return `accepted ${ep.selected}: the probation branch recovered after ${p.epochs} epoch(s)`;
  if (ep.selected) return `accepted ${ep.selected}`;
  const flat = ep.flat_epochs ? ` (flat epoch ${ep.flat_epochs})` : "";
  if (p?.state === "continued") return `rejected (${p.rejections.join(", ")}); on probation, training continues from ${p.focus} (probation epoch ${p.epoch_in_branch})${flat}`;
  if (ep.continued) return `not accepted yet; training continues from ${ep.continued}${flat}`;
  if (p?.state === "discarded") return `probation ended after ${p.epochs} epoch(s) without a safe improvement; restored the best checkpoint${flat}`;
  if (ep.restored) return `no safe candidate; restored the best checkpoint${flat}`;
  return `no candidate accepted${flat}`;
}
function scoreCell(s) { return s ? `${s.exact_correct}/${s.exact_total}` : "—"; }
function onTeachEvent(e) {
  if (e.type === "group_started") {
    for (const i of e.indices) fileState[i] = el("span", { class: "tag" }, "teaching together…");
    renderLessons();
    $("progressTitle").textContent = `Teaching ${active?.name ?? ""} · ${e.indices.length} lessons together`;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7 }, el("b", {}, `Taught together (one mixed session): ${e.names.join(", ")}`),
      e.earlier?.length ? el("span", { class: "note" }, ` · with ${e.earlier.length} saved earlier lesson(s) trained alongside and guarded: ${e.earlier.join(", ")}`) : "")));
  } else if (e.type === "group_scored") {
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7, class: "note" },
      `After attempt ${e.attempt + 1}, each lesson on its own held-out questions: ` + e.indices.map((ix, k) => `${lessonsList[ix]?.name ?? "lesson " + (ix + 1)} ${(100 * e.rates[k]).toFixed(0)}%${e.passed[k] ? " (passed)" : ""}`).join(" · ")
      + (e.earlier_rates?.length ? ` · saved earlier lessons: ${e.earlier_rates.map(r => (100 * r).toFixed(0) + "%").join(", ")}` : ""))));
  } else if (e.type === "file_started") {
    progress.index = e.index;
    fileState[e.index] = el("span", { class: "tag" }, e.pass > 1 ? "trying again…" : "teaching…");
    renderLessons();
    $("progressTitle").textContent = `Teaching ${active?.name ?? ""} · lesson ${e.index + 1} of ${e.files}: ${e.name}${e.pass > 1 ? " (second try, after other lessons)" : ""}`;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7 }, el("b", {}, `Lesson ${e.index + 1}: ${e.name}`))));
  } else if (e.type === "attempt_started") {
    progress.epochs = e.epochs;
    const what = e.attempt === 0 ? "first attempt" : `retry ${e.attempt} of ${e.attempts - 1}${e.remedy ? ": " + e.remedy : ""}`;
    progress.rateSet = e.learning_rate;
    progress.attempt = `${what} · learning rate ${rateText(e.learning_rate)} · ${e.strategy} study · ${e.epochs} epochs`;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7, class: "note" }, progress.attempt)));
  } else if (e.type === "diagnosed") {
    const d = e.diagnosis;
    const cause = { unstable: "unstable training", forgetting: "harming earlier lessons", interference: "new learning overwrites known answers", not_learned: "not learned yet", not_generalising: "memorised, not generalised" }[d.cause] || d.cause;
    $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7 }, el("span", { class: "warn" },
      `Not passed (held-out ${(100 * d.heldout_rate).toFixed(0)}%, training questions ${(100 * d.train_rate).toFixed(0)}%) → diagnosis: ${cause}. Next: ${d.remedy}`))));
  } else if (e.type === "session") {
    const s = e.event;
    if (s.type === "learning_rate_lowered") {
      log("INFO", `Epoch ${s.epoch}: ${s.reason} (now ${Math.round(100 * s.scale)}% of the scheduled rate).`);
      $("candidateTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 7, class: "note warn" }, `Learning rate halved after epoch ${s.epoch}: ${s.reason}; now ${Math.round(100 * s.scale)}% of the scheduled rate.`)));
    } else if (s.type === "checkpoint") {
      if (s.error) log("WARN", `Checkpoint of epoch ${s.epoch} could not be written: ${s.error}`);
      else log("INFO", `Accepted epoch ${s.epoch} written to the checkpoint folder (step ${fmt(s.step)}); the model itself is saved when the session ends.`);
    } else if (s.type === "started") {
      $("progressLine").textContent = `Baseline: held-out loss ${s.baseline.heldout_loss.toFixed(4)}, exact ${scoreCell(s.baseline)}; ${fmt(s.train_units)} units to learn, ${fmt(s.heldout_units)} held out, ${fmt(s.new_words)} new word units.`;
      const g = s.growth;
      if (g) log(g.earlier_before[0] > g.earlier_after[0] || g.judged_before[0] > g.judged_after[0] ? "WARN" : "INFO",
        `${fmt(g.words)} word units added. Just from adding them (before any training): earlier-lesson answers checked ${g.earlier_before[0]} → ${g.earlier_after[0]} of ${g.earlier_after[1]}, questions judged ${g.judged_before[0]} → ${g.judged_after[0]} of ${g.judged_after[1]}.` + (g.guarded_before ? " The earlier answers are guarded as they were before." : " That loss is not guarded (the guards start from the model with the new words)."));
      const room = s.replies;
      if (room && room.longest_answer) log("INFO", `Held-out answers take up to ${fmt(room.longest_answer)} tokens with this vocabulary (median ${fmt(room.median_answer)}); replies may use up to ${fmt(room.longest_limit)} (each reply gets room to spell its answer one letter at a time).`);
      if (room?.too_long) log("WARN", `${fmt(room.too_long)} of the ${fmt(room.exact)} held-out answers graded word for word are longer than a reply may be (24 tokens with fitted reply limits off), so they can never be answered exactly right: their loss can still fall, but exact answers cannot rise on them. Turn on "Fit the reply limit to each answer" next to the training device.`);
    } else if (s.type === "batch") {
      progress.epoch = s.epoch;
      progress.candidate = Math.max(0, progress.foci.indexOf(s.focus));
      if (progress.candidates === 1) progress.candidate = 0;
      const inFile = (((s.epoch - 1) * progress.candidates + progress.candidate) * s.batches + s.batch) / (progress.epochs * progress.candidates * s.batches);
      $("progressBar").style.width = (100 * Math.min(1, (progress.done + Math.min(1, inFile)) / progress.files)).toFixed(1) + "%";
      $("progressLine").textContent = `${progress.attempt} — epoch ${s.epoch}/${progress.epochs} · ${s.focus} candidate · batch ${s.batch}/${s.batches} · loss ${s.loss.toFixed(4)} · ${rateLine(s.learning_rate, progress.rateSet, s.epoch === 1 && s.learning_rate < progress.rateSet)}`;
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
        else if (c.continued) row.lastChild.replaceChildren(el("span", { class: "warn" }, "training continues from this one (safe, not yet better enough to accept)"));
        else if (c.eligible && !c.reasons.length) row.lastChild.replaceChildren(el("span", {}, "eligible; another candidate was better"));
      }
      // The whole brain after this epoch (the accepted model): progress at a glance.
      const a = s.accepted_score;
      $("candidateTable").tBodies[0].append(el("tr", { class: "epoch-summary" }, el("td", { colspan: 7 },
        el("b", {}, `Whole brain after epoch ${s.epoch}: `),
        `exact ${scoreCell(a)} · held-out loss ${a.heldout_loss.toFixed(4)}`,
        a.dense_reliance == null ? "" : ` · transformer alone ${a.without_dense_loss.toFixed(4)} · Dense contribution ${a.dense_reliance.toFixed(3)}`,
        ` · ${epochDecision(s)}`)));
      log("INFO", `Epoch ${s.epoch}: ${epochDecision(s)}`);
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
        learningText(e) ? el("div", { class: "note" }, learningText(e)) : "",
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

// ---- settings kept on this device and updates ------------------------------------------------
// The Classroom and Conversation pages' settings are kept in Studio's settings file (in the app's
// data folder), so they survive restarts and updates. Passwords and files are never kept.
const KEPT_PAGES = { classroom: "page-classroom", conversation: "page-conversation" };
function formState(root) {
  const out = {};
  for (const e of root.querySelectorAll("input[id], select[id]")) {
    if (e.type === "password" || e.type === "file" || e.type === "button") continue;
    out[e.id] = e.type === "checkbox" ? e.checked : e.value;
  }
  return out;
}
function keptValueFits(e, v) {
  if (e.type === "checkbox") return typeof v === "boolean";
  if (e.tagName === "SELECT") return [...e.options].some(o => o.value === String(v));
  if (e.type === "number") {
    const n = Number(v);
    if (v === "" || !Number.isFinite(n)) return false;
    if (e.min !== "" && n < Number(e.min)) return false;
    if (e.max !== "" && n > Number(e.max)) return false;
    return true;
  }
  return typeof v === "string";
}
async function restorePageSettings() {
  for (const [page, id] of Object.entries(KEPT_PAGES)) {
    const root = $(id);
    if (!root) continue;
    const saved = await invoke("page_settings", { page }).catch(() => null);
    for (const [key, v] of Object.entries(saved || {})) {
      const e = root.querySelector("#" + CSS.escape(key));
      if (!e || e.type === "password" || e.type === "file") continue;
      // Only values the control can hold: a choice that still exists, a number in range. Anything
      // else keeps the control's default (and is reported), never a blank or broken value.
      if (!keptValueFits(e, v)) { log("WARN", `Saved setting ${key} = ${JSON.stringify(v)} no longer fits; its default is used`); continue; }
      if (e.type === "checkbox") e.checked = !!v; else e.value = v;
      e.dispatchEvent(new Event("change", { bubbles: true }));
    }
    let timer = null;
    root.addEventListener("change", ev => {
      if (!ev.target.id || ev.target.type === "password") return;
      clearTimeout(timer);
      timer = setTimeout(() => invoke("save_page_settings", { page, settings: formState(root) }).catch(e => log("ERROR", e)), 400);
    });
  }
}

// Updates: Studio asks the release page for a newer build; installing keeps everything in the
// app's data folder and restarts Studio.
async function checkUpdate(asked) {
  try {
    const u = await invoke("check_update");
    if (u) {
      $("updateNote").textContent = `Version ${u.version} is available (this is ${u.current}).`;
      $("installUpdate").hidden = false;
    } else if (asked) {
      $("updateNote").textContent = "Studio is up to date.";
    }
  } catch (e) {
    if (asked) $("updateNote").textContent = String(e);
  }
}
$("checkUpdate").addEventListener("click", safe(() => checkUpdate(true), "Check for updates"));
$("installUpdate").addEventListener("click", safe(async () => {
  if (!await askYes("Download and install the update now? Studio restarts when it is installed; your models, datasets, chats and settings are kept.")) return;
  $("installUpdate").disabled = true;
  $("updateNote").textContent = "Downloading the update…";
  try { await invoke("install_update"); } finally { $("installUpdate").disabled = false; }
}, "Update"));
if (tauri?.event?.listen) {
  tauri.event.listen("update-progress", ev => {
    const p = ev.payload;
    $("updateNote").textContent = `Downloading the update: ${mb(p.received)}${p.total ? " of " + mb(p.total) : ""}…`;
  });
}

// ---- reading (corpus pre-training) ---------------------------------------------------------
let sourcesList = [];
let corporaList = [];
let familiesList = [];
const mb = bytes => (bytes / 1048576).toFixed(1) + " MB";
// A download's progress as the backend reports it: rows read of the dataset's rows, documents
// kept by the quality filter, the Parquet file being fetched. The bar follows what limits the
// download (rows read, or the size asked for); the estimates come from the measured pace.
const downloadPace = {};
function downloadFraction(e) {
  const kept = e.total ? e.received / e.total : 0;
  let rows = e.rowsTotal && e.rows ? e.rows / e.rowsTotal : null;
  if (rows == null && e.file) {
    const [i, n] = e.file;
    const part = e.fileBytes && e.fileBytes[1] ? e.fileBytes[0] / e.fileBytes[1] : 0;
    rows = (i - 1 + part) / n;
  }
  return Math.min(1, Math.max(kept, rows ?? 0));
}
function downloadText(e) {
  if (e.rows == null && !e.file) return `${mb(e.received)}${e.total ? " of " + mb(e.total) : ""} received`;
  const parts = [];
  if (e.file && e.fileBytes && e.fileBytes[0] < e.fileBytes[1]) parts.push(`file ${e.file[0]} of ${e.file[1]}: ${mb(e.fileBytes[0])} of ${mb(e.fileBytes[1])} downloaded`);
  if (e.rows != null) parts.push(`read ${fmt(e.rows)}${e.rowsTotal ? ` of ${fmt(e.rowsTotal)}` : ""} rows`);
  if (e.kept != null) parts.push(`kept ${fmt(e.kept)} (${mb(e.received)}${e.total ? `; stops at ${mb(e.total)}` : ""})`);
  const pace = downloadPace[e.source] ||= { start: Date.now(), rows0: e.rows || 0 };
  const secs = (Date.now() - pace.start) / 1000;
  if (e.rowsTotal && e.rows > 5000) {
    const share = e.kept / e.rows;
    const final = e.received * e.rowsTotal / e.rows;
    if (e.total && final < 0.9 * e.total) parts.push(`the filter keeps ${(100 * share).toFixed(1)}% of rows, so about ${mb(final)} in all`);
    const rate = (e.rows - pace.rows0) / Math.max(1, secs);
    if (secs > 20 && rate > 0) {
      const left = Math.min(e.rowsTotal - e.rows, e.total && e.received ? (e.total - e.received) / e.received * e.rows : Infinity) / rate;
      if (left > 30) parts.push(`about ${left > 5400 ? `${(left / 3600).toFixed(1)} h` : `${Math.round(left / 60)} min`} left`);
    }
  }
  return parts.join(" · ");
}
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
          if (used + want * 1048576 > BUDGET && !await askYes(`This would bring the corpora on this device to about ${mb(used + want * 1048576)}, over the 1 GB budget. Download anyway?`)) return;
          const start = await invoke("download_corpus", { source: s.id, maxMb: want });
          if (start.queued) { say(`${s.name} will download after the ${start.position === 1 ? "one running" : `${start.position - 1} waiting before it`}.`); return; }
          $("downloadEntry").hidden = false;
          $("downloadTitle").textContent = `Downloading ${s.name}`;
          $("downloadBar").style.width = "0%";
          $("downloadLine").textContent = "Connecting…";
          $("downloadNote").hidden = true;
          $("cancelDownload").hidden = false;
          $("finishDownload").hidden = false;
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
        if (!await askYes(`Delete the corpus "${c.name}" from this device?`)) return;
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
// The Reading page's settings are remembered between runs of the app.
const READ_FIELDS = ["readTokens", "readBatch", "readRate", "readWindow", "readMaxWords", "readMode", "genDifficulty", "genMb", "importCategory"];
let readSettingsLoaded = false;
let readSaveTimer = null;
function readSettings() {
  const values = {};
  for (const id of READ_FIELDS) values[id] = $(id).value;
  const stages = {};
  for (const sel of document.querySelectorAll("#stageMap select")) stages[sel.dataset.cat] = sel.value;
  const families = {};
  for (const i of document.querySelectorAll("#familyList input")) families[i.value] = i.checked;
  // Ticks and shares of the corpora still on this device.
  const corpora = {};
  for (const c of corporaList) if (mix[c.id]) corpora[c.id] = { read: !!mix[c.id].read, share: mix[c.id].share };
  return { values, readWords: $("readWords").checked, stages, families, corpora };
}
function saveReadSettings() {
  if (!readSettingsLoaded) return;  // never overwrite saved settings with the page's defaults
  clearTimeout(readSaveTimer);
  readSaveTimer = setTimeout(() => {
    invoke("save_page_settings", { page: "reading", settings: readSettings() }).catch(e => log("ERROR", e));
  }, 400);
}
function applyReadSettings(saved) {
  if (saved && typeof saved === "object") {
    for (const id of READ_FIELDS) {
      const v = saved.values?.[id];
      if (v == null) continue;
      const input = $(id);
      // A select keeps its value only if the option still exists.
      if (input.tagName === "SELECT" ? [...input.options].some(o => o.value === v) : v !== "") input.value = v;
    }
    if (typeof saved.readWords === "boolean") $("readWords").checked = saved.readWords;
    for (const sel of document.querySelectorAll("#stageMap select")) {
      const v = saved.stages?.[sel.dataset.cat];
      if (STAGE_OPTIONS.some(([o]) => o === v)) sel.value = v;
    }
    for (const i of document.querySelectorAll("#familyList input")) {
      if (typeof saved.families?.[i.value] === "boolean") i.checked = saved.families[i.value];
    }
    for (const [id, m] of Object.entries(saved.corpora || {})) {
      if (m && typeof m === "object") mix[id] = { read: !!m.read, share: Math.min(100, Math.max(1, Number(m.share) || 25)) };
    }
  }
  $("stageMap").hidden = $("readMode").value !== "bicameral";
  readSettingsLoaded = true;
}
for (const ev of ["input", "change"]) $("page-reading").addEventListener(ev, saveReadSettings);
async function refreshCorpora() { corporaList = await invoke("corpora"); renderCorpora(); }
function pick(id) { (mix[id] ??= { read: true, share: 25 }).read = true; saveReadSettings(); }
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
$("finishDownload").addEventListener("click", safe(async () => { await invoke("finish_download"); say("Finishing the download: what it has so far is kept."); }, "Finish download"));
// Downloads run one at a time; the ones asked for meanwhile wait their turn. Both download
// panels (§ 2 Reading, § 4 Conversation) list the queue.
function renderDownloadQueue(q) {
  const list = waiting => waiting.length
    ? [el("p", { class: "note" }, `Waiting (${waiting.length}), downloaded one after another:`),
       el("ol", {}, ...waiting.map(w => el("li", {}, `${w.name} · ${w.maxMb} MB `,
         el("button", { type: "button", class: "small", onclick: safe(async () => renderDownloadQueue(await invoke("remove_queued_download", { source: w.source })), "Remove") }, "Remove"))))]
    : [];
  for (const id of ["downloadQueue", "convDownloadQueue"]) $(id)?.replaceChildren(...list(q.waiting));
  // A queued download that starts: the Reading panel shows it when it is a reading source.
  if (q.current && sourcesList.some(s => s.id === q.current.source)) {
    $("downloadEntry").hidden = false;
    $("downloadTitle").textContent = `Downloading ${q.current.name}`;
    $("cancelDownload").hidden = false;
    $("finishDownload").hidden = false;
  }
  window.dispatchEvent(new CustomEvent("download-queue", { detail: q }));
}
function onDownload(e) {
  if (!sourcesList.some(s => s.id === e.source)) return;
  if (!e.done && e.note != null) {
    // Waiting for the server (rate limit, busy): shown beside the progress, which is kept.
    $("downloadNote").textContent = e.note;
    $("downloadNote").hidden = !e.note;
    return;
  }
  if (e.done) {
    $("downloadNote").textContent = e.note || "";
    $("downloadNote").hidden = !e.note || !!e.error;
  }
  if (!e.done) {
    $("downloadBar").style.width = e.total || e.rowsTotal || e.file ? (100 * downloadFraction(e)).toFixed(1) + "%" : "50%";
    $("downloadLine").textContent = downloadText(e);
    return;
  }
  delete downloadPace[e.source];
  $("cancelDownload").hidden = true;
  $("finishDownload").hidden = true;
  if (e.error) { $("downloadLine").textContent = e.error; log("ERROR", e.error); return; }
  $("downloadBar").style.width = "100%";
  const c = e.corpus;
  $("downloadLine").textContent = `Saved ${c.name}: ${mb(c.bytes)}, ${fmt(c.documents)} documents (${fmt(c.heldOut)} held out).`;
  // Chosen for reading only when downloaded for reading (not a § 4 Conversation dataset).
  if (sourcesList.some(s => s.id === e.source)) pick(c.id);
  refreshCorpora();
  say(`Downloaded ${c.name}.`);
}

let readStart = 0;
let readParts = [];
// The Reading page's settings, as the backend (and a plan's reading step) takes them.
function readingOptions() {
  const parts = chosen();
  if (!parts.length) throw new Error("Tick the corpora to read on § 2 Reading (download one, generate practice or use a text file first)");
  const stages = {};
  for (const sel of document.querySelectorAll("#stageMap select")) stages[sel.dataset.cat] = sel.value;
  return {
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
}
window.readingOptions = readingOptions;
$("read").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open or create a model first");
  const options = readingOptions();
  if (!await window.beforeNewRun()) return;
  const parts = chosen();
  readParts = parts;
  $("evalTable").tBodies[0].replaceChildren();
  $("readEntry").hidden = false;
  cpuFallback = false;
  $("readTitle").textContent = `${active.name} is reading ${parts.map(c => c.name).join(" + ")}${options.mode === "bicameral" ? " (bicameral)" : ""}${deviceLabel()}`;
  await invoke("read_corpus", { options });
  readStart = 0;
  readRateSet = options.learningRate;
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
    // Shown in the Reading card; also kept in the diagnostics log.
    log("INFO", `Reading: ${fmt(e.train_documents)} documents (${fmt(e.train_tokens)} tokens), ${fmt(e.held_out_documents)} held out; ${fmt(e.new_words)} new word units; ${fmt(e.steps)} steps${parts ? " · " + parts : ""}. Starting held-out loss ${e.baseline_loss.toFixed(4)}.`);
    $("evalTable").tBodies[0].append(el("tr", {}, el("td", { class: "num" }, "0"), el("td", {}, ""), el("td", { class: "num" }, e.baseline_loss.toFixed(4)), el("td", { class: "num" }, (e.baseline_dense_reliance ?? 0).toFixed(3)), el("td", {}, "starting point"), el("td", {}, "")));
    if (e.resumed_from != null) {
      // A continued run: the checks before its recovery point, then a marker.
      for (const ev of pendingReadPrevious || []) onReadEvent({ type: "evaluated", ...ev });
      pendingReadPrevious = null;
      $("evalTable").tBodies[0].append(el("tr", {}, el("td", { colspan: 6, class: "note" }, `Continued from the recovery point after step ${fmt(e.resumed_from)} of ${fmt(e.steps)}`)));
      log("INFO", `Reading continues from its recovery point after step ${fmt(e.resumed_from)}.`);
    }
  } else if (e.type === "batch") {
  } else if (e.type === "evaluated") {
    $("evalTable").tBodies[0].append(el("tr", { class: e.accepted ? "current-row" : "" },
      el("td", { class: "num" }, fmt(e.tokens)),
      el("td", { class: "num" }, e.train_loss.toFixed(4)),
      el("td", { class: "num" }, e.held_out_loss.toFixed(4)),
      el("td", { class: "num" }, (e.dense_reliance ?? 0).toFixed(3)),
      el("td", {}, e.accepted ? el("b", { class: "ok" }, "kept (new best)") : el("span", { class: "warn" }, e.reason || `not better than ${e.best_loss.toFixed(4)}`)),
      el("td", { class: "sample" }, ...(e.parts?.length ? e.parts : [{ name: "", held_out_loss: e.held_out_loss, sample_prompt: e.sample_prompt, sample: e.sample }]).map(p =>
        el("div", { class: "part-line" }, p.name ? el("b", {}, `${CATEGORY[p.category] || ""}${p.name ? " · " + p.name : ""}: ${p.held_out_loss.toFixed(4)} `) : "",
          el("span", { class: "note" }, p.sample_prompt.replace(/\n/g, " ⏎ ")), " ", sampleText(p.sample))))));
  } else if (e.type === "finished" && e.early_stop) {
    log("INFO", "Reading stopped early: the held-out loss stopped improving.");
  }
}
function onReadFinished(f) {
  setTeaching(false);
  if (f.model) showActive(f.model);
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
    if (answer.memories.length) addLine("memory", answer.memories.join(" · "), "memory");
    conversation.push([message, answer.reply]);
    addLine(active.name, answer.reply || "(empty reply)", "model");
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
  const neuro = !!m.trace.neuro;
  const design = active?.design && { ...active.design, blocks: active.blocks };
  Brain.setArchitecture(neuro, design);
  const { attention } = neuro ? Brain.showNeuro(m, idx, design) : Brain.show(m, idx);
  // The readings name the paths of the architecture shown.
  const reading = (id, bica, neuroText) => { const th = $(id).previousElementSibling; if (th) th.textContent = neuro ? neuroText : bica; };
  reading("gateMeans", "Gate openings (exchange, rounds)", "Write-back gate (per round)");
  reading("effectRight", "Output change without right → left", "Output change without the networks' write-back");
  reading("effectLeft", "Output change without left → right", "Output change without language core → bus");
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
  window.showNeuroActivity?.(m);
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
    if (r.reading?.type === "evaluated") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), `Read ${fmt(r.reading.tokens)} tokens: held-out loss ${r.reading.held_out_loss.toFixed(4)} — ${r.reading.accepted ? "kept" : "not better"}.`, el("div", { class: "note mono" }, `${r.reading.sample_prompt} → ${sampleText(r.reading.sample)}`));
    if (r.type === "course_started") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), `Course started: ${r.files} file(s), ${r.options.epochs} epoch(s), ${r.options.foundation ? "Foundation" : "Joint only"}.`);
    if (r.type === "course_finished") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)), r.error ? el("span", { class: "bad" }, "Course ended with an error: " + r.error) : `Course ${r.stopped ? "stopped" : "finished"}.`);
    if (e?.type === "file_finished") return el("div", { class: "record" }, el("span", { class: "when" }, when(r.time)),
      el("b", {}, e.name), ` — ${e.outcome.replace("_", " ")}${e.pass > 1 ? " (second pass)" : ""}, steps ${fmt(e.start_step)} → ${fmt(e.end_step)}, exact ${scoreCell(e.baseline)} → ${scoreCell(e.final_score)}, ${e.attempts?.length ?? 0} attempt(s)`, e.error ? ` (${e.error})` : "",
      learningText(e) ? el("div", { class: "note" }, learningText(e)) : "",
      el("div", { class: "note mono" }, "sha256 " + e.sha256));
    if (e?.type === "session" && e.event.type === "epoch") {
      const ep = e.event;
      const d = el("details", { class: "record" }, el("summary", {}, el("span", { class: "when" }, when(r.time)),
        `Epoch ${ep.epoch}: ${epochDecision(ep)} · exact ${scoreCell(ep.accepted_score)} · held-out loss ${ep.accepted_score.heldout_loss.toFixed(4)}`));
      const t = el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, "Candidate"), el("th", { class: "num" }, "Held-out loss"), el("th", { class: "num" }, "Exact"), el("th", {}, "Decision"))));
      const b = el("tbody");
      for (const c of ep.candidates) b.append(el("tr", {}, el("td", {}, c.focus), el("td", { class: "num" }, c.score.heldout_loss.toFixed(4)), el("td", { class: "num" }, scoreCell(c.score)), el("td", {}, c.selected ? "selected" : c.continued ? "training continued from it (not yet accepted)" : (c.reasons.join("; ") || "eligible; another candidate was better"))));
      t.append(b);
      d.append(t);
      for (const [p, exp, got] of ep.sample_replies) d.append(el("p", { class: "note" }, `${p.replace(/\n/g, " ⏎ ")} → expected "${exp}", replied "${got}"`));
      return d;
    }
    return el("div", { class: "record note" }, el("span", { class: "when" }, when(r.time)), e?.type === "session" ? `${e.event.type} (lesson ${e.index + 1})` : (r.type || e?.type || "record"));
  }));
}
$("refreshHistory").addEventListener("click", safe(loadHistory, "Library"));

// ---- unfinished training (recovery points) --------------------------------------------------
const RUN_KIND = { reading: "Reading", conversation: "Conversation practice" };
let recoveryMinutes = 10;
function runDetails(r) {
  const pct = r.steps ? ` (${Math.round(100 * r.step / r.steps)}%)` : "";
  const state = r.interrupted ? "Interrupted (Studio closed, crashed or the computer restarted while it trained)"
    : r.status === "stopped" ? "Stopped" : r.status === "failed" ? `Ended with an error: ${r.error}` : r.status;
  const lines = [
    `${state}.`,
    `Started ${when(r.started)} with Studio ${r.appVersion}${r.resumed ? `; continued ${r.resumed} time(s)` : ""}.`,
    r.snapshot ? `Last recovery point: ${when(r.snapshot.time)}, after step ${fmt(r.snapshot.step)} of ${fmt(r.steps)}${pct}, ${fmt(r.tokens)} tokens.`
      : "It ended before its first recovery point, so there is nothing to continue; the accepted model is safe.",
    r.snapshot ? (r.snapshot.stopped ? "Nothing needs repeating: the recovery point was saved when it stopped."
      : r.interrupted ? `The training after that point is repeated: at most about ${recoveryMinutes} minute(s) at the current setting (checks the run made after it are made again).` : "") : "",
    r.check || "",
    r.datasets.length ? `Datasets: ${r.datasets.map(d => d.name).join(", ")}.` : "",
    r.plan ? `A step of the plan "${r.plan.planName}": Resume on the Planner continues it.` : "",
    r.failedSaves ? `${r.failedSaves} recovery point(s) could not be saved (${r.saveError}).` : "",
  ];
  return lines.filter(Boolean);
}
async function discardRun(r) {
  const yes = await askYes(`Discard the unfinished ${RUN_KIND[r.kind] || r.kind} of ${r.modelName}? Its recovery points are deleted and it cannot be continued. The model keeps everything it accepted so far.`);
  if (!yes) return false;
  await invoke("discard_run", { modelId: r.modelId, runId: r.id });
  say("Discarded the unfinished run; the accepted model is unchanged.");
  return true;
}
let pendingReadPrevious = null;
async function continueRun(r) {
  if (r.plan) { go("planner"); say(`Resume the plan "${r.plan.planName}" to continue its step from the recovery point.`); return; }
  if (teaching) throw new Error("Wait until the training now running has finished (or stop it) first");
  const info = await invoke("resume_run", { modelId: r.modelId, runId: r.id });
  showActive(info.meta);
  if (info.note) log("WARN", info.note);
  if (info.kind === "reading") {
    go("reading");
    $("evalTable").tBodies[0].replaceChildren();
    $("readEntry").hidden = false;
    cpuFallback = false;
    $("readTitle").textContent = `${info.meta.name} continues ${info.label.replace(/^Reading: /, "")} from step ${fmt(info.step)} of ${fmt(info.steps)}${deviceLabel()}`;
    pendingReadPrevious = info.previous || [];
    readStart = 0;
    setTeaching(true);
    say("Continuing the reading from its recovery point…");
  } else {
    window.resumeConversation?.(info);
  }
  refreshUnfinished(false).catch(e => log("ERROR", e));
}
async function refreshUnfinished(startup) {
  recoveryMinutes = await invoke("recovery_minutes").catch(() => recoveryMinutes);
  const runs = await invoke("unfinished_runs", { verify: !!startup });
  $("unfinishedBox").hidden = !runs.length;
  $("unfinishedList").replaceChildren(...runs.map(r => {
    const cont = el("button", { type: "button", class: "primary small" }, r.plan ? "Open the Planner" : "Continue");
    cont.disabled = (!r.resumable && !r.plan) || teaching;
    cont.addEventListener("click", safe(() => continueRun(r), "Continue"));
    const drop = el("button", { type: "button", class: "small quiet" }, "Discard…");
    drop.disabled = teaching;
    drop.addEventListener("click", safe(async () => { if (await discardRun(r)) await refreshUnfinished(false); }, "Discard"));
    return el("div", { class: "run-card" }, el("b", {}, `${RUN_KIND[r.kind] || r.kind} · ${r.modelName}`), ` — ${r.label}`,
      ...runDetails(r).map(t => el("div", { class: "note" }, t)), el("div", { class: "actions" }, cont, drop));
  }));
  if (!startup) return;
  // An interrupted run is never continued or discarded without asking.
  for (const r of runs.filter(x => x.interrupted)) {
    const content = el("div", {}, el("p", {}, el("b", {}, `${RUN_KIND[r.kind] || r.kind} of ${r.modelName}`), ` — ${r.label}`),
      ...runDetails(r).map(t => el("p", { class: "note" }, t)),
      el("p", { class: "note" }, "Continuing uses the run's own settings and data (both are checked first) and its optimizer state, so it carries on where the recovery point was. The accepted model stays as it is until a check accepts a better one."));
    const buttons = r.resumable && !r.plan ? [["later", "Decide later"], ["discard", "Discard…"], ["new", "Start a new run…"], ["continue", "Continue training", true]]
      : r.plan ? [["later", "Decide later"], ["discard", "Discard…"], ["plan", "Open the Planner", true]]
      : [["later", "Decide later"], ["discard", "Discard…", true]];
    const choice = await choose("Interrupted training detected", content, buttons);
    if (choice === "continue") { await continueRun(r); break; }
    if (choice === "plan") { await continueRun(r); break; }
    if (choice === "discard") await discardRun(r);
    if (choice === "new" && await discardRun(r)) {
      showActive(await invoke("open_model", { id: r.modelId }));
      go(r.kind === "conversation" ? "conversation" : "reading");
      say("The interrupted run is discarded. Set up the new run on this page and start it.");
      break;
    }
  }
  // Classroom sessions keep their own recovery (accepted epochs in the model's checkpoints).
  for (const c of await invoke("interrupted_classroom").catch(() => [])) {
    const content = el("div", {}, el("p", {}, `${c.modelName} has ${c.checkpoints} accepted epoch checkpoint(s) from a Classroom session that did not finish (newest ${when(c.newest)}).`),
      el("p", { class: "note" }, "Open the model to restore the newest checkpoint (the next run can continue the interrupted schedule) or to discard them. Nothing is changed until you choose."));
    const choice = await choose("Interrupted Classroom session", content, [["later", "Decide later"], ["open", "Open the model", true]]);
    if (choice === "open") {
      if (teaching) break;
      showActive(await invoke("open_model", { id: c.modelId }));
      go("models");
      break;
    }
  }
  await refreshUnfinished(false);
}
window.refreshUnfinished = refreshUnfinished;
// Before new training on the open model: an unfinished run of it (stopped, interrupted or failed)
// is continued or discarded first, as the owner chooses. Resolves true to go on with the new run.
window.beforeNewRun = async () => {
  const runs = (await invoke("unfinished_runs")).filter(r => r.modelId === active?.id);
  for (const r of runs) {
    const content = el("div", {}, el("p", {}, `${active.name} has an unfinished ${(RUN_KIND[r.kind] || r.kind).toLowerCase()}: ${r.label}.`),
      ...runDetails(r).map(t => el("p", { class: "note" }, t)),
      el("p", { class: "note" }, "New training would change the model under it, so it could no longer be continued. Discarding deletes only its recovery points; the model keeps everything it accepted."));
    const buttons = [["cancel", "Cancel"], ["discard", "Discard it and start the new run"]];
    if (r.resumable || r.plan) buttons.push(["continue", r.plan ? "Open the Planner" : "Continue it instead", true]);
    const choice = await choose("Unfinished training", content, buttons);
    if (choice === "continue") { await continueRun(r); return false; }
    if (choice !== "discard") return false;
    await invoke("discard_run", { modelId: r.modelId, runId: r.id });
  }
  if (runs.length) await refreshUnfinished(false);
  return true;
};
$("recoveryMinutes").addEventListener("change", safe(async () => {
  const v = Number($("recoveryMinutes").value);
  $("recoveryMinutes").value = await invoke("set_recovery_minutes", { minutes: Math.max(0, Math.min(240, Math.round(v))) });
  say(v ? `A recovery point is saved every ${v} minutes while reading or practising conversations.` : "Recovery points are saved only when a run is stopped.");
}, "Recovery"));

// ---- switches that only work together ---------------------------------------------------------
// A switch never sits on without effect: one that needs another is kept consistent with it, and
// one that cannot act is disabled with the reason beside it.
(() => {
  const answers = $("optEarlierAnswers"), every = $("optEarlierEvery");
  answers.addEventListener("change", () => {
    if (answers.checked && !every.checked) { every.checked = true; every.dispatchEvent(new Event("change", { bubbles: true })); say("Judging earlier lessons by their answers needs the check of every earlier lesson, so that is on too."); }
  });
  every.addEventListener("change", () => {
    if (!every.checked && answers.checked) { answers.checked = false; answers.dispatchEvent(new Event("change", { bubbles: true })); say("Judging earlier lessons by their answers is off too: it works only with the check of every earlier lesson."); }
  });
  const listening = $("convListening"), understanding = $("convUnderstanding");
  const why = el("span", { class: "note" });
  understanding.parentElement.append(" ", why);
  const sync = () => {
    const off = !(Number(listening.value) > 0);
    understanding.disabled = off;
    why.textContent = off ? "(needs listening practice above 0%: it is part of it)" : "";
  };
  for (const ev of ["input", "change"]) listening.addEventListener(ev, sync);
  sync();
})();

// ---- start ---------------------------------------------------------------------------------
async function boot() {
  if (!tauri?.core) { say("Open this page inside the Yendory Studio app."); return; }
  await tauri.event.listen("teach-event", ev => { try { onTeachEvent(ev.payload); } catch (err) { log("ERROR", err); } });
  await tauri.event.listen("teach-finished", ev => onTeachFinished(ev.payload));
  await tauri.event.listen("compare-event", ev => { try { onCompareEvent(ev.payload); } catch (err) { log("ERROR", err); } });
  await tauri.event.listen("compare-finished", ev => {
    setTeaching(false);
    if (ev.payload) { log("ERROR", ev.payload); $("compareLine").textContent = ev.payload; }
    else say("The comparison finished.");
  });
  await tauri.event.listen("read-event", ev => { try { onReadEvent(ev.payload); } catch (err) { log("ERROR", err); } });
  await tauri.event.listen("read-finished", ev => onReadFinished(ev.payload));
  await tauri.event.listen("corpus-download", ev => onDownload(ev.payload));
  await tauri.event.listen("download-queue", ev => renderDownloadQueue(ev.payload));
  invoke("download_queue").then(renderDownloadQueue).catch(e => log("ERROR", e));
  await tauri.event.listen("benchmark-progress", ev => { $("benchNote").textContent = `Benchmarking: ${ev.payload}…`; });
  await tauri.event.listen("device-notice", ev => onDeviceNotice(ev.payload));
  await tauri.event.listen("recovery-notice", ev => { log("WARN", ev.payload); say(ev.payload); });
  // Closing while training: the run stops at a safe point and saves a recovery point first.
  await tauri.event.listen("closing", () => {
    document.body.append(el("div", { class: "modal" }, el("div", { class: "modal-box", role: "alertdialog" },
      el("h3", {}, "Saving before closing…"),
      el("p", {}, "Training stops at the next safe point and saves a recovery point; Studio then closes. Next time it offers to continue where it stopped."))));
  });
  for (const n of await invoke("settings_notices").catch(() => [])) { log("WARN", n); say(n); }
  invoke("recovery_minutes").then(m => { $("recoveryMinutes").value = m; }).catch(e => log("ERROR", e));
  sourcesList = await invoke("corpus_sources");
  renderSources();
  familiesList = await invoke("practice_families");
  renderFamilies();
  applyReadSettings(await invoke("page_settings", { page: "reading" }).catch(() => null));
  await restorePageSettings();
  checkUpdate(false);
  await refreshCorpora();
  updateReadNote();
  const s = await invoke("status");
  showActive(s.active);
  setTeaching(s.teaching);
  sizes = await invoke("model_sizes");
  renderSizes();
  invoke("training_device").then(showDevice).catch(e => log("ERROR", e));
  await updateVocabNote();
  await refreshModels();
  await refreshLessons();
  say(`Yendory Studio ${s.engine} · every reply and measurement comes from the model itself.`);
  await refreshUnfinished(!s.teaching).catch(e => log("ERROR", e));
}
document.addEventListener("DOMContentLoaded", safe(boot, "Startup"));
