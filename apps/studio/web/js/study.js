"use strict";
// Study page: Study Mode, and the memory inspector. Studying, retrieval and the teaching hooks
// run in the Rust Engine (yendory_engine::memory).

let memoryView = null;
let memoryFilter = null;

function showStudyState(s) {
  studyModeOn = !!s.on;
  $("studyModeState").textContent = s.on ? "on" : "off";
  $("studyModeState").className = s.on ? "ok" : "";
  $("studyToggle").textContent = s.on ? "Turn off" : "Turn on";
  $("studyToggle").className = s.on ? "" : "primary";
  $("studyModeNote").textContent = s.on
    ? `Reading blocks in memory or combined mode are studied into memory when their lesson is taught; lessons train with related memories in front of their questions, and the score with memory is reported apart from the grade (always without memory). Conversations put related memories in front of your message. ${s.model ? `${s.model} has ${fmt(s.memories)} memories.` : ""}`
    : `Nothing is written to or read from memory: reading, teaching and conversation work on the network alone. Saved memories are kept${s.memories ? ` (${fmt(s.memories)} for ${s.model})` : ""} and come back when Study Mode is turned on again.`;
  for (const id of ["memoryAdd", "studyTextGo"]) $(id).disabled = !s.on;
}

async function refreshStudy() {
  showStudyState(await invoke("study_state"));
  if (!active) { $("memorySummary").textContent = "Open a model to see its memory."; return; }
  memoryView = await invoke("memory_view");
  renderMemory();
}

const purposeName = { training: "training", evaluation: "grading with memory", conversation: "conversation", reflection: "reflection", review: "review" };

function renderMemory() {
  const v = memoryView;
  const st = v.store;
  const byId = new Map(st.entries.map(m => [m.id, m]));
  const studied = st.entries.filter(m => m.origin === "studied").length;
  $("memorySummary").textContent = `${fmt(st.entries.length)} memories (${fmt(studied)} kept by Nova while studying, ${fmt(st.entries.length - studied)} written by you), ${fmt(st.entries.filter(m => !m.enabled).length)} switched off. Model step ${fmt(v.step)}. ${v.on ? "" : "Study Mode is off: memories are kept but not used."}`;
  const contradictions = v.relations.filter(r => r.kind === "contradicts");
  $("memoryConflicts").replaceChildren(...contradictions.map(r => el("div", { class: "conflict" },
    el("b", {}, "May contradict: "), `#${r.a} "${byId.get(r.a)?.text}" and #${r.b} "${byId.get(r.b)?.text}". Check them and switch off or edit the wrong one.`)));
  $("memoryConcepts").replaceChildren(...(v.concepts.length ? v.concepts.slice(0, 60).map(([t, n]) => el("button", {
    class: "chip quiet" + (memoryFilter === t ? " on" : ""), onclick: () => { memoryFilter = memoryFilter === t ? null : t; renderMemory(); },
  }, `${t} ${n}`)) : [el("span", { class: "note" }, "No shared concepts yet.")]));
  const body = $("memoryTable").tBodies[0];
  body.replaceChildren();
  const rows = st.entries.filter(m => !memoryFilter || m.terms.includes(memoryFilter));
  if (!rows.length) body.append(el("tr", {}, el("td", { colspan: 7, class: "empty" }, st.entries.length ? "No memory with this concept." : "No memories yet. Study Mode stores them while lessons with memory-mode reading are taught, or study a text below.")));
  for (const m of rows) {
    const decide = (action, text) => safe(async () => { await invoke("memory_decide", { id: m.id, action, text }); await refreshStudy(); }, "Memory");
    const textCell = el("td", {}, m.text, m.edited ? el("span", { class: "tag" }, "edited") : "", m.verified ? el("span", { class: "tag ok" }, "checked") : "",
      el("div", { class: "note" }, `${m.origin === "teacher" ? "written by you" : "kept by Nova"} · step ${fmt(m.step)}${m.seen > 1 ? ` · studied ${m.seen} times` : ""}`));
    body.append(el("tr", { class: m.enabled ? null : "off" },
      el("td", { class: "num" }, String(m.id)),
      textCell,
      el("td", { class: "note" }, m.sources.map(s => `${s.lesson}${s.sentence ? ` (sentence ${s.sentence})` : ""}`).join("; ") || "—"),
      el("td", { class: "num" }, m.origin === "teacher" ? "—" : m.surprise.toFixed(2)),
      el("td", { class: "num" }, fmt(m.retrieved)),
      el("td", { class: "num" }, m.usefulness == null ? "—" : (m.usefulness >= 0 ? "+" : "") + m.usefulness.toFixed(3)),
      el("td", {},
        el("button", { class: "quiet", onclick: decide(m.enabled ? "disabled" : "enabled") }, m.enabled ? "switch off" : "switch on"),
        m.verified ? "" : el("button", { class: "quiet", onclick: decide("verified") }, "checked"),
        el("button", { class: "quiet", onclick: () => {
          const input = el("input", { type: "text", value: m.text, maxlength: 300 });
          const save = el("button", {}, "save");
          save.addEventListener("click", safe(async () => { await invoke("memory_decide", { id: m.id, action: "edited", text: input.value }); await refreshStudy(); }, "Edit memory"));
          textCell.replaceChildren(input, save);
          input.focus();
        } }, "edit"),
        el("button", { class: "quiet", onclick: async () => { if (await askYes(`Delete memory #${m.id}?`)) await decide("deleted")(); } }, "delete"))));
  }
  const log = $("memoryLog").tBodies[0];
  log.replaceChildren(...(v.log.length ? v.log.map(r => el("tr", {},
    el("td", { class: "note" }, when(r.time)), el("td", {}, purposeName[r.purpose] || r.purpose), el("td", {}, r.query),
    el("td", { class: "note" }, r.ids.map(id => `#${id} ${byId.get(id)?.text ?? "(deleted)"}`).join(" · ")))) : [el("tr", {}, el("td", { colspan: 4, class: "empty" }, "Nothing retrieved yet."))]));
}

function studyReportView(r) {
  return el("div", { class: "note" },
    el("p", {}, `${r.lesson}: ${r.candidates} sentence(s); kept ${r.stored.length}, already known or ranked lower ${r.not_kept.length}, already in memory ${r.merged.length}${r.held_out.length ? `, ${r.held_out.length} matching held-out questions (never stored)` : ""}.`),
    r.stored.length ? el("ul", {}, ...r.stored.map(([id, t, s]) => el("li", {}, `#${id} ${t} `, el("span", { class: "tag" }, `surprise ${s.toFixed(2)}`)))) : "",
    r.not_kept.length ? el("details", {}, el("summary", {}, "Not kept"), el("ul", {}, ...r.not_kept.map(([t, s]) => el("li", {}, `${t} (surprise ${s.toFixed(2)})`)))) : "",
    r.conflicts.length ? el("p", { class: "warn" }, `${r.conflicts.length} new memory(ies) may contradict an existing one; see above.`) : "");
}

$("studyToggle").addEventListener("click", safe(async () => {
  showStudyState(await invoke("set_study_mode", { on: !studyModeOn }));
  say(studyModeOn ? "Study Mode is on." : "Study Mode is off; memories are kept.");
  await refreshStudy();
}, "Study Mode"));
$("memoryTry").addEventListener("click", safe(async () => {
  const q = $("memoryQuery").value.trim();
  if (!q) return;
  const got = await invoke("memory_try", { query: q });
  $("memoryTried").replaceChildren(...(got.length ? got.map(r => el("li", {}, `#${r.id} ${r.text} `, el("span", { class: "tag" }, `score ${r.score.toFixed(2)}`)))
    : [el("li", {}, "Nothing related in memory.")]));
}, "Try retrieval"));
$("memoryQuery").addEventListener("keydown", e => { if (e.key === "Enter") $("memoryTry").click(); });
$("memoryAdd").addEventListener("click", safe(async () => {
  const t = $("memoryNew").value.trim();
  if (!t) return;
  await invoke("memory_add", { text: t });
  $("memoryNew").value = "";
  await refreshStudy();
}, "Add memory"));
$("studyTextGo").addEventListener("click", safe(async () => {
  const text = $("studyTextInput").value;
  if (!text.trim()) throw new Error("Paste some text to study");
  $("studyTextGo").disabled = true;
  try {
    say("Studying…");
    const r = await invoke("study_text", { text, label: $("studyTextLabel").value });
    $("studyTextReport").replaceChildren(studyReportView(r));
    say(`Studied: kept ${r.stored.length} of ${r.candidates} sentences.`);
    await refreshStudy();
  } finally { $("studyTextGo").disabled = !studyModeOn; }
}, "Study text"));

const studyPrevPage = window.onPage;
window.onPage = page => {
  studyPrevPage?.(page);
  if (page === "study") refreshStudy().catch(e => log("ERROR", e));
};
document.addEventListener("DOMContentLoaded", () => { if (tauri?.core) invoke("study_state").then(showStudyState).catch(() => {}); });

// Teaching reports what was studied, and the score with memory.
const studyBaseTeach = onTeachEvent;
onTeachEvent = e => {
  studyBaseTeach(e);
  if (e.type === "studied") {
    const r = e.report;
    log("INFO", `Studied ${r.lesson}: kept ${r.stored.length} of ${r.candidates} sentences; ${e.copies} question(s) train with memories in front.`);
    if (typeof teachCard !== "undefined" && !teachCard.container.hidden)
      teachCard.step("L" + e.index, { detail: `Studied its reading: kept ${r.stored.length} of ${r.candidates} sentences; ${e.copies} question(s) train with memories in front.` });
  } else if (e.type === "file_finished" && e.with_memory && typeof teachCard !== "undefined" && !teachCard.container.hidden) {
    const s = teachCard.steps().find(x => x.id === "L" + e.index);
    if (s) teachCard.step(s.id, { detail: `${s.detail} With memory: ${e.with_memory[0]} of ${e.with_memory[1]} held-out questions right (the grade is without memory).` });
  }
};
const studyBasePlan = onPlanEvent;
onPlanEvent = payload => {
  studyBasePlan(payload);
  const e = payload.event;
  if (e.type === "lesson" && e.event.type === "studied") {
    const r = e.event.report;
    planLog(`Studied ${r.lesson}: kept ${r.stored.length} of ${r.candidates} sentences; ${e.event.copies} question(s) train with memories in front.`);
  } else if (e.type === "lesson_taught" && e.record?.with_memory) {
    planLog(`${e.name} with memory: ${e.record.with_memory[0]} of ${e.record.with_memory[1]} held-out questions right (the grade is without memory).`);
  }
};
