"use strict";
// Practice: Nova answers, the teacher marks, an approved training pass learns from the marks.
// Answers, training and scoring run in the Rust Engine (learn::practice).

let practice = null;
let practiceTimer = null;
const practiceCard = new TaskCard($("practiceCard"));
const practiceTrack = { epochs: 1, candidates: 1, epochNow: 0, seen: [] };
const ORIGIN_TEXT = { nova_confirmed: "Nova's answer, confirmed", teacher_correction: "your correction", lesson_answer: "the lesson's answer" };
// Grading as the lesson would: its first accepted form, normalised like sentence grading.
const norm = s => String(s || "").toLowerCase().replace(/n't/g, " not").replace(/[^a-z0-9]+/g, " ").trim();
const matchesLesson = it => it.expected && it.expected.split("|").map(x => x.trim()).filter(x => x && !/^(must|never):|^!/.test(x)).some(x => norm(x) === norm(it.answer));

async function refreshPractice() {
  const sources = await invoke("practice_sources").catch(() => []);
  const sel = $("practiceSource");
  const keep = sel.value;
  sel.replaceChildren(el("option", { value: "" }, sources.length ? "Choose a lesson…" : "No lesson with practice questions (Designer Practice blocks or the Classroom list)"),
    ...sources.map(s => el("option", { value: `${s.kind}:${s.id}` }, `${s.kind === "design" ? "Designer" : "Classroom"}: ${s.name} (${s.questions} questions)`)));
  if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
  const list = await invoke("list_practice");
  const body = $("practiceList").tBodies[0];
  body.replaceChildren(...(list.length ? list.map(p => el("tr", { class: practice?.id === p.id ? "current-row" : null },
    el("td", {}, el("b", {}, p.lesson)), el("td", {}, p.modelName), el("td", { class: "num" }, `${p.marked}/${p.questions}`),
    el("td", {}, p.trained ? el("span", { class: "ok" }, "trained") : "—"), el("td", { class: "note" }, when(p.created)),
    el("td", {}, el("button", { class: "quiet", onclick: safe(async () => showPractice(await invoke("practice_detail", { id: p.id })), "Open practice") }, "open"))))
    : [el("tr", {}, el("td", { colspan: 6, class: "empty" }, "No practice yet."))]));
}

function practiceSteps(p) {
  const marked = p.items.filter(i => i.mark).length;
  const trainable = p.items.filter(i => i.mark === "right" && i.answer.trim() || i.mark === "wrong" && (i.correction.trim() || i.expected)).length;
  const r = p.result;
  return [
    { id: "answer", title: "Nova answered", state: "done", detail: `${p.items.length} question(s), answered by the network${p.studyMode ? " with Study Mode memories in front" : ""}.` },
    { id: "review", title: "Teacher review", state: marked === p.items.length ? "done" : r ? "done" : "running",
      detail: `${marked} of ${p.items.length} marked; ${trainable} will be trained.` },
    { id: "train", title: "Neural training", state: r ? (r.learned ? "done" : "skipped") : p.error ? "failed" : "waiting",
      detail: r ? (r.learned ? `Accepted: held-out ${r.baseline.exact_correct} → ${r.finalScore.exact_correct} of ${r.finalScore.exact_total} right, held-out loss ${r.baseline.heldout_loss.toFixed(3)} → ${r.finalScore.heldout_loss.toFixed(3)} (steps ${fmt(r.stepBefore)} → ${fmt(r.stepAfter)}).` : `No change was accepted (held-out loss ${r.baseline.heldout_loss.toFixed(3)}); the model is unchanged.`)
        : p.error ? p.error : "Starts when you approve the marked answers." },
    { id: "after", title: "Asked again", state: r ? "done" : "waiting", detail: r ? `${p.items.filter(i => i.after != null && i.after !== i.answer).length} answer(s) changed after training.${r.withMemory ? ` With memory: ${r.withMemory[0]} of ${r.withMemory[1]} held-out questions right.` : ""}` : "Nova answers the same questions after training." },
    { id: "reflect", title: "Reflection", state: "waiting", detail: "Available after the training pass (Reflection, below)." },
  ];
}

function showPractice(p) {
  practice = p;
  $("practiceEntry").hidden = false;
  $("practiceTitle").textContent = `Practice: ${p.lesson} · ${p.modelName}`;
  practiceCard.set({ title: `Practice: ${p.lesson}`, subtitle: p.modelName, status: p.result ? "Trained" : p.error ? "Error" : "Reviewing",
    statusKind: p.result ? "good" : p.error ? "bad" : "", footer: [p.studyMode ? "Study Mode: on when answered" : "Study Mode: off when answered", "Grading without memory"],
    steps: practiceSteps(p) });
  const done = !!p.result;
  const body = $("practiceTable").tBodies[0];
  body.replaceChildren();
  p.items.forEach((it, k) => {
    const right = el("input", { type: "radio", name: `pm${k}`, checked: it.mark === "right" ? "" : null, disabled: done ? "" : null, onchange: () => { it.mark = "right"; marksChanged(); } });
    const wrong = el("input", { type: "radio", name: `pm${k}`, checked: it.mark === "wrong" ? "" : null, disabled: done ? "" : null, onchange: () => { it.mark = "wrong"; marksChanged(); } });
    const correction = el("input", { type: "text", value: it.correction, placeholder: "Correction (trained instead of her answer)", disabled: done ? "" : null, oninput: e => { it.correction = e.target.value; marksChanged(); } });
    const note = el("input", { type: "text", value: it.note, placeholder: "Note (not taught)", disabled: done ? "" : null, oninput: e => { it.note = e.target.value; marksChanged(); } });
    body.append(el("tr", {},
      el("td", { class: "num" }, String(k + 1)),
      el("td", {}, it.question, it.expected ? el("div", { class: "note" }, `Lesson's answer: ${it.expected}`) : ""),
      el("td", {}, el("span", { class: "mono" }, it.answer || "(empty)"),
        matchesLesson(it) ? el("div", { class: "note ok" }, "matches the lesson's answer") : "",
        it.memories.length ? el("div", { class: "note" }, "Memories in front: " + it.memories.join(" · ")) : "",
        it.after != null ? el("div", { class: "note" }, "After training: ", el("span", { class: "mono" }, it.after || "(empty)")) : "",
        it.target ? el("div", { class: "note" }, `Trained on ${ORIGIN_TEXT[it.target[1]]}: ${it.target[0]}`) : ""),
      el("td", {}, el("label", { class: "check-inline" }, right, "right"), " ", el("label", { class: "check-inline" }, wrong, "wrong")),
      el("td", {}, correction, note)));
  });
  for (const id of ["practiceTrain", "practiceAllRight"]) $(id).disabled = done || teaching;
  $("practiceResult").replaceChildren(p.error ? el("p", { class: "bad" }, p.error) : "");
  refreshPractice().catch(() => {});
}

function marksChanged() {
  $("practiceSaved").textContent = "Unsaved marks…";
  practiceCard.step("review", practiceSteps(practice)[1]);
  clearTimeout(practiceTimer);
  practiceTimer = setTimeout(() => saveMarks().catch(e => log("ERROR", e)), 600);
}
async function saveMarks() {
  clearTimeout(practiceTimer);
  if (!practice || practice.result) return;
  practice = await invoke("save_practice_marks", { id: practice.id, marks: practice.items.map(i => ({ mark: i.mark, correction: i.correction, note: i.note })) });
  $("practiceSaved").textContent = `Saved ${new Date().toLocaleTimeString()}`;
}

$("practiceStart").addEventListener("click", safe(async () => {
  const v = $("practiceSource").value;
  if (!v) throw new Error("Choose a lesson to practise");
  if (!active) throw new Error("Open the model that should practise");
  const [kind, ...rest] = v.split(":");
  $("practiceStart").disabled = true;
  try {
    say("Nova is answering…");
    showPractice(await invoke("start_practice", { kind, id: rest.join(":") }));
    say("Nova answered. Mark her answers.");
  } finally { $("practiceStart").disabled = false; }
}, "Start practice"));
$("practiceAllRight").addEventListener("click", () => {
  if (!practice) return;
  let n = 0;
  for (const it of practice.items) if (!it.mark && matchesLesson(it)) { it.mark = "right"; n++; }
  showPractice(practice);
  marksChanged();
  say(`${n} answer(s) matching the lesson's marked right. Check them before training.`);
});
$("practiceTrain").addEventListener("click", safe(async () => {
  await saveMarks();
  const epochs = Number($("practiceEpochs").value);
  Object.assign(practiceTrack, { epochs, candidates: 1, epochNow: 0, seen: [] });
  await invoke("train_practice", { id: practice.id, epochs, learningRate: Number($("practiceRate").value) });
  setTeaching(true);
  $("practiceStop").disabled = false;
  $("practiceTrain").disabled = true;
  practiceCard.set({ status: "Training", statusKind: "" });
  practiceCard.step("train", { state: "running", progress: 0, detail: "Learning from the marked answers; held-out questions judge every step." });
}, "Train practice"));
$("practiceStop").addEventListener("click", safe(async () => { await invoke("stop_teaching"); $("practiceStop").disabled = true; }, "Stop practice"));
$("practiceDelete").addEventListener("click", safe(async () => {
  if (!practice || !confirm("Delete this practice session? The model is not affected.")) return;
  await invoke("delete_practice", { id: practice.id });
  practice = null;
  $("practiceEntry").hidden = true;
  refreshPractice();
}, "Delete practice"));

document.addEventListener("DOMContentLoaded", safe(async () => {
  if (!tauri?.event) return;
  await tauri.event.listen("practice-event", ev => {
    const { id, event: s } = ev.payload;
    if (practice?.id !== id) return;
    if (s.type === "started") practiceCard.step("train", { detail: `Starts with ${s.baseline.exact_correct} of ${s.baseline.exact_total} held-out questions right; ${s.train_units} answer(s) to learn.` });
    else if (s.type === "batch") practiceCard.step("train", { progress: attemptProgress(practiceTrack, s), progressLabel: batchLabel(practiceTrack, s) });
    else if (s.type === "epoch") practiceCard.step("train", { detail: `After epoch ${s.epoch}: ${s.accepted_score.exact_correct} of ${s.accepted_score.exact_total} held-out right, held-out loss ${s.accepted_score.heldout_loss.toFixed(3)}.` });
  });
  await tauri.event.listen("practice-finished", ev => {
    setTeaching(false);
    $("practiceStop").disabled = true;
    if (practice?.id === ev.payload.id) showPractice(ev.payload);
    say(ev.payload.error ? `Practice training ended: ${ev.payload.error}` : "Practice training finished.");
    refreshModels();
  });
}, "Practice"));

const practicePrevPage = window.onPage;
window.onPage = page => {
  practicePrevPage?.(page);
  if (page === "study") refreshPractice().catch(e => log("ERROR", e));
};
