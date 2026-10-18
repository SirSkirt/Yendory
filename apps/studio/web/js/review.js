"use strict";
// Review Training: memory -> review material (a Designer lesson the teacher checks) -> initial
// assessment -> neural training -> assessment after -> comparison (Rust Engine: learn::review).

let review = null;
const reviewCard = new TaskCard($("reviewCard"));
const reviewTrack = { epochs: 1, candidates: 1, epochNow: 0, seen: [] };
const reviewPick = { concepts: new Set(), lessons: new Set() };

async function refreshReview() {
  if (!active) return;
  const t = await invoke("review_topics");
  $("reviewGate").hidden = t.studyMode;
  $("reviewSetup").hidden = !t.studyMode;
  if (t.studyMode) {
    $("reviewMemCount").textContent = `${fmt(t.memories)} memories. Choose concepts or lessons to review (none chosen: all memories).`;
    const chip = (set, name, n) => el("button", { class: "chip quiet" + (set.has(name) ? " on" : ""), onclick: e => {
      if (set.has(name)) set.delete(name); else set.add(name);
      e.target.classList.toggle("on", set.has(name));
    } }, `${name} ${n}`);
    $("reviewConcepts").replaceChildren(el("span", { class: "note" }, "Concepts: "), ...t.concepts.slice(0, 40).map(([c, ids]) => chip(reviewPick.concepts, c, ids.length)));
    $("reviewLessons").replaceChildren(el("span", { class: "note" }, "Read in: "), ...t.lessons.map(([l, ids]) => chip(reviewPick.lessons, l, ids.length)));
  }
  const list = await invoke("list_reviews");
  $("reviewList").tBodies[0].replaceChildren(...(list.length ? list.map(r => el("tr", { class: review?.session.id === r.id ? "current-row" : null },
    el("td", {}, el("b", {}, r.name)), el("td", {}, r.modelName),
    el("td", { class: "num" }, r.before ? `${r.before.correct}/${r.before.total}${r.after ? ` → ${r.after.correct}/${r.after.total}` : ""}` : "—"),
    el("td", { class: "note" }, when(r.created)),
    el("td", {}, el("button", { class: "quiet", onclick: safe(async () => showReview(await invoke("review_detail", { id: r.id })), "Open review") }, "open"))))
    : [el("tr", {}, el("td", { colspan: 5, class: "empty" }, "No review yet."))]));
}

function reviewSteps(v) {
  const s = v.session;
  const tr = s.training;
  const unchecked = v.errors.length;
  return [
    { id: "select", title: "Review Training selected", state: "done", detail: `Study Mode memories of ${s.model_name || s.modelName}.` },
    { id: "analyze", title: "Memory analysed", state: "done", detail: `${s.memories.length} memory(ies) chosen.` },
    { id: "generate", title: "Material generated", state: "done", detail: `${v.examples} worked example(s), ${v.heldOut} held-out review question(s); every answer marked generated.` },
    { id: "review", title: "Teacher review", state: unchecked ? "running" : "done",
      detail: unchecked ? `${unchecked} problem(s) to fix in the Designer (e.g. ${v.errors[0]}).` : "Every answer checked; nothing trains before you start it." },
    { id: "before", title: "Initial assessment (without memory)", state: s.before ? "done" : "waiting",
      detail: s.before ? `${s.before.correct} of ${s.before.total} right${s.before.with_memory ? `; with memory ${s.before.with_memory[0]} of ${s.before.with_memory[1]}` : ""} (step ${fmt(s.before.step)}).` : "Optional; taken automatically before training if you skip it." },
    { id: "train", title: "Review training", state: tr ? (tr.learned ? "done" : "skipped") : s.error ? "failed" : "waiting",
      detail: tr ? `${tr.learned ? "Accepted" : "Not accepted; the model is unchanged"}: steps ${fmt(tr.stepBefore)} → ${fmt(tr.stepAfter)}, ${tr.epochs} epoch(s).` : s.error || "Neural training on the worked examples; the held-out questions are never trained." },
    { id: "after", title: "Post-training assessment", state: s.after ? "done" : "waiting",
      detail: s.after ? `${s.after.correct} of ${s.after.total} right${s.after.with_memory ? `; with memory ${s.after.with_memory[0]} of ${s.after.with_memory[1]}` : ""}.` : "The same held-out questions, without memory." },
  ];
}

function mark(ok) { return el("span", { class: ok ? "ok" : "bad" }, ok ? "✓" : "✕"); }

function reviewResults(v) {
  const s = v.session;
  if (!s.before) return el("p", { class: "note" }, "No assessment yet.");
  const after = new Map((s.after?.asked || []).map(a => [a.question, a]));
  const rows = s.before.asked.map(b => {
    const a = after.get(b.question);
    return el("tr", {},
      el("td", {}, b.question, el("div", { class: "note" }, `Expected: ${b.expected}`)),
      el("td", {}, mark(b.correct), " ", el("span", { class: "mono" }, b.reply || "(empty)"),
        b.with_memory ? el("div", { class: "note" }, "with memory: ", mark(b.with_memory[1]), " ", b.with_memory[0]) : ""),
      el("td", {}, ...(a ? [mark(a.correct), " ", el("span", { class: "mono" }, a.reply || "(empty)"),
        a.with_memory ? el("div", { class: "note" }, "with memory: ", mark(a.with_memory[1]), " ", a.with_memory[0]) : ""] : ["—"])));
  });
  const box = el("div", {},
    el("h4", {}, "Held-out review questions"),
    el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, "Question"), el("th", {}, "Before"), el("th", {}, "After"))), el("tbody", {}, ...rows)));
  const c = s.comparison;
  if (c) {
    const list = (title, items, cls) => items.length ? el("p", { class: cls || "note" }, el("b", {}, `${title} (${items.length}): `), items.join(" · ")) : "";
    box.append(el("h4", {}, "Comparison (without memory)"),
      list("Improved", c.improved, "ok"), list("Regressed", c.regressed, "bad"), list("Still to learn", c.still_wrong, "warn"), list("Kept", c.still_right),
      el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, "Concept"), el("th", { class: "num" }, "Before"), el("th", { class: "num" }, "After"))),
        el("tbody", {}, ...c.concepts.slice(0, 30).map(([t, b, a, n]) => el("tr", {}, el("td", {}, t), el("td", { class: "num" }, `${b}/${n}`),
          el("td", { class: "num" + (a > b ? " ok" : a < b ? " bad" : "") }, `${a}/${n}`))))));
    box.append(el("p", { class: "note" }, "Remembered knowledge is what Nova answers with her memories in front; neurally learned knowledge is what she answers without them. Only the second counts as learned."));
  }
  return box;
}

function showReview(v) {
  review = v;
  const s = v.session;
  $("reviewEntry").hidden = false;
  $("reviewTitle").textContent = `${s.name} · ${s.modelName}`;
  reviewCard.set({ title: s.name, subtitle: s.modelName, status: s.after ? "Finished" : s.error ? "Error" : v.errors.length ? "Teacher review" : "Ready",
    statusKind: s.after ? "good" : s.error ? "bad" : "", footer: [studyFooter(), "Assessments without memory"], steps: reviewSteps(v) });
  const locked = !!s.training || teaching;
  $("reviewBaseline").disabled = locked || v.errors.length > 0;
  $("reviewTrain").disabled = locked || v.errors.length > 0;
  $("reviewResults").replaceChildren(reviewResults(v));
  refreshReview().catch(() => {});
}

$("reviewStudyOn").addEventListener("click", safe(async () => {
  showStudyState(await invoke("set_study_mode", { on: true }));
  await refreshReview();
}, "Study Mode"));
$("reviewGenerate").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open a model first");
  const s = await invoke("generate_review", { concepts: [...reviewPick.concepts], lessons: [...reviewPick.lessons], difficulty: Number($("reviewDifficulty").value) });
  showReview(await invoke("review_detail", { id: s.id }));
  say("Review material generated as a Designer lesson. Check every answer before assessing or training.");
}, "Generate review"));
$("reviewEdit").addEventListener("click", safe(async () => {
  if (!review) return;
  go("designer");
  await openDesign(review.session.designId);
}, "Open in Designer"));
$("reviewBaseline").addEventListener("click", safe(async () => {
  $("reviewBaseline").disabled = true;
  reviewCard.step("before", { state: "running", detail: "Asking the held-out review questions without memory…" });
  showReview(await invoke("review_assess", { id: review.session.id }));
}, "Initial assessment"));
$("reviewTrain").addEventListener("click", safe(async () => {
  const epochs = Number($("reviewEpochs").value);
  Object.assign(reviewTrack, { epochs, candidates: 1, epochNow: 0, seen: [] });
  await invoke("train_review", { id: review.session.id, epochs, learningRate: Number($("reviewRate").value) });
  setTeaching(true);
  $("reviewStop").disabled = false;
  $("reviewTrain").disabled = true;
  reviewCard.set({ status: "Training" });
  if (!review.session.before) reviewCard.step("before", { state: "running", detail: "Taking the initial assessment first…" });
}, "Review training"));
$("reviewStop").addEventListener("click", safe(async () => { await invoke("stop_teaching"); $("reviewStop").disabled = true; }, "Stop review"));
$("reviewDelete").addEventListener("click", safe(async () => {
  if (!review || !await askYes("Delete this review session? Its Designer lesson and the model are kept.")) return;
  await invoke("delete_review", { id: review.session.id });
  review = null;
  $("reviewEntry").hidden = true;
  refreshReview();
}, "Delete review"));

document.addEventListener("DOMContentLoaded", safe(async () => {
  if (!tauri?.event) return;
  await tauri.event.listen("review-event", ev => {
    const p = ev.payload;
    if (review?.session.id !== p.id) return;
    if (p.phase === "baseline") reviewCard.step("before", { state: "running", detail: "Asking the held-out review questions without memory…" });
    else if (p.phase === "after") { reviewCard.step("train", { state: "done", progress: null }); reviewCard.step("after", { state: "running", detail: "Asking the same questions again…" }); }
    else if (p.event?.type === "started") { reviewCard.step("before", { state: "done" }); reviewCard.step("train", { state: "running", progress: 0, detail: `${p.event.train_units} example(s) to learn.` }); }
    else if (p.event?.type === "batch") reviewCard.step("train", { state: "running", progress: attemptProgress(reviewTrack, p.event), progressLabel: batchLabel(reviewTrack, p.event) });
  });
  await tauri.event.listen("review-finished", ev => {
    setTeaching(false);
    $("reviewStop").disabled = true;
    if (review?.session.id === ev.payload.session.id) showReview(ev.payload);
    say(ev.payload.session.error ? `Review training ended: ${ev.payload.session.error}` : "Review training finished.");
    refreshModels();
  });
}, "Review"));

const reviewPrevPage = window.onPage;
window.onPage = page => {
  reviewPrevPage?.(page);
  if (page === "study") refreshReview().catch(e => log("ERROR", e));
};
