"use strict";
// Progress cards for Reading, the Classroom and the Planner: what is being taught as a
// timeline of steps. They follow the same Engine events as the tables and logs below them
// (each handler is wrapped, so the existing views are unchanged). Progress bars show counted
// batches and steps only.

const deviceText = () => deviceLabel().replace(/^ · /, "");
const OUTCOME_STATE = {
  passed: "done", learned: "done", progressed: "done", deferred: "waiting", kept_previous: "failed",
  unresolved: "failed", duplicate: "skipped", failed: "failed", stopped: "skipped",
};
const CAUSE_TEXT = {
  unstable: "unstable training", forgetting: "harming earlier lessons", interference: "new learning overwrites known answers",
  not_learned: "not learned yet", not_generalising: "memorised, not generalised",
};
const outcomeText = r => r.outcome === "unresolved"
  ? (r.attempts?.some(a => a.learned) ? "Attempts used up; the progress made is kept" : "Attempts used up; nothing was learned")
  : ({ passed: "Passed", learned: "Learned", progressed: "Improved, below the pass mark", deferred: "Not passed yet: moved to the end of the queue",
    kept_previous: "Not learned; the previous model is kept", duplicate: "Same text as an earlier lesson; not taught again", failed: "Not taught", stopped: "Stopped" }[r.outcome] || r.outcome);

// Where a session is within an attempt, from counted batches: epochs × candidates × batches.
function attemptProgress(t, s) {
  if (s.epoch !== t.epochNow) { t.epochNow = s.epoch; t.seen = []; }
  if (!t.seen.includes(s.focus)) t.seen.push(s.focus);
  const cands = Math.max(t.candidates || 1, t.seen.length);
  const ci = Math.max(0, t.seen.indexOf(s.focus));
  return Math.min(1, ((s.epoch - 1) + (ci + s.batch / Math.max(1, s.batches)) / cands) / Math.max(1, t.epochs || 1));
}
const batchLabel = (t, s) => `Epoch ${s.epoch} of ${t.epochs} · ${s.focus} · batch ${s.batch} of ${s.batches} · loss ${s.loss.toFixed(3)}`;

// ---- Reading -------------------------------------------------------------------------------
const readCard = new TaskCard($("readCard"));
const readTrack = { checks: 0, kept: 0, best: null, started: 0 };

$("read").addEventListener("click", () => {
  const parts = corporaList.filter(c => mix[c.id]?.read);
  if (!active || !parts.length) return;
  Object.assign(readTrack, { checks: 0, kept: 0, best: null, started: performance.now() });
  readCard.set({
    title: `Reading: ${parts.map(c => c.name).join(" + ")}`,
    subtitle: `${active.name} · ${fmt(Number($("readTokens").value))} tokens${$("readMode").value === "bicameral" ? " · bicameral stages" : ""}`,
    status: "Preparing", statusKind: "",
    footer: [studyFooter(), deviceText()],
    steps: [
      { id: "prepare", title: "Preparing the corpora", detail: "Splitting documents, holding one in ten out, learning word units.", state: "running" },
      { id: "read", title: "Reading", detail: "Waiting for the corpora.", state: "waiting" },
      { id: "keep", title: "Keeping the best checkpoint", detail: "Only checkpoints that predict unseen documents better are kept.", state: "waiting" },
    ],
  });
  // If the start was refused (an error), nothing runs: hide the card again.
  setTimeout(() => { if (!teaching && readCard.steps()[0]?.state === "running") readCard.hide(); }, 2500);
});

const baseReadEvent = onReadEvent;
onReadEvent = e => {
  baseReadEvent(e);
  if (e.type === "started") {
    readTrack.best = e.baseline_loss;
    readTrack.started = performance.now();
    readCard.set({ status: "Reading", statusKind: "" });
    readCard.step("prepare", { state: "done", title: "Corpora prepared",
      detail: `${fmt(e.train_documents)} documents to read (${fmt(e.train_tokens)} tokens), ${fmt(e.held_out_documents)} held out; ${fmt(e.new_words)} new word units. Starting held-out loss ${e.baseline_loss.toFixed(3)}.` });
    readCard.step("read", { state: "running", progress: 0, detail: `${fmt(e.steps)} training steps.` });
  } else if (e.type === "batch") {
    const secs = (performance.now() - readTrack.started) / 1000;
    const rate = e.tokens / Math.max(secs, 1e-3);
    const left = (e.steps - e.step) * (e.tokens / Math.max(1, e.step)) / Math.max(rate, 1e-9);
    const part = readParts.length > 1 ? `${readParts[e.part]?.name ?? ""} · ` : "";
    readCard.step("read", { state: "running", progress: e.step / Math.max(1, e.steps),
      progressLabel: `Step ${fmt(e.step)} of ${fmt(e.steps)} · ${part}${fmt(Math.round(rate))} tokens/s · about ${Math.max(1, Math.ceil(left / 60))} min left`,
      detail: `Train loss ${e.loss.toFixed(4)} · ${rateLine(e.learning_rate, readRateSet, e.step <= Math.min(Math.floor(e.steps / 20), 200))} · ${fmt(e.tokens)} tokens read. ` +
        (readTrack.checks ? `${readTrack.checks} held-out check(s), ${readTrack.kept} kept; best held-out loss ${readTrack.best.toFixed(3)}.` : "The first held-out check comes after about a twentieth of the steps.") });
  } else if (e.type === "evaluated") {
    readTrack.checks++;
    if (e.accepted) { readTrack.kept++; readTrack.best = e.held_out_loss; }
    readCard.step("keep", { state: readTrack.kept ? "active" : "waiting",
      detail: e.accepted ? `New best at ${fmt(e.tokens)} tokens: held-out loss ${e.held_out_loss.toFixed(3)}, saved.` : `Check at ${fmt(e.tokens)} tokens not better (${e.held_out_loss.toFixed(3)}); the best so far stays.` });
  }
};
const baseReadFinished = onReadFinished;
onReadFinished = f => {
  baseReadFinished(f);
  if (readCard.container.hidden) return;
  readCard.step("read", { state: f.error ? "failed" : f.stopped ? "skipped" : "done", progress: null,
    title: f.error ? "Reading ended with an error" : f.stopped ? "Reading stopped" : f.earlyStop ? "Reading finished early" : "Reading finished",
    detail: f.error ? f.error : `${fmt(f.tokens)} tokens read${f.earlyStop ? "; the held-out loss stopped improving" : ""}.` });
  if (readCard.steps()[0].state === "running") readCard.step("prepare", { state: f.error ? "failed" : "skipped" });
  readCard.step("keep", { state: f.error ? "failed" : f.accepted ? "done" : "skipped",
    title: f.accepted ? "Best checkpoint saved" : "Model unchanged",
    detail: f.accepted ? `Held-out loss ${f.baselineLoss.toFixed(3)} → ${f.bestLoss.toFixed(3)}.` : f.error ? "Saved checkpoints are unchanged." : "No checkpoint lowered the held-out loss." });
  readCard.set({ status: f.error ? "Error" : f.stopped ? "Stopped" : "Finished", statusKind: f.error ? "bad" : f.accepted ? "good" : "warn" });
};

// ---- Classroom -----------------------------------------------------------------------------
const teachCard = new TaskCard($("teachCard"));
const teachTrack = { epochs: 1, candidates: 1, epochNow: 0, seen: [], attempt: "" };

$("teach").addEventListener("click", () => {
  if (!active || !lessonsList.length) return;
  const education = $("optEducation").value;
  Object.assign(teachTrack, { epochs: Number($("optEpochs").value), candidates: stageFoci(education, $("optFoundation").checked).length, epochNow: 0, seen: [], attempt: "" });
  teachCard.set({
    title: `Classroom: ${lessonsList.length} lesson${lessonsList.length === 1 ? "" : "s"}`,
    subtitle: `${active.name}${education !== "off" ? " · " + $("optEducation").selectedOptions[0].textContent.split(":")[0] : ""}`,
    status: "Starting", statusKind: "",
    footer: [studyFooter(), deviceText()],
    steps: lessonsList.map((l, i) => ({ id: "L" + i, title: `Lesson ${i + 1} — ${l.name}`, detail: `${fmt(l.train)} units to teach, ${fmt(l.heldOut)} held out.`, state: "waiting" })),
  });
  setTimeout(() => { if (!teaching && teachCard.steps().every(s => s.state === "waiting")) teachCard.hide(); }, 2500);
});

const baseTeachEvent = onTeachEvent;
onTeachEvent = e => {
  baseTeachEvent(e);
  if (teachCard.container.hidden) return;
  if (e.type === "file_started") {
    teachCard.set({ subtitle: `Lesson ${e.index + 1} of ${e.files} — ${e.name}`, status: e.pass > 1 ? "Second try" : "Training" });
    teachCard.step("L" + e.index, { state: "running", progress: 0, detail: e.pass > 1 ? "Second try, after the other lessons." : "Measuring where the model starts." });
  } else if (e.type === "attempt_started") {
    teachTrack.epochs = e.epochs; teachTrack.epochNow = 0; teachTrack.seen = [];
    teachTrack.attempt = e.attempt === 0 ? "First attempt" : `Retry ${e.attempt} of ${e.attempts - 1}${e.remedy ? ": " + e.remedy : ""}`;
    teachCard.step("L" + e.index, { detail: teachTrack.attempt, progress: 0 });
  } else if (e.type === "diagnosed") {
    teachCard.step("L" + e.index, { detail: `Not passed (held-out ${(100 * e.diagnosis.heldout_rate).toFixed(0)}%): ${CAUSE_TEXT[e.diagnosis.cause] || e.diagnosis.cause}. Next: ${e.diagnosis.remedy}` });
  } else if (e.type === "session") {
    const s = e.event;
    if (s.type === "started") {
      teachCard.step("L" + e.index, { detail: `${teachTrack.attempt}. Starts with ${s.baseline.exact_correct} of ${s.baseline.exact_total} held-out questions right; ${fmt(s.train_units)} units to learn.` });
    } else if (s.type === "batch") {
      teachCard.step("L" + e.index, { progress: attemptProgress(teachTrack, s), progressLabel: batchLabel(teachTrack, s) });
    } else if (s.type === "epoch") {
      const a = s.accepted_score;
      teachCard.step("L" + e.index, { detail: `${teachTrack.attempt}. After epoch ${s.epoch}: ${a.exact_correct} of ${a.exact_total} held-out questions right, held-out loss ${a.heldout_loss.toFixed(3)}.` });
    }
  } else if (e.type === "file_finished") {
    teachCard.step("L" + e.index, { state: OUTCOME_STATE[e.outcome] || "done", progress: null,
      detail: [outcomeText(e), learningText(e)].filter(Boolean).join(". ") + "." });
  }
};
const baseTeachFinished = onTeachFinished;
onTeachFinished = f => {
  baseTeachFinished(f);
  if (teachCard.container.hidden) return;
  for (const s of teachCard.steps()) if (s.state === "running") teachCard.step(s.id, { state: f.error ? "failed" : "skipped", progress: null });
  const passed = (f.files || []).filter(r => r.outcome === "passed").length;
  teachCard.set({
    subtitle: f.error ? f.error : `${passed} of ${(f.files || []).length} lesson(s) passed`,
    status: f.error ? "Error" : f.stopped ? "Stopped" : "Finished", statusKind: f.error ? "bad" : passed ? "good" : "warn",
  });
};

// ---- Planner -------------------------------------------------------------------------------
const planCard = new TaskCard($("planCard"));
const planTrack = { epochs: 1, candidates: 1, epochNow: 0, seen: [], attempt: "", course: "" };

const basePlanEvent = onPlanEvent;
onPlanEvent = payload => {
  basePlanEvent(payload);
  const { planId, event: e } = payload;
  if (plan?.id !== planId) return;
  const footer = [studyFooter(), deviceText()];
  if (e.type === "course_started") {
    planTrack.course = e.name;
    const steps = Array.from({ length: e.lessons }, (_, i) => ({ id: "L" + i, title: `Lesson ${i + 1}`, state: "waiting" }));
    steps.push({ id: "grade", title: "Course grade", detail: "Lessons re-graded at the end of the course (70%) and the final exam (30%).", state: "waiting" });
    planCard.set({ title: `Course: ${e.name}`, subtitle: `Plan ${plan.name}${e.resumed ? " · resumed" : ""}`, status: "Training", statusKind: "", footer, steps });
    // Lesson names, from the course itself.
    const course = plan.courses.find(c => c.name === e.name);
    if (course) invoke("course_detail", { id: course.id }).then(c => {
      c.lessons.forEach((l, i) => { const s = planCard.steps().find(x => x.id === "L" + i); if (s && s.title === `Lesson ${i + 1}`) s.title = `Lesson ${i + 1} — ${l.name}`; });
      planCard.render();
    }).catch(() => {});
  } else if (e.type === "pretest") {
    planCard.set({ subtitle: `Plan ${plan.name} · pre-test ${percent(e.score)} (ungraded)` });
  } else if (e.type === "lesson") {
    const c = e.event, id = "L" + c.index;
    if (c.type === "file_started") {
      planCard.set({ subtitle: `Lesson ${c.index + 1} — ${c.name} · ${stageName(e.stage)}`, status: c.pass > 1 ? "Second try" : "Training" });
      planCard.step(id, { title: `Lesson ${c.index + 1} — ${c.name}`, state: "running", progress: 0, detail: `${stageName(e.stage)}${c.pass > 1 ? " · second try" : ""}.` });
    } else if (c.type === "attempt_started") {
      Object.assign(planTrack, { epochs: c.epochs, epochNow: 0, seen: [], candidates: 1 });
      planTrack.attempt = c.attempt === 0 ? "First attempt" : `Retry ${c.attempt}${c.remedy ? ": " + c.remedy : ""}`;
      planCard.step(id, { detail: planTrack.attempt, progress: 0 });
    } else if (c.type === "diagnosed") {
      planCard.step(id, { detail: `Not passed yet (${percent(c.diagnosis.heldout_rate)}): ${CAUSE_TEXT[c.diagnosis.cause] || c.diagnosis.cause}.` });
    } else if (c.type === "session" && c.event.type === "batch") {
      planCard.step(id, { progress: attemptProgress(planTrack, c.event), progressLabel: batchLabel(planTrack, c.event) });
    }
  } else if (e.type === "lesson_taught") {
    planCard.step("L" + e.index, { state: OUTCOME_STATE[e.outcome] || "done", progress: null,
      detail: [`${outcomeText(e)}: ${percent(e.when_taught)} of its evaluation questions right`, e.record ? learningText(e.record) : ""].filter(Boolean).join(". ") + "." });
  } else if (e.type === "graded") {
    const g = e.grade;
    planCard.step("grade", { state: g.passed ? "done" : "failed", title: e.after_attempt ? `Grade after remediation ${e.after_attempt}` : "Course grade",
      detail: `${percent(g.grade)} (required ${percent(g.required)})${g.failed_lessons.length ? `; below their mark: ${g.failed_lessons.join(", ")}` : ""}.` });
  } else if (e.type === "remediation_started") {
    Object.assign(planTrack, { epochNow: 0, seen: [], epochs: 1 });
    planCard.step("R" + e.attempt, { title: `Remediation ${e.attempt} of 3`, state: "running", progress: null, detail: `${e.description}. Lessons: ${e.target_names.join(", ")}.` });
    planCard.set({ status: "Remediation" });
  } else if (e.type === "remediation") {
    if (e.event.type === "batch") planCard.step("R" + e.attempt, { progress: null, detail: `${e.name} · epoch ${e.event.epoch} · ${e.event.focus} · batch ${e.event.batch} of ${e.event.batches}` });
  } else if (e.type === "remediation_finished") {
    planCard.step("R" + e.attempt, { state: e.reverted ? "failed" : "done", detail: e.grade_after != null ? `Grade ${percent(e.grade_after)}${e.reverted ? ": lower, so the best checkpoint was restored" : ""}.` : "Finished." });
  } else if (e.type === "course_finished") {
    planCard.set({ status: { passed: "Passed", failed: "Failed", timed_out: "Out of time" }[e.phase] || e.phase, statusKind: e.phase === "passed" ? "good" : "bad",
      subtitle: `Plan ${plan.name} · best grade ${percent(e.best_grade)}` });
  }
};
