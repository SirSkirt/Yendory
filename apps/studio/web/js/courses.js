"use strict";
// Course Builder and Lesson Planner. Grading, remediation and resuming run in the Rust Engine;
// this page only edits courses and plans and shows what the Engine reports.

const STAGE_NAME = { dense_knowledge: "Dense (A)", transformer_language: "Transformer (B)", cooperative: "Co-op (C)" };
const stageName = s => STAGE_NAME[s] || "Whole brain";
const SECTION_NAME = { dense: "Dense", transformer: "Transformer", coop: "Co-op", whole_brain: "Whole brain" };
const percent = x => x == null ? "—" : (100 * x).toFixed(1) + "%";

// ---- courses -------------------------------------------------------------------------------
let courses = [];
let course = null;

async function refreshCourses() {
  courses = await invoke("list_courses");
  const body = $("courseList").tBodies[0];
  body.replaceChildren();
  if (!courses.length) body.append(el("tr", {}, el("td", { colspan: 5, class: "empty" }, "No courses yet.")));
  for (const c of courses) body.append(el("tr", { class: course?.id === c.id ? "current-row" : null },
    el("td", {}, el("b", {}, c.name)),
    el("td", { class: "num" }, fmt(c.lessons)),
    el("td", { class: "num" }, percent(c.requiredGrade)),
    el("td", {}, c.ready ? el("span", { class: "ok" }, "ready") : el("span", { class: "warn" }, "needs attention")),
    el("td", {}, el("button", { class: "quiet", onclick: safe(async () => showCourse(await invoke("course_detail", { id: c.id })), "Open course") }, "open"))));
  renderPlanCourseChoices();
}

const edit = async e => showCourse(await invoke("edit_course", { id: course.id, edit: e }));

function stageChoice(l, i) {
  const value = l.stageFrom === "name" ? "name" : l.stageFrom === "default" ? "default"
    : ({ dense_knowledge: "dense", transformer_language: "transformer", cooperative: "cooperative" }[l.stage] || "whole");
  const s = el("select", { class: "stage", onchange: safe(e => edit({ type: "lessonStage", index: i, stage: e.target.value }), "Stage") },
    el("option", { value: "name" }, l.stageFrom === "name" ? `From name: ${stageName(l.stage)}` : "From name"),
    el("option", { value: "default" }, `Course default: ${stageName(course.defaultStage)}`),
    el("option", { value: "dense" }, "Dense (A)"),
    el("option", { value: "transformer" }, "Transformer (B)"),
    el("option", { value: "cooperative" }, "Co-op (C)"),
    el("option", { value: "whole" }, "Whole brain"));
  s.value = value;
  return s;
}

function showCourse(c) {
  course = c;
  $("courseEditor").hidden = false;
  $("courseTitle").textContent = c.name;
  $("courseNotes").replaceChildren(...(c.notes || []).map(n => el("p", { class: "note" }, n)));
  const checks = c.check.lessons;
  const body = $("courseLessons").tBodies[0];
  body.replaceChildren();
  if (!c.lessons.length) body.append(el("tr", {}, el("td", { colspan: 8, class: "empty" }, "Add lesson files: their names give their order and stage.")));
  c.lessons.forEach((l, i) => {
    const k = checks.find(x => x.name === l.name) || {};
    const mark = el("input", { type: "number", class: "mark", min: 0, max: 100, step: 5, value: Math.round(l.passMark * 100),
      onchange: safe(e => edit({ type: "lessonPassMark", index: i, percent: Number(e.target.value) }), "Pass mark") });
    const practice = el("div", {}, ...l.practice.map((p, j) => el("div", { class: "note" }, p, " ",
      el("button", { class: "quiet", onclick: safe(() => edit({ type: "removePractice", index: i, practice: j }), "Remove practice") }, "×"))),
      l.hasExtraPracticeBlock ? el("div", { class: "note" }, "[Extra Practice] block in the file") : "",
      k.practice != null ? el("div", { class: "note" }, `${fmt(k.practice)} practice question(s)${k.practice_excluded ? `, ${k.practice_excluded} excluded (match evaluation or exam)` : ""}`) : "",
      el("button", { class: "quiet", onclick: safe(async () => showCourse(await invoke("attach_practice", { id: course.id, index: i })), "Attach practice") }, "attach…"));
    body.append(el("tr", {},
      el("td", {}, el("input", { type: "checkbox", checked: l.together ? "" : null, "aria-label": `Teach ${l.name} together with the others of its stage`,
        onchange: safe(e => edit({ type: "lessonTogether", index: i, on: e.target.checked }), "Together") })),
      el("td", { class: "num" }, l.number ?? "—"),
      el("td", {}, l.name, l.nameProblem && l.stageFrom !== "manual" ? el("div", { class: "bad note" }, l.nameProblem) : ""),
      el("td", {}, stageChoice(l, i), l.stageFrom === "default" ? el("div", { class: "note" }, `taught as ${stageName(l.effectiveStage)}`) : ""),
      el("td", { class: "num" }, mark),
      el("td", { class: "num" }, k.train == null ? "—" : `${fmt(k.train)} · ${fmt(k.eval)} · ${fmt(k.exam)}`),
      el("td", {}, practice),
      el("td", {}, el("button", { class: "quiet", onclick: safe(() => edit({ type: "removeLesson", index: i }), "Remove lesson") }, "remove"))));
  });
  for (const [j, p] of (c.unmatchedPractice || []).entries()) body.append(el("tr", {},
    el("td", {}, ""), el("td", {}, "—"), el("td", { colspan: 5, class: "warn" }, `${p}: practice file without a lesson of the same number and stage`),
    el("td", {}, el("button", { class: "quiet", onclick: safe(() => edit({ type: "removeUnmatched", index: j }), "Remove") }, "remove"))));

  // Taught together: the ticked lessons of each stage form one mixed session.
  const together = c.lessons.filter(l => l.together);
  const byStage = new Map();
  for (const l of together) byStage.set(stageName(l.effectiveStage), [...(byStage.get(stageName(l.effectiveStage)) || []), l.name]);
  $("courseNotes").append(el("p", { class: "note" },
    el("b", {}, "Taught together: "),
    together.length < 2 ? "no group (every lesson is taught on its own). " : [...byStage].map(([st, names]) => names.length > 1 ? `${names.length} ${st} lessons in one mixed session` : `${names[0]} on its own (the only ticked ${st} lesson)`).join("; ") + ". ",
    "Ticked lessons of the same stage are taught in one session, every batch drawing on all of them like reading; the rest one at a time afterwards. Remediation practises them together too. ",
    el("button", { type: "button", class: "quiet", onclick: safe(() => edit({ type: "allTogether", on: true }), "Together") }, "Tick all"), " ",
    el("button", { type: "button", class: "quiet", onclick: safe(() => edit({ type: "allTogether", on: false }), "Together") }, "Untick all")));

  // Exam and checks.
  const ck = c.check;
  const sections = ck.sections.map(([s, n, w]) => `${SECTION_NAME[s] || s} ${fmt(n)} question(s), ${(100 * w).toFixed(1)}%`).join(" · ");
  $("courseCheck").replaceChildren(
    el("p", {}, el("b", {}, "Final exam: "), c.exam ? `${c.exam} (teacher-written) ` : "assembled from about 10% of each lesson's questions, never trained and never used for the lesson grades ",
      c.exam ? el("button", { class: "quiet", onclick: safe(() => edit({ type: "removeExam" }), "Remove exam") }, "remove") : "",
      el("div", { class: "note" }, sections || "no exam questions yet")),
    el("p", {}, el("b", {}, "To pass: "), `course grade (70% lessons at course end + 30% exam) of at least ${percent(ck.required_grade)}, the average of the lesson pass marks`,
      c.examMinimum != null ? `, and an exam score of at least ${percent(c.examMinimum)}` : "", "."),
    ck.problems.length ? el("ul", { class: "check-list bad" }, ...ck.problems.map(p => el("li", {}, p))) : el("p", { class: "ok" }, "Ready to teach."),
    ck.warnings.length ? el("ul", { class: "check-list warn" }, ...ck.warnings.map(p => el("li", {}, p))) : "",
    el("p", { class: "note mono" }, "version " + ck.fingerprint.slice(0, 12)));

  // Settings.
  $("cDefaultStage").value = c.defaultStage || "";
  $("cAllMarks").value = Math.round(c.defaultPassMark * 100);
  $("wDense").value = c.sectionWeights.dense;
  $("wTransformer").value = c.sectionWeights.transformer;
  $("wCoop").value = c.sectionWeights.coop;
  $("wWhole").value = c.sectionWeights.whole_brain;
  $("cExamMinOn").checked = c.examMinimum != null;
  if (c.examMinimum != null) $("cExamMin").value = Math.round(c.examMinimum * 100);
  $("cPretest").checked = c.pretest;
  const s = c.settings;
  $("cEpochs").value = s.epochs; $("cBatch").value = s.batch_size; $("cRate").value = s.learning_rate;
  $("cRetries").value = s.retries; $("cFlat").value = s.flat_epochs ?? 4; $("cTimeLimit").value = s.time_limit_hours ?? "";
  $("cFoundation").checked = s.foundation; $("cRehearsal").checked = s.rehearsal;
  $("cProtect").checked = s.protect_answers; $("cRequeue").checked = s.requeue;
  $("cTrainEarlier").checked = s.train_earlier ?? true; $("cRecoverEarlier").checked = s.recover_earlier ?? true;
  $("cBackoff").checked = s.backoff ?? true; $("cOverrideFirst").checked = s.override_first_epoch ?? false; $("cProbation").value = s.probation_epochs ?? 0;
  refreshCoursesQuiet();
}
const refreshCoursesQuiet = () => refreshCourses().catch(e => log("ERROR", e));

$("newCourse").addEventListener("click", safe(async () => {
  const name = await askText("Name of the new course (e.g. Basic English I)");
  if (!name) return;
  showCourse(await invoke("create_course", { name }));
}, "New course"));
$("addCourseFiles").addEventListener("click", safe(async () => showCourse(await invoke("add_course_files", { id: course.id })), "Add files"));
$("chooseExam").addEventListener("click", safe(async () => showCourse(await invoke("choose_exam", { id: course.id })), "Exam"));
$("renameCourse").addEventListener("click", safe(async () => {
  const name = await askText("New name for this course", course.name);
  if (name) await edit({ type: "rename", name });
}, "Rename course"));
$("deleteCourse").addEventListener("click", safe(async () => {
  if (!await askYes(`Delete the course "${course.name}"? Plans that use it will stop at it.`)) return;
  await invoke("delete_course", { id: course.id });
  course = null;
  $("courseEditor").hidden = true;
  await refreshCourses();
}, "Delete course"));
$("cApplyMarks").addEventListener("click", safe(() => edit({ type: "allPassMarks", percent: Number($("cAllMarks").value) }), "Pass marks"));
$("cDefaultStage").addEventListener("change", safe(() => edit({ type: "defaultStage", stage: $("cDefaultStage").value || null }), "Default stage"));
for (const id of ["wDense", "wTransformer", "wCoop", "wWhole"]) $(id).addEventListener("change", safe(() => edit({ type: "weights", weights: {
  dense: Number($("wDense").value), transformer: Number($("wTransformer").value), coop: Number($("wCoop").value), whole_brain: Number($("wWhole").value),
} }), "Weights"));
for (const id of ["cExamMinOn", "cExamMin"]) $(id).addEventListener("change", safe(() => edit({ type: "examMinimum", percent: $("cExamMinOn").checked ? Number($("cExamMin").value) : null }), "Exam minimum"));
$("cPretest").addEventListener("change", safe(() => edit({ type: "pretest", on: $("cPretest").checked }), "Pre-test"));
for (const id of ["cEpochs", "cBatch", "cRate", "cRetries", "cFlat", "cTimeLimit", "cFoundation", "cRehearsal", "cProtect", "cRequeue", "cTrainEarlier", "cRecoverEarlier", "cBackoff", "cOverrideFirst", "cProbation"]) $(id).addEventListener("change", safe(() => edit({ type: "settings", settings: {
  // Settings not shown here keep their values.
  ...course.settings,
  epochs: Number($("cEpochs").value), batch_size: Number($("cBatch").value), learning_rate: Number($("cRate").value),
  retries: Number($("cRetries").value), flat_epochs: Number($("cFlat").value), time_limit_hours: $("cTimeLimit").value === "" ? null : Number($("cTimeLimit").value),
  foundation: $("cFoundation").checked, rehearsal: $("cRehearsal").checked, protect_answers: $("cProtect").checked, requeue: $("cRequeue").checked,
  train_earlier: $("cTrainEarlier").checked, recover_earlier: $("cRecoverEarlier").checked,
  backoff: $("cBackoff").checked, override_first_epoch: $("cOverrideFirst").checked, probation_epochs: Math.max(0, Math.min(10, Number($("cProbation").value) || 0)),
} }), "Training settings"));

// ---- plans ---------------------------------------------------------------------------------
let plan = null;
let planRunning = false;
let overrideArmed = false;
const live = { lessons: 0, taught: 0, line: "" };

async function refreshPlans() {
  const plans = await invoke("list_plans");
  const body = $("planList").tBodies[0];
  body.replaceChildren();
  if (!plans.length) body.append(el("tr", {}, el("td", { colspan: 5, class: "empty" }, "No plans yet.")));
  for (const p of plans) body.append(el("tr", { class: plan?.id === p.id ? "current-row" : null },
    el("td", {}, el("b", {}, p.name)), el("td", {}, p.modelName),
    el("td", { class: "num" }, `${Math.min(p.position, p.courses)} of ${p.courses}`),
    el("td", {}, statusLabel(p.status)),
    el("td", {}, el("button", { class: "quiet", onclick: safe(async () => showPlan(await invoke("plan_detail", { id: p.id })), "Open plan") }, "open"))));
}
function statusLabel(s) {
  const t = { idle: ["", "ready"], running: ["ok", "running"], stopped: ["warn", "stopped (resumable)"], finished: ["ok", "finished"], failed: ["bad", "stopped: course failed"], needs_attention: ["warn", "needs attention"] }[s] || ["", s];
  return el("span", { class: t[0] }, t[1]);
}
function renderPlanCourseChoices() {
  const sel = $("planAddCourse");
  sel.replaceChildren(...courses.map(c => el("option", { value: c.id }, c.name)));
}

const setCourses = async ids => showPlan(await invoke("edit_plan", { id: plan.id, name: null, courses: ids }));

function showPlan(p) {
  plan = p;
  overrideArmed = false;
  $("planEditor").hidden = false;
  $("planTitle").textContent = `${p.name} · teaches ${p.modelName}`;
  $("planMessage").replaceChildren(statusLabel(p.status), p.message ? " — " + p.message : "");
  // Passed courses, the course in progress and a course that failed stay where they are.
  const lastRun = [...p.results].reverse().find(r => r.courseId === p.courses[p.position]?.id);
  const fixed = p.position + (p.run || ["failed", "needs_attention"].includes(p.status) || (lastRun && lastRun.phase !== "passed") ? 1 : 0);
  const body = $("planCourses").tBodies[0];
  body.replaceChildren();
  if (!p.courses.length) body.append(el("tr", {}, el("td", { colspan: 6, class: "empty" }, "Add reading, saved courses and conversation practice in the order they should happen.")));
  const ids = p.courses.map(c => c.id);
  p.courses.forEach((c, i) => {
    const movable = i >= fixed && !planRunning;
    const move = (d) => { const a = [...ids]; [a[i], a[i + d]] = [a[i + d], a[i]]; return setCourses(a); };
    const cls = { passed: "ok", failed: "bad", current: "", pending: "", missing: "bad" }[c.state];
    const step = c.kind && c.kind !== "course";
    body.append(el("tr", { class: c.state === "current" ? "current-row" : null },
      el("td", { class: "num" }, i + 1), el("td", {}, step ? el("span", { class: "plan-kind " + c.kind }, c.kind === "reading" ? "Reading" : "Conversation") : "",
        step ? " " + c.name.replace(/^(Reading|Conversation): /, "") : c.name,
        c.lessons ? el("div", { class: "note" },
        c.together >= 2 ? `${c.lessons} lessons, ${c.together === c.lessons ? "all" : c.together} taught together` : `${c.lessons} lessons, taught one at a time`) : "",
        c.summary ? el("div", { class: "note" }, c.summary) : ""),
      el("td", { class: "num" }, step ? "—" : percent(c.required)), el("td", { class: "num" }, step || c.state === "pending" ? "—" : percent(c.grade)),
      el("td", {}, el("span", { class: cls }, c.state)),
      el("td", {}, movable && i > fixed ? el("button", { class: "quiet", onclick: safe(() => move(-1), "Move") }, "↑") : "",
        movable && i < ids.length - 1 ? el("button", { class: "quiet", onclick: safe(() => move(1), "Move") }, "↓") : "",
        movable ? el("button", { class: "quiet", onclick: safe(() => setCourses(ids.filter((_, j) => j !== i)), "Remove") }, "remove") : "")));
  });
  const r = p.restart;
  $("planRestart").hidden = !r || planRunning;
  $("planRestart").textContent = r === "fresh" ? "Restart revised course" : r === "needs_override" ? "Give one more chance (override)…" : "Restart course";
  $("planRestart").disabled = r === "refused";
  $("planRestart").title = r === "refused" ? "This version already failed after its override: revise the course first" :
    r === "needs_override" ? "This version of the course failed before; one override is allowed per version" : "The revised course starts fresh with three remediation attempts; the model keeps what it learned";
  if (r === "refused") $("planMessage").append(el("div", { class: "warn" }, "This version of the course has used its override: revise it (practice material, lessons, pass marks or exam) to restart it."));
  updatePlanButtons();
  renderResults(p);
  refreshPlans().catch(e => log("ERROR", e));
}

function updatePlanButtons() {
  const busy = teaching || planRunning;
  const onStep = plan && plan.courses[plan.position] && plan.courses[plan.position].kind !== "course";
  $("planStart").disabled = busy || !plan || plan.status === "finished" || (plan.status === "failed" && !onStep);
  $("planStart").textContent = plan && (plan.status === "stopped" || plan.run) ? "Resume" : "Start";
  $("planStop").disabled = !(planRunning || (teaching && plan?.status === "running"));
  $("planAdd").disabled = planRunning;
  $("planAddReading").disabled = planRunning;
  $("planAddConversation").disabled = planRunning;
  $("planDelete").disabled = planRunning;
  $("newPlan").disabled = planRunning;
}
window.onTeachingChanged = () => updatePlanButtons();

function gradeBlock(g, title) {
  const lessons = el("table", { class: "ledger compact" },
    el("thead", {}, el("tr", {}, el("th", {}, "Lesson"), el("th", {}, "Stage"), el("th", { class: "num" }, "Mark"), el("th", { class: "num" }, "When taught"), el("th", { class: "num" }, "Course end"), el("th", { class: "num" }, "Change"), el("th", {}, ""))),
    el("tbody", {}, ...g.lessons.map(l => el("tr", {},
      el("td", {}, l.name), el("td", {}, stageName(l.stage)), el("td", { class: "num" }, percent(l.pass_mark)),
      el("td", { class: "num" }, percent(l.when_taught)), el("td", { class: "num" }, percent(l.course_end)),
      el("td", { class: "num" }, l.when_taught == null ? "—" : `${l.course_end >= l.when_taught ? "+" : ""}${(100 * (l.course_end - l.when_taught)).toFixed(1)}`),
      el("td", {}, l.passed ? el("span", { class: "ok" }, "met") : el("span", { class: "bad" }, "below mark"))))));
  const exam = el("table", { class: "ledger compact" },
    el("thead", {}, el("tr", {}, el("th", {}, "Exam section"), el("th", { class: "num" }, "Weight"), el("th", { class: "num" }, "Questions"), el("th", { class: "num" }, "Whole brain"), el("th", { class: "num" }, "Dense cut"))),
    el("tbody", {}, ...g.exam.sections.map(s => el("tr", {},
      el("td", {}, SECTION_NAME[s.section] || s.section), el("td", { class: "num" }, percent(s.weight)), el("td", { class: "num" }, fmt(s.questions)),
      el("td", { class: "num" }, percent(s.score)), el("td", { class: "num" }, percent(s.without_dense))))));
  return el("div", {},
    el("p", {}, el("b", {}, title + ": "), el("span", { class: g.passed ? "ok" : "bad" }, `${percent(g.grade)} ${g.passed ? "≥" : "<"} ${percent(g.required)} required`),
      ` — lessons ${percent(g.lesson_average)} × 70% + exam ${percent(g.exam.score)} × 30%`,
      g.exam_minimum != null ? ` (exam minimum ${percent(g.exam_minimum)})` : "",
      ` · exam with Dense cut ${percent(g.exam.without_dense)}, Dense reliance ${g.exam.dense_reliance.toFixed(3)} · step ${fmt(g.step)}`),
    g.failed_lessons.length ? el("p", { class: "warn note" }, `Below their own mark: ${g.failed_lessons.join(", ")}`) : "",
    lessons, exam);
}

function runReport(entry) {
  const box = el("details", { class: "report", open: "" });
  const last = entry.grades[entry.grades.length - 1];
  const phase = { passed: ["ok", "passed"], failed: ["bad", "failed after three remediation attempts"], timed_out: ["bad", "ran out of time"] }[entry.phase] || ["", entry.phase];
  box.append(el("summary", {}, el("b", {}, entry.name), " — ", el("span", { class: phase[0] }, phase[1]),
    last ? ` · grade ${percent(last.grade)}` : "", entry.overridden ? " · override run" : "",
    entry.finished ? el("span", { class: "note" }, ` · ${when(entry.finished)}`) : ""));
  if (entry.pretest) box.append(el("p", {}, el("b", {}, "Pre-test: "), `${percent(entry.pretest.score)} (ungraded)`, last ? ` → final exam ${percent(last.exam.score)}` : ""));
  const outcomes = (entry.lessons || []).filter(([, o]) => o && o !== "passed");
  if (outcomes.length) box.append(el("p", { class: "note" }, "Teaching: " + outcomes.map(([n, o]) => `${n} ${OUTCOME_LABEL[o] || o}`).join("; ")));
  entry.grades.forEach((g, k) => box.append(gradeBlock(g, k === 0 ? "Course grade after teaching" : `After remediation ${k}`)));
  for (const r of entry.remediation) box.append(el("p", { class: "note" },
    `Remediation ${r.attempt}: ${r.cause.replace("_", " ")} → ${r.description}; lessons: ${r.target_names.join(", ")}`,
    r.grade_after != null ? ` → grade ${percent(r.grade_after)}${r.reverted ? " (worse: best checkpoint restored)" : ""}` : ""));
  return box;
}

function stepReport(r) {
  return el("details", { class: "report" },
    el("summary", {}, el("b", {}, r.label), " — ", el("span", { class: r.accepted ? "ok" : "warn" }, r.accepted ? "kept the best checkpoint" : r.stopped ? "stopped" : "unchanged"),
      ` · held-out loss ${r.baselineLoss.toFixed(3)} → ${r.bestLoss.toFixed(3)} · ${fmt(r.tokens)} tokens`,
      r.finished ? el("span", { class: "note" }, ` · ${when(r.finished)}`) : ""),
    r.note ? el("p", { class: "warn note" }, r.note) : "",
    el("p", { class: "note" }, `Model step ${fmt(r.startedStep)} → ${fmt(r.endStep)}. The full record is in § 12 Library${r.kind === "conversation" ? " and on § 4 Conversation" : ""}.`));
}

function renderResults(p) {
  const box = $("planResults");
  const items = [...p.results.map(r => ({ t: r.finished, v: runReport(r) })), ...(p.stepResults || []).map(r => ({ t: r.finished, v: stepReport(r) }))]
    .sort((a, b) => b.t - a.t).map(x => x.v);
  if (p.run && p.status !== "failed") {
    const r = p.run;
    if (r.grades.length || r.pretest) items.unshift(runReport({ name: r.name + " (in progress)", phase: r.phase, overridden: r.overridden, pretest: r.pretest, grades: r.grades, remediation: r.remediation, lessons: [], finished: 0 }));
  }
  box.replaceChildren(...(items.length ? items : [el("p", { class: "empty" }, "No course finished yet.")]));
}

function planLog(text, cls) {
  const p = el("p", { class: cls || null }, el("span", { class: "when" }, new Date().toLocaleTimeString() + " "), text);
  $("planLog").prepend(p);
  while ($("planLog").childNodes.length > 300) $("planLog").lastChild.remove();
}

function onPlanEvent({ planId, event: e }) {
  if (plan?.id !== planId) return;
  $("planProgress").hidden = false;
  if (e.type === "course_started") {
    live.lessons = e.lessons; live.taught = 0;
    $("planNow").textContent = `${e.name} · ${e.resumed ? "resumed" : "started"}${deviceLabel()}`;
    planLog(`${e.resumed ? "Resumed" : "Started"} ${e.name} (${e.lessons} lessons).`);
  } else if (e.type === "pretest") {
    planLog(`Pre-test (ungraded): ${percent(e.score)}.`);
  } else if (e.type === "lesson") {
    const c = e.event;
    if (c.type === "group_started") { $("planNow").textContent = `${c.indices.length} lessons together · ${stageName(e.stage)}: ${c.names.join(", ")}`; planLog(`Taught together: ${c.names.join(", ")}${c.earlier?.length ? `, with ${c.earlier.length} saved earlier lesson(s) trained alongside (still guarded)` : ""}.`); }
    else if (c.type === "group_scored") planLog(`After attempt ${c.attempt + 1}: ` + c.indices.map((i, k) => `lesson ${i + 1} ${percent(c.rates[k])}${c.passed[k] ? " (passed)" : ""}`).join(" · "));
    else if (c.type === "file_started") $("planNow").textContent = `Lesson ${c.index + 1} of ${c.files}: ${c.name} · ${stageName(e.stage)}${c.pass > 1 ? " · second try" : ""}`;
    else if (c.type === "attempt_started") { live.line = c.attempt === 0 ? "first attempt" : `retry ${c.attempt}${c.remedy ? ": " + c.remedy : ""}`; if (c.attempt) planLog(`${live.line}`); }
    else if (c.type === "session" && c.event.type === "learning_rate_lowered") planLog(`Epoch ${c.event.epoch}: ${c.event.reason} (now ${Math.round(100 * c.event.scale)}% of the scheduled rate).`, "warn");
    else if (c.type === "session" && c.event.type === "checkpoint") planLog(c.event.error ? `Checkpoint of epoch ${c.event.epoch} could not be written: ${c.event.error}` : `Accepted epoch ${c.event.epoch} written to the checkpoint folder (step ${fmt(c.event.step)}).`, c.event.error ? "warn" : undefined);
    else if (c.type === "diagnosed") planLog(`Not passed yet (${percent(c.diagnosis.heldout_rate)}): ${c.diagnosis.cause.replace("_", " ")}.`, "warn");
    else if (c.type === "session" && c.event.type === "batch") {
      const s = c.event;
      $("planLine").textContent = `${live.line} · epoch ${s.epoch} · ${s.focus} · batch ${s.batch}/${s.batches} · loss ${s.loss.toFixed(4)}`;
    }
  } else if (e.type === "lesson_taught" || e.type === "graded") {
    invoke("status").then(st => showActive(st.active)).catch(() => {});
  }
  if (e.type === "lesson_taught") {
    live.taught += e.outcome === "deferred" ? 0 : 1;
    $("planBar").style.width = (100 * Math.min(1, live.taught / Math.max(1, live.lessons))).toFixed(1) + "%";
    planLog(`${e.name}: ${OUTCOME_LABEL[e.outcome] || e.outcome}, ${percent(e.when_taught)} of its evaluation questions right.${e.record && learningText(e.record) ? " " + learningText(e.record) + "." : ""}`, OUTCOME_CLASS[e.outcome]);
  } else if (e.type === "graded") {
    const g = e.grade;
    planLog(`${e.after_attempt ? `After remediation ${e.after_attempt}` : "Course grade"}: ${percent(g.grade)} (required ${percent(g.required)}) — ${g.passed ? "passed" : "not passed"}${g.failed_lessons.length ? `; below their mark: ${g.failed_lessons.join(", ")}` : ""}.`, g.passed ? "ok" : "warn");
  } else if (e.type === "remediation_started") {
    $("planNow").textContent = `Remediation ${e.attempt} of 3: ${e.strategy.replace("_", " ")}`;
    planLog(`Remediation ${e.attempt}: ${e.cause.replace("_", " ")} → ${e.description}. Lessons: ${e.target_names.join(", ")}.`, "warn");
  } else if (e.type === "remediation") {
    if (e.event.type === "batch") $("planLine").textContent = `remediation ${e.attempt} · ${e.name} · epoch ${e.event.epoch} · ${e.event.focus} · batch ${e.event.batch}/${e.event.batches} · loss ${e.event.loss.toFixed(4)}`;
  } else if (e.type === "remediation_finished") {
    if (e.reverted) planLog(`Remediation ${e.attempt} lowered the grade; the best checkpoint was restored.`, "warn");
  } else if (e.type === "course_finished") {
    $("planNow").textContent = `Course ${e.phase.replace("_", " ")}`;
    planLog(`Course ${e.phase.replace("_", " ")} (best grade ${percent(e.best_grade)}).`, e.phase === "passed" ? "ok" : "bad");
  }
}

$("newPlan").addEventListener("click", safe(async () => {
  if (!active) throw new Error("Open the model the plan should teach first");
  const name = await askText(`Name of the new plan for ${active.name}`);
  if (!name) return;
  showPlan(await invoke("create_plan", { name }));
}, "New plan"));
$("planAdd").addEventListener("click", safe(async () => {
  const id = $("planAddCourse").value;
  if (!id) throw new Error("Create a course first");
  await setCourses([...plan.courses.map(c => c.id), id]);
}, "Add course"));
$("planAddReading").addEventListener("click", safe(async () => {
  showPlan(await invoke("add_plan_step", { id: plan.id, kind: "reading", options: window.readingOptions() }));
  say("Reading added to the plan with the § 2 settings.");
}, "Add reading"));
$("planAddConversation").addEventListener("click", safe(async () => {
  showPlan(await invoke("add_plan_step", { id: plan.id, kind: "conversation", options: await window.conversationOptions() }));
  say("Conversation practice added to the plan with the § 4 settings.");
}, "Add conversation"));
// Progress of a plan's reading or conversation step.
function onPlanStep({ planId, kind, label, event: e }) {
  if (plan?.id !== planId) return;
  $("planProgress").hidden = false;
  const STAGE = ["short exchanges", "a few turns", "whole conversations"];
  if (e.type === "started") {
    $("planNow").textContent = label;
    planLog(`${label}: started (held-out loss ${(e.baseline_loss ?? 0).toFixed(4)}${e.baseline_context ? `, context tests ${e.baseline_context.passed}/${e.baseline_context.total}` : ""}${e.classroom_total ? `, Classroom ${e.baseline_classroom}/${e.classroom_total}` : ""}).`);
  } else if (e.type === "batch") {
    $("planBar").style.width = (100 * e.step / Math.max(1, e.steps)).toFixed(1) + "%";
    $("planLine").textContent = `${kind === "reading" ? "reading" : "conversation · " + STAGE[e.stage]} · step ${fmt(e.step)}/${fmt(e.steps)} · ${fmt(e.tokens)} tokens · loss ${e.loss.toFixed(4)}`;
  } else if (e.type === "evaluated" || e.type === "checked") {
    planLog(`${kind === "reading" ? "Reading" : "Conversation"} check at ${fmt(e.tokens)} tokens: held-out loss ${e.held_out_loss.toFixed(4)}` +
      (e.context ? `, context tests ${e.context.passed}/${e.context.total}` : "") +
      (e.classroom_total ? `, Classroom ${e.classroom_correct}/${e.classroom_total} (−${e.classroom_lost} +${e.classroom_gained})` : "") +
      ` — ${e.accepted ? "kept" : "not kept: " + (e.reason || "")}`, e.accepted ? "ok" : undefined);
  } else if (e.type === "stage") {
    planLog(`Conversation: now ${e.name}. ${e.reason}`, e.mastered ? "ok" : "warn");
  } else if (e.type === "learning_rate_lowered") {
    planLog(`${e.reason}: learning rate ×${e.scale}.`, "warn");
  } else if (e.type === "finished") {
    planLog(`${label}: ${e.accepted ? "finished, best checkpoint kept" : "finished, model unchanged"}${e.stuck ? ` — ${e.stuck}` : ""}.`, e.accepted ? "ok" : "warn");
  }
}
if (window.__TAURI__?.event?.listen) window.__TAURI__.event.listen("plan-step-event", ev => { try { onPlanStep(ev.payload); } catch (err) { log("ERROR", err); } });
$("planStart").addEventListener("click", safe(async () => {
  $("planLog").replaceChildren();
  $("planBar").style.width = "0%";
  cpuFallback = false;
  const v = await invoke("start_plan", { id: plan.id });
  planRunning = true;
  setTeaching(true);
  showPlan(v);
  say(`Plan ${v.name} is running; it continues unattended and stops only at a failed course.`);
}, "Start plan"));
$("planStop").addEventListener("click", safe(async () => {
  await invoke("stop_teaching");
  $("planStop").disabled = true;
  say("Stopping at the next batch; the lesson in progress starts again on Resume.");
}, "Stop plan"));
// The override is confirmed in the page (a second click), with the warning shown next to it.
$("planRestart").addEventListener("click", safe(async () => {
  if (plan.restart === "needs_override" && !overrideArmed) {
    overrideArmed = true;
    $("planRestart").textContent = "Confirm: retry this unchanged version (uses its one override)";
    $("planMessage").append(el("div", { class: "warn" }, "This exact version of the course failed before. Its one override lets it run again unchanged, and is recorded in the model's history; after that it must be revised. A revised course always gets a fresh start."));
    return;
  }
  const confirmOverride = overrideArmed;
  overrideArmed = false;
  showPlan(await invoke("restart_course", { id: plan.id, confirmOverride }));
}, "Restart course"));
$("planDelete").addEventListener("click", safe(async () => {
  if (!await askYes(`Delete the plan "${plan.name}"? The model and courses are kept.`)) return;
  await invoke("delete_plan", { id: plan.id });
  plan = null;
  $("planEditor").hidden = true;
  await refreshPlans();
}, "Delete plan"));

window.onPage = page => {
  if (page === "courses") refreshCourses().catch(e => log("ERROR", e));
  if (page === "planner") { refreshCourses().catch(e => log("ERROR", e)); refreshPlans().catch(e => log("ERROR", e)); }
};

document.addEventListener("DOMContentLoaded", safe(async () => {
  if (!tauri?.event) return;
  await tauri.event.listen("plan-event", ev => { try { onPlanEvent(ev.payload); } catch (err) { log("ERROR", err); } });
  await tauri.event.listen("plan-finished", ev => {
    planRunning = false;
    setTeaching(false);
    const v = ev.payload;
    if (plan?.id === v.id) showPlan(v);
    say(`Plan ${v.name}: ${v.message || v.status}`);
    refreshModels();
  });
  await refreshCourses();
}, "Courses"));

// Built-in courses: Basic English, Reasoning and Math, generated by the Engine (the same files in
// every copy of Yendory), for testing that a model learns from lessons apart from your own.
let builtinList = [];
async function renderBuiltins() {
  try { builtinList = await invoke("builtin_courses"); } catch (e) { log("ERROR", e); return; }
  const box = $("builtinCourses");
  box.replaceChildren(
    el("h3", {}, "Built-in courses"),
    el("p", { class: "note" }, "Shipped with Yendory. Each lesson teaches one pattern with many examples; its held-out questions are new combinations of the same pattern (new names, colours or numbers), so a model passes only by learning the pattern. If a model learns these but not your lessons, look at the lessons; if it learns neither, look at the model or the training settings."),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Course"), el("th", {}, "Lessons (taught · held out)"), el("th", {}, ""))),
      el("tbody", {}, ...builtinList.map(c => el("tr", {},
        el("td", {}, el("b", {}, c.name), el("div", { class: "note" }, c.description)),
        el("td", { class: "note" }, c.lessons.map(l => `${l.name.replace(/\.txt$/, "")} (${l.taught} · ${l.heldOut})`).join("; ")),
        el("td", {}, el("button", { type: "button", class: "quiet", onclick: safe(async () => {
          showCourse(await invoke("add_builtin_course", { id: c.id }));
          await refreshCourses();
          say(`${c.name} added to your courses.`);
        }, "Add built-in course") }, "Add as course")))))));
  const sel = $("builtinLessons");
  sel.replaceChildren(...builtinList.map(c => el("option", { value: c.id }, c.name)));
}
$("addBuiltinLessons").addEventListener("click", safe(async () => {
  const id = $("builtinLessons").value;
  const c = builtinList.find(x => x.id === id);
  if (!c) return;
  if (lessonsList.length && await askYes(`Replace the ${lessonsList.length} lesson(s) in the list with ${c.name}? Choose No to add them alongside (lessons are taught in file-number order, so they would mix).`)) {
    lessonsList = await invoke("remove_lesson", { index: null });
  }
  lessonsList = await invoke("add_builtin_lessons", { id });
  for (const k of Object.keys(fileState)) delete fileState[k];
  renderLessons();
  say(`${c.name}: ${c.lessons.length} lessons in the Classroom list.`);
}, "Built-in lessons"));
document.addEventListener("DOMContentLoaded", () => renderBuiltins());
