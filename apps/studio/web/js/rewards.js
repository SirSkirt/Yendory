"use strict";
// Rewards: settings, the open model's ledger (concepts, sessions, questions with their training
// emphasis, the audit log), the reward status line on the Classroom and Planner pages, and
// reward lines in the progress cards. Grading, the ledger and the emphasis are computed by the
// Rust Engine (learn::reward); this page only displays them.

let rewardConfig = null;
const RW_VALUES = [
  ["progress_budget", "Progress budget of a question's reward cycle", 0, 10, 0.05],
  ["partial_cap", "Partial milestones together: at most this share of it", 0, 1, 0.05],
  ["discovery_attempts", "Partial credit stops after this many attempts", 1, 1000, 1],
  ["first_letter", "Spelling lessons: first correct letter", 0, 1, 0.01],
  ["per_letter", "Spelling lessons: each further correct letter", 0, 1, 0.01],
  ["retention_first", "Retention: first correct answer at a checkpoint", 0, 10, 0.05],
  ["retention_repeat", "Retention: next checkpoint (then halves each time)", 0, 10, 0.05],
  ["retention_floor", "Retention: smallest bonus", 0, 1, 0.01],
  ["generalization", "Generalization: first correct never-trained question", 0, 10, 0.05],
  ["memory_share", "Answers with memories: share of the credit", 0, 1, 0.05],
  ["incorrect_reward", "Recorded for a certainly wrong answer (0 or negative)", -1, 0, 0.01],
  ["relevance_word", "Relevance (diagnostic only): per recognised word", 0, 0.1, 0.005],
  ["relevance_cap", "Relevance (diagnostic only): cap per cycle", 0, 1, 0.01],
  ["quiz_limit", "Taught questions asked per session (and per epoch with Stage 2)", 0, 500, 1],
  ["min_emphasis", "Stage 2: least practice of a unit (never zero)", 0.05, 1, 0.05],
  ["max_emphasis", "Stage 2: most practice of a unit", 1, 4, 0.1],
  ["weights.gap", "Stage 2: added at a 0% answer", 0, 5, 0.05],
  ["weights.failures", "Stage 2: added for repeated failures", 0, 5, 0.05],
  ["weights.forgotten", "Stage 2: added when an answer was forgotten", 0, 5, 0.05],
  ["weights.concept", "Stage 2: added for a weak concept", 0, 5, 0.05],
  ["weights.variation", "Stage 2: added when new variations fail at first sight", 0, 5, 0.05],
  ["weights.difficulty", "Stage 2: added for long answers", 0, 5, 0.05],
  ["weights.correct_factor", "Stage 2: a correct answer's practice times", 0.05, 1, 0.05],
  ["weights.retained_factor", "Stage 2: times this per checkpoint retained", 0.05, 1, 0.05],
];
const getPath = (o, k) => k.split(".").reduce((x, p) => x?.[p], o);
const setPath = (o, k, v) => { const ps = k.split("."); const last = ps.pop(); ps.reduce((x, p) => x[p], o)[last] = v; };
const num2 = x => x == null ? "—" : Number(x).toFixed(2);
const gradeText = p => p == null ? "uncertain" : `${Math.round(p)}%`;
const ORIGIN = { training: "taught (checkpoint)", session: "taught (during training)", held_out: "held-out", review: "review question", practice: "practice (teacher)" };

function rewardStatusText(c) {
  if (!c || !c.record) return "Rewards: off. Answers are not graded into a ledger (§ 8 Rewards).";
  return `Rewards: grading and recording on${c.weighted_training ? "; reward-guided training ON (experimental): units are practised by need" : "; reward-guided training off (ordinary training)"}${c.letter_prefix ? "; letter-prefix progress on for spelling lessons" : ""}. Details on § 8 Rewards.`;
}
function showRewardStatus() {
  document.querySelectorAll("[data-reward-status]").forEach(p => { p.textContent = rewardStatusText(rewardConfig); });
}

function renderSettings(c) {
  rewardConfig = c;
  $("rwRecord").checked = c.record;
  $("rwWeighted").checked = c.weighted_training;
  $("rwPrefix").checked = c.letter_prefix;
  $("rwValues").replaceChildren(...RW_VALUES.map(([k, label, min, max, step]) =>
    el("label", {}, `${label} (${min} to ${max})`, el("input", { type: "number", "data-key": k, min, max, step, value: getPath(c, k) }))));
  showRewardStatus();
}
function readSettings() {
  const c = structuredClone(rewardConfig);
  c.record = $("rwRecord").checked;
  c.weighted_training = $("rwWeighted").checked;
  c.letter_prefix = $("rwPrefix").checked;
  for (const input of $("rwValues").querySelectorAll("input")) {
    const v = Number(input.value);
    if (!Number.isFinite(v)) throw new Error(`${input.parentElement.firstChild.textContent}: enter a number`);
    setPath(c, input.dataset.key, input.dataset.key === "quiz_limit" || input.dataset.key === "discovery_attempts" ? Math.round(v) : v);
  }
  return c;
}

async function refreshRewards() {
  const v = await invoke("reward_view");
  renderSettings(v.config);
  const s = v.summary;
  if (!v.model || !s) {
    $("rwSummary").textContent = "Open a model to see its ledger.";
    $("rwTotals").hidden = true;
    $("rwFlags").replaceChildren();
    return;
  }
  const t = s.totals;
  $("rwSummary").textContent = s.questions
    ? `${v.model}: ${fmt(s.questions)} question(s) recorded (${fmt(s.taught)} taught, ${fmt(s.unseen)} never trained), ${fmt(t.answers)} answers graded (${fmt(t.uncertain)} uncertain: recorded, never credited). ${fmt(s.protected_refused)} answer(s) to protected exam questions refused.`
    : `${v.model}: nothing recorded yet. Teach with recording on, or use Practice or Review Training.`;
  $("rwTotals").hidden = !s.questions;
  $("rwTotals").replaceChildren(
    el("tr", {}, el("th", {}, "Progress credit"), el("td", {}, num2(t.progress)), el("th", {}, "Complete cycles"), el("td", {}, fmt(s.complete_cycles))),
    el("tr", {}, el("th", {}, "Retention credit"), el("td", {}, num2(t.retention)), el("th", {}, "Right now, network alone"), el("td", {}, fmt(s.neural_right))),
    el("tr", {}, el("th", {}, "Generalization credit"), el("td", {}, num2(t.generalization)), el("th", {}, "Right with memories"), el("td", {}, fmt(s.memory_right))),
    el("tr", {}, el("th", {}, "Relevance (diagnostic)"), el("td", {}, num2(t.relevance)), el("th", {}, "Forgotten · stuck"), el("td", {}, `${fmt(s.forgotten)} · ${fmt(s.stuck)}`)));
  $("rwFlags").replaceChildren(...s.flags.map(f => el("div", { class: "conflict" }, el("b", {}, "Investigate: "), f.detail,
    el("span", { class: "note" }, " (a pattern to look into, not a verdict)"))));
  const cb = $("rwConcepts").tBodies[0];
  const frac = ([r, n]) => n ? `${r}/${n} (${Math.round(100 * r / n)}%)` : "—";
  cb.replaceChildren(...(s.concepts.length ? s.concepts.slice(0, 40).map(c => el("tr", {},
    el("td", { class: "mono" }, c.template), el("td", { class: "num" }, fmt(c.questions)),
    el("td", { class: "num" }, frac(c.taught)), el("td", { class: "num" }, frac(c.first_sight)), el("td", { class: "num" }, frac(c.unseen))))
    : [el("tr", {}, el("td", { colspan: 5, class: "empty" }, "Nothing recorded yet."))]));
  const sb = $("rwSessions").tBodies[0];
  sb.replaceChildren(...(s.sessions.length ? s.sessions.map(x => el("tr", {},
    el("td", { class: "note" }, when(x.time)), el("td", {}, x.lesson),
    el("td", {}, x.weighted ? el("span", { class: "tag" }, "reward-guided") : "ordinary", x.learned ? "" : el("span", { class: "note" }, " (unchanged)")),
    el("td", { class: "num" }, num2(x.progress)), el("td", { class: "num" }, num2(x.retention)), el("td", { class: "num" }, num2(x.generalization)),
    el("td", { class: "num" }, x.quiz_rate == null ? "—" : `${Math.round(100 * x.quiz_rate)}%`),
    el("td", { class: "num" }, `${Math.round(100 * x.heldout_before)}% → ${Math.round(100 * x.heldout_after)}%`),
    el("td", { class: "num" }, `${x.heldout_loss_before.toFixed(3)} → ${x.heldout_loss_after.toFixed(3)}`)))
    : [el("tr", {}, el("td", { colspan: 9, class: "empty" }, "No session recorded yet."))]));
  await showQuestions();
  await showAudit(null);
}

async function showQuestions() {
  if (!active) return;
  const rows = await invoke("reward_questions", { filter: $("rwFilter").value, search: $("rwSearch").value });
  const body = $("rwQuestions").tBodies[0];
  body.replaceChildren(...(rows.length ? rows.map(q => el("tr", {},
    el("td", {}, q.question, el("div", { class: "note" }, `${q.lesson} · answer: ${q.expected}${q.unseen ? " · never trained" : ""}`),
      q.notes.length ? el("div", { class: "note" }, "Your notes: " + q.notes.join(" · ")) : ""),
    el("td", { class: "num" }, gradeText(q.lastPercent),
      q.lastMemoryPercent != null ? el("div", { class: "note" }, `with memories ${gradeText(q.lastMemoryPercent)}`) : "",
      q.firstPercent != null ? el("div", { class: "note" }, `first sight ${gradeText(q.firstPercent)}`) : ""),
    el("td", { class: "note" }, q.milestones.length ? q.milestones.join(" → ") : "—", q.complete ? el("span", { class: "ok" }, " ✓") : "",
      q.forgotten ? el("div", { class: "warn" }, "forgotten") : "", q.retained ? el("div", {}, `kept ${q.retained} checkpoint(s)`) : ""),
    el("td", { class: "num" }, num2(q.retention)),
    el("td", { class: "num" }, num2(q.generalization)),
    el("td", { class: "num", title: q.reasons.join("\n") }, q.emphasis == null ? "—" : q.emphasis.toFixed(2),
      q.reasons.length ? el("div", { class: "note" }, q.reasons[0]) : ""),
    el("td", {},
      el("button", { class: "quiet", onclick: safe(() => showAudit(q), "Reward history") }, "history"),
      el("button", { class: "quiet", onclick: safe(async () => {
        if (!await askYes("Start this question's reward cycle again? Its progress milestones can then be credited again. Grades and history are kept.")) return;
        await invoke("reward_reset_cycle", { id: q.id });
        await refreshRewards();
      }, "Reset cycle") }, "reset cycle"))))
    : [el("tr", {}, el("td", { colspan: 7, class: "empty" }, active ? "No question matches." : "Open a model."))]));
}

async function showAudit(q) {
  if (!active) return;
  const entries = await invoke("reward_audit", { question: q?.id ?? null, limit: q ? 200 : 100 });
  $("rwAuditTitle").textContent = q ? `Reward history: ${q.question}` : "Audit log";
  const credit = e => [
    e.progress ? `progress +${num2(e.progress)}${e.milestone ? ` (${e.milestone})` : ""}` : "",
    e.retention ? `retention +${num2(e.retention)}` : "",
    e.generalization ? `generalization +${num2(e.generalization)}` : "",
    e.relevance ? `relevance +${num2(e.relevance)} (diagnostic: ${e.recognized.join(", ")})` : "",
    e.accuracy ? `incorrect ${num2(e.accuracy)}` : "",
  ].filter(Boolean);
  $("rwAudit").tBodies[0].replaceChildren(...(entries.length ? entries.map(e => el("tr", {},
    el("td", { class: "note" }, when(e.time), el("div", {}, `step ${fmt(e.step)}`)),
    el("td", {}, e.question, el("div", { class: "note" }, `expected: ${e.expected}`)),
    el("td", { class: "note" }, ORIGIN[e.origin] || e.origin, e.with_memory ? el("div", {}, "with memories") : "", e.event ? el("div", {}, e.event) : ""),
    el("td", { class: "mono" }, e.reply || "(empty)", e.note ? el("div", { class: "note" }, `Note: ${e.note}`) : ""),
    el("td", {}, el("b", { class: e.grade.correct ? "ok" : e.grade.percent == null ? "warn" : "" }, gradeText(e.grade.percent)), el("div", { class: "note" }, e.grade.reason)),
    el("td", { class: "note" }, credit(e).join("; ") || "none", e.previous.length ? el("div", {}, `already credited: ${e.previous.join(", ")}`) : ""),
    el("td", { class: "note" }, e.withheld.join("; ") || "—")))
    : [el("tr", {}, el("td", { colspan: 7, class: "empty" }, "No answers recorded yet."))]));
}

$("rwSave").addEventListener("click", safe(async () => {
  renderSettings(await invoke("save_reward_settings", { config: readSettings() }));
  $("rwSaved").textContent = `Saved ${new Date().toLocaleTimeString()}`;
}, "Save reward settings"));
$("rwDefaults").addEventListener("click", safe(async () => {
  if (!await askYes("Restore the default reward settings? (Recording on, reward-guided training off.)")) return;
  renderSettings(await invoke("reset_reward_settings"));
  $("rwSaved").textContent = "Defaults restored";
}, "Restore reward defaults"));
$("rwWeighted").addEventListener("change", () => { if ($("rwWeighted").checked) $("rwRecord").checked = true; });
$("rwRecord").addEventListener("change", () => { if (!$("rwRecord").checked) $("rwWeighted").checked = false; });
$("rwShow").addEventListener("click", safe(showQuestions, "Show questions"));
$("rwFilter").addEventListener("change", safe(showQuestions, "Show questions"));
$("rwAuditAll").addEventListener("click", safe(() => showAudit(null), "Audit log"));
$("rwClear").addEventListener("click", safe(async () => {
  if (!active || !await askYes(`Clear the whole reward ledger of ${active.name}? Its grades, credit and history are deleted; the model itself is not affected.`)) return;
  await invoke("reward_clear");
  await refreshRewards();
}, "Clear ledger"));

// ---- Reward lines in progress cards ---------------------------------------------------------
function rewardedText(s) {
  const credit = [s.progress && `progress +${num2(s.progress)}`, s.retention && `retention +${num2(s.retention)}`,
    s.generalization && `generalization +${num2(s.generalization)}`].filter(Boolean).join(", ");
  if (s.when === "start") return `Rewards: asked ${s.quizzed} taught question(s), ${s.quiz_right} right (network alone)${credit ? `; ${credit}` : ""}.`;
  return `Rewards this session: ${credit || "no new credit"}${s.flags.length ? `; ${s.flags.length} pattern(s) to investigate (§ 8 Rewards)` : ""}.`;
}
function emphasisText(e) {
  const p = e.plan;
  const top = p.top[0];
  return `Reward-guided epoch ${e.epoch}: ${p.quiz_right} of ${p.quizzed} asked were right; ${p.raised} unit(s) practised more, ${p.lowered} less (${p.at_minimum} at the minimum, never left out); ${p.copies} units scheduled for ${p.units}.${top ? ` Most practice: "${top.question}" ×${top.value.toFixed(2)} (${top.reasons[0] || ""}).` : ""}`;
}
function rewardStep(card, id) {
  return s => {
    if (card.container.hidden) return;
    if (s.type === "rewarded") card.step(id, { detail: rewardedText(s) });
    else if (s.type === "emphasis") card.step(id, { detail: emphasisText(s) });
  };
}
{
  const prevTeach = onTeachEvent;
  onTeachEvent = e => {
    prevTeach(e);
    if (e.type === "session") rewardStep(teachCard, "L" + e.index)(e.event);
  };
  const prevPlan = onPlanEvent;
  onPlanEvent = payload => {
    prevPlan(payload);
    const { planId, event: e } = payload;
    if (plan?.id !== planId) return;
    if (e.type === "lesson" && e.event.type === "session") rewardStep(planCard, "L" + e.event.index)(e.event.event);
    else if (e.type === "remediation") rewardStep(planCard, "R" + e.attempt)(e.event);
  };
}
document.addEventListener("DOMContentLoaded", safe(async () => {
  try { rewardConfig = await invoke("reward_settings"); showRewardStatus(); } catch { /* outside the app */ }
  if (!tauri?.event) return;
  await tauri.event.listen("practice-event", ev => { if (practice?.id === ev.payload.id) rewardStep(practiceCard, "train")(ev.payload.event); });
  await tauri.event.listen("review-event", ev => { if (ev.payload.event && typeof reviewCard !== "undefined") rewardStep(reviewCard, "train")(ev.payload.event); });
}, "Rewards"));

const rewardsPrevPage = window.onPage;
window.onPage = page => {
  rewardsPrevPage?.(page);
  if (page === "rewards") refreshRewards().catch(e => log("ERROR", e));
  if (page === "classroom" || page === "planner") invoke("reward_settings").then(c => { rewardConfig = c; showRewardStatus(); }).catch(() => {});
};
