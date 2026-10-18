"use strict";
// Reflection: Yendory proposes and measures connections, Nova writes her own observations, the
// teacher marks them and may approve training (Rust Engine: learn::reflect).

let reflection = null;
let reflectTimer = null;
const reflectCard = new TaskCard($("reflectCard"));
const reflectTrack = { epochs: 1, candidates: 1, epochNow: 0, seen: [] };

async function refreshReflection() {
  const sources = await invoke("reflection_sources").catch(() => []);
  const sel = $("reflectSource");
  const keep = sel.value;
  sel.replaceChildren(el("option", { value: "" }, sources.length ? "Choose what to revisit…" : "Nothing to revisit yet (Study Mode memories, or lessons with Reading blocks)"),
    ...sources.map(s => el("option", { value: `${s.kind}:${s.id}` }, `${s.name} (${s.statements} statements)`)));
  if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
  const list = await invoke("list_reflections");
  $("reflectList").tBodies[0].replaceChildren(...(list.length ? list.map(r => el("tr", { class: reflection?.id === r.id ? "current-row" : null },
    el("td", {}, el("b", {}, r.material)), el("td", {}, r.modelName), el("td", { class: "num" }, `${r.marked}/${r.observations}`),
    el("td", {}, r.trained ? el("span", { class: "ok" }, "trained") : "—"), el("td", { class: "note" }, when(r.created)),
    el("td", {}, el("button", { class: "quiet", onclick: safe(async () => showReflection(await invoke("reflection_detail", { id: r.id })), "Open reflection") }, "open"))))
    : [el("tr", {}, el("td", { colspan: 6, class: "empty" }, "No reflection yet."))]));
}

function reflectSteps(s) {
  const marked = s.observations.filter(o => o.mark).length;
  const approved = s.observations.filter(o => o.mark === "right" && o.observation.trim() || o.mark === "wrong" && o.correction.trim()).length;
  const r = s.result;
  const helped = s.observations.filter(o => o.connection.help > 0).length;
  return [
    { id: "revisit", title: "Material revisited", state: "done", detail: `${s.statements} statement(s) from ${s.material}${s.memoryMode ? " (shown to Nova as memory lines)" : ""}.` },
    { id: "measure", title: "Connections measured", state: "done", detail: `${s.observations.length} pair(s) sharing a concept; in ${helped} of them one statement lowered Nova's loss on the other.` },
    { id: "nova", title: "Nova's observations", state: "done", detail: "Written by her network, unedited." },
    { id: "review", title: "Teacher review", state: r || marked === s.observations.length ? "done" : "running", detail: `${marked} of ${s.observations.length} marked; ${approved} approved for training (at least 3 needed; a quarter judge the training).` },
    { id: "train", title: "Neural training (optional)", state: r ? (r.learned ? "done" : "skipped") : s.error ? "failed" : "waiting",
      detail: r ? `${r.learned ? "Accepted" : "Not accepted; the model is unchanged"}: held-out observations ${r.baseline.exact_correct} → ${r.final_score.exact_correct} of ${r.final_score.exact_total}, held-out loss ${r.baseline.heldout_loss.toFixed(3)} → ${r.final_score.heldout_loss.toFixed(3)}; loss on the material read alone ${r.material_before.toFixed(3)} → ${r.material_after.toFixed(3)} (${r.trained} trained, ${r.held_out} held out).`
        : s.error || "Only after your approval." },
    { id: "after", title: "Reflects again", state: r ? "done" : "waiting", detail: r ? `${s.observations.filter(o => o.after != null && o.after !== o.observation).length} observation(s) changed.` : "Nova writes her observations again after training." },
  ];
}

function showReflection(s) {
  reflection = s;
  $("reflectEntry").hidden = false;
  $("reflectTitle").textContent = `Reflection: ${s.material} · ${s.modelName}`;
  reflectCard.set({ title: `Reflection: ${s.material}`, subtitle: `${s.modelName} · step ${fmt(s.step)}${s.stepAfter != null ? ` → ${fmt(s.stepAfter)}` : ""}`,
    status: s.result ? "Trained" : s.error ? "Error" : "Reviewing", statusKind: s.result ? "good" : s.error ? "bad" : "",
    footer: [s.memoryMode ? "Study Mode memories" : "Reading (Study Mode not needed)", "Observations are Nova's own"], steps: reflectSteps(s) });
  const done = !!s.result;
  const body = $("reflectTable").tBodies[0];
  body.replaceChildren();
  s.observations.forEach((o, k) => {
    const c = o.connection;
    const right = el("input", { type: "radio", name: `rm${k}`, checked: o.mark === "right" ? "" : null, disabled: done ? "" : null, onchange: () => { o.mark = "right"; reflectChanged(); } });
    const wrong = el("input", { type: "radio", name: `rm${k}`, checked: o.mark === "wrong" ? "" : null, disabled: done ? "" : null, onchange: () => { o.mark = "wrong"; reflectChanged(); } });
    const tag = st => st.memory != null ? el("span", { class: "tag" }, `memory #${st.memory}`) : el("span", { class: "tag" }, st.source);
    body.append(el("tr", {},
      el("td", { class: "num" }, String(k + 1)),
      el("td", {}, el("div", {}, c.a.text, " ", tag(c.a)), el("div", {}, c.b.text, " ", tag(c.b)),
        el("div", { class: "note" }, `shares "${c.shared.join(", ")}" · ${c.help >= 0 ? "helps" : "does not help"} ${c.help.toFixed(3)} (${c.a_helps_b ? "first → second" : "second → first"})`)),
      el("td", {}, el("span", { class: "mono" }, o.observation || "(empty)"),
        o.after != null ? el("div", { class: "note" }, "After training: ", el("span", { class: "mono" }, o.after || "(empty)")) : "",
        o.target ? el("div", { class: "note" }, `${o.held_out ? "Held out to judge training" : "Trained on"} ${o.target[1] === "teacher_correction" ? "your correction" : "her observation, confirmed"}: ${o.target[0]}`) : ""),
      el("td", {}, el("label", { class: "check-inline" }, right, "right"), " ", el("label", { class: "check-inline" }, wrong, "wrong")),
      el("td", {},
        el("input", { type: "text", value: o.correction, placeholder: "Your version of the connection", disabled: done ? "" : null, oninput: e => { o.correction = e.target.value; reflectChanged(); } }),
        el("input", { type: "text", value: o.note, placeholder: "Note (not taught)", disabled: done ? "" : null, oninput: e => { o.note = e.target.value; reflectChanged(); } }))));
  });
  $("reflectTrain").disabled = done || teaching;
  refreshReflection().catch(() => {});
}

function reflectChanged() {
  $("reflectSaved").textContent = "Unsaved…";
  reflectCard.step("review", reflectSteps(reflection)[3]);
  clearTimeout(reflectTimer);
  reflectTimer = setTimeout(() => saveReflectMarks().catch(e => log("ERROR", e)), 600);
}
async function saveReflectMarks() {
  clearTimeout(reflectTimer);
  if (!reflection || reflection.result) return;
  reflection = await invoke("save_reflection_marks", { id: reflection.id, marks: reflection.observations.map(o => ({ mark: o.mark, correction: o.correction, note: o.note })) });
  $("reflectSaved").textContent = `Saved ${new Date().toLocaleTimeString()}`;
}

$("reflectStart").addEventListener("click", safe(async () => {
  const v = $("reflectSource").value;
  if (!v) throw new Error("Choose what Nova should revisit");
  if (!active) throw new Error("Open the model that should reflect");
  const [kind, ...rest] = v.split(":");
  $("reflectStart").disabled = true;
  try {
    say("Measuring connections; Nova is writing her observations…");
    showReflection(await invoke("start_reflection", { kind, id: rest.join(":"), count: Number($("reflectCount").value) }));
    say("Nova reflected. Review her observations.");
  } finally { $("reflectStart").disabled = false; }
}, "Reflect"));
$("reflectTrain").addEventListener("click", safe(async () => {
  await saveReflectMarks();
  const epochs = Number($("reflectEpochs").value);
  Object.assign(reflectTrack, { epochs, candidates: 1, epochNow: 0, seen: [] });
  await invoke("train_reflection", { id: reflection.id, epochs, learningRate: Number($("reflectRate").value) });
  setTeaching(true);
  $("reflectStop").disabled = false;
  $("reflectTrain").disabled = true;
  reflectCard.set({ status: "Training" });
  reflectCard.step("train", { state: "running", progress: 0, detail: "Learning from the approved observations; a quarter of them judge every step." });
}, "Train reflection"));
$("reflectStop").addEventListener("click", safe(async () => { await invoke("stop_teaching"); $("reflectStop").disabled = true; }, "Stop reflection"));
$("reflectDelete").addEventListener("click", safe(async () => {
  if (!reflection || !await askYes("Delete this reflection? The model is not affected.")) return;
  await invoke("delete_reflection", { id: reflection.id });
  reflection = null;
  $("reflectEntry").hidden = true;
  refreshReflection();
}, "Delete reflection"));

document.addEventListener("DOMContentLoaded", safe(async () => {
  if (!tauri?.event) return;
  await tauri.event.listen("reflection-event", ev => {
    const { id, event: s } = ev.payload;
    if (reflection?.id !== id) return;
    if (s.type === "batch") reflectCard.step("train", { progress: attemptProgress(reflectTrack, s), progressLabel: batchLabel(reflectTrack, s) });
  });
  await tauri.event.listen("reflection-finished", ev => {
    setTeaching(false);
    $("reflectStop").disabled = true;
    if (reflection?.id === ev.payload.id) showReflection(ev.payload);
    say(ev.payload.error ? `Reflection training ended: ${ev.payload.error}` : "Reflection training finished.");
    refreshModels();
  });
}, "Reflection"));

const reflectPrevPage = window.onPage;
window.onPage = page => {
  reflectPrevPage?.(page);
  if (page === "study") refreshReflection().catch(e => log("ERROR", e));
};
