"use strict";
// Lesson Designer. Analysis, checks and the lesson file come from the Rust Engine
// (learn::designer); this page edits drafts and saves them.

const KIND_LABEL = {
  reading: "Reading", lesson: "Lesson", examples: "Examples", practice: "Practice",
  extra_practice: "Extra Practice", evaluation: "Evaluation", raw: "Unified text", unsorted: "Unsorted",
};
const KIND_NOTE = {
  reading: "Nova reads this text (neural reading, Study Mode memory, or both).",
  lesson: "Instruction: the concept, its principles and relationships. Taught as explanation before the examples after it.",
  examples: "Worked examples and demonstrations with their answers (trained).",
  practice: "Questions Nova answers herself. Trained like examples when taught automatically; asked one by one in a practice session (§ 7 Study).",
  extra_practice: "Reserve exercises a course uses when the lesson needs more practice.",
  evaluation: "Held-out questions: never taught, never studied into memory; they grade the lesson.",
  raw: "Unified lesson text kept as written.",
  unsorted: "Material the analysis could not place. Never taught: move it into a block or delete it.",
};
const ITEM_KINDS = ["examples", "practice", "extra_practice", "evaluation"];
const ANSWER_LABEL = { source: "from the file", generated: "generated: verify", teacher: "teacher", missing: "missing" };
const ORIGIN_LABEL = { imported: "imported", generated: "generated", teacher: "teacher" };

let designs = [];
let design = null;   // the last view from the Engine (issues, lesson file)
let draft = null;    // the draft being edited
let designTimer = null;
let designDirty = false;
const designEditing = new Set();  // blocks shown as source instead of preview

const newId = p => p + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

async function refreshDesigns() {
  designs = await invoke("list_designs");
  const body = $("designList").tBodies[0];
  body.replaceChildren();
  if (!designs.length) body.append(el("tr", {}, el("td", { colspan: 6, class: "empty" }, "No lessons yet. Import a dataset or start an empty lesson.")));
  for (const d of designs) body.append(el("tr", { class: draft?.id === d.id ? "current-row" : null },
    el("td", {}, el("b", {}, d.name)),
    el("td", { class: "note" }, d.sourceFile || "—"),
    el("td", { class: "num" }, `${fmt(d.blocks)} · ${fmt(d.items)}`),
    el("td", {}, d.finished ? el("span", { class: "ok" }, "finished") : "draft",
      d.errors ? el("span", { class: "warn" }, ` · ${d.errors} to fix`) : ""),
    el("td", { class: "note" }, when(d.updated)),
    el("td", {}, el("button", { class: "quiet", onclick: safe(() => openDesign(d.id), "Open lesson") }, "open"))));
}

async function openDesign(id) {
  await flushDesign();
  showDesign(await invoke("design_detail", { id }));
  refreshDesigns();
  $("designEditor").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showDesign(view) {
  design = view;
  draft = structuredClone(view.draft);
  designEditing.clear();
  $("designEditor").hidden = false;
  $("designName").value = draft.name;
  $("designStage").value = draft.stage || "";
  $("designFinished").checked = !!draft.finished;
  $("designNotes").value = draft.notes || "";
  $("designSource").textContent = !draft.source_file ? "Started from nothing."
    : draft.source_sha256 ? `Imported from ${draft.source_file} (fingerprint ${draft.source_sha256.slice(0, 12)}…). Every block and question keeps the line it came from.`
    : `Made from ${draft.source_file}. Every question names what it was made from.`;
  $("designPreviewPane").hidden = true;
  $("designFilePane").hidden = true;
  renderChecks();
  renderBlocks();
  fillCourseChoice();
}

// ---- saving --------------------------------------------------------------------------------
function changed(structural = false) {
  designDirty = true;
  $("designSaved").textContent = "Unsaved changes…";
  if (structural) renderBlocks();
  clearTimeout(designTimer);
  designTimer = setTimeout(() => flushDesign().catch(e => log("ERROR", e)), 700);
}

async function flushDesign() {
  clearTimeout(designTimer);
  if (!draft || !designDirty) return;
  designDirty = false;
  const view = await invoke("save_design", { draft });
  design = view;
  $("designSaved").textContent = `Saved ${new Date().toLocaleTimeString()}`;
  renderChecks();
  if (!$("designFilePane").hidden) showLessonFile();
  refreshDesigns();
}

// ---- checks --------------------------------------------------------------------------------
function renderChecks() {
  const issues = design?.issues || [];
  const lesson = design?.lesson;
  const ok = lesson && lesson.Ok;
  $("designSummary").textContent = ok
    ? `As a lesson file (${design.fileName}): ${fmt(ok.taught)} units taught, ${fmt(ok.held_out)} held out for grading, ${fmt(ok.reading)} reading unit(s)${ok.reading_memory ? ` (${ok.reading_memory} studied into memory with Study Mode on)` : ""}, ${fmt(ok.extra_practice)} extra practice.${ok.skipped.length ? ` Left out: ${ok.skipped.length} (${ok.skipped.slice(0, 3).join("; ")}${ok.skipped.length > 3 ? "; …" : ""}).` : ""}`
    : `The lesson file cannot be written yet: ${lesson?.Err || "—"}`;
  const order = { error: 0, warning: 1, info: 2 };
  const list = [...issues].sort((a, b) => order[a.severity] - order[b.severity]);
  $("designIssues").replaceChildren(...(list.length ? list.map(i => el("li", {
    class: i.severity,
    onclick: () => {
      const target = (i.item && document.querySelector(`[data-item="${CSS.escape(i.item)}"]`)) || (i.block && document.querySelector(`[data-block="${CSS.escape(i.block)}"]`));
      if (target) { target.scrollIntoView({ behavior: "smooth", block: "center" }); target.classList.remove("flash"); void target.offsetWidth; target.classList.add("flash"); }
    },
  }, i.message)) : [el("li", { class: "info" }, "No problems found.")]));
  // Items with problems are highlighted in their tables.
  document.querySelectorAll(".design-items tr.issue").forEach(r => r.classList.remove("issue"));
  for (const i of issues) if (i.item && i.severity !== "info") document.querySelector(`[data-item="${CSS.escape(i.item)}"]`)?.classList.add("issue");
}

function showLessonFile() {
  const ok = design?.lesson?.Ok;
  $("designFilePane").textContent = ok ? ok.text : `Cannot be written: ${design?.lesson?.Err}`;
}

// ---- blocks --------------------------------------------------------------------------------
function blankBlock(kind) {
  return { id: newId("t"), kind, title: "", text: "", items: [], mode: "neural", use_in_teaching: true, origin: "teacher", edited: false, lines: null };
}
function blankItem() {
  return { id: newId("q"), question: "", answer: "", answer_source: "missing", verified: false, origin: "teacher", edited: false, source: null, flags: [], note: "" };
}

function originTag(b) {
  const where = b.lines ? `lines ${b.lines[0]}–${b.lines[1]}` : "";
  return el("span", { class: "tag" }, [ORIGIN_LABEL[b.origin] || b.origin, where, b.edited ? "edited" : ""].filter(Boolean).join(" · "));
}

function moveBlock(i, delta) {
  const j = i + delta;
  if (j < 0 || j >= draft.blocks.length) return;
  [draft.blocks[i], draft.blocks[j]] = [draft.blocks[j], draft.blocks[i]];
  changed(true);
}

function mergeNext(i) {
  const a = draft.blocks[i], b = draft.blocks[i + 1];
  if (!b) return;
  const aItems = ITEM_KINDS.includes(a.kind), bItems = ITEM_KINDS.includes(b.kind);
  if (aItems !== bItems) { say("Only blocks of questions merge with blocks of questions, and text with text."); return; }
  if (aItems) a.items.push(...b.items); else a.text = [a.text, b.text].filter(t => t.trim()).join("\n\n");
  if (a.lines && b.lines) a.lines = [Math.min(a.lines[0], b.lines[0]), Math.max(a.lines[1], b.lines[1])];
  a.edited = true;
  draft.blocks.splice(i + 1, 1);
  changed(true);
}

function renderBlocks() {
  const host = $("designBlocks");
  host.replaceChildren();
  if (!draft.blocks.length) host.append(el("p", { class: "empty" }, "No blocks yet. Add one above."));
  draft.blocks.forEach((b, i) => host.append(blockCard(b, i)));
  renderChecks();
}

function blockCard(b, i) {
  const card = el("div", { class: `design-block k-${b.kind}${b.use_in_teaching ? "" : " unused"}`, "data-block": b.id });
  const kind = el("select", { "aria-label": "Block type", onchange: e => {
    const to = e.target.value;
    const fromItems = ITEM_KINDS.includes(b.kind), toItems = ITEM_KINDS.includes(to);
    if (fromItems !== toItems && (b.items.length || b.text.trim())) {
      // Text <-> questions: keep the content in the form the new type uses.
      if (toItems) { b.items = b.text.split("\n").map(l => l.trim()).filter(Boolean).map(l => ({ ...blankItem(), question: l, origin: b.origin })); b.text = ""; }
      else { b.text = b.items.map(it => it.answer ? `${it.question}\n${it.answer}` : it.question).join("\n\n"); b.items = []; }
    }
    b.kind = to; b.edited = true;
    if (to === "unsorted") b.use_in_teaching = false;
    changed(true);
  } }, ...Object.entries(KIND_LABEL).map(([v, t]) => el("option", { value: v }, t)));
  kind.value = b.kind;
  const title = el("input", { type: "text", class: "title", placeholder: "Title (optional)", value: b.title || "", oninput: e => { b.title = e.target.value; b.edited = true; changed(); } });
  const use = el("label", { class: "check-inline" }, el("input", { type: "checkbox", checked: b.use_in_teaching ? "" : null, disabled: b.kind === "unsorted" ? "" : null,
    onchange: e => { b.use_in_teaching = e.target.checked; card.classList.toggle("unused", !b.use_in_teaching); changed(); } }), "Use in teaching");
  const tools = el("span", { class: "db-tools" },
    el("button", { class: "quiet", title: "Move up", onclick: () => moveBlock(i, -1) }, "↑"),
    el("button", { class: "quiet", title: "Move down", onclick: () => moveBlock(i, 1) }, "↓"),
    el("button", { class: "quiet", title: "Merge with the next block", onclick: () => mergeNext(i) }, "merge"),
    el("button", { class: "quiet", title: "Delete block", onclick: async () => {
      if ((b.items.length || b.text.trim()) && !await askYes(`Delete this ${KIND_LABEL[b.kind]} block and its content?`)) return;
      draft.blocks.splice(i, 1); changed(true);
    } }, "delete"));
  card.append(el("div", { class: "db-head" }, kind, title, use, originTag(b), tools),
    el("p", { class: "note" }, KIND_NOTE[b.kind] || ""));
  if (b.kind === "reading") {
    const mode = el("select", { "aria-label": "Reading mode", onchange: e => { b.mode = e.target.value; b.edited = true; changed(); } },
      el("option", { value: "neural" }, "Neural reading: the weights learn from the text"),
      el("option", { value: "memory" }, "Study Mode memory only: no weight change"),
      el("option", { value: "combined" }, "Combined: neural reading and Study Mode memory"));
    mode.value = b.mode || "neural";
    card.append(el("div", { class: "actions" }, mode,
      b.mode !== "neural" ? el("span", { class: studyModeOn ? "note" : "note warn" },
        studyModeOn ? "Study Mode is on: memories are formed when the lesson is taught." : "Study Mode is off: only neural reading happens until it is turned on (§ 7 Study).") : ""));
  }
  if (ITEM_KINDS.includes(b.kind)) card.append(itemTable(b));
  else card.append(textEditor(b));
  return card;
}

function textEditor(b) {
  const wrap = el("div");
  const editing = designEditing.has(b.id) || !b.text.trim();
  const toggle = el("button", { class: "quiet", onclick: () => {
    if (designEditing.has(b.id)) designEditing.delete(b.id); else designEditing.add(b.id);
    wrap.replaceWith(textEditor(b));
  } }, editing ? "Preview" : "Edit");
  if (editing) {
    const ta = el("textarea", { "aria-label": `${KIND_LABEL[b.kind]} text (Markdown)`, placeholder: b.kind === "raw" ? "[Dialogue Lesson] ... [/Dialogue Lesson]" : "Markdown: # heading, **bold**, *italic*, - lists, | tables |, ``` code ```" });
    ta.value = b.text;
    ta.addEventListener("input", () => { b.text = ta.value; b.edited = true; changed(); });
    designEditing.add(b.id);
    wrap.append(el("div", { class: "actions" }, toggle, el("span", { class: "note" }, b.kind === "raw" ? "Unified lesson format" : "Markdown is for you; Nova is taught the plain text.")), ta);
  } else {
    wrap.append(el("div", { class: "actions" }, toggle), b.kind === "raw" ? el("pre", { class: "lesson-file" }, b.text) : renderMarkdown(b.text));
  }
  return wrap;
}

function itemTable(b) {
  const table = el("table", { class: "design-items" });
  table.append(el("thead", {}, el("tr", {}, el("th", {}, "#"), el("th", {}, "Question"), el("th", {}, "Answer"), el("th", {}, ""))));
  const body = el("tbody");
  const otherBlocks = () => draft.blocks.filter(x => x !== b && ITEM_KINDS.includes(x.kind));
  b.items.forEach((it, k) => {
    const q = el("input", { type: "text", value: it.question, "aria-label": "Question", oninput: e => { it.question = e.target.value; it.edited = true; b.edited = true; changed(); } });
    const a = el("input", { type: "text", value: it.answer, "aria-label": "Answer", placeholder: "Answer (accepted forms separated by |)",
      oninput: e => {
        it.answer = e.target.value; it.edited = true; b.edited = true;
        it.answer_source = it.answer.trim() ? "teacher" : "missing";
        it.verified = !!it.answer.trim();
        meta.replaceChildren(...metaParts());
        changed();
      } });
    const verify = el("label", { class: "check-inline item-meta" }, el("input", { type: "checkbox", checked: it.verified ? "" : null,
      onchange: e => { it.verified = e.target.checked; changed(); } }), "verified");
    const metaParts = () => [
      `${ORIGIN_LABEL[it.origin] || it.origin}${it.source ? (it.source.line ? ` · line ${it.source.line}` : ` · from ${it.source.file}`) : ""}${it.edited ? " · edited" : ""} · answer ${ANSWER_LABEL[it.answer_source] || it.answer_source}${it.verified ? " · checked" : ""}`,
      ...(it.verified || it.edited ? [] : it.flags.map(f => el("div", { class: "flag" }, f))),
    ];
    const meta = el("div", { class: "item-meta" }, ...metaParts());
    const note = el("input", { type: "text", value: it.note || "", placeholder: "Teacher note (not taught)", "aria-label": "Teacher note", oninput: e => { it.note = e.target.value; changed(); } });
    const move = el("select", { "aria-label": "Move to block", onchange: e => {
      const target = draft.blocks.find(x => x.id === e.target.value);
      if (!target) return;
      b.items.splice(k, 1); target.items.push(it); b.edited = target.edited = true; changed(true);
    } }, el("option", { value: "" }, "Move to…"), ...otherBlocks().map(x => el("option", { value: x.id }, `${KIND_LABEL[x.kind]}${x.title ? ": " + x.title : ""}`)));
    body.append(el("tr", { "data-item": it.id },
      el("td", { class: "num" }, String(k + 1)),
      el("td", {}, q, meta),
      el("td", {}, a, note),
      el("td", {}, verify, move,
        el("span", { class: "db-tools" },
          el("button", { class: "quiet", title: "Up", onclick: () => { if (k > 0) { [b.items[k - 1], b.items[k]] = [b.items[k], b.items[k - 1]]; changed(true); } } }, "↑"),
          el("button", { class: "quiet", title: "Down", onclick: () => { if (k + 1 < b.items.length) { [b.items[k + 1], b.items[k]] = [b.items[k], b.items[k + 1]]; changed(true); } } }, "↓"),
          el("button", { class: "quiet", title: "Delete question", onclick: () => { b.items.splice(k, 1); b.edited = true; changed(true); } }, "×")))));
  });
  if (!b.items.length) body.append(el("tr", {}, el("td", { colspan: 4, class: "empty" }, "No questions.")));
  table.append(body);
  const unchecked = b.items.filter(it => !it.verified && it.answer.trim()).length;
  return el("div", {}, table, el("div", { class: "actions" },
    el("button", { onclick: () => { b.items.push(blankItem()); b.edited = true; changed(true); } }, "Add a question"),
    unchecked ? el("button", { class: "quiet", onclick: async () => {
      if (!await askYes(`Mark the ${unchecked} answer(s) in this block as checked? Only do this after reading every one of them.`)) return;
      for (const it of b.items) if (it.answer.trim()) it.verified = true;
      changed(true);
    } }, `Mark all ${unchecked} answer(s) as checked`) : ""));
}

// ---- whole-lesson preview ------------------------------------------------------------------
function lessonPreview() {
  const out = el("div", { class: "md" });
  out.append(el("h3", {}, draft.name));
  for (const b of draft.blocks.filter(x => x.use_in_teaching && x.kind !== "unsorted")) {
    out.append(el("h4", {}, `[${KIND_LABEL[b.kind].toUpperCase()}]${b.title ? " " + b.title : ""}${b.kind === "reading" ? ` (${b.mode})` : ""}`));
    if (ITEM_KINDS.includes(b.kind)) {
      const list = el("ol");
      for (const it of b.items) {
        const li = el("li");
        li.append(mdInline(it.question), el("br"), el("span", { class: "note" }, "Answer: "), mdInline(it.answer || "—"));
        list.append(li);
      }
      out.append(list);
    } else if (b.kind === "raw") out.append(el("pre", {}, b.text));
    else out.append(renderMarkdown(b.text));
  }
  return out;
}

// ---- course choice -------------------------------------------------------------------------
async function fillCourseChoice() {
  const list = await invoke("list_courses").catch(() => []);
  const sel = $("designCourse");
  const keep = sel.value;
  sel.replaceChildren(el("option", { value: "" }, "New course…"), ...list.map(c => el("option", { value: c.id }, c.name)));
  if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
  $("designNewCourse").hidden = !!sel.value;
}

// ---- wiring --------------------------------------------------------------------------------
$("importDesign").addEventListener("click", safe(async () => {
  await flushDesign();
  say("Reading the dataset…");
  const made = await invoke("import_design", { split: $("designSplit").checked });
  if (!made.length) { say("No file chosen."); return; }
  await refreshDesigns();
  say(made.length > 1 ? `The dataset was divided into ${made.length} lessons. Review each one: the organisation is only a proposal.` : "Proposed a lesson. Review it: the organisation is only a proposal.");
  await openDesign(made[0].id);
}, "Import dataset"));
$("newDesign").addEventListener("click", safe(async () => {
  await flushDesign();
  const v = await invoke("create_design", { name: "New lesson" });
  showDesign(v);
  refreshDesigns();
}, "New lesson"));
$("designName").addEventListener("input", e => { if (draft) { draft.name = e.target.value; changed(); } });
$("designStage").addEventListener("change", e => { if (draft) { draft.stage = e.target.value || null; changed(); } });
$("designFinished").addEventListener("change", e => { if (draft) { draft.finished = e.target.checked; changed(); } });
$("designNotes").addEventListener("input", e => { if (draft) { draft.notes = e.target.value; changed(); } });
$("designAddKind").addEventListener("change", e => {
  if (!draft || !e.target.value) return;
  const b = blankBlock(e.target.value);
  draft.blocks.push(b);
  designEditing.add(b.id);
  e.target.value = "";
  changed(true);
  document.querySelector(`[data-block="${CSS.escape(b.id)}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
});
$("designPreview").addEventListener("click", () => {
  const pane = $("designPreviewPane");
  pane.hidden = !pane.hidden;
  $("designPreview").textContent = pane.hidden ? "Preview the whole lesson" : "Hide the preview";
  if (!pane.hidden) pane.replaceChildren(lessonPreview());
});
$("designFile").addEventListener("click", safe(async () => {
  await flushDesign();
  const pane = $("designFilePane");
  pane.hidden = !pane.hidden;
  $("designFile").textContent = pane.hidden ? "Show the lesson file" : "Hide the lesson file";
  if (!pane.hidden) showLessonFile();
}, "Lesson file"));
$("designCourse").addEventListener("change", e => { $("designNewCourse").hidden = !!e.target.value; });
$("designToClassroom").addEventListener("click", safe(async () => {
  await flushDesign();
  const msg = await invoke("design_to_classroom", { id: draft.id });
  say(msg);
  await refreshLessons();
}, "Add to Classroom"));
$("designToCourse").addEventListener("click", safe(async () => {
  await flushDesign();
  const course = $("designCourse").value || null;
  const newCourse = course ? null : $("designNewCourse").value.trim();
  if (!course && !newCourse) throw new Error("Choose a course or type a name for a new one");
  const msg = await invoke("design_to_course", { id: draft.id, course, newCourse });
  say(msg);
  log("INFO", msg);
  await fillCourseChoice();
  if (typeof refreshCourses === "function") refreshCourses().catch(() => {});
}, "Save into course"));
$("designDelete").addEventListener("click", safe(async () => {
  if (!draft || !await askYes(`Delete the lesson draft "${draft.name}"? Lessons already saved in courses are not affected.`)) return;
  clearTimeout(designTimer);
  designDirty = false;
  await invoke("delete_design", { id: draft.id });
  draft = null; design = null;
  $("designEditor").hidden = true;
  refreshDesigns();
}, "Delete lesson"));

const designerPrevPage = window.onPage;
window.onPage = page => {
  designerPrevPage?.(page);
  if (page === "designer") refreshDesigns().catch(e => log("ERROR", e));
  else flushDesign().catch(e => log("ERROR", e));
};
window.addEventListener("beforeunload", () => { if (designDirty) flushDesign(); });
