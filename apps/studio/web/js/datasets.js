// § 14 Datasets: importing a dataset of your choosing (imports.rs). Inspect a source, tick files
// of one split, read their columns, choose what becomes a document, preview, import.
(() => {
  let inspection = null;
  let sample = null;
  let roles = {};
  const size = n => (n ? (n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(1)} GB` : `${(n / 1048576).toFixed(1)} MB`) : "—");
  const FORMAT = { parquet: "Parquet", json: "JSON", jsonl: "JSON lines", csv: "CSV", tsv: "TSV", text: "text", zip: "ZIP", script: "Python script (not run)", other: "not readable" };

  function showInspection(i) {
    inspection = i;
    sample = null;
    $("impMapBox").hidden = true;
    $("impRunBox").hidden = true;
    $("impFilesBox").hidden = false;
    $("impInfo").replaceChildren(el("b", {}, i.name), ` · ${i.source}`, i.revision ? ` · revision ${i.revision}` : "", i.license ? ` · licence ${i.license}` : "");
    $("impNotes").replaceChildren(...i.notes.map(n => el("li", {}, n)));
    const readable = f => !["script", "other"].includes(f.format);
    // The first readable split is ticked (the training split when there is one).
    const first = i.files.find(f => readable(f) && f.split === "train") || i.files.find(readable);
    $("impFiles").tBodies[0].replaceChildren(...i.files.map((f, k) => el("tr", {},
      el("td", {}, el("input", { type: "checkbox", "data-k": k, disabled: readable(f) ? null : "", checked: readable(f) && first && f.split === first.split && (f.converted != null) === (first.converted != null) ? "" : null })),
      el("td", {}, f.name), el("td", {}, f.split || "—"), el("td", {}, FORMAT[f.format] + (f.gz ? " (gzip)" : "")), el("td", { class: "num" }, size(f.size)))));
    $("impLicense").value = i.license || "";
    $("impName").value = i.name;
  }
  const chosen = () => [...$("impFiles").querySelectorAll("input:checked")].map(b => inspection.files[Number(b.dataset.k)]);

  const columnSelect = (id, value, empty) => {
    const sel = el("select", { id }, ...(empty ? [el("option", { value: "" }, empty)] : []), ...sample.columns.map(c => el("option", { value: c.name }, c.name)));
    if (value != null) sel.value = value;
    return sel;
  };
  // The layout's fields: which columns (and, for messages, which fields of each message).
  function renderFields(m) {
    const f = $("impFields");
    const layout = $("impLayout").value;
    const label = (t, x) => el("label", {}, t, " ", x);
    if (layout === "messages") {
      f.replaceChildren(label("Messages column", columnSelect("impColumn", m?.column)),
        label("role field", el("input", { type: "text", id: "impRoleField", value: m?.roleField || "role", style: "width: 7em" })),
        label("text field", el("input", { type: "text", id: "impContentField", value: m?.contentField || "content", style: "width: 7em" })));
    } else if (layout === "pair") {
      f.replaceChildren(label("Prompt column", columnSelect("impPrompt", m?.prompt)), label("Response column", columnSelect("impResponse", m?.response)));
    } else if (layout === "turns") {
      f.replaceChildren(label("Turns column", columnSelect("impColumn", m?.column)));
    } else if (layout === "rows") {
      f.replaceChildren(label("Conversation column", columnSelect("impConversation", m?.conversation)),
        label("Speaker column", columnSelect("impSpeaker", m?.speaker)), label("Text column", columnSelect("impText", m?.text)));
    } else {
      const picked = new Set(m?.columns || []);
      f.replaceChildren(el("span", {}, "Text columns: "), ...sample.columns.filter(c => c.kind === "text").map(c =>
        el("label", { class: "check-inline" }, el("input", { type: "checkbox", class: "impTextCol", value: c.name, checked: picked.has(c.name) ? "" : null }), ` ${c.name}`)));
    }
    const conv = $("impCategory").value === "conversation";
    $("impConvOptions").hidden = !conv;
    $("impFilterBox").hidden = !conv;
    $("impRoles").hidden = !conv || !["messages", "rows"].includes(layout);
  }
  function renderRoles(found) {
    const names = Object.keys(found || {});
    if (!names.length) { $("impRoles").replaceChildren(); return; }
    $("impRoles").replaceChildren(el("p", { class: "note" }, "Roles found (with how often). Leave every name empty to have roles that are names kept and the others named for you (system messages left out). Once you name one, name them all: a role without a name is reported as not mapped."),
      el("table", { class: "ledger compact" }, el("tbody", {}, ...names.map(r => el("tr", {},
        el("td", {}, `${r} (${fmt(found[r])})`),
        el("td", {}, el("input", { type: "text", class: "impRoleName", "data-role": r, value: roles[r]?.name || "", placeholder: "name" })),
        el("td", {}, el("label", { class: "check-inline" }, el("input", { type: "checkbox", class: "impRoleSkip", "data-role": r, checked: roles[r]?.skip ? "" : null }), " leave out")))))));
  }
  function mapping() {
    const layout = $("impLayout").value;
    const m = { category: $("impCategory").value, layout };
    const v = id => $(id)?.value || "";
    if (layout === "messages") Object.assign(m, { column: v("impColumn"), roleField: v("impRoleField"), contentField: v("impContentField") });
    else if (layout === "pair") Object.assign(m, { prompt: v("impPrompt"), response: v("impResponse") });
    else if (layout === "turns") Object.assign(m, { column: v("impColumn") });
    else if (layout === "rows") Object.assign(m, { conversation: v("impConversation"), speaker: v("impSpeaker"), text: v("impText") });
    else {
      m.columns = [...document.querySelectorAll(".impTextCol:checked")].map(b => b.value);
      if (!m.columns.length) throw new Error("Tick at least one text column");
    }
    // Roles: only when one was named or left out (else automatic).
    for (const i of document.querySelectorAll(".impRoleName")) roles[i.dataset.role] = { ...roles[i.dataset.role], name: i.value.trim() };
    for (const i of document.querySelectorAll(".impRoleSkip")) roles[i.dataset.role] = { ...roles[i.dataset.role], skip: i.checked };
    const touched = Object.values(roles).some(r => r.name || r.skip);
    m.roles = {};
    if (touched) for (const [r, x] of Object.entries(roles)) { if (x.skip) m.roles[r] = ""; else if (x.name) m.roles[r] = x.name; }
    m.names = [v("impNameA").trim(), v("impNameB").trim()].filter(Boolean).length ? [v("impNameA").trim(), v("impNameB").trim()] : [];
    m.situation = v("impSituation") || null;
    return m;
  }
  // A mapping's layout as the backend writes it ("messages", ...) and its fields in camelCase.
  const fromBackend = m => {
    if (!m) return null;
    const out = { ...m };
    if (m.role_field) out.roleField = m.role_field;
    if (m.content_field) out.contentField = m.content_field;
    return out;
  };

  $("impInspect").addEventListener("click", safe(async () => {
    const source = $("impSource").value.trim();
    if (!source) throw new Error("Give a Hugging Face dataset (owner/name or its link) or an https link to a file");
    $("impWait").textContent = "Looking…";
    try { showInspection(await invoke("import_inspect", { source, revision: $("impRevision").value.trim() || null })); }
    finally { $("impWait").textContent = ""; }
  }, "Inspect"));
  $("impLocal").addEventListener("click", safe(async () => {
    const i = await invoke("import_pick_local");
    if (i) showInspection(i);
  }, "Files"));
  $("impSample").addEventListener("click", safe(async () => {
    const files = chosen();
    if (!files.length) throw new Error("Tick at least one file");
    $("impWait").textContent = "Reading the chosen files (downloading them first when they are on Hugging Face)…";
    try { sample = await invoke("import_sample", { files }); }
    finally { $("impWait").textContent = ""; }
    roles = {};
    $("impColumns").tBodies[0].replaceChildren(...sample.columns.map(c => el("tr", {}, el("td", {}, c.name), el("td", {}, c.kind), el("td", { class: "note" }, c.example))));
    const s = fromBackend(sample.suggested);
    $("impCategory").value = s?.category || "conversation";
    $("impLayout").value = s?.layout || "text";
    $("impSituation").replaceChildren(el("option", { value: "" }, "none"), ...sample.columns.map(c => el("option", { value: c.name }, c.name)));
    $("impSituation").value = s?.situation || "";
    renderFields(s);
    renderRoles({});
    $("impPreviewBox").replaceChildren(el("p", { class: "note" }, s ? "A mapping is suggested from the columns; check it and press Preview." : "Choose what becomes a document, then Preview."));
    $("impMapBox").hidden = false;
    $("impRunBox").hidden = true;
  }, "Read files"));
  for (const id of ["impLayout", "impCategory"]) $(id).addEventListener("change", () => sample && renderFields(null));
  $("impPreview").addEventListener("click", safe(async () => {
    const p = await invoke("import_preview", { files: sample.files, mapping: mapping() });
    renderRoles(p.report.roles);
    $("impPreviewBox").replaceChildren(el("p", {}, p.summary),
      ...(p.docs.length ? p.docs.map(d => el("pre", { class: "sample" }, d.length > 1500 ? d.slice(0, 1500) + " …" : d)) : [el("p", { class: "warn" }, "Nothing converted: check the columns and the layout.")]));
    $("impRunBox").hidden = !p.docs.length;
  }, "Preview"));
  $("impRun").addEventListener("click", safe(async () => {
    $("impRun").disabled = true;
    $("impCancel").hidden = false;
    $("impResult").textContent = "";
    try {
      const info = await invoke("import_run", { request: {
        kind: inspection.kind, source: inspection.source, revision: inspection.revision, files: sample.files, mapping: mapping(),
        name: $("impName").value, license: $("impLicense").value, filterAtPractice: $("impFilter").checked, maxMb: Number($("impMaxMb").value),
      } });
      const r = info.import?.report;
      $("impResult").replaceChildren(el("b", { class: "ok" }, `Imported ${info.name}: `), `${mb(info.bytes)}, ${fmt(info.documents)} documents. `,
        r ? `${r.accepted} of ${r.examined} records imported` : "", info.category === "conversation" ? " It is on § 4 Conversation." : " It is on § 2 Reading.");
      say(`Imported ${info.name}.`);
      refreshCorpora?.();
      window.refreshConversation?.();
    } finally {
      $("impRun").disabled = false;
      $("impCancel").hidden = true;
      $("impProgress").textContent = "";
    }
  }, "Import"));
  $("impCancel").addEventListener("click", () => invoke("import_cancel").catch(e => log("ERROR", e)));
  if (window.__TAURI__?.event?.listen) {
    window.__TAURI__.event.listen("import-progress", ev => {
      const p = ev.payload;
      $("impProgress").textContent = `${p.what}${p.total ? `: ${mb(p.done)} of ${mb(p.total)}` : p.done ? `: ${mb(p.done)}` : ""}`;
    });
  }
})();
