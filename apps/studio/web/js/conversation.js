"use strict";
// § 4 Conversation: Nova practises real conversations between two people after the Classroom.
// The training runs in the Engine (learn::conversation); this page chooses the material and
// shows the checks.
(() => {
  let sources = [];
  let material = { corpora: [], chatReplies: 0, lessonUnits: 0, classroomQuestions: 0 };
  const chosen = new Set();
  let downloading = null;
  let started = 0;

  function renderSources() {
    $("convSources").replaceChildren(...sources.map(s => {
      const size = el("input", { type: "number", min: 1, max: Math.min(400, s.sizeMb), value: s.defaultMb, title: "Megabytes to download", class: "inline-num" });
      return el("div", { class: "source" },
        el("b", {}, s.name),
        s.sizeMb > s.defaultMb ? el("span", {}, "first ", size, ` MB of about ${fmt(s.sizeMb)} MB`) : el("span", {}, `about ${s.sizeMb} MB`),
        el("button", { type: "button", onclick: safe(async () => {
          const want = s.sizeMb > s.defaultMb ? Number(size.value) || s.defaultMb : s.sizeMb;
          await invoke("download_corpus", { source: s.id, maxMb: want });
          downloading = s.id;
          $("convDownload").hidden = false;
          $("convDownloadTitle").textContent = `Downloading ${s.name}`;
          $("convDownloadBar").style.width = "0%";
          $("convDownloadLine").textContent = "Connecting…";
          $("convCancelDownload").hidden = false;
        }, "Download") }, "Download"),
        el("p", { class: "note" }, `${s.description} `, el("b", {}, "Licence: "), `${s.license}. `, el("b", {}, "Credit: "), `${s.attribution}. Source: ${s.page}`));
    }));
  }

  function renderMaterial() {
    const body = $("convTable").tBodies[0];
    body.replaceChildren();
    if (!material.corpora.length) {
      body.append(el("tr", {}, el("td", { colspan: 5, class: "empty" }, "No conversation dataset yet. Download one above.")));
    }
    for (const c of material.corpora) {
      const box = el("input", { type: "checkbox", checked: chosen.has(c.id) ? "" : null, disabled: teaching ? "" : null,
        onchange: e => { if (e.target.checked) chosen.add(c.id); else chosen.delete(c.id); note(); } });
      body.append(el("tr", {},
        el("td", {}, box),
        el("td", {}, c.name, c.attribution ? el("div", { class: "note" }, "Credit: " + c.attribution) : ""),
        el("td", { class: "num" }, mb(c.bytes)),
        el("td", { class: "num" }, fmt(c.documents)),
        el("td", {}, c.license)));
    }
    $("convChatCount").textContent = active
      ? (material.chatReplies ? `(${fmt(material.chatReplies)} marked so far)` : "(none marked yet: mark replies with ✓ Consistent in a chat)")
      : "";
    note();
  }

  function note() {
    if (!active) { $("convNote").textContent = "Open a model on § 1 Models first."; return; }
    const sets = material.corpora.filter(c => chosen.has(c.id));
    const parts = [];
    parts.push(sets.length ? `${sets.length} dataset(s), ${fmt(sets.reduce((a, c) => a + c.documents, 0))} conversations` : "Choose at least one conversation dataset");
    if ($("convChats").checked && material.chatReplies) parts.push(`${fmt(material.chatReplies)} marked chat replies`);
    parts.push(material.lessonUnits
      ? `${$("convEarlier").value}% of rows from ${fmt(material.lessonUnits)} taught lesson units`
      : "no Classroom lessons taught yet, so none are mixed in");
    parts.push(material.classroomQuestions
      ? `up to ${fmt(Math.min(material.classroomQuestions, Number($("convGuard").value) || 0))} held-out Classroom questions checked at each step`
      : "no held-out Classroom questions to check");
    $("convNote").textContent = parts.join(" · ") + ".";
  }

  async function refresh() {
    material = await invoke("conversation_material");
    for (const id of [...chosen]) if (!material.corpora.some(c => c.id === id)) chosen.delete(id);
    // A new dataset is chosen by default.
    if (!chosen.size) for (const c of material.corpora) chosen.add(c.id);
    renderMaterial();
    renderHistory();
  }

  function renderHistory() {
    const runs = active?.conversation || [];
    if (!runs.length) { $("convHistory").replaceChildren(el("p", { class: "empty" }, "None yet.")); return; }
    $("convHistory").replaceChildren(el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "When"), el("th", {}, "Conversations"), el("th", { class: "num" }, "Tokens"), el("th", { class: "num" }, "Held-out loss"), el("th", {}, "Result"))),
      el("tbody", {}, ...runs.slice().reverse().map(r => el("tr", {},
        el("td", {}, when(r.time)),
        el("td", {}, r.sources.map(s => s.name).join(" + ") + (r.chatReplies ? ` + ${fmt(r.chatReplies)} chat replies` : "")),
        el("td", { class: "num" }, fmt(r.tokens)),
        el("td", { class: "num" }, `${r.baselineLoss.toFixed(3)} → ${r.bestLoss.toFixed(3)}`),
        el("td", { class: r.accepted ? "ok" : "" }, r.accepted ? "kept the best checkpoint" : r.stopped ? "stopped, unchanged" : "no checkpoint was better: unchanged"))))));
  }

  // ---- a run ---------------------------------------------------------------------------------
  function sampleView(samples) {
    return el("div", { class: "conv-samples" }, ...samples.map(s => el("div", { class: "conv-sample" },
      el("div", { class: "mono note" }, s.prompt.replace(/\n/g, " ⏎ ")),
      el("div", {}, el("b", {}, "Nova: "), s.reply || "(empty)"),
      el("div", { class: "note" }, "In the conversation: ", s.original))));
  }
  function onEvent(e) {
    if (e.type === "started") {
      $("convTitle").textContent = `Practising ${fmt(e.practice)} conversations (${fmt(e.heldOutDialogues)} more held out)`;
      $("convLine").textContent = `Held-out loss before: ${e.baselineLoss.toFixed(4)}` +
        (e.classroomTotal ? ` · Classroom: ${fmt(e.baselineClassroom)} of ${fmt(e.classroomTotal)} answers right` : "") +
        (e.leftOutAssistant ? ` · ${fmt(e.leftOutAssistant)} conversation(s) in an assistant's voice left out` : "");
      $("convChecks").tBodies[0].replaceChildren();
    } else if (e.type === "batch") {
      $("convBar").style.width = (100 * e.step / e.steps).toFixed(1) + "%";
      const secs = (Date.now() - started) / 1000;
      $("convLine").textContent = `Step ${fmt(e.step)} of ${fmt(e.steps)} · ${fmt(e.tokens)} tokens · loss ${e.loss.toFixed(4)} · learning rate ${rateText(e.learningRate)}` +
        (secs > 2 ? ` · ${fmt(Math.round(e.tokens / secs))} tokens/s` : "");
    } else if (e.type === "checked") {
      $("convChecks").tBodies[0].prepend(el("tr", {},
        el("td", { class: "num" }, fmt(e.tokens)),
        el("td", { class: "num" }, e.trainLoss.toFixed(4)),
        el("td", { class: "num" }, e.heldOutLoss.toFixed(4)),
        el("td", { class: "num" }, e.classroomTotal ? `${fmt(e.classroomCorrect)} / ${fmt(e.classroomTotal)}` : "—"),
        el("td", { class: e.accepted ? "ok" : "" }, e.accepted ? "kept" : `not kept: ${e.reason}`),
        el("td", {}, sampleView(e.samples))));
    }
  }
  function onFinished(f) {
    setTeaching(false);
    $("stopConverse").disabled = true;
    if (f.model) showActive(f.model);
    const text = f.error ? `Conversation practice ended with an error: ${f.error}`
      : f.accepted ? `${f.stopped ? "Stopped" : f.earlyStop ? "Finished early (no further improvement)" : "Finished"} after ${fmt(f.tokens)} tokens: held-out loss of Nova's lines ${f.baselineLoss.toFixed(4)} → ${f.bestLoss.toFixed(4)}. The best checkpoint is saved.`
      : `${f.stopped ? "Stopped" : "Finished"}: no checkpoint was better on held-out conversations while keeping the Classroom answers, so the model is unchanged.`;
    $("convTitle").textContent = text;
    if (f.error) log("ERROR", f.error); else say(text);
    refresh().catch(e => log("ERROR", e));
    refreshModels();
  }

  $("converse").addEventListener("click", safe(async () => {
    if (!active) throw new Error("Open or create a model first");
    const corpusIds = material.corpora.filter(c => chosen.has(c.id)).map(c => c.id);
    if (!corpusIds.length) throw new Error("Choose at least one conversation dataset");
    const options = {
      corpusIds,
      chats: $("convChats").checked,
      tokens: Number($("convTokens").value),
      batchSize: Number($("convBatch").value),
      learningRate: Number($("convRate").value),
      bothSides: $("convBoth").checked,
      ownName: $("convName").value,
      situations: $("convSituations").checked,
      earlierPercent: Number($("convEarlier").value),
      guardQuestions: Number($("convGuard").value),
      guardDropPercent: Number($("convDrop").value),
    };
    await invoke("converse", { options });
    started = Date.now();
    setTeaching(true);
    $("stopConverse").disabled = false;
    $("convEntry").hidden = false;
    $("convBar").style.width = "0%";
    $("convTitle").textContent = "Preparing the conversations…";
    $("convLine").textContent = "Reading the datasets and measuring the model before practice.";
    $("convChecks").tBodies[0].replaceChildren();
    say("Practising conversations…");
  }, "Conversation"));
  $("stopConverse").addEventListener("click", safe(async () => { await invoke("stop_teaching"); say("Stopping after this batch…"); }, "Stop"));
  $("convCancelDownload").addEventListener("click", () => invoke("cancel_download").catch(e => log("ERROR", e)));
  $("convAddCourse").addEventListener("click", () => {
    const sel = $("builtinLessons");
    if (![...sel.options].some(o => o.value === "deciding")) { say("The built-in courses are still loading."); return; }
    sel.value = "deciding";
    go("classroom");
    $("addBuiltinLessons").click();
  });
  for (const id of ["convChats", "convEarlier", "convGuard"]) $(id).addEventListener("input", note);

  const prevTeaching = window.onTeachingChanged;
  window.onTeachingChanged = on => {
    prevTeaching?.(on);
    $("converse").disabled = on;
    renderMaterial();
  };
  const prevPage = window.onPage;
  window.onPage = page => {
    prevPage?.(page);
    if (page !== "conversation") return;
    refresh().catch(e => log("ERROR", e));
  };
  // A different model (or new figures for the same one): its own name and record.
  window.showConversation = meta => {
    if (meta && $("convName").dataset.model !== meta.id) {
      $("convName").dataset.model = meta.id;
      $("convName").value = meta.name.replace(/[^A-Za-z -]/g, "").trim().slice(0, 32) || "Nova";
    }
    renderHistory();
  };

  if (tauri?.event?.listen) {
    tauri.event.listen("converse-event", ev => { try { onEvent(ev.payload); } catch (err) { log("ERROR", err); } });
    tauri.event.listen("converse-finished", ev => onFinished(ev.payload));
    tauri.event.listen("corpus-download", ev => {
      const e = ev.payload;
      if (e.source !== downloading) return;
      if (!e.done) {
        if (e.note) { $("convDownloadLine").textContent = e.note; return; }
        $("convDownloadBar").style.width = e.total ? (100 * e.received / e.total).toFixed(1) + "%" : "50%";
        $("convDownloadLine").textContent = `${mb(e.received)}${e.total ? " of " + mb(e.total) : ""} received`;
        return;
      }
      downloading = null;
      $("convCancelDownload").hidden = true;
      if (e.error) { $("convDownloadLine").textContent = e.error; return; }
      $("convDownloadBar").style.width = "100%";
      $("convDownloadLine").textContent = `Saved ${e.corpus.name}: ${mb(e.corpus.bytes)}, ${fmt(e.corpus.documents)} conversations.`;
      chosen.add(e.corpus.id);
      refresh().catch(err => log("ERROR", err));
    });
    invoke("conversation_sources").then(s => { sources = s; renderSources(); }).catch(e => log("ERROR", e));
  }
})();
