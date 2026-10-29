"use strict";
// § 4 Conversation: Nova practises real conversations between two people after the Classroom.
// The training runs in the Engine (learn::conversation); this page chooses the material and
// shows the checks.
(() => {
  let sources = [];
  let material = { corpora: [], chatReplies: 0, lessonUnits: 0, classroomQuestions: 0 };
  const chosen = new Set();
  // The conversation download running (its source id), shown in this page's panel.
  let downloading = null;
  function showDownload(name) {
    $("convDownload").hidden = false;
    $("convDownloadTitle").textContent = `Downloading ${name}`;
    $("convDownloadBar").style.width = "0%";
    $("convDownloadLine").textContent = "Connecting…";
    $("convCancelDownload").hidden = false;
  }
  let started = 0;

  function renderSources() {
    $("convSources").replaceChildren(...sources.map(s => {
      const size = el("input", { type: "number", min: 1, max: Math.min(400, s.sizeMb), value: s.defaultMb, title: "Megabytes to download", class: "inline-num" });
      return el("div", { class: "source" },
        el("b", {}, s.name),
        s.sizeMb > s.defaultMb ? el("span", {}, "first ", size, ` MB of about ${fmt(s.sizeMb)} MB`) : el("span", {}, `about ${s.sizeMb} MB`),
        el("button", { type: "button", onclick: safe(async () => {
          const want = s.sizeMb > s.defaultMb ? Number(size.value) || s.defaultMb : s.sizeMb;
          const start = await invoke("download_corpus", { source: s.id, maxMb: want });
          $("convDownload").hidden = false;
          if (start.queued) { say(`${s.name} will download after the ${start.position === 1 ? "one running" : `${start.position - 1} waiting before it`}.`); return; }
          showDownload(s.name);
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
        onchange: e => { if (e.target.checked) chosen.add(c.id); else chosen.delete(c.id); note(); saveChoice(); } });
      body.append(el("tr", {},
        el("td", {}, box),
        el("td", {}, c.name, c.attribution ? el("div", { class: "note" }, "Credit: " + c.attribution) : "",
          c.filter ? el("div", { class: "note" }, "Quality filter: " + filterText(c.filter)) : ""),
        el("td", { class: "num" }, mb(c.bytes)),
        el("td", { class: "num" }, fmt(c.documents)),
        el("td", {}, c.license)));
    }
    $("convChatCount").textContent = active
      ? (material.chatReplies ? `(${fmt(material.chatReplies)} marked so far)` : "(none marked yet: mark replies with ✓ Consistent in a chat)")
      : "";
    note();
  }

  // The download filter's counts: kept of read, and why the rest was left out.
  function filterText(r) {
    const why = [[r.assistantStyle, "assistant-style"], [r.notEnglish, "not English"], [r.flagged, "flagged by the dataset"],
      [r.tooLong, "too long or code"], [r.lowQuality, "low quality"], [r.unsafeContent, "unsafe or inappropriate"],
      [r.turnOrder, "broken turn order"], [r.malformed, "malformed"], [r.duplicates, "duplicates"], [r.overlapsTests, "overlapping Yendory's tests"]]
      .filter(([n]) => n > 0).map(([n, w]) => `${fmt(n)} ${w}`);
    return `kept ${fmt(r.kept)} of ${fmt(r.read)} conversations` + (why.length ? ` (left out: ${why.join(", ")})` : "");
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
    $("convReadingNote").textContent = material.reading?.length
      ? `Read text rehearsed and guarded: ${material.reading.join(", ")} (a sample of up to 3,000 documents).`
      : "None of the text this model read is on this device, so no reading is rehearsed or guarded (download or import it again on § 2 to include it).";
  }

  // The ticked datasets are kept between runs of Studio: the ticked ones and the ones known
  // then (a dataset added since is ticked, as a new dataset always was).
  let savedChoice;
  const saveChoice = () => invoke("save_page_settings", { page: "conversationDatasets",
    settings: { ids: [...chosen], seen: material.corpora.map(c => c.id) } }).catch(e => log("ERROR", e));
  async function refresh() {
    material = await invoke("conversation_material");
    if (savedChoice === undefined) {
      savedChoice = await invoke("page_settings", { page: "conversationDatasets" }).catch(() => null);
      if (Array.isArray(savedChoice?.ids) && Array.isArray(savedChoice?.seen)) {
        for (const c of material.corpora) {
          if (savedChoice.ids.includes(c.id) || !savedChoice.seen.includes(c.id)) chosen.add(c.id);
        }
      } else {
        savedChoice = null;
      }
    }
    for (const id of [...chosen]) if (!material.corpora.some(c => c.id === id)) chosen.delete(id);
    // A new dataset is chosen by default.
    if (!chosen.size && !savedChoice) for (const c of material.corpora) chosen.add(c.id);
    renderMaterial();
    renderHistory();
    saveChoice();
  }

  function renderHistory() {
    const runs = active?.conversation || [];
    if (!runs.length) { $("convHistory").replaceChildren(el("p", { class: "empty" }, "None yet.")); return; }
    $("convHistory").replaceChildren(el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "When"), el("th", {}, "Conversations"), el("th", { class: "num" }, "Tokens"), el("th", { class: "num" }, "Held-out loss"), el("th", { class: "num" }, "Context tests"), el("th", { class: "num", title: "Understanding-test pairs passed, teacher-forced / free-running: before → the kept checkpoint" }, "Understanding"), el("th", {}, "Result"))),
      el("tbody", {}, ...runs.slice().reverse().map(r => el("tr", {},
        el("td", {}, when(r.time)),
        el("td", {}, r.sources.map(s => s.name).join(" + ") + (r.chatReplies ? ` + ${fmt(r.chatReplies)} chat replies` : "")),
        el("td", { class: "num" }, fmt(r.tokens)),
        el("td", { class: "num" }, `${r.baselineLoss.toFixed(3)} → ${r.bestLoss.toFixed(3)}`),
        el("td", { class: "num" }, r.baselineContext ? `${r.baselineContext.passed}/${r.baselineContext.total} → ${r.bestCheck?.context ? r.bestCheck.context.passed : "—"}` : "—"),
        el("td", { class: "num" }, r.baselineUnderstanding ? `${understandingText(r.baselineUnderstanding)} → ${r.bestCheck?.understanding ? understandingText(r.bestCheck.understanding) : "—"}` : "—"),
        el("td", { class: r.accepted ? "ok" : "" }, r.accepted ? "kept the best checkpoint" : r.stopped ? "stopped, unchanged" : "no checkpoint was better: unchanged",
          r.stage != null ? el("div", { class: "note" }, `reached: ${STAGE[r.stage] || "—"}`) : "",
          r.stuck ? el("div", { class: "warn note" }, r.stuck) : ""))))));
  }

  // ---- a run ---------------------------------------------------------------------------------
  // A conversation as Nova read it ("U: ..." / "N: ..." lines), one turn per line, then her
  // reply and (for held-out conversations) what the person really said.
  function transcript(prompt, reply, extra) {
    const turns = prompt.split("\n").filter(l => l.trim() && l !== "N: ");
    return el("div", { class: "conv-transcript" },
      ...turns.map(l => {
        const nova = l.startsWith("N: ");
        return el("div", { class: "conv-turn" + (nova ? " nova" : "") }, el("span", { class: "conv-who" }, nova ? "Nova" : "Them"), l.replace(/^[UN]: /, ""));
      }),
      el("div", { class: "conv-turn nova reply" }, el("span", { class: "conv-who" }, "Nova now"), reply || "(empty reply)"),
      extra || "");
  }
  function sampleView(samples) {
    if (!samples.length) return "";
    return el("details", { class: "conv-samples" },
      el("summary", {}, `${samples.length} held-out conversation(s): Nova's reply after the whole conversation so far`),
      ...samples.map(s => transcript(s.prompt, s.reply,
        el("div", { class: "conv-turn original" }, el("span", { class: "conv-who" }, "Real reply"), s.original))));
  }
  function pairsView(c, title) {
    if (!c?.pairs?.length) return "";
    const mark = ok => el("span", { class: ok ? "ok" : "bad" }, ok ? " ✓" : " ✗");
    return el("details", { class: "conv-samples" },
      el("summary", {}, title || "Context tests: one pair of each kind, in full"),
      ...c.pairs.map(p => el("div", { class: "conv-pair" },
        el("div", { class: "conv-pair-skill" }, SKILL[p.skill] || p.skill, p.a_right && p.b_right ? el("span", { class: "ok" }, " · passed") : el("span", { class: "warn" }, " · not passed")),
        el("div", { class: "conv-pair-sides" },
          transcript(p.a, p.a_reply, mark(p.a_right)),
          transcript(p.b, p.b_reply, mark(p.b_right))))));
  }
  const STAGE = ["short exchanges", "a few turns", "whole conversations"];
  const SKILL = { names: "names", locations: "locations", changing_facts: "changing facts", corrections: "corrections", references: "references", follow_ups: "follow-ups", uncertainty: "uncertainty",
    repeated_greetings: "greeted again", paraphrases: "same question, other words", who_said_what: "who said what", latest_message: "the latest message",
    topic_changes: "topic changes", misunderstandings: "misunderstandings", repeated_questions: "asked again" };
  // Understanding tests: teacher-forced · free-running pairs passed.
  const understandingText = u => `${u.teacher_forced.passed} · ${u.free_running.passed} /${u.teacher_forced.total}`;
  function understandingCell(u) {
    if (!u) return "—";
    const per = (c, k) => { const s = c.skills.find(x => x[0] === k); return s ? `${s[1]}/${s[2]}` : "—"; };
    const title = u.teacher_forced.skills.map(([k]) => `${SKILL[k] || k}: ${per(u.teacher_forced, k)} teacher-forced, ${per(u.free_running, k)} free-running`).join("\n");
    return el("span", { title }, `${u.teacher_forced.passed}/${u.teacher_forced.total}`, el("div", { class: "note" }, `free ${u.free_running.passed}/${u.free_running.total}`));
  }
  const pctOf = (a, b) => (b ? `${Math.round(100 * a / b)}%` : "—");
  function contextCell(c) {
    if (!c) return "—";
    return el("span", { title: c.skills.map(([k, p, t]) => `${SKILL[k] || k}: ${p}/${t}`).join("\n") + `\nanswers that changed with the fact: ${c.changed}/${c.total}` },
      `${c.passed}/${c.total}`, el("div", { class: "note" }, `${c.changed} changed`));
  }
  function checksView(r) {
    if (!r || !r.replies) return "";
    const n = r.replies;
    return el("div", { class: "note conv-checks" }, `Reply checks (${n}): on topic ${pctOf(r.on_topic, n)} · asks ${pctOf(r.questions, n)} · repeats ${pctOf(r.repeats, n)} · echoes ${pctOf(r.echoes, n)} · empty ${pctOf(r.empty, n)}` +
      (r.wrong_name ? ` · other name ${r.wrong_name}` : "") + (r.assistant ? ` · assistant's voice ${r.assistant}` : ""));
  }
  let baseline = null;
  let pendingPrevious = null;
  function onEvent(e) {
    if (e.type === "started") {
      baseline = e;
      $("convTitle").textContent = `Practising ${fmt(e.practice)} conversations (${fmt(e.held_out_dialogues)} more held out)`;
      $("convLine").textContent = `Before: held-out loss ${e.baseline_loss.toFixed(4)} (${e.baseline_stage_losses.map(l => l.toFixed(3)).join(" · ")})` +
        (e.baseline_listening != null ? ` · listening ${e.baseline_listening.toFixed(3)}` : "") +
        (e.baseline_context ? ` · context tests ${e.baseline_context.passed}/${e.baseline_context.total}` : "") +
        (e.baseline_understanding ? ` · understanding ${e.baseline_understanding.teacher_forced.passed}/${e.baseline_understanding.teacher_forced.total} (free-running ${e.baseline_understanding.free_running.passed})` : "") +
        (e.examples ? ` · ${fmt(e.examples)} example conversations` : "") +
        (e.classroom_total ? ` · Classroom ${fmt(e.baseline_classroom)}/${fmt(e.classroom_total)}` : "") +
        (e.baseline_reading != null ? ` · reading ${e.baseline_reading.toFixed(3)}` : "") +
        (e.left_out_assistant ? ` · ${fmt(e.left_out_assistant)} conversation(s) in an assistant's voice left out` : "");
      const body = $("convChecks").tBodies[0];
      body.replaceChildren(el("tr", {},
        el("td", { class: "num" }, "0"), el("td", {}, "before"),
        el("td", { class: "num" }, e.baseline_loss.toFixed(4), el("div", { class: "note nowrap", title: "short exchanges · a few turns · whole conversations" }, e.baseline_stage_losses.map(l => l.toFixed(2)).join(" · "))),
        el("td", { class: "num" }, e.baseline_listening != null ? e.baseline_listening.toFixed(3) : "—"),
        el("td", { class: "num" }, contextCell(e.baseline_context)),
        el("td", { class: "num" }, understandingCell(e.baseline_understanding)),
        el("td", { class: "num" }, e.classroom_total ? `${fmt(e.baseline_classroom)} / ${fmt(e.classroom_total)}` : "—"),
        el("td", { class: "num" }, e.baseline_reading != null ? e.baseline_reading.toFixed(3) : "—"),
        el("td", {}, "starting point"), el("td", {}, checksView(e.baseline_reply_checks))));
      if (e.resumed_from != null) {
        // A continued run: the checks before its recovery point, then a marker.
        for (const c of pendingPrevious || []) onEvent({ type: "checked", ...c });
        pendingPrevious = null;
        body.prepend(el("tr", {}, el("td", { colspan: 10, class: "note" }, `Continued from the recovery point after step ${fmt(e.resumed_from)} of ${fmt(e.steps)}`)));
        log("INFO", `Conversation practice continues from its recovery point after step ${fmt(e.resumed_from)}.`);
      }
    } else if (e.type === "batch") {
      $("convBar").style.width = (100 * e.step / e.steps).toFixed(1) + "%";
      const secs = (Date.now() - started) / 1000;
      $("convLine").textContent = `Stage: ${STAGE[e.stage]} · step ${fmt(e.step)} of ${fmt(e.steps)} · ${fmt(e.tokens)} tokens · loss ${e.loss.toFixed(4)} · learning rate ${rateText(e.learning_rate)}` +
        (secs > 2 ? ` · ${fmt(Math.round(e.tokens / secs))} tokens/s` : "");
    } else if (e.type === "checked") {
      $("convChecks").tBodies[0].prepend(el("tr", { class: e.accepted ? "current-row" : "" },
        el("td", { class: "num" }, fmt(e.tokens)),
        el("td", {}, STAGE[e.stage], e.learning_rate_scale < 1 ? el("div", { class: "note" }, `rate ×${e.learning_rate_scale}`) : ""),
        el("td", { class: "num" }, e.held_out_loss.toFixed(4), el("div", { class: "note nowrap", title: "short exchanges · a few turns · whole conversations" }, e.stage_losses.map(l => l.toFixed(2)).join(" · "))),
        el("td", { class: "num" }, e.listening != null ? e.listening.toFixed(3) : "—"),
        el("td", { class: "num" }, contextCell(e.context)),
        el("td", { class: "num" }, understandingCell(e.understanding)),
        el("td", { class: "num" }, e.classroom_total ? `${fmt(e.classroom_correct)} / ${fmt(e.classroom_total)}` : "—",
          e.classroom_total ? el("div", { class: "note" }, `−${e.classroom_lost} +${e.classroom_gained}`) : ""),
        el("td", { class: "num" }, e.reading_loss != null ? e.reading_loss.toFixed(3) : "—"),
        el("td", { class: e.accepted ? "ok" : "" }, e.accepted ? "kept" : `not kept: ${e.reason}`),
        el("td", {}, checksView(e.reply_checks), sampleView(e.samples), pairsView(e.context), pairsView(e.understanding?.teacher_forced, "Understanding tests (teacher-forced): one pair of each kind, in full"))));
    } else if (e.type === "stage") {
      log("INFO", `Conversation: now ${e.name}. ${e.reason}`);
      $("convChecks").tBodies[0].prepend(el("tr", {}, el("td", { colspan: 10, class: e.mastered ? "ok" : "warn" }, `→ ${e.name}: ${e.reason}`)));
    } else if (e.type === "learning_rate_lowered") {
      $("convChecks").tBodies[0].prepend(el("tr", {}, el("td", { colspan: 10, class: "warn" }, `${e.reason}: learning rate now ×${e.scale}`)));
    }
  }
  function onFinished(f) {
    setTeaching(false);
    $("stopConverse").disabled = true;
    if (f.model) showActive(f.model);
    const text = f.error ? `Conversation practice ended with an error: ${f.error}`
      : f.accepted ? `${f.stopped ? "Stopped" : f.earlyStop ? "Finished early (no further improvement)" : "Finished"} after ${fmt(f.tokens)} tokens: held-out loss of Nova's lines ${f.baselineLoss.toFixed(4)} → ${f.bestLoss.toFixed(4)}. The best checkpoint is saved.`
      : `${f.stopped ? "Stopped" : "Finished"}: no checkpoint was better on held-out conversations while keeping the Classroom answers and reading, so the model is unchanged.`;
    const balanced = f.balanced ? ` The best balanced checkpoint (most context tests passed, then listening, among those that learned and kept the Classroom answers) is saved as its own model: "${f.balanced}".` : "";
    $("convTitle").textContent = (f.stuck ? `${text} ${f.stuck}.` : text) + balanced;
    if (f.error) log("ERROR", f.error); else say(text + balanced);
    refresh().catch(e => log("ERROR", e));
    refreshModels();
  }

  // The page's settings, as the backend (and a plan's conversation step) takes them.
  function options() {
    const corpusIds = material.corpora.filter(c => chosen.has(c.id)).map(c => c.id);
    if (!corpusIds.length) throw new Error("Choose at least one conversation dataset on § 4 Conversation");
    const n = id => Number($(id).value);
    return {
      corpusIds,
      chats: $("convChats").checked,
      tokens: n("convTokens"),
      batchSize: n("convBatch"),
      learningRate: n("convRate"),
      bothSides: $("convBoth").checked,
      ownName: $("convName").value,
      situations: $("convSituations").checked,
      earlierPercent: n("convEarlier"),
      guardQuestions: n("convGuard"),
      guardDropPercent: n("convDrop"),
      curriculum: $("convCurriculum").checked,
      competencyLoss: n("convMastery"),
      competencyGainPercent: n("convMasteryGain"),
      advanceWhenStuck: $("convAdvanceStuck").checked,
      reviewPercent: n("convReview"),
      targeted: $("convTargeted").checked,
      contextWeight: n("convContextWeight"),
      listeningPercent: n("convListening"),
      contextTests: n("convTests"),
      replyChecks: $("convReplyChecks").checked,
      readingPercent: n("convReading"),
      readingRisePercent: n("convReadingRise"),
      skipAiVoice: $("convSkipAi").checked,
      keepBalanced: $("convBalanced").checked,
      examplesPercent: n("convExamples"),
      understandingPractice: $("convUnderstanding").checked,
      understandingTests: n("convUnderstandingTests"),
    };
  }
  window.conversationOptions = async () => {
    if (!material.corpora.length) await refresh();
    return options();
  };

  $("converse").addEventListener("click", safe(async () => {
    if (!active) throw new Error("Open or create a model first");
    const o = options();
    if (!await window.beforeNewRun()) return;
    await invoke("converse", { options: o });
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
  // Continuing a run from its recovery point (§ 1 Models, Unfinished training).
  window.resumeConversation = info => {
    go("conversation");
    started = Date.now();
    setTeaching(true);
    $("stopConverse").disabled = false;
    $("convEntry").hidden = false;
    $("convBar").style.width = (100 * info.step / Math.max(1, info.steps)).toFixed(1) + "%";
    $("convTitle").textContent = `Continuing: ${info.label} (from step ${fmt(info.step)} of ${fmt(info.steps)})`;
    $("convLine").textContent = "Reading the datasets and checking that they match the recovery point.";
    $("convChecks").tBodies[0].replaceChildren();
    pendingPrevious = info.previous || [];
    say("Continuing conversation practice from its recovery point…");
  };
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
    // A queued conversation download that starts is shown here (app.js renders the queue).
    window.addEventListener("download-queue", ev => {
      const q = ev.detail;
      if (q.waiting.length) $("convDownload").hidden = false;
      const cur = q.current && sources.find(s => s.id === q.current.source);
      if (cur && downloading !== cur.id) { downloading = cur.id; showDownload(cur.name); }
    });
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
      $("convDownloadLine").textContent = `Saved ${e.corpus.name}: ${mb(e.corpus.bytes)}, ${fmt(e.corpus.documents)} conversations.` +
        (e.corpus.filter ? ` Quality filter: ${filterText(e.corpus.filter)}.` : "");
      chosen.add(e.corpus.id);
      refresh().catch(err => log("ERROR", err));
    });
    invoke("conversation_sources").then(s => { sources = s; renderSources(); }).catch(e => log("ERROR", e));
    // The Hugging Face token for gated datasets (only whether one is set comes back).
    const tokenNote = set => { $("hfTokenNote").textContent = set ? "A token is set." : "No token set (needed only for gated datasets)."; };
    invoke("hf_token_set").then(tokenNote).catch(e => log("ERROR", e));
    $("hfTokenSave").addEventListener("click", safe(async () => {
      const token = $("hfToken").value.trim();
      if (!token) throw new Error("Paste your Hugging Face access token first");
      tokenNote(await invoke("set_hf_token", { token }));
      $("hfToken").value = "";
    }, "Save token"));
    $("hfTokenClear").addEventListener("click", safe(async () => {
      tokenNote(await invoke("set_hf_token", { token: "" }));
    }, "Remove token"));
    // Your own conversations: a file, or text written or pasted here.
    const added = c => {
      chosen.add(c.id);
      $("convOwnNote").textContent = `Saved ${c.name}: ${fmt(c.documents)} conversation(s). It is ticked in the list above.`;
      say(`Saved ${c.name}.`);
      refresh().catch(err => log("ERROR", err));
    };
    $("convImport").addEventListener("click", safe(async () => {
      const c = await invoke("import_conversations");
      if (c) added(c);
    }, "Add conversations"));
    $("convOwnSave").addEventListener("click", safe(async () => {
      const text = $("convOwnText").value;
      if (!text.trim()) throw new Error("Write or paste some conversations first");
      added(await invoke("save_conversations", { name: $("convOwnName").value, text }));
    }, "Save conversations"));
  }
})();
