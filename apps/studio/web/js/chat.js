"use strict";
// § 13 Chat: saved conversations with the open model. Replies are written by the model in the
// Engine token by token and arrive here as "chat-token" events; this page only displays them.
(() => {
  const DEFAULTS = {
    context: "", history: "all", turns: 6, replyTokens: 0, fixedSeed: false, memories: true, rememberSelf: true, trace: false,
    sampling: { temperature: 0, topK: 0, topP: 1, repetitionPenalty: 1, seed: 0 },
  };
  const SAMPLED_TEMPERATURE = 0.8;
  let chat = null;          // the open chat, or null for a new chat not yet sent
  let settings = structuredClone(DEFAULTS);
  let list = [];
  let pending = [];         // attachments waiting to be sent
  let editing = null;       // index of the message being edited
  let writing = null;       // { chat, node, text, started } while a reply is written
  let shownModel = null;
  let visible = false;

  const log$ = () => $("chatLog");
  const modelName = () => active?.name || "Model";
  const copy = text => navigator.clipboard?.writeText(text).then(() => say("Copied."), () => say("Copying is not allowed here."));
  const ago = secs => {
    if (!secs) return "";
    const d = Date.now() / 1000 - secs;
    if (d < 60) return "just now";
    if (d < 3600) return `${Math.floor(d / 60)} min ago`;
    if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
    return new Date(secs * 1000).toLocaleDateString();
  };
  const bytes = n => n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : n >= 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`;
  const narrow = () => window.matchMedia("(max-width: 900px)").matches;

  // ---- the list of chats -------------------------------------------------------------------
  function group(c) {
    if (c.pinned) return "Pinned";
    const day = 86400, now = new Date();
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
    if (c.updated >= start) return "Today";
    if (c.updated >= start - day) return "Yesterday";
    if (c.updated >= start - 7 * day) return "Previous 7 days";
    return "Older";
  }
  function renderList() {
    const box = $("chatItems");
    const q = $("chatSearch").value.trim().toLowerCase();
    const shown = list.filter(c => !q || c.title.toLowerCase().includes(q) || c.preview.toLowerCase().includes(q));
    box.replaceChildren();
    if (!active) { box.append(el("p", { class: "empty" }, "Open a model to see its chats.")); return; }
    if (!shown.length) { box.append(el("p", { class: "empty" }, q ? "No chat matches." : "No chats yet.")); return; }
    let last = null;
    for (const c of shown) {
      const g = group(c);
      if (g !== last) { box.append(el("div", { class: "chat-group" }, g)); last = g; }
      box.append(el("button", {
        type: "button", class: "chat-item" + (chat?.id === c.id ? " is-open" : ""), "aria-current": chat?.id === c.id ? "true" : null,
        onclick: safe(async () => { await openChat(c.id); if (narrow()) showList(false); }, "Open chat"),
      }, el("span", { class: "chat-item-title" }, c.pinned ? "📌 " : "", c.title),
        el("span", { class: "chat-item-meta" }, c.preview || "Empty", " · ", ago(c.updated))));
    }
  }
  async function refreshList() {
    list = active ? await invoke("chats") : [];
    renderList();
  }
  function showList(open) {
    $("chatShell").classList.toggle("list-open", open);
  }

  // ---- messages ----------------------------------------------------------------------------
  function avatar(role) {
    return el("span", { class: "chat-avatar " + role, "aria-hidden": "true" }, role === "user" ? "You"[0] : (modelName()[0] || "N").toUpperCase());
  }
  function attachmentsView(atts) {
    if (!atts?.length) return "";
    return el("div", { class: "chat-atts" }, ...atts.map(a => a.kind === "image"
      ? el("button", { type: "button", class: "chat-thumb", title: `${a.name} (${bytes(a.bytes)}). Kept with the chat; the model's current networks read text only.`, onclick: () => lightbox(a) },
        el("img", { src: a.data, alt: a.name }))
      : el("span", { class: "chat-file", title: `${fmt(a.tokens)} tokens${a.dropped ? `; ${fmt(a.dropped)} character(s) the model cannot read were left out` : ""}` },
        el("span", { class: "chat-file-icon", "aria-hidden": "true" }, "TXT"), el("span", {}, a.name), el("span", { class: "chat-file-size" }, `${fmt(a.tokens)} tokens`))));
  }
  function lightbox(a) {
    const close = () => { box.remove(); document.removeEventListener("keydown", key, true); };
    const key = e => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); } };
    document.addEventListener("keydown", key, true);
    const box = el("div", { class: "modal chat-lightbox", onclick: close },
      el("figure", {}, el("img", { src: a.data, alt: a.name }),
        el("figcaption", {}, `${a.name} · ${bytes(a.bytes)} · kept with this chat for networks that can see images; the model does not read it now.`)));
    document.body.append(box);
  }
  function samplingText(s) {
    if (!s || !(s.temperature > 0)) return s && s.repetitionPenalty > 1 ? `most likely · penalty ${s.repetitionPenalty}` : "most likely";
    const parts = [`T ${s.temperature}`];
    if (s.topP < 1) parts.push(`top-p ${s.topP}`);
    if (s.topK > 0) parts.push(`top-k ${s.topK}`);
    if (s.repetitionPenalty > 1) parts.push(`penalty ${s.repetitionPenalty}`);
    parts.push(`seed ${s.seed}`);
    return "sampled · " + parts.join(" · ");
  }
  function infoView(m, i) {
    const f = m.info;
    if (!f) return "";
    const secs = f.millis / 1000;
    const rate = secs > 0 ? `${(f.tokens / secs).toFixed(1)} tokens/s` : "";
    const end = f.stopped ? "stopped" : f.finished ? "" : "reached the length limit";
    const summary = [`${fmt(f.tokens)} tokens`, `${secs.toFixed(1)} s`, rate, end].filter(Boolean).join(" · ");
    const details = el("details", { class: "chat-info" }, el("summary", {}, summary),
      el("dl", {},
        el("dt", {}, "Prompt"), el("dd", {}, `${fmt(f.promptTokens)} of ${fmt(f.context)} tokens (up to ${fmt(f.replyLimit)} kept for the reply)`),
        el("dt", {}, "Earlier turns"), el("dd", {}, `${fmt(f.turnsUsed)} in the prompt${f.turnsLeftOut ? `, ${fmt(f.turnsLeftOut)} left out` : ""}`),
        f.attachmentTokensCut ? el("dt", {}, "Attached text") : "", f.attachmentTokensCut ? el("dd", {}, `${fmt(f.attachmentTokensCut)} tokens cut to fit`) : "",
        el("dt", {}, "Words chosen"), el("dd", {}, samplingText(f.sampling)),
        el("dt", {}, "Model"), el("dd", {}, `step ${fmt(f.step)}`),
        f.memories?.length ? el("dt", {}, "Memories") : "", f.memories?.length ? el("dd", {}, el("ul", {}, ...f.memories.map(t => el("li", {}, t)))) : ""));
    return details;
  }
  // The teacher's mark on a reply: by consistency with what Nova said before and whether it is
  // its own voice, never by agreement. Replies marked consistent can be practised in § 4.
  function markButtons(m, i) {
    const set = mark => safe(async () => {
      chat = await invoke("chat_mark", { id: chat.id, index: i, mark: m.mark === mark ? null : mark });
      render();
    }, "Mark")();
    const hint = "Mark by consistency with what Nova said before and whether it gave its own reason, not by whether you agree.";
    return el("span", { class: "chat-marks", title: hint },
      el("button", { type: "button", class: "chat-act chat-mark" + (m.mark === "consistent" ? " on good" : ""), "aria-pressed": String(m.mark === "consistent"), onclick: () => set("consistent") }, "✓ Consistent"),
      el("button", { type: "button", class: "chat-act chat-mark" + (m.mark === "contradicts" ? " on bad" : ""), "aria-pressed": String(m.mark === "contradicts"), onclick: () => set("contradicts") }, "✗ Contradicts itself"));
  }
  function messageView(m, i) {
    const mine = m.role === "user";
    const last = i === chat.messages.length - 1;
    const actions = el("div", { class: "chat-actions" },
      el("button", { type: "button", class: "chat-act", onclick: () => copy(m.text) }, "Copy"),
      mine ? el("button", { type: "button", class: "chat-act", onclick: () => startEdit(i) }, "Edit") : "",
      !mine && last ? el("button", { type: "button", class: "chat-act", onclick: safe(regenerate, "Regenerate") }, "Regenerate") : "",
      !mine && i > 0 ? el("button", { type: "button", class: "chat-act", title: m.info?.trace ? "Open the trace recorded while Nova wrote this reply (§ 11 Brain)" : "Replay this reply through the model on § 11 Brain", onclick: () => window.openTrace?.(chat.id, i) }, m.info?.trace ? "Brain trace" : "Trace on Brain page") : "",
      mine ? "" : markButtons(m, i));
    const body = el("div", { class: "chat-body" },
      el("div", { class: "chat-who" }, mine ? "You" : modelName(), el("time", {}, m.time ? new Date(m.time * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "")),
      attachmentsView(m.attachments),
      el("div", { class: "chat-text" + (m.text ? "" : " empty-reply") }, m.text || "(an empty reply: the model's first token ended the line)"),
      m.dropped ? el("p", { class: "chat-flag" }, `${fmt(m.dropped)} character(s) the model cannot read (emoji, other alphabets) were left out.`) : "",
      m.info?.memories?.length ? el("p", { class: "chat-flag memory" }, `${m.info.memories.length} Study Mode memor${m.info.memories.length === 1 ? "y" : "ies"} were in front of the message.`) : "",
      m.info?.remembered?.length ? el("p", { class: "chat-flag said" }, "Remembered what it said about itself: ", m.info.remembered.map(r => `“${r[1]}”`).join(" ")) : "",
      m.info?.mayContradict?.length ? el("p", { class: "chat-flag" }, "This may contradict what it said before: ", m.info.mayContradict.map(t => `“${t}”`).join(" "), " You can mark the reply below, or change the older memory on § 8 Study.") : "",
      mine ? "" : infoView(m, i), actions);
    return el("article", { class: "chat-msg " + (mine ? "user" : "model") }, mine ? "" : avatar("model"), body);
  }
  function renderEmpty() {
    const box = el("div", { class: "chat-welcome" });
    if (!active) {
      box.append(el("p", { class: "chat-welcome-title" }, "No model is open"), el("p", { class: "note" }, "Open or create a model on § 1 Models to chat with it."));
    } else {
      box.append(el("div", { class: "chat-welcome-mark", "aria-hidden": "true" }, (active.name[0] || "N").toUpperCase()),
        el("p", { class: "chat-welcome-title" }, `Chat with ${active.name}`),
        el("p", { class: "note" }, `Step ${fmt(active.step)} · ${fmt(active.words)} words learned · reads ${fmt(active.context)} tokens at once. ` +
          "It replies in the way it was taught: one line per turn. Attach text files for it to read; images are kept with the chat for networks that can see them later."));
    }
    return box;
  }
  function render() {
    const box = log$();
    box.replaceChildren();
    $("chatTitle").textContent = chat ? chat.title : "New chat";
    $("chatSub").textContent = active ? `${active.name} · ${chat ? `${fmt(chat.messages.length)} message(s)` : "not saved until you send"}` : "";
    if (!chat || !chat.messages.length) box.append(renderEmpty());
    else chat.messages.forEach((m, i) => box.append(messageView(m, i)));
    if (writing && (!chat || writing.chat === chat.id)) box.append(writing.node);
    box.scrollTop = box.scrollHeight;
    renderList();
  }
  function writingNode() {
    const text = el("span");
    const caret = el("span", { class: "chat-caret", "aria-hidden": "true" });
    const node = el("article", { class: "chat-msg model writing" }, avatar("model"),
      el("div", { class: "chat-body" }, el("div", { class: "chat-who" }, modelName(), el("time", {}, "writing…")), el("div", { class: "chat-text" }, text, caret)));
    return { node, text };
  }

  // ---- settings ----------------------------------------------------------------------------
  function showSettings() {
    const s = settings, sp = s.sampling;
    $("chatContext").value = s.context;
    $("chatHistory").value = s.history;
    $("chatTurns").value = s.turns;
    $("chatTurns").hidden = s.history !== "last";
    $("chatReplyTokens").value = s.replyTokens;
    $("chatMode").value = sp.temperature > 0 ? "sample" : "greedy";
    $("chatSampling").hidden = !(sp.temperature > 0);
    $("chatTemp").value = sp.temperature > 0 ? sp.temperature : SAMPLED_TEMPERATURE;
    $("chatTopP").value = sp.topP;
    $("chatTopK").value = sp.topK;
    $("chatPenalty").value = sp.repetitionPenalty;
    $("chatFixedSeed").checked = s.fixedSeed;
    $("chatSeed").value = sp.seed;
    $("chatSeed").hidden = !s.fixedSeed;
    $("chatMemories").checked = s.memories;
    $("chatRememberSelf").checked = s.rememberSelf;
    $("chatTrace").checked = !!s.trace;
    outputs();
    windowNote();
  }
  function outputs() {
    $("chatTempOut").textContent = Number($("chatTemp").value).toFixed(2);
    $("chatTopPOut").textContent = Number($("chatTopP").value).toFixed(2);
    $("chatPenaltyOut").textContent = Number($("chatPenalty").value).toFixed(2);
  }
  function readSettings() {
    const sample = $("chatMode").value === "sample";
    const num = (id, d) => { const v = Number($(id).value); return Number.isFinite(v) ? v : d; };
    settings = {
      context: $("chatContext").value,
      history: $("chatHistory").value,
      turns: Math.max(1, Math.round(num("chatTurns", 6))),
      replyTokens: Math.max(0, Math.round(num("chatReplyTokens", 0))),
      fixedSeed: $("chatFixedSeed").checked,
      memories: $("chatMemories").checked,
      rememberSelf: $("chatRememberSelf").checked,
      trace: $("chatTrace").checked,
      sampling: {
        temperature: sample ? num("chatTemp", SAMPLED_TEMPERATURE) : 0,
        topP: num("chatTopP", 1), topK: Math.max(0, Math.round(num("chatTopK", 0))),
        repetitionPenalty: num("chatPenalty", 1), seed: Math.max(0, Math.round(num("chatSeed", 0))),
      },
    };
    $("chatTurns").hidden = settings.history !== "last";
    $("chatSampling").hidden = !sample;
    $("chatSeed").hidden = !settings.fixedSeed;
    outputs();
  }
  function windowNote() {
    if (!active) { $("chatWindowNote").textContent = "—"; return; }
    const def = Math.min(512, Math.max(48, Math.floor(active.context / 4)));
    $("chatReplyNote").textContent = `tokens (0: Studio's default, ${fmt(def)} for this model)`;
    $("chatWindowNote").textContent = `${active.name} reads ${fmt(active.context)} tokens at once: the context text, memories, earlier turns, attached text and your message, plus the reply. ` +
      "What does not fit is left out, oldest turns first; the meter under the message box shows how it is spent.";
    $("chatMemoryNote").textContent = studyModeOn ? "Study Mode is on." : "Study Mode is off (§ 8 Study), so no memories are added.";
  }
  let saveTimer = 0;
  function settingsChanged() {
    readSettings();
    measureSoon();
    clearTimeout(saveTimer);
    if (!chat) return;
    const id = chat.id;
    saveTimer = setTimeout(safe(async () => {
      const saved = await invoke("chat_settings", { id, settings });
      if (chat?.id === id) { chat.settings = saved.settings; $("chatContextNote").textContent = contextNote(saved.settings.context); }
    }, "Chat settings"), 400);
  }
  const contextNote = text => text && text !== $("chatContext").value.trim()
    ? "Saved as the model reads it: characters outside its alphabet were changed or left out." : "";
  function showSettingsPanel(open) {
    $("chatSettings").hidden = !open;
    $("chatShell").classList.toggle("settings-open", open);
    $("chatSettingsOpen").setAttribute("aria-expanded", String(open));
    if (open) showSettings();
  }

  // ---- composing ---------------------------------------------------------------------------
  function grow() {
    const t = $("chatInput");
    t.style.height = "auto";
    t.style.height = Math.min(t.scrollHeight + 2, Math.max(120, window.innerHeight * 0.35)) + "px";
  }
  function renderPending() {
    const box = $("chatPending");
    box.replaceChildren(...pending.map((a, i) => el("span", { class: "chat-chip" + (a.kind === "image" ? " image" : "") },
      a.kind === "image" ? el("img", { src: a.data, alt: "" }) : el("span", { class: "chat-file-icon", "aria-hidden": "true" }, "TXT"),
      el("span", { class: "chat-chip-name", title: a.name }, a.name),
      el("span", { class: "chat-chip-size" }, a.kind === "image" ? bytes(a.bytes) : `${fmt(a.tokens)} tokens`),
      el("button", { type: "button", class: "chat-chip-x", "aria-label": `Remove ${a.name}`, onclick: () => { pending.splice(i, 1); renderPending(); measureSoon(); } }, "✕"))));
    const images = pending.some(a => a.kind === "image");
    const dropped = pending.reduce((n, a) => n + (a.dropped || 0), 0);
    if (images || dropped) {
      box.append(el("p", { class: "chat-flag" },
        images ? "Images are kept with the chat; the model's current networks read text only, so it will not see them. " : "",
        dropped ? `${fmt(dropped)} character(s) in attached text are outside the model's alphabet and were left out.` : ""));
    }
  }
  function addPending(atts) {
    for (const a of atts) {
      if (pending.length >= 8) { say("At most 8 files can be attached to a message."); break; }
      pending.push(a);
    }
    renderPending();
    measureSoon();
  }
  let measureTimer = 0;
  function measureSoon() {
    clearTimeout(measureTimer);
    measureTimer = setTimeout(measure, 250);
  }
  async function measure() {
    const meter = $("chatMeter");
    if (!active) { meter.textContent = ""; return; }
    try {
      const b = await invoke("chat_measure", { id: chat?.id ?? null, settings, text: $("chatInput").value, attachments: pending, edit: editing });
      const used = b.promptTokens + b.replyLimit;
      const parts = [`${fmt(b.promptTokens)} + ${fmt(b.replyLimit)} reply of ${fmt(b.context)} tokens`];
      if (b.turnsLeftOut) parts.push(`${fmt(b.turnsLeftOut)} earlier turn(s) left out`);
      if (b.attachmentTokensCut) parts.push(`${fmt(b.attachmentTokensCut)} attached tokens cut`);
      meter.replaceChildren(el("span", { class: "chat-gauge", "aria-hidden": "true" }, el("i", { style: `width:${Math.min(100, 100 * used / b.context).toFixed(1)}%` })),
        el("span", {}, parts.join(" · ")));
      meter.title = `Context text ${fmt(b.contextTextTokens)} · earlier turns ${fmt(b.historyTokens)} (${fmt(b.turnsUsed)}) · attached text ${fmt(b.attachmentTokens)} · message ${fmt(b.messageTokens)} tokens` +
        (studyModeOn && settings.memories ? " · Study Mode memories are added when you send" : "");
      meter.classList.remove("over");
    } catch (e) {
      meter.textContent = typeof e === "string" ? e : e?.message || String(e);
      meter.classList.add("over");
    }
  }
  function startEdit(i) {
    if (writing) return;
    const m = chat.messages[i];
    editing = i;
    $("chatInput").value = m.text;
    pending = [...(m.attachments || [])];
    renderPending();
    $("chatEditing").hidden = false;
    grow();
    $("chatInput").focus();
    measureSoon();
  }
  function cancelEdit() {
    editing = null;
    $("chatEditing").hidden = true;
    $("chatInput").value = "";
    pending = [];
    renderPending();
    grow();
    measureSoon();
  }
  function setWriting(on) {
    $("chatSend").hidden = on;
    $("chatStop").hidden = !on;
    $("chatAttach").disabled = on;
  }

  async function runReply(work, chatId) {
    const w = writingNode();
    writing = { chat: chatId, node: w.node, text: w.text, started: performance.now(), count: 0 };
    setWriting(true);
    render();
    try {
      const updated = await work();
      if (chat?.id === updated.id) chat = updated;
    } finally {
      writing = null;
      setWriting(false);
      if (chat) chat = await invoke("chat_open", { id: chat.id }).catch(() => chat);
      render();
      await refreshList();
      measureSoon();
    }
  }
  async function send() {
    if (!active) throw new Error("Open a model first");
    if (writing) return;
    const text = $("chatInput").value;
    if (!text.trim()) { $("chatInput").focus(); return; }
    const attachments = pending, edit = editing;
    if (!chat) {
      chat = await invoke("chat_new", { settings });
    }
    const id = chat.id;
    // Shown at once; the saved chat replaces it when the reply is done.
    const shown = { role: "user", text: text.trim(), time: Date.now() / 1000, attachments, dropped: 0 };
    chat.messages = edit == null ? [...chat.messages, shown] : [...chat.messages.slice(0, edit), shown];
    $("chatInput").value = "";
    pending = [];
    editing = null;
    $("chatEditing").hidden = true;
    renderPending();
    grow();
    try {
      await runReply(() => invoke("chat_send", { id, text, attachments, edit }), id);
    } catch (e) {
      // A message that could not be sent was not saved: it goes back in the message box (and a
      // chat created for it goes again).
      const saved = chat?.messages.length && chat.messages[chat.messages.length - 1].role === "user";
      if (!saved && !$("chatInput").value) {
        $("chatInput").value = text; pending = attachments; editing = edit; $("chatEditing").hidden = edit == null; renderPending(); grow();
      }
      if (chat && !chat.messages.length) {
        await invoke("chat_delete", { id: chat.id }).catch(() => {});
        chat = null;
        render();
        await refreshList();
      }
      throw e;
    }
  }
  async function regenerate() {
    if (!chat || writing) return;
    const id = chat.id;
    while (chat.messages.length && chat.messages[chat.messages.length - 1].role === "model") chat.messages.pop();
    await runReply(() => invoke("chat_regenerate", { id }), id);
  }
  function onToken(p) {
    if (!writing || p.chat !== writing.chat) return;
    writing.text.append(p.text);
    writing.count++;
    const box = log$();
    if (box.scrollHeight - box.scrollTop - box.clientHeight < 160) box.scrollTop = box.scrollHeight;
    const secs = (performance.now() - writing.started) / 1000;
    const t = writing.node.querySelector("time");
    if (t) t.textContent = `writing… ${fmt(writing.count)} tokens · ${secs.toFixed(1)} s`;
  }

  // ---- opening and managing chats ----------------------------------------------------------
  async function openChat(id) {
    if (writing) { say("Wait for the reply, or stop it, before opening another chat."); return; }
    chat = await invoke("chat_open", { id });
    settings = { ...structuredClone(DEFAULTS), ...chat.settings, sampling: { ...DEFAULTS.sampling, ...chat.settings.sampling } };
    if (editing != null) cancelEdit();
    showSettings();
    $("chatContextNote").textContent = "";
    render();
    measureSoon();
  }
  function newChat() {
    if (writing) { say("Wait for the reply, or stop it, before starting another chat."); return; }
    chat = null;  // keeps the current settings
    if (editing != null) cancelEdit();
    render();
    measureSoon();
    if (narrow()) showList(false);
    $("chatInput").focus();
  }
  function menu() {
    const btn = $("chatMore");
    document.querySelector(".chat-menu")?.remove();
    const close = () => { m.remove(); btn.setAttribute("aria-expanded", "false"); document.removeEventListener("mousedown", outside, true); };
    const outside = e => { if (!m.contains(e.target) && e.target !== btn) close(); };
    const item = (label, fn, disabled) => el("button", { type: "button", role: "menuitem", disabled: disabled || null, onclick: () => { close(); safe(fn, label)(); } }, label);
    const m = el("div", { class: "chat-menu", role: "menu" },
      item("Rename…", rename, !chat),
      item(chat?.pinned ? "Unpin" : "Pin to the top", pin, !chat),
      item("Save as a text file…", exportChat, !chat || !chat.messages.length),
      item("Delete this chat…", remove, !chat || !!writing));
    const r = btn.getBoundingClientRect();
    m.style.top = r.bottom + 4 + "px";
    m.style.right = Math.max(8, window.innerWidth - r.right) + "px";
    document.body.append(m);
    btn.setAttribute("aria-expanded", "true");
    document.addEventListener("mousedown", outside, true);
    m.addEventListener("keydown", e => { if (e.key === "Escape") { close(); btn.focus(); } });
    m.querySelector("button:not(:disabled)")?.focus();
  }
  async function rename() {
    const title = await askText("Title of this chat", chat.title);
    if (title == null) return;
    chat = await invoke("chat_rename", { id: chat.id, title });
    render();
    await refreshList();
  }
  async function pin() {
    chat = await invoke("chat_pin", { id: chat.id, pinned: !chat.pinned });
    await refreshList();
  }
  async function exportChat() {
    const name = await invoke("chat_export", { id: chat.id });
    if (name) say(`Saved ${name}.`);
  }
  async function remove() {
    if (!(await askYes(`Delete “${chat.title}”? Its messages and attachments are removed for good.`))) return;
    await invoke("chat_delete", { id: chat.id });
    chat = null;
    render();
    await refreshList();
  }

  // ---- the open model changed --------------------------------------------------------------
  window.showChat = async meta => {
    const id = meta?.id ?? null;
    windowNote();
    if (id === shownModel) {
      // The same model with new figures (its step after teaching, say).
      if (meta) $("chatSub").textContent = `${meta.name} · ${chat ? `${fmt(chat.messages.length)} message(s)` : "not saved until you send"}`;
      return;
    }
    shownModel = id;
    if (writing) return;
    chat = null;
    pending = [];
    editing = null;
    $("chatEditing").hidden = true;
    renderPending();
    await refreshList().catch(e => log("ERROR", e));
    if (list.length) await openChat(list[0].id).catch(e => log("ERROR", e)); else render();
  };

  // ---- events ------------------------------------------------------------------------------
  const prevPage = window.onPage;
  window.onPage = page => {
    prevPage?.(page);
    visible = page === "chat";
    if (!visible) return;
    if (shownModel !== (active?.id ?? null)) window.showChat(active);
    else { render(); refreshList().catch(e => log("ERROR", e)); }
    windowNote();
    measureSoon();
    setTimeout(() => $("chatInput").focus(), 0);
  };
  $("chatComposer").addEventListener("submit", e => { e.preventDefault(); safe(send, "Chat")(); });
  $("chatInput").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); safe(send, "Chat")(); }
    else if (e.key === "Escape" && editing != null) { e.preventDefault(); cancelEdit(); }
  });
  $("chatInput").addEventListener("input", () => { grow(); measureSoon(); });
  $("chatInput").addEventListener("paste", e => {
    const files = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith("image/"));
    if (!files.length) return;
    e.preventDefault();
    for (const f of files) {
      if (f.size > 8 << 20) { say(`${f.name || "The pasted image"} is larger than 8 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => addPending([{ kind: "image", name: f.name && f.name !== "image.png" ? f.name : `pasted image ${new Date().toLocaleTimeString()}.png`, data: r.result, bytes: f.size, text: "", dropped: 0, tokens: 0 }]);
      r.readAsDataURL(f);
    }
  });
  document.addEventListener("keydown", e => {
    if (visible && writing && e.key === "Escape" && !document.querySelector(".modal")) { e.preventDefault(); invoke("chat_stop"); }
  });
  $("chatSend").addEventListener("click", e => { e.preventDefault(); safe(send, "Chat")(); });
  $("chatStop").addEventListener("click", () => invoke("chat_stop").catch(e => log("ERROR", e)));
  $("chatAttach").addEventListener("click", safe(async () => {
    if (!active) throw new Error("Open a model first");
    addPending(await invoke("chat_attach"));
  }, "Attach"));
  $("chatEditCancel").addEventListener("click", cancelEdit);
  $("chatNew").addEventListener("click", newChat);
  $("chatSearch").addEventListener("input", renderList);
  $("chatListOpen").addEventListener("click", () => showList(true));
  $("chatListClose").addEventListener("click", () => showList(false));
  $("chatMore").addEventListener("click", menu);
  $("chatSettingsOpen").addEventListener("click", () => showSettingsPanel($("chatSettings").hidden));
  $("chatSettingsClose").addEventListener("click", () => showSettingsPanel(false));
  $("chatContextChange").addEventListener("click", () => $("contextLimit").click());
  for (const ev of ["input", "change"]) $("chatSettings").addEventListener(ev, settingsChanged);
  $("chatMode").addEventListener("change", () => {
    if ($("chatMode").value === "sample" && !(Number($("chatTemp").value) > 0)) $("chatTemp").value = SAMPLED_TEMPERATURE;
  });

  // Files dropped on the window (desktop): attached while this page is shown.
  if (tauri?.event?.listen) {
    const drop = $("chatDrop");
    tauri.event.listen("chat-token", ev => onToken(ev.payload));
    tauri.event.listen("tauri://drag-enter", () => { if (visible && active && !writing) drop.hidden = false; });
    tauri.event.listen("tauri://drag-leave", () => { drop.hidden = true; });
    tauri.event.listen("tauri://drag-drop", ev => {
      drop.hidden = true;
      const paths = ev.payload?.paths || [];
      if (!visible || !active || writing || !paths.length) return;
      safe(async () => addPending(await invoke("chat_attach_paths", { paths })), "Attach")();
    });
  }
  showSettings();
  render();
})();
