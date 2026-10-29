"use strict";
// Shared pieces: a safe Markdown renderer and the training progress card.

// ---- Markdown ------------------------------------------------------------------------------
// Builds DOM nodes directly (never innerHTML), so HTML or scripts in imported text are shown as
// text and never run. Links are shown as underlined text without a target: following a link
// would navigate the app's window.
function mdInline(text) {
  const frag = document.createDocumentFragment();
  let i = 0;
  let plain = "";
  const flush = () => { if (plain) { frag.append(document.createTextNode(plain)); plain = ""; } };
  const find = (s, from) => { const j = text.indexOf(s, from); return j < 0 ? null : j; };
  while (i < text.length) {
    const c = text[i];
    if (c === "\\" && i + 1 < text.length && "\\`*_[]()#+-.!|<>".includes(text[i + 1])) { plain += text[i + 1]; i += 2; continue; }
    if (c === "`") {
      const j = find("`", i + 1);
      if (j != null) { flush(); frag.append(el("code", {}, text.slice(i + 1, j))); i = j + 1; continue; }
    }
    if ((c === "*" || c === "_") && text[i + 1] === c) {
      const j = find(c + c, i + 2);
      if (j != null && j > i + 2) { flush(); const b = el("strong"); b.append(mdInline(text.slice(i + 2, j))); frag.append(b); i = j + 2; continue; }
    }
    if ((c === "*" || c === "_") && text[i + 1] && text[i + 1] !== " ") {
      const wordEdge = c === "*" || i === 0 || !/[A-Za-z0-9]/.test(text[i - 1]);
      const j = find(c, i + 1);
      if (wordEdge && j != null && j > i + 1 && text[j - 1] !== " ") {
        flush(); const e = el("em"); e.append(mdInline(text.slice(i + 1, j))); frag.append(e); i = j + 1; continue;
      }
    }
    if (c === "!" && text[i + 1] === "[" || c === "[") {
      const start = c === "!" ? i + 2 : i + 1;
      const close = find("]", start);
      if (close != null && text[close + 1] === "(") {
        const end = find(")", close + 2);
        if (end != null) {
          flush();
          const label = text.slice(start, close);
          frag.append(c === "!" ? el("span", { class: "md-image", title: text.slice(close + 2, end) }, `[image: ${label}]`)
            : el("span", { class: "md-link", title: text.slice(close + 2, end) }, label));
          i = end + 1;
          continue;
        }
      }
    }
    plain += c;
    i++;
  }
  flush();
  return frag;
}

function renderMarkdown(md) {
  const root = el("div", { class: "md" });
  const lines = String(md || "").replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  const isTableRow = l => /^\s*\|.*\|\s*$/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();
    if (!t) { i++; continue; }
    if (t.startsWith("```") || t.startsWith("~~~")) {
      const fence = t.slice(0, 3);
      const code = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence)) code.push(lines[i++]);
      i++;
      root.append(el("pre", {}, el("code", {}, code.join("\n"))));
      continue;
    }
    const h = /^(#{1,6})\s+(.*?)\s*#*$/.exec(t);
    if (h) { const e = el("h" + Math.min(6, h[1].length + 2)); e.append(mdInline(h[2])); root.append(e); i++; continue; }
    if (/^([-*_])(\s*\1){2,}$/.test(t)) { root.append(el("hr")); i++; continue; }
    if (t.startsWith(">")) {
      const q = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) q.push(lines[i++].trim().replace(/^>\s?/, ""));
      const b = el("blockquote");
      b.append(renderMarkdown(q.join("\n")));
      root.append(b);
      continue;
    }
    if (isTableRow(line)) {
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) rows.push(lines[i++]);
      const cells = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      const table = el("table", { class: "md-table" });
      let header = null;
      const body = [];
      rows.forEach((r, k) => {
        const cs = cells(r);
        if (cs.every(c => /^:?-{2,}:?$/.test(c))) return;
        if (k === 0 && rows[1] && cells(rows[1]).every(c => /^:?-{2,}:?$/.test(c))) header = cs; else body.push(cs);
      });
      if (header) {
        const tr = el("tr");
        for (const c of header) { const th = el("th"); th.append(mdInline(c)); tr.append(th); }
        table.append(el("thead", {}, tr));
      }
      const tb = el("tbody");
      for (const r of body) { const tr = el("tr"); for (const c of r) { const td = el("td"); td.append(mdInline(c)); tr.append(td); } tb.append(tr); }
      table.append(tb);
      root.append(table);
      continue;
    }
    const bullet = /^([-*+])\s+(.*)$/;
    const number = /^(\d+)[.)]\s+(.*)$/;
    if (bullet.test(t) || number.test(t)) {
      const ordered = number.test(t);
      const list = el(ordered ? "ol" : "ul");
      while (i < lines.length) {
        const m = (ordered ? number : bullet).exec(lines[i].trim());
        if (!m) break;
        const li = el("li");
        li.append(mdInline(m[2]));
        list.append(li);
        i++;
      }
      root.append(list);
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|~~~|>|[-*+]\s|\d+[.)]\s)/.test(lines[i].trim()) && !isTableRow(lines[i])) {
      para.push(lines[i++].trim());
    }
    if (!para.length) { para.push(t); i++; }
    const p = el("p");
    p.append(mdInline(para.join(" ")));
    root.append(p);
  }
  return root;
}

// ---- Progress card -------------------------------------------------------------------------
// A card with a header (what is being taught, the current part, a status), a timeline of steps
// (done, running with real progress, active: under way alongside the running step, waiting,
// failed, skipped) and a footer line.
//   const card = new TaskCard($("someContainer"));
//   card.set({ title, subtitle, status, footer, steps: [{ id, title, detail, state, progress }] });
//   card.step("train", { state: "running", progress: 0.4, detail: "..." });
class TaskCard {
  constructor(container) {
    this.container = container;
    this.data = { title: "", subtitle: "", status: "", footer: "", steps: [] };
  }
  set(data) {
    if (data.steps) this.index = null;
    this.data = { ...this.data, ...data, steps: data.steps ?? this.data.steps };
    this.container.hidden = false; // shown at once (callers check it), drawn on the next update
    this.schedule();
  }
  step(id, patch) {
    let s = this.index?.get(id);
    if (!s) {
      s = this.data.steps.find(x => x.id === id);
      if (!s) { s = { id }; this.data.steps.push(s); }
      (this.index ??= new Map()).set(id, s);
    }
    Object.assign(s, patch);
    this.schedule();
  }
  // Changes are drawn together, at most about ten times a second: a group of 151 lessons
  // updates every lesson for every batch, and redrawing the card each time fell so far behind
  // that the page stopped answering (Stop included).
  schedule() {
    if (this.pending) return;
    this.pending = setTimeout(() => { this.pending = null; this.render(); }, 100);
  }
  steps() { return this.data.steps; }
  hide() {
    if (this.pending) { clearTimeout(this.pending); this.pending = null; }
    this.container.hidden = true;
  }
  render() {
    if (this.pending) { clearTimeout(this.pending); this.pending = null; }
    const d = this.data;
    this.index = new Map(d.steps.map(s => [s.id, s]));
    this.container.hidden = false;
    const icon = s => el("span", { class: "tc-icon " + (s.state || "waiting"), "aria-hidden": "true" },
      s.state === "done" ? "✓" : s.state === "failed" ? "✕" : s.state === "skipped" ? "–" : "");
    const label = { done: "done", running: "in progress", active: "under way", waiting: "waiting", failed: "failed", skipped: "skipped" };
    const list = el("ol", { class: "tc-steps" }, ...d.steps.map(s => el("li", { class: "tc-step " + (s.state || "waiting") },
      icon(s),
      el("div", { class: "tc-body" },
        el("div", { class: "tc-title" }, s.title, el("span", { class: "sr-only" }, ` (${label[s.state || "waiting"]})`)),
        s.detail ? el("div", { class: "tc-detail" }, s.detail) : "",
        s.state === "running" && s.progress != null ? el("div", { class: "tc-progress" },
          el("div", { class: "tc-progress-label" }, el("span", {}, s.progressLabel || "Progress"), el("span", { class: "num" }, `${Math.round(100 * Math.max(0, Math.min(1, s.progress)))}%`)),
          el("div", { class: "tc-bar" }, el("i", { style: `width:${(100 * Math.max(0, Math.min(1, s.progress))).toFixed(1)}%` }))) : ""))));
    this.container.replaceChildren(el("div", { class: "task-card" },
      el("div", { class: "tc-head" },
        el("span", { class: "tc-mark", "aria-hidden": "true" }, "§"),
        el("div", { class: "tc-names" }, el("div", { class: "tc-name" }, d.title || "—"), d.subtitle ? el("div", { class: "tc-sub" }, d.subtitle) : ""),
        d.status ? el("span", { class: "tc-status " + (d.statusKind || "") }, d.status) : ""),
      list,
      d.footer ? el("div", { class: "tc-foot" }, ...[].concat(d.footer).map(f => el("span", {}, f))) : ""));
  }
}

// Study Mode, shown in the footer of progress cards (set by the Study page).
let studyModeOn = false;
const studyFooter = () => `Study Mode: ${studyModeOn ? "on" : "off"}`;

// ---- Dialogs -------------------------------------------------------------------------------
// In-page confirmation and text dialogs that wait for the answer. (The webview's confirm() and
// prompt() do not reliably wait on every platform, so they are not used.)
function modal(message, withInput, initial) {
  return new Promise(resolve => {
    const before = document.activeElement;
    const input = withInput ? el("input", { type: "text", value: initial ?? "", "aria-label": message }) : null;
    const ok = el("button", { class: "primary" }, "OK");
    const cancel = el("button", {}, "Cancel");
    const overlay = el("div", { class: "modal" }, el("div", { class: "modal-box", role: "dialog", "aria-modal": "true" },
      el("p", {}, message), input || "", el("div", { class: "actions" }, cancel, ok)));
    const answer = yes => withInput ? (yes ? input.value : null) : yes;
    const done = yes => {
      overlay.remove();
      document.removeEventListener("keydown", key, true);
      before?.focus?.();
      resolve(answer(yes));
    };
    const key = e => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(false); }
      else if (e.key === "Enter" && document.activeElement !== cancel) { e.preventDefault(); e.stopPropagation(); done(true); }
      else if (e.key === "Tab") { e.preventDefault(); (document.activeElement === ok ? (input || cancel) : document.activeElement === input ? cancel : ok).focus(); }
    };
    ok.addEventListener("click", () => done(true));
    cancel.addEventListener("click", () => done(false));
    overlay.addEventListener("click", e => { if (e.target === overlay) done(false); });
    document.addEventListener("keydown", key, true);
    document.body.append(overlay);
    if (input) { input.focus(); input.select(); } else ok.focus();
  });
}
/** A dialog with a title, content and buttons ([key, label, primary?]); resolves the key of the
 *  button chosen, or null (Escape or a click outside). */
function choose(title, content, buttons) {
  return new Promise(resolve => {
    const before = document.activeElement;
    const btns = buttons.map(([key, label, primary]) => {
      const b = el("button", primary ? { class: "primary" } : {}, label);
      b.addEventListener("click", () => done(key));
      return b;
    });
    const overlay = el("div", { class: "modal" }, el("div", { class: "modal-box wide", role: "dialog", "aria-modal": "true" },
      el("h3", {}, title), content, el("div", { class: "actions" }, ...btns)));
    const done = key => {
      overlay.remove();
      document.removeEventListener("keydown", esc, true);
      before?.focus?.();
      resolve(key);
    };
    const esc = e => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); } };
    overlay.addEventListener("click", e => { if (e.target === overlay) done(null); });
    document.addEventListener("keydown", esc, true);
    document.body.append(overlay);
    (btns.find((_, i) => buttons[i][2]) || btns[0])?.focus();
  });
}
/** Resolves true (OK) or false (Cancel). */
const askYes = message => modal(message, false);
/** Resolves the text entered, or null (Cancel). */
const askText = (message, initial) => modal(message, true, initial);
