"use strict";
// Yendory dropdowns: every <select> in Studio is shown as a dropdown drawn by Studio itself, in
// the lab-notebook design, instead of the operating system's (Windows and Android draw their
// own, each differently). The <select> stays in the page, hidden, and remains the source of
// truth: choosing an item sets its value and fires its "input" and "change" events, so the code
// that reads or listens to it is unchanged. Setting `select.value` (or an option's `selected`)
// from code updates the dropdown too.
//
// An <option> may carry a `data-note` attribute: a second, smaller line in the list explaining
// the choice.
(() => {
  const OPEN = new Set();
  let uid = 0;

  function closeAll(except) {
    for (const d of [...OPEN]) if (d !== except) d.close();
  }

  class Dropdown {
    constructor(select) {
      this.select = select;
      select._ydd = this;
      this.id = `ydd${++uid}`;
      this.button = document.createElement("button");
      this.button.type = "button";
      this.button.className = "ydd-button";
      this.button.setAttribute("role", "combobox");
      this.button.setAttribute("aria-haspopup", "listbox");
      this.button.setAttribute("aria-expanded", "false");
      this.button.setAttribute("aria-controls", this.id + "-list");
      this.label = document.createElement("span");
      this.label.className = "ydd-label";
      const chevron = document.createElement("span");
      chevron.className = "ydd-chevron";
      chevron.setAttribute("aria-hidden", "true");
      this.button.append(this.label, chevron);
      this.wrap = document.createElement("span");
      this.wrap.className = "ydd";
      this.wrap.append(this.button);
      this.list = document.createElement("div");
      this.list.className = "ydd-list";
      this.list.id = this.id + "-list";
      this.list.setAttribute("role", "listbox");
      this.list.hidden = true;
      this.active = -1;
      this.typed = "";
      this.typedAt = 0;
      select.classList.add("ydd-native");
      select.setAttribute("tabindex", "-1");
      select.setAttribute("aria-hidden", "true");
      select.after(this.wrap);
      this.button.addEventListener("click", () => (this.list.hidden ? this.open() : this.close()));
      this.button.addEventListener("keydown", e => this.key(e));
      this.button.addEventListener("blur", () => setTimeout(() => {
        if (!this.list.contains(document.activeElement)) this.close();
      }, 0));
      // A <label for=...> pointing at the select focuses and opens the dropdown instead.
      if (select.id) {
        for (const l of document.querySelectorAll(`label[for="${CSS.escape(select.id)}"]`)) {
          l.addEventListener("click", e => { e.preventDefault(); this.button.focus(); });
          if (!l.id) l.id = this.id + "-label";
          this.button.setAttribute("aria-labelledby", l.id);
        }
      }
      new MutationObserver(() => this.refresh()).observe(select, {
        childList: true, subtree: true, characterData: true,
        attributes: true, attributeFilter: ["disabled", "hidden", "selected", "label", "title", "aria-label", "data-note", "class", "style"],
      });
      this.refresh();
    }
    options() {
      return [...this.select.options];
    }
    refresh() {
      const s = this.select;
      if (s.nextElementSibling !== this.wrap) s.after(this.wrap);
      const o = s.options[s.selectedIndex];
      this.label.textContent = o ? o.textContent : "";
      this.button.title = o ? (o.dataset.note || o.title || o.textContent) : "";
      this.button.disabled = s.disabled;
      this.wrap.hidden = s.hidden || s.style.display === "none";
      const aria = s.getAttribute("aria-label");
      if (aria && !this.button.hasAttribute("aria-labelledby")) this.button.setAttribute("aria-label", aria);
      // Classes a select was styled with carry over to the dropdown (e.g. a narrow table cell).
      this.wrap.className = "ydd " + [...s.classList].filter(c => c !== "ydd-native").map(c => "ydd-" + c).join(" ");
      if (!this.list.hidden) this.renderList();
    }
    renderList() {
      this.list.replaceChildren();
      let group = null;
      this.options().forEach((o, i) => {
        const parent = o.parentElement;
        if (parent && parent.tagName === "OPTGROUP" && parent !== group) {
          group = parent;
          const g = document.createElement("div");
          g.className = "ydd-group";
          g.textContent = parent.label;
          this.list.append(g);
        }
        const item = document.createElement("div");
        item.className = "ydd-option";
        item.id = `${this.id}-o${i}`;
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", String(i === this.select.selectedIndex));
        if (o.disabled) item.setAttribute("aria-disabled", "true");
        if (o.hidden) item.hidden = true;
        const text = document.createElement("span");
        text.className = "ydd-text";
        text.textContent = o.textContent;
        item.append(text);
        if (o.dataset.note) {
          const note = document.createElement("span");
          note.className = "ydd-note";
          note.textContent = o.dataset.note;
          item.append(note);
        }
        item.addEventListener("mousedown", e => e.preventDefault()); // keep focus on the button
        item.addEventListener("click", () => { if (!o.disabled) { this.choose(i); this.close(); this.button.focus(); } });
        item.addEventListener("mousemove", () => this.highlight(i, false));
        this.list.append(item);
      });
      this.highlight(this.active, true);
    }
    place() {
      const r = this.button.getBoundingClientRect();
      const below = window.innerHeight - r.bottom - 8;
      const above = r.top - 8;
      const width = Math.max(r.width, 180);
      this.list.style.minWidth = width + "px";
      this.list.style.maxWidth = Math.min(Math.max(width, 520), window.innerWidth - 16) + "px";
      this.list.style.left = Math.max(8, Math.min(r.left, window.innerWidth - Math.min(Math.max(width, 520), window.innerWidth - 16) - 8)) + "px";
      const want = Math.min(this.list.scrollHeight, 360);
      if (below >= Math.min(want, 200) || below >= above) {
        this.list.style.top = r.bottom + 2 + "px";
        this.list.style.bottom = "";
        this.list.style.maxHeight = Math.max(120, below) + "px";
      } else {
        this.list.style.top = "";
        this.list.style.bottom = window.innerHeight - r.top + 2 + "px";
        this.list.style.maxHeight = Math.max(120, above) + "px";
      }
    }
    open() {
      if (this.button.disabled || !this.list.hidden) return;
      closeAll(this);
      document.body.append(this.list);
      this.list.hidden = false;
      this.active = this.select.selectedIndex;
      this.renderList();
      this.place();
      this.button.setAttribute("aria-expanded", "true");
      this.wrap.classList.add("open");
      OPEN.add(this);
      this.scrollTo(this.active);
    }
    close() {
      if (this.list.hidden) return;
      this.list.hidden = true;
      this.list.remove();
      this.button.setAttribute("aria-expanded", "false");
      this.button.removeAttribute("aria-activedescendant");
      this.wrap.classList.remove("open");
      OPEN.delete(this);
    }
    highlight(i, scroll) {
      this.active = i;
      for (const item of this.list.querySelectorAll(".ydd-option")) item.classList.toggle("active", item.id === `${this.id}-o${i}`);
      if (i >= 0) this.button.setAttribute("aria-activedescendant", `${this.id}-o${i}`);
      if (scroll) this.scrollTo(i);
    }
    scrollTo(i) {
      const item = this.list.querySelector(`#${this.id}-o${i}`);
      if (item) item.scrollIntoView({ block: "nearest" });
    }
    choose(i) {
      const s = this.select;
      if (i < 0 || i >= s.options.length || s.options[i].disabled) return;
      if (s.selectedIndex !== i) {
        s.selectedIndex = i;
        s.dispatchEvent(new Event("input", { bubbles: true }));
        s.dispatchEvent(new Event("change", { bubbles: true }));
      }
      this.refresh();
    }
    step(from, dir) {
      const opts = this.options();
      for (let i = from + dir; i >= 0 && i < opts.length; i += dir) if (!opts[i].disabled && !opts[i].hidden) return i;
      return from;
    }
    key(e) {
      const open = !this.list.hidden;
      const last = this.options().length - 1;
      switch (e.key) {
        case "ArrowDown":
        case "ArrowUp": {
          e.preventDefault();
          const dir = e.key === "ArrowDown" ? 1 : -1;
          if (!open && e.altKey) { this.open(); return; }
          if (!open) { this.choose(this.step(this.select.selectedIndex, dir)); return; }
          this.highlight(this.step(this.active, dir), true);
          return;
        }
        case "Home":
        case "End":
          if (!open) return;
          e.preventDefault();
          this.highlight(e.key === "Home" ? this.step(-1, 1) : this.step(last + 1, -1), true);
          return;
        case "Enter":
        case " ":
          e.preventDefault();
          if (open) { this.choose(this.active); this.close(); } else this.open();
          return;
        case "Escape":
          if (open) { e.preventDefault(); e.stopPropagation(); this.close(); }
          return;
        case "Tab":
          this.close();
          return;
        default:
          if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
            // Type to jump to the first item that starts with what was typed.
            const now = Date.now();
            this.typed = (now - this.typedAt < 700 ? this.typed : "") + e.key.toLowerCase();
            this.typedAt = now;
            const i = this.options().findIndex(o => !o.disabled && o.textContent.trim().toLowerCase().startsWith(this.typed));
            if (i >= 0) { if (open) this.highlight(i, true); else this.choose(i); }
          }
      }
    }
  }

  function enhance(root) {
    if (root.tagName === "SELECT") { if (!root._ydd && !root.multiple) new Dropdown(root); return; }
    if (root.querySelectorAll) for (const s of root.querySelectorAll("select")) if (!s._ydd && !s.multiple) new Dropdown(s);
  }

  // Values set from code update the dropdown.
  const hook = (proto, prop, owner) => {
    const d = Object.getOwnPropertyDescriptor(proto, prop);
    if (!d || !d.set) return;
    Object.defineProperty(proto, prop, {
      ...d,
      set(v) {
        d.set.call(this, v);
        const s = owner(this);
        if (s && s._ydd) s._ydd.refresh();
      },
    });
  };
  hook(HTMLSelectElement.prototype, "value", x => x);
  hook(HTMLSelectElement.prototype, "selectedIndex", x => x);
  hook(HTMLSelectElement.prototype, "disabled", x => x);
  hook(HTMLOptionElement.prototype, "selected", o => o.parentElement && (o.parentElement.tagName === "SELECT" ? o.parentElement : o.parentElement.parentElement));
  hook(HTMLElement.prototype, "hidden", x => (x.tagName === "SELECT" ? x : null));

  document.addEventListener("mousedown", e => {
    for (const d of [...OPEN]) if (!d.wrap.contains(e.target) && !d.list.contains(e.target)) d.close();
  });
  window.addEventListener("resize", () => closeAll());
  document.addEventListener("scroll", e => {
    for (const d of [...OPEN]) if (!d.list.contains(e.target)) d.place();
  }, true);

  const start = () => {
    enhance(document.body);
    new MutationObserver(records => {
      for (const r of records) for (const n of r.addedNodes) if (n.nodeType === 1) enhance(n);
    }).observe(document.body, { childList: true, subtree: true });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
  window.YendoryDropdown = { enhance, closeAll };
})();
