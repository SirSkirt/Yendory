"use strict";
// Game skills of the open model (Nova Neuro since Engine 0.9.4, Bicameral since 0.9.5, linked to
// the core since 0.9.6): networks the model carries in its own file to play games, trained in
// Yendory Tools. Studio keeps them as they are and shows them: on the page bar, the model card,
// and the Brain page (the plate's skills region and one card per skill, drawn from its saved
// weights).

let skillViews = [];

const skillTime = t => !t ? "—" : new Date(t > 1e12 ? t : t * 1000).toLocaleString();
const SKILL_GROUPS = {
  vision: "Vision (convolutions, encoder)",
  memory: "Working memory",
  heads: "Controls (heads)",
  interconnect: "Goal input and link to the core",
};

function skillDot(level, detail, show) {
  const d = el("span", { class: "skill-dot", title: detail, tabindex: "0", role: "img", "aria-label": detail });
  d.style.opacity = (.15 + .85 * Math.min(1, Math.max(0, level))).toFixed(2);
  d.addEventListener("click", () => show(detail));
  d.addEventListener("focus", () => show(detail));
  return d;
}

// One layer of a skill's network: its name, size and a dot per unit.
function skillLayer(k, layer, title, size, show, extra = "") {
  const box = el("div", { class: "skill-layer" + (layer?.id === "goal" ? " goal" : "") }, el("h5", {}, title), el("div", { class: "size" }, size));
  if (layer) {
    // Brightness compares a unit with the largest of its own layer.
    const peak = Math.max(1e-9, ...layer.units);
    const dots = el("div", { class: "skill-dots" });
    layer.units.forEach((v, i) => dots.append(skillDot(v / peak,
      `${k.name} · ${title} · unit ${i + 1} · weight size ${v.toFixed(5)} · moved ${layer.changed[i].toFixed(5)} since the accepted weights${extra}`, show)));
    box.append(dots);
  }
  return box;
}

// The controls: one chip per key, button and mouse step, with its head's weight size.
function skillControls(k, show) {
  const box = el("div", { class: "skill-layer" }, el("h5", {}, "Controls"));
  const parts = [];
  if (k.keys.length) parts.push(`${k.keys.length} key${k.keys.length === 1 ? "" : "s"}`);
  if (k.buttons.length) parts.push(`${k.buttons.length} button${k.buttons.length === 1 ? "" : "s"}`);
  if (k.arch.mouseBins.length) parts.push(`mouse, ${k.arch.mouseBins.length} steps per axis`);
  box.append(el("div", { class: "size" }, parts.join(" · ") || "none"));
  const chips = el("div", { class: "skill-controls" });
  const layer = id => k.layers.find(l => l.id === id);
  const add = (id, names, what) => {
    const l = layer(id);
    if (!l) return;
    const peak = Math.max(1e-9, ...l.units);
    l.units.forEach((v, i) => {
      const name = names[i];
      const detail = `${k.name} · ${what} ${name} · weight size ${v.toFixed(5)} · moved ${l.changed[i].toFixed(5)} since the accepted weights`;
      const chip = el("span", { class: "skill-control", title: detail, tabindex: "0" }, skillDot(v / peak, detail, show), name);
      chip.addEventListener("click", () => show(detail));
      chips.append(chip);
    });
  };
  add("keys", k.keys, "key");
  add("buttons", k.buttons, "button");
  const bins = k.arch.mouseBins.map(b => (b > 0 ? "+" : "") + b);
  add("mouseX", bins.map(b => `x ${b}`), "mouse step");
  add("mouseY", bins.map(b => `y ${b}`), "mouse step");
  box.append(chips);
  return box;
}

function skillCard(k) {
  const detail = el("p", { class: "skill-detail", "aria-live": "polite" }, "Hover or tap a dot for its figures.");
  const show = text => { detail.textContent = text; };
  const a = k.arch;
  const layer = id => k.layers.find(l => l.id === id);
  const arrow = t => el("span", { class: "skill-arrow", "aria-hidden": "true" }, t);

  const net = el("div", { class: "skill-net" });
  net.append(skillLayer(k, null, "Pictures", `${a.width}×${a.height} ${a.channels === 1 ? "grey" : "colour"} · the last ${a.frames}`, show));
  a.convs.forEach((c, i) => {
    const [ch, h, w] = k.shapes[i + 1];
    net.append(arrow("→"), skillLayer(k, layer(`conv${i}`), `Convolution ${i + 1}`, `${c.kernel}×${c.kernel}, stride ${c.stride} → ${ch}×${h}×${w}`, show));
  });
  net.append(arrow("→"), skillLayer(k, layer("encoder"), "Encoder", `${fmt(a.hidden)} units`, show));
  const link = a.cortex;
  if (link) {
    // The link to the model's core: the core reads the encoder's sight (link in, and for
    // Bicameral the right hemisphere's input) and its state comes back as the goal input.
    const box = el("div", { class: "skill-layer goal" }, el("h5", {}, "Link to the core"),
      el("div", { class: "size" }, `core ${fmt(link.width)} wide${link.right ? ` · right input ${fmt(link.right)}` : ""} · ${k.linkProblem ? "does not fit this core" : "fits this core"}`));
    for (const [id, what] of [["linkIn", "in, to each of the core's units"], ["linkRight", "in, to the right hemisphere's input"], ["linkOut", "back, to each goal input"]]) {
      const l = layer(id);
      if (!l) continue;
      const peak = Math.max(1e-9, ...l.units);
      const dots = el("div", { class: "skill-dots" });
      l.units.forEach((v, i) => dots.append(skillDot(v / peak,
        `${k.name} · link ${what} · unit ${i + 1} · weight size ${v.toFixed(5)} · moved ${l.changed[i].toFixed(5)} since the accepted weights${id === "linkOut" && v === 0 ? " (the way back starts at zero: until it is trained the skill plays as without the core)" : ""}`, show)));
      box.append(el("div", { class: "size" }, `${what}`), dots);
    }
    net.append(arrow("⇄"), box);
  }
  if (a.goal) {
    net.append(arrow("+"), skillLayer(k, layer("goal"), "Goal input",
      `${a.goal} wide · ${link ? "the core's state, through the link" : "zeros until Command Mode gives a command"}`, show,
      " (a goal input's figure is its column in the encoder's goal weights)"));
  }
  if (a.memory) {
    const gate = layer("memoryGate");
    const box = skillLayer(k, layer("memory"), "Working memory", `${fmt(a.memory)} units (minGRU), carried across ticks`, show);
    if (gate) {
      // The update gate's weights, per unit, in the same box's figures.
      box.querySelectorAll(".skill-dot").forEach((d, i) => {
        const t = `${d.title} · update gate weight size ${gate.units[i].toFixed(5)}`;
        d.title = t;
        d.setAttribute("aria-label", t);
        d.addEventListener("click", () => show(t));
        d.addEventListener("focus", () => show(t));
      });
    }
    net.append(arrow("→"), box);
  }
  net.append(arrow("→"), skillControls(k, show));

  const lineage = k.lineage;
  const groups = el("table", { class: "figures" },
    el("tr", {}, el("th", {}, "Part"), el("th", {}, "Weights"), el("th", {}, "Weight size"), el("th", {}, "Moved since accepted")),
    ...k.groups.map(g => el("tr", {}, el("th", {}, SKILL_GROUPS[g.group] || g.group), el("td", {}, fmt(g.parameters)),
      el("td", {}, g.rms.toFixed(5)), el("td", {}, g.changed.toFixed(5)))));

  const notes = k.notes && (typeof k.notes !== "object" || Object.keys(k.notes).length)
    ? (() => {
      const text = JSON.stringify(k.notes, null, 2);
      const cut = 20000;
      return el("details", {}, el("summary", {}, "Trainer's notes (Yendory Tools)"),
        el("pre", {}, text.length > cut ? text.slice(0, cut) + `\n… (${fmt(text.length - cut)} more characters)` : text));
    })()
    : el("p", { class: "note" }, "No trainer's notes.");

  return el("article", { class: "skill-card" },
    el("h4", {}, k.name, el("span", { class: "game" }, k.game)),
    el("table", { class: "figures" },
      el("tr", {}, el("th", {}, "Training"), el("td", {}, `${fmt(k.steps)} steps; the accepted weights (the ones that play) were kept at step ${fmt(k.acceptedStep)}`)),
      el("tr", {}, el("th", {}, "Lineage"), el("td", {}, `generation ${lineage.generation} · ${lineage.method}${lineage.parent ? ` · from ${lineage.parent}` : " · a new network"}`)),
      el("tr", {}, el("th", {}, "Network"), el("td", {}, `${k.size} · ${fmt(k.parameters)} weights`)),
      el("tr", {}, el("th", {}, "Core"), k.arch.cortex
        ? el("td", {}, k.linkProblem
          ? el("span", { class: "link-problem" }, `linked, but ${k.linkProblem.replace(/^The skill's/, "its")}. It cannot play with this core until Yendory Tools links it again.`)
          : "linked: the model's core reads what this skill sees, and its state comes back as the goal input. Training the model in Studio also changes how it plays.")
        : el("td", {}, "not linked: the skill plays on its own")),
      el("tr", {}, el("th", {}, "Made"), el("td", {}, skillTime(k.created)))),
    net, detail, el("div", { class: "table-scroll" }, groups), notes);
}

// The open model's skills, wherever Studio shows them.
async function showSkills(meta) {
  const listed = meta?.skills || [];
  let views = [];
  if (listed.length) {
    try { views = await invoke("model_skills"); } catch (e) { log("warn", `Game skills: ${e}`); }
  }
  if (meta !== active) return; // another model was opened meanwhile
  skillViews = views;
  Brain.setSkills(views);
  $("skillSection").hidden = !views.length;
  $("skillList").replaceChildren(...views.map(skillCard));
  $("activeSkillsRow").hidden = !listed.length;
  $("activeSkills").textContent = listed.map(s => `${s.name} (${s.game}): ${fmt(s.steps)} steps, generation ${s.generation}`).join(" · ");
}
window.showSkills = showSkills;

// Page bar: "· 2 game skills" opens the Brain page at the skills.
function skillsLink(meta) {
  const n = (meta?.skills || []).length;
  if (!n) return null;
  const a = el("a", { href: "#", class: "skills-link", title: (meta.skills || []).map(s => s.name).join(", ") }, `${n} game skill${n === 1 ? "" : "s"}`);
  a.addEventListener("click", e => {
    e.preventDefault();
    go("brain");
    $("skillSection").scrollIntoView({ block: "start" });
  });
  return a;
}
window.skillsLink = skillsLink;
