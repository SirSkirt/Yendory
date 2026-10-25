"use strict";
// Nova Neuro brain builder (experimental): a brain built from networks the user chooses, from a
// single network to all of them. Each network is a dropdown section with a switch, a plain
// explanation of what it does and its settings as Yendory dropdowns. The Engine computes the
// exact parameter count of every part, the memory needed and where it can train.

let neuroPresetsList = [];
let neuroDesign = null;     // the builder's current input (camelCase, as the backend expects)
let neuroPlanTimer = null;
let neuroPlan = null;       // the latest plan from the Engine
const neuroOpen = new Set(["transformer"]); // sections shown open

// Choices offered in each setting's dropdown: [value, note]. The current value is always added
// when it is not in the list (designs from presets or older versions).
const NEURO_CHOICES = {
  width: [[64], [96], [128, "Nova-2 size"], [160], [192, "4M size"], [224], [256, "8M size"], [320, "15M size"], [384], [448], [512], [640], [768], [1024]],
  heads: [[1], [2], [4], [6], [8], [10], [12], [16], [24], [32]],
  blocks: [[1], [2], [3], [4], [5], [6], [8], [10], [12], [16], [24]],
  ff: [[128], [192], [256], [288], [384], [448], [512], [640], [736], [768], [1024], [1536], [2048], [3072], [4096]],
  rounds: [[1, "one pass: fastest, no back-and-forth"], [2, "the baseline: the networks answer, the transformer uses it"], [3], [4, "most thinking per token, slowest"]],
  bus: [[32], [48], [64], [96], [128], [192], [256], [320], [384], [512], [768], [1024]],
  controller: [[16], [32], [48], [64], [96], [128], [192], [256], [384], [512]],
  memLayers: [[1, "one layer"], [2, "the baseline"], [3], [4]],
  memWidth: [[64], [96], [128], [160], [192], [224], [256], [320], [384], [448], [512], [640], [768], [912], [1024], [1536], [2048], [3072], [4096]],
  patterns: [[128], [256], [384], [512], [896], [1024], [1536], [2048], [2176], [2560], [3392], [4096], [6496], [8192], [16384]],
  key: [[16], [32], [48], [64], [96], [128], [192], [256], [512]],
  slots: [[4, "a few things at once"], [8], [12], [16, "about a dozen people, places or objects"], [24], [32], [48], [64, "many things at once; more memory while training"]],
  slotWidth: [[16], [24], [32], [48], [64], [96], [128], [192], [256]],
  hops: [[1, "direct facts: who is where"], [2, "one link: where the person with the apple is"], [3], [4, "long chains of links"]],
  experts: [[2], [3], [4], [6], [8], [12], [16]],
  expertWidth: [[32], [48], [64], [96], [128], [160], [192], [256], [384], [512], [768], [1024]],
  code: [[8], [16], [24], [32], [48], [64], [96], [128], [256]],
  hidden: [[48], [64], [96], [128], [160], [192], [224], [256], [384], [512], [768], [1024]],
};

// Each network: what it is, in plain words, and its settings.
const NEURO_SECTIONS = [
  {
    id: "transformer", name: "Language core (Transformer)", part: "transformer", toggle: "transformer",
    short: "Reads the words in context and writes the reply.",
    what: "The part most language models are made of. Every word looks back at the earlier words through attention and combines what it finds. Lower blocks run once per word; upper blocks run again in every round, after the other networks have answered.",
    good: "Language, grammar, combining everything into a reply. Switch it off to see what the other networks can do on their own.",
    fields: [
      ["heads", "Attention heads", "heads", "How many different things each word can look for at once. The main width must divide by it, with an even number per head."],
      ["blocks", "Blocks", "blocks", "Layers of attention and feed-forward. More blocks: deeper understanding, slower."],
      ["lower", "Lower blocks", null, "Blocks that run once, before the other networks. The rest are upper blocks and run every round."],
      ["ff", "Feed-forward width", "ff", "The size of each block's thinking layer. Larger stores more knowledge."],
    ],
  },
  {
    id: "memory", name: "Working memory (minGRU)", part: "working memory", toggle: "memory",
    short: "Keeps track of what matters in the text being read.",
    what: "A recurrent network: as each word is read, it decides how much of its state to keep and how much to overwrite. Its state belongs to the current conversation and starts empty each time.",
    good: "Following a story or conversation as it goes; it was the network Neuro-Core relied on most in the review.",
    defaults: { layers: 2, width: 224 },
    fields: [["layers", "Layers", "memLayers", "Stacked memory layers."], ["width", "State width", "memWidth", "How much the memory can hold at once."]],
  },
  {
    id: "associative", name: "Associative memory (modern Hopfield)", part: "associative memory", toggle: "associative",
    short: "Recalls learned patterns that resemble the current situation.",
    what: "A set of patterns learned in training. Each round it compares the current state with every pattern and blends the most similar ones back in. With consolidation on, it compares using consolidation's short code. Separate from Study Mode's text memory.",
    good: "Recalling what usually goes together. In the review at 1.2M it made no measurable difference yet.",
    defaults: { patterns: 896, key: 64 },
    fields: [["patterns", "Patterns", "patterns", "How many patterns it can learn."], ["key", "Key width", "key", "How much detail each comparison uses."]],
  },
  {
    id: "relational", name: "Relational network (entity slots)", part: "relational network", toggle: "relational",
    short: "Keeps a slot for each thing being talked about and connects them.",
    what: "Each word writes itself into a few of the slots (learned, never assigned), so a slot gathers what has been read about one person, place or object, using only words already read. Each round the other networks ask the slots a question and can follow the answer to a second slot (hops).",
    good: "Who is where, who has what, and how that changes. Meant to address following new names through a story, which no model has managed yet.",
    defaults: { slots: 16, width: 32, hops: 2 },
    fields: [["slots", "Slots", "slots", "How many things it can keep apart."], ["width", "Slot width", "slotWidth", "How much it keeps about each one."], ["hops", "Hops", "hops", "How many links a question can follow."]],
  },
  {
    id: "experts", name: "Mixture of Experts", part: "experts", toggle: "experts",
    short: "Several small specialist networks, chosen by a learned router.",
    what: "A router looks at the current state and decides how much each expert contributes. What each expert specialises in is learned in training, never assigned.",
    good: "Different kinds of text or tasks (stories, maths, questions) handled by different specialists.",
    defaults: { count: 4, width: 48 },
    fields: [["count", "Experts", "experts", "How many specialists."], ["width", "Expert width", "expertWidth", "The size of each specialist."]],
  },
  {
    id: "consolidation", name: "Consolidation (autoencoder)", part: "consolidation", toggle: "consolidation",
    short: "Squeezes what is going on into a short code.",
    what: "Compresses the bus into a short code and expands it again, so only what matters most gets through. Associative memory uses the code to compare with its patterns.",
    good: "Summaries of the situation; more compact recall.",
    defaults: { code: 32, hidden: 96 },
    fields: [["code", "Code width", "code", "How short the summary is."], ["hidden", "Hidden width", "hidden", "The size of the compressor and expander."]],
  },
  {
    id: "communication", name: "Communication and control", part: null, toggle: null,
    short: "How the networks share information and take turns.",
    what: "The main width is the width of the word embeddings and of the stream every network reads and writes. The bus carries information between the networks in both directions, and the Executive Controller learns how strongly to use each network for each word and round. Rounds are how many times the networks and the transformer go back and forth per word.",
    good: "Always present when the brain has more than the transformer.",
    fields: [
      ["width", "Main width", "width", "Width of the word embeddings and the main stream (the transformer's width)."],
      ["rounds", "Rounds", "rounds", "Back-and-forth passes per word."],
      ["bus", "Bus width", "bus", "Shared width every network reads from and writes to."],
      ["controller", "Controller width", "controller", "Size of the Executive Controller."],
    ],
  },
];

const NEURO_FUTURE = [
  ["Classic GRU, LSTM, State Space Model", "alternative working memories, for comparison"],
  ["CNN", "images, audio, spatial input"],
  ["Liquid network", "adaptive timing"],
  ["Spiking network", "event-driven processing"],
  ["Diffusion network", "generating images or audio"],
];

function neuroInput() {
  if (!neuroDesign) throw new Error("Choose a Nova Neuro design first");
  return neuroDesign;
}

// Defaults for a network switched on: the "every network" preset nearest in width.
function neuroDefaults(key, fallback) {
  const full = neuroPresetsList.filter(p => p.id.startsWith("neuro-full") && p.input[key]);
  full.sort((a, b) => Math.abs(a.input.width - neuroDesign.width) - Math.abs(b.input.width - neuroDesign.width));
  return structuredClone(full[0]?.input[key] || fallback);
}

// A Yendory dropdown (a <select> enhanced by dropdown.js) for one setting.
function choiceField(obj, key, label, list, hint, onChange, values) {
  const options = (values || NEURO_CHOICES[list]).map(([v, note]) => [v, note]);
  if (!options.some(([v]) => v === obj[key])) options.push([obj[key], "from this design"]);
  options.sort((a, b) => a[0] - b[0]);
  const sel = el("select", { "aria-label": label, title: hint, "data-key": `${list || key}:${key}` },
    ...options.map(([v, note]) => el("option", { value: v, selected: v === obj[key] ? "" : null, "data-note": note || null }, fmt(v))));
  sel.addEventListener("change", () => { obj[key] = Number(sel.value); onChange(); });
  return el("label", { class: "nb-field", title: hint }, el("span", {}, label), sel, el("span", { class: "nb-hint" }, hint));
}

function partParams(part) {
  if (!neuroPlan || !part) return null;
  return neuroPlan.parts.find(x => x.name === part)?.parameters ?? null;
}

function renderNeuroBuilder() {
  const box = $("neuroBuilder");
  // Keep focus where it was (a redraw replaces every control).
  const focused = document.activeElement;
  const focusKey = focused && box.contains(focused)
    ? (focused.closest(".ydd")?.previousElementSibling?.dataset.key && "sel:" + focused.closest(".ydd").previousElementSibling.dataset.key)
      || (focused.dataset.focus && "btn:" + focused.dataset.focus)
    : null;
  queueMicrotask(() => setTimeout(() => {
    if (!focusKey) return;
    const [kind, key] = [focusKey.slice(0, 4), focusKey.slice(4)];
    const target = kind === "sel:" ? box.querySelector(`select[data-key="${CSS.escape(key)}"]`)?._ydd?.button
      : box.querySelector(`[data-focus="${CSS.escape(key)}"]`);
    target?.focus();
  }, 0));
  const d = neuroDesign;
  if (d.transformer == null) d.transformer = true;
  const changed = () => { d._preset = "custom"; renderNeuroBuilder(); schedulePlan(); };
  const presetSel = el("select", { id: "neuroPreset", "aria-label": "Nova Neuro design" },
    ...neuroPresetsList.map(p => el("option", { value: p.id, selected: d._preset === p.id ? "" : null, "data-note": presetNote(p) }, `${p.name}: ${fmt(p.parameters)} parameters`)),
    el("option", { value: "custom", selected: d._preset === "custom" ? "" : null, "data-note": "your own combination of networks and sizes" }, "Custom design"));
  presetSel.addEventListener("change", () => {
    const p = neuroPresetsList.find(x => x.id === presetSel.value);
    if (p) { neuroDesign = { ...structuredClone(p.input), _preset: p.id }; renderNeuroBuilder(); schedulePlan(); }
  });
  const on = sec => sec.toggle == null || (sec.toggle === "transformer" ? d.transformer : d[sec.toggle] != null);
  const sections = NEURO_SECTIONS.map(sec => {
    const isOn = on(sec);
    const open = neuroOpen.has(sec.id);
    const params = partParams(sec.part);
    const head = el("button", { type: "button", class: "nb-head", "aria-expanded": String(open), "aria-controls": "nb-body-" + sec.id, "data-focus": "head-" + sec.id },
      el("span", { class: "nb-chevron", "aria-hidden": "true" }),
      el("span", { class: "nb-title" }, el("b", {}, sec.name), el("span", { class: "nb-short" }, sec.short)),
      el("span", { class: "nb-count num" }, sec.toggle == null ? "" : isOn ? (params != null ? fmt(params) : "…") : "off"));
    head.addEventListener("click", () => { if (open) neuroOpen.delete(sec.id); else neuroOpen.add(sec.id); renderNeuroBuilder(); });
    let toggle = "";
    if (sec.toggle) {
      toggle = el("button", { type: "button", role: "switch", class: "nb-switch", "aria-checked": String(isOn), "aria-label": `${sec.name}: ${isOn ? "on" : "off"}`, "data-focus": "switch-" + sec.id }, el("i", {}));
      toggle.addEventListener("click", () => {
        if (sec.toggle === "transformer") d.transformer = !d.transformer;
        else d[sec.toggle] = isOn ? null : neuroDefaults(sec.toggle, sec.defaults);
        if (!isOn) neuroOpen.add(sec.id);
        changed();
      });
    }
    const target = sec.toggle && sec.toggle !== "transformer" ? d[sec.toggle] : d;
    const fields = !isOn ? []
      : sec.fields.map(([key, label, list, hint]) => {
        if (key === "lower") {
          const values = Array.from({ length: Math.max(1, d.blocks) }, (_, k) => [k, k === 0 ? "the networks read the word embeddings" : null]);
          if (d.lower >= d.blocks) d.lower = Math.max(0, d.blocks - 1);
          return choiceField(d, "lower", label, null, hint, changed, values);
        }
        return choiceField(target, key, label, list, hint, changed);
      });
    return el("section", { class: "nb-section" + (isOn ? "" : " off") + (open ? " open" : "") },
      el("div", { class: "nb-row" }, toggle || el("span", { class: "nb-switch-space" }), head),
      el("div", { class: "nb-body", id: "nb-body-" + sec.id, hidden: open ? null : "" },
        el("p", {}, sec.what),
        el("p", { class: "note" }, el("b", {}, "Good for: "), sec.good),
        isOn ? el("div", { class: "nb-fields" }, ...fields) : el("p", { class: "note" }, "Switched off: not part of this brain. Use the switch to add it.")));
  });
  const networks = [d.transformer && "language core", d.memory && "working memory", d.associative && "associative memory", d.relational && "relational network", d.experts && "experts", d.consolidation && "consolidation"].filter(Boolean);
  const futureOpen = neuroOpen.has("future");
  const futureHead = el("button", { type: "button", class: "nb-head", "aria-expanded": String(futureOpen) },
    el("span", { class: "nb-chevron", "aria-hidden": "true" }),
    el("span", { class: "nb-title" }, el("b", {}, "Future networks"), el("span", { class: "nb-short" }, "Not available yet")), el("span", {}));
  futureHead.addEventListener("click", () => { if (futureOpen) neuroOpen.delete("future"); else neuroOpen.add("future"); renderNeuroBuilder(); });
  box.replaceChildren(
    el("p", { class: "note" }, "Nova Neuro is an experimental architecture: a brain made of networks you choose, from a single one to all of them. Every network is trained, and how strongly each is used is learned by the Executive Controller, never set by rules. It trains on the GPU (DirectX 12 or Vulkan) or the CPU; replies are generated on the CPU. Bicameral models are not affected."),
    el("div", { class: "line-form" }, el("label", { for: "neuroPreset" }, "Design"), presetSel),
    el("p", { class: "nb-summary" }, networks.length ? `This brain: ${networks.join(", ")}.` : "No network chosen yet: switch at least one on.",
      neuroPlan && neuroPlan.parameters ? ` ${fmt(neuroPlan.parameters)} parameters.` : ""),
    el("div", { class: "nb-sections" }, ...sections,
      el("section", { class: "nb-section off" + (futureOpen ? " open" : "") },
        el("div", { class: "nb-row" }, el("span", { class: "nb-switch-space" }), futureHead),
        el("div", { class: "nb-body", hidden: futureOpen ? null : "" },
          el("ul", {}, ...NEURO_FUTURE.map(([n, r]) => el("li", {}, el("b", {}, n), ": ", r)))))),
    el("div", { id: "neuroPlan" }, neuroPlanView()));
}

function presetNote(p) {
  const i = p.input;
  const parts = [i.transformer !== false && "transformer", i.memory && "working memory", i.associative && "associative", i.relational && "relational", i.experts && "experts", i.consolidation && "consolidation"].filter(Boolean);
  return parts.join(" + ");
}

function schedulePlan() {
  clearTimeout(neuroPlanTimer);
  neuroPlanTimer = setTimeout(() => showNeuroPlan().catch(e => log("ERROR", e)), 150);
}

async function showNeuroPlan() {
  if (!neuroDesign) return;
  let limit = null, context = 256;
  try { limit = chosenLimit(); context = chosenContext(); } catch { /* shown by the vocabulary note */ }
  const input = { ...neuroDesign };
  delete input._preset;
  neuroPlan = await invoke("neuro_plan", { input, wordLimit: limit, context });
  renderNeuroBuilder();
  $("createModel").disabled = teaching || neuroPlan.errors.length > 0;
}

function neuroPlanView() {
  const p = neuroPlan;
  if (!p) return el("p", { class: "note" }, "Calculating…");
  const bar = share => el("span", { class: "nb-bar" }, el("i", { style: `width:${(100 * share).toFixed(1)}%` }));
  return el("div", {},
    el("h4", {}, "Exact plan (computed by the Engine)"),
    p.errors.length ? el("div", { class: "conflict" }, ...p.errors.map(e => el("div", {}, e))) : "",
    p.parameters ? el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Part"), el("th", { class: "num" }, "Parameters"), el("th", { class: "num" }, "Share"), el("th", {}, ""))),
      el("tbody", {}, ...p.parts.map(x => el("tr", {}, el("td", {}, x.name[0].toUpperCase() + x.name.slice(1)), el("td", { class: "num" }, fmt(x.parameters)),
        el("td", { class: "num" }, (100 * x.share).toFixed(1) + "%"), el("td", {}, bar(x.share)))),
        el("tr", {}, el("td", {}, el("b", {}, "Total")), el("td", { class: "num" }, el("b", {}, fmt(p.parameters))), el("td", {}, ""), el("td", {}, "")))) : "",
    p.parameters ? el("p", { class: "note" }, `File ${bytes(p.fileBytes)} · memory ${[p.chatBytes, p.readingBytes, p.teachingBytes].map(bytes).join(" · ")} (talking · reading · teaching; estimated from the Engine's activation count, to be checked by measurement).`) : "",
    el("ul", { class: "note" }, ...p.hardware.map(h => el("li", {}, h))));
}

function setArchitecture() {
  const neuro = $("archChoice").value === "neuro";
  $("neuroBuilder").hidden = !neuro;
  $("sizeTable").hidden = neuro;
  $("measureContext").hidden = neuro;
  $("costNote").hidden = neuro;
  $("createModel").textContent = neuro ? "Create untrained Nova Neuro model" : "Create untrained model";
  if (neuro && !neuroDesign && neuroPresetsList.length) {
    neuroDesign = { ...structuredClone(neuroPresetsList[0].input), _preset: neuroPresetsList[0].id };
    renderNeuroBuilder();
    schedulePlan();
  } else if (neuro) {
    schedulePlan();
  } else {
    $("createModel").disabled = teaching;
  }
}

document.addEventListener("DOMContentLoaded", safe(async () => {
  try { neuroPresetsList = await invoke("neuro_presets"); } catch { return; }
  $("archChoice").addEventListener("change", setArchitecture);
  for (const id of ["vocabChoice", "vocabCustom", "contextChoice", "contextCustom"]) {
    $(id).addEventListener("change", () => { if ($("archChoice").value === "neuro") schedulePlan(); });
  }
}, "Nova Neuro builder"));

// How much the next-token distribution changes when each Nova Neuro system is switched off.
function effectsTable(e) {
  if (!e) return "";
  const row = (label, v, what) => v == null ? "" : el("tr", {}, el("th", {}, label), el("td", { class: "num" }, v.toFixed(3)), el("td", { class: "note" }, what));
  return el("div", {},
    el("h4", {}, "Each system's effect on this token (measured by switching it off)"),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Switched off"), el("th", { class: "num" }, "Output change"), el("th", {}, ""))),
      el("tbody", {},
        row("Working memory", e.without_memory, "its output set to zero"),
        row("Associative memory", e.without_associative, "its output set to zero"),
        row("Executive Controller", e.without_controller, "every gate fully open instead of its choices"),
        row("Relational network", e.without_relational, "its output set to zero"),
        row("Experts", e.without_experts, "their output set to zero"),
        row("Consolidation", e.without_consolidation, "its output set to zero (associative memory still reads its code)"),
        row("Extra computation rounds", e.one_round, "one round instead of the model's rounds"))),
    el("p", { class: "note" }, "0 means the reply's next-token probabilities are unchanged; 1 means completely different. A system with a small effect is not yet used much by the trained model."));
}

// Brain page: Nova Neuro's measured activity for the inspected token (from the Engine's trace of
// a real forward pass). The plate above shows the language core's attention.
// What a network did this round, in a few words (from the trace).
function detail(name, r) {
  const pct = v => (100 * v).toFixed(1) + "%";
  const top = (v, n) => v.map((w, i) => [i, w]).sort((a, b) => b[1] - a[1]).slice(0, n);
  if (name === "associative memory") return r.patterns.slice(0, 5).map(([i, w]) => `pattern #${i} ${pct(w)}`).join(" · ");
  if (name === "relational network" && r.slot_reads?.length) return r.slot_reads.map((p, h) => `hop ${h + 1}: ` + top(p, 2).map(([i, w]) => `slot ${i + 1} ${pct(w)}`).join(", ")).join(" · ");
  if (name === "experts" && r.experts?.length) return top(r.experts, 3).map(([i, w]) => `expert ${i + 1} ${pct(w)}`).join(" · ");
  if (name === "consolidation" && r.code?.length) return `code size ${Math.sqrt(r.code.reduce((a, x) => a + x * x, 0)).toFixed(3)}`;
  return "";
}

window.showNeuroActivity = m => {
  const box = $("neuroActivity");
  const n = m.trace.neuro;
  box.hidden = !n;
  if (!n) return;
  const pct = v => (100 * v).toFixed(1) + "%";
  const rows = [];
  n.rounds.forEach((r, ri) => {
    n.modules.forEach((name, k) => rows.push(el("tr", {},
      el("td", {}, `Round ${ri + 1}`), el("td", {}, name),
      el("td", { class: "num" }, pct(r.gates[k] ?? 0)), el("td", { class: "num" }, (r.outputs[k] ?? 0).toFixed(3)),
      el("td", { class: "note" }, detail(name, r)))));
  });
  box.replaceChildren(
    el("h3", {}, "Nova Neuro activity (measured)"),
    el("p", { class: "note" }, "For the inspected token: how strongly the Executive Controller opened each network's gate in each round, the size of each network's output before its gate, and what it did: the patterns associative memory recalled most, the slots each relational hop read, the experts the router chose. The plate above draws the same measurements: choose a block in “Attention shown” to see the networks in that block's round."),
    effectsTable(m.neuro_effects),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Round"), el("th", {}, "Network"), el("th", { class: "num" }, "Controller gate"), el("th", { class: "num" }, "Output size"), el("th", {}, "Recalled patterns"))),
      el("tbody", {}, ...rows)),
    n.slot_write?.length ? el("p", { class: "note" }, "Relational network: this token was written mostly into " + n.slot_write.map((w, i) => [i, w]).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([i, w]) => `slot ${i + 1} (${pct(w)})`).join(", ") + ".") : "",
    el("table", { class: "figures" },
      el("tr", {}, el("th", {}, "Bus state size per round"), el("td", {}, n.rounds.map(r => r.bus_norm.toFixed(3)).join(" · ")),
        el("th", {}, "Write-back size per round"), el("td", {}, n.rounds.map(r => r.write_norm.toFixed(3)).join(" · "))),
      el("tr", {}, el("th", {}, "Working memory change per layer"), el("td", {}, n.memory_change.map(v => v.toFixed(3)).join(" · ") || "—"),
        el("th", {}, "Update gate per layer (0 keep, 1 overwrite)"), el("td", {}, n.memory_update.map(pct).join(" · ") || "—")),
      el("tr", {}, el("th", {}, "Output change with the networks' write-back cut"), el("td", {}, m.ablation.without_right_to_left.toFixed(3)),
        el("th", {}, "Output change with the transformer-to-bus path cut"), el("td", {}, m.ablation.without_left_to_right.toFixed(3)))));
};
