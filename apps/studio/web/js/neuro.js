"use strict";
// Nova Neuro brain builder (experimental): choose the networks and their sizes; the Engine
// computes the exact parameter count of every part, the memory needed and where it can train.

let neuroPresetsList = [];
let neuroDesign = null;     // the builder's current input (camelCase, as the backend expects)
let neuroPlanTimer = null;

const NEURO_FIELDS = {
  core: [
    ["width", "Width", 16, 1024, 16, "Transformer width (also the embedding width; embeddings are tied with the output layer)"],
    ["blocks", "Blocks", 2, 24, 1, "Transformer blocks"],
    ["heads", "Heads", 1, 32, 1, "Attention heads (the width must divide into them, with an even width per head)"],
    ["ff", "Feed-forward width", 16, 4096, 16, "SwiGLU feed-forward width of each block"],
    ["lower", "Lower blocks", 1, 23, 1, "Blocks run once before the modules; the rest run once per round"],
    ["rounds", "Computation rounds", 1, 4, 1, "Rounds of module exchange and upper blocks (2 is the baseline)"],
  ],
  bus: [["bus", "Bus width", 8, 2048, 8, "Shared width every network reads from and writes to"]],
  controller: [["controller", "Controller width", 8, 2048, 8, "Hidden width of the Executive Controller"]],
  memory: [
    ["layers", "Layers", 1, 4, 1, "minGRU layers"],
    ["width", "State width", 8, 4096, 8, "Working-memory state per layer"],
  ],
  associative: [
    ["patterns", "Patterns", 16, 16384, 16, "Learned patterns (each a key and a value)"],
    ["key", "Key width", 8, 1024, 8, "Width the query and pattern keys are compared in"],
  ],
};

function neuroInput() {
  if (!neuroDesign) throw new Error("Choose a Nova Neuro design first");
  return neuroDesign;
}

function numberField(obj, [key, label, min, max, step, hint], onChange) {
  const input = el("input", { type: "number", min, max, step, value: obj[key], class: "narrow-num", title: hint, "aria-label": label });
  input.addEventListener("input", () => { const v = Number(input.value); if (Number.isFinite(v)) { obj[key] = Math.round(v); onChange(); } });
  return el("label", { title: hint }, label, input);
}

function renderNeuroBuilder() {
  const box = $("neuroBuilder");
  const changed = () => { $("neuroPreset").value = "custom"; schedulePlan(); };
  const d = neuroDesign;
  const presetSel = el("select", { id: "neuroPreset", "aria-label": "Nova Neuro preset" },
    ...neuroPresetsList.map(p => el("option", { value: p.id }, `${p.name}: ${fmt(p.parameters)} parameters`)),
    el("option", { value: "custom" }, "Custom design"));
  presetSel.addEventListener("change", () => {
    const p = neuroPresetsList.find(x => x.id === presetSel.value);
    if (p) { neuroDesign = structuredClone(p.input); renderNeuroBuilder(); presetSel.value = p.id; }
  });
  const module = (title, role, key, fields, defaults) => {
    const on = d[key] != null;
    const check = el("input", { type: "checkbox", checked: on ? "" : null });
    check.addEventListener("change", () => { d[key] = check.checked ? (d[key] || structuredClone(defaults)) : null; renderNeuroBuilder(); changed(); });
    return el("fieldset", { class: "nb-module" + (on ? "" : " off") },
      el("legend", {}, el("label", { class: "check-inline" }, check, el("b", {}, title))),
      el("p", { class: "note" }, role),
      on ? el("div", { class: "nb-fields" }, ...fields.map(f => numberField(d[key], f, changed))) : el("p", { class: "note" }, "Not included."));
  };
  const fixed = (title, role, fields) => el("fieldset", { class: "nb-module" },
    el("legend", {}, el("b", {}, title)), el("p", { class: "note" }, role),
    el("div", { class: "nb-fields" }, ...fields.map(f => numberField(d, f, changed))));
  box.replaceChildren(
    el("p", { class: "note" }, "Nova Neuro is an experimental architecture. Every network below is a trainable neural network; how strongly each one is used is learned by the Executive Controller, never set by rules. It trains on the GPU (DirectX 12 or Vulkan) or the CPU, whichever the device setting chooses; replies are generated on the CPU. Bicameral models are not affected."),
    el("div", { class: "line-form" }, el("label", { for: "neuroPreset" }, "Design"), presetSel),
    el("div", { class: "nb-grid" },
      fixed("Language core (Transformer)", "Language, context and combining what the other networks send into a reply.", NEURO_FIELDS.core),
      module("Working memory (minGRU)", "Keeps and updates what matters in the text being read; its state belongs to the current conversation.", "memory", NEURO_FIELDS.memory, { layers: 2, width: 224 }),
      module("Associative memory (modern Hopfield)", "Learned patterns recalled by similarity; separate from Study Mode's text memory.", "associative", NEURO_FIELDS.associative, { patterns: 896, key: 64 }),
      fixed("Communication bus", "Carries information from the transformer to the networks and back, every round.", NEURO_FIELDS.bus),
      fixed("Executive Controller", "Learns how strongly to use each network for each token and round.", NEURO_FIELDS.controller)),
    el("div", { id: "neuroOffers" }),
    el("div", { id: "neuroPlan" }, el("p", { class: "note" }, "Calculating…")));
  schedulePlan();
}

function schedulePlan() {
  clearTimeout(neuroPlanTimer);
  neuroPlanTimer = setTimeout(() => showNeuroPlan().catch(e => log("ERROR", e)), 150);
}

async function showNeuroPlan() {
  if (!neuroDesign) return;
  let limit = null, context = 256;
  try { limit = chosenLimit(); context = chosenContext(); } catch { /* shown by the vocabulary note */ }
  const p = await invoke("neuro_plan", { input: neuroDesign, wordLimit: limit, context });
  const statusText = { available: "available", "next stage": "next stage", future: "future research" };
  $("neuroOffers").replaceChildren(
    el("h4", {}, "Networks"),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Network"), el("th", {}, "Role"), el("th", {}, "Status"), el("th", {}, "GPU training"))),
      el("tbody", {}, ...p.offers.map(o => el("tr", { class: o.status === "available" ? "" : "nb-later" },
        el("td", {}, o.name), el("td", { class: "note" }, o.role),
        el("td", {}, el("span", { class: "tag" + (o.status === "available" ? " ok" : "") }, statusText[o.status] || o.status)),
        el("td", { class: "note" }, o.gpu ? "yes" : "not yet"))))));
  const bar = share => el("span", { class: "nb-bar" }, el("i", { style: `width:${(100 * share).toFixed(1)}%` }));
  $("neuroPlan").replaceChildren(
    el("h4", {}, "Exact plan (computed by the Engine)"),
    p.errors.length ? el("div", { class: "conflict" }, ...p.errors.map(e => el("div", {}, e))) : "",
    p.parameters ? el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Part"), el("th", { class: "num" }, "Parameters"), el("th", { class: "num" }, "Share"), el("th", {}, ""))),
      el("tbody", {}, ...p.parts.map(x => el("tr", {}, el("td", {}, x.name[0].toUpperCase() + x.name.slice(1)), el("td", { class: "num" }, fmt(x.parameters)),
        el("td", { class: "num" }, (100 * x.share).toFixed(1) + "%"), el("td", {}, bar(x.share)))),
        el("tr", {}, el("td", {}, el("b", {}, "Total")), el("td", { class: "num" }, el("b", {}, fmt(p.parameters))), el("td", {}, ""), el("td", {}, "")))) : "",
    p.parameters ? el("p", { class: "note" }, `File ${bytes(p.fileBytes)} · memory ${[p.chatBytes, p.readingBytes, p.teachingBytes].map(bytes).join(" · ")} (talking · reading · teaching; estimated from the Engine's activation count, to be checked by measurement).`) : "",
    el("ul", { class: "note" }, ...p.hardware.map(h => el("li", {}, h))));
  $("createModel").disabled = teaching || p.errors.length > 0;
}

function setArchitecture() {
  const neuro = $("archChoice").value === "neuro";
  $("neuroBuilder").hidden = !neuro;
  $("sizeTable").hidden = neuro;
  $("measureContext").hidden = neuro;
  $("costNote").hidden = neuro;
  $("createModel").textContent = neuro ? "Create untrained Nova Neuro model" : "Create untrained model";
  if (neuro && !neuroDesign && neuroPresetsList.length) {
    neuroDesign = structuredClone(neuroPresetsList[0].input);
    renderNeuroBuilder();
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

// Brain page: Nova Neuro's measured activity for the inspected token (from the Engine's trace of
// a real forward pass). The plate above shows the language core's attention.
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
      el("td", { class: "note" }, name === "associative memory" ? r.patterns.map(([i, w]) => `#${i} ${pct(w)}`).join(" · ") : ""))));
  });
  box.replaceChildren(
    el("h3", {}, "Nova Neuro activity (measured)"),
    el("p", { class: "note" }, "For the inspected token: how strongly the Executive Controller opened each network's gate in each round, the size of each network's output before its gate, and, for associative memory, the patterns it recalled most. The plate above shows the language core's attention. A fuller Neuro visualizer comes in a later stage."),
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Round"), el("th", {}, "Network"), el("th", { class: "num" }, "Controller gate"), el("th", { class: "num" }, "Output size"), el("th", {}, "Recalled patterns"))),
      el("tbody", {}, ...rows)),
    el("table", { class: "figures" },
      el("tr", {}, el("th", {}, "Bus state size per round"), el("td", {}, n.rounds.map(r => r.bus_norm.toFixed(3)).join(" · ")),
        el("th", {}, "Write-back size per round"), el("td", {}, n.rounds.map(r => r.write_norm.toFixed(3)).join(" · "))),
      el("tr", {}, el("th", {}, "Working memory change per layer"), el("td", {}, n.memory_change.map(v => v.toFixed(3)).join(" · ") || "—"),
        el("th", {}, "Update gate per layer (0 keep, 1 overwrite)"), el("td", {}, n.memory_update.map(pct).join(" · ") || "—")),
      el("tr", {}, el("th", {}, "Output change with the networks' write-back cut"), el("td", {}, m.ablation.without_right_to_left.toFixed(3)),
        el("th", {}, "Output change with the transformer-to-bus path cut"), el("td", {}, m.ablation.without_left_to_right.toFixed(3)))));
};
