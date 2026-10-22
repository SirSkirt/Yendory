"use strict";
// Network growth (docs/GROWTH.md): the open model's growth setting, its original and current
// size, the growth steps the Engine offers for its architecture (each keeps what the network
// computes), what cannot be grown and why, and the growth history with rollback.

const GROWTH_MODES = {
  off: "Off: the network keeps its size",
  manual: "Manual: you choose when and how it grows",
  automatic: "Automatic: Studio may grow it when training shows it needs capacity (within your limits)",
};

function pctMore(after, before) {
  return `+${(100 * (after - before) / Math.max(1, before)).toFixed(0)}%`;
}

// Training memory a growth step adds: model state, and activations for the Reading settings'
// batch (rows x window tokens).
function growthMemory(g, o) {
  const rows = Number($("readBatch")?.value) || 16;
  const window = Number($("readWindow")?.value) || 128;
  const act = (floats) => 4 * floats * rows * window;
  const added = o.stateBytesAdded + act(o.activation_floats_per_token) - act(g.activationFloatsPerToken);
  return { added, total: o.stateBytes + act(o.activation_floats_per_token), rows, window };
}

async function renderGrowth() {
  const box = $("growthPanel");
  if (!box) return;
  if (!active) { box.replaceChildren(); return; }
  let g;
  try { g = await invoke("growth_view"); } catch (e) { box.replaceChildren(el("p", { class: "note" }, String(e))); return; }
  const mode = el("select", { id: "growthMode", "aria-label": "Growth" },
    ...Object.entries(GROWTH_MODES).map(([v, t]) => el("option", { value: v, selected: g.mode === v ? "" : null }, t)));
  mode.addEventListener("change", safe(async () => {
    await invoke("set_growth_mode", { mode: mode.value });
    await renderGrowth();
    say(`Growth for ${active.name}: ${mode.value}.`);
  }, "Growth setting"));
  const grown = g.currentParameters !== g.originalParameters;
  const rows = g.options.map(o => {
    const m = growthMemory(g, o);
    const button = el("button", { type: "button", class: "quiet", disabled: g.mode === "off" || teaching ? "" : null }, "Expand");
    button.addEventListener("click", safe(async () => {
      const ok = await askYes(`Grow ${active.name}: ${o.label}? ${fmt(o.parameters_before)} → ${fmt(o.parameters_after)} parameters (${pctMore(o.parameters_after, o.parameters_before)}). The model as it is now is kept as a recovery checkpoint; the grown network is checked to give the same outputs before it replaces the original. New capacity is used only after further training.`);
      if (!ok) return;
      button.disabled = true;
      await invoke("expand_model", { option: o.id });
      showActive((await invoke("status")).active);
      await refreshModels();
      say(`${active.name} grew to ${fmt(active.parameters)} parameters; its outputs were checked to be unchanged.`);
    }, "Expand"));
    return el("tr", {},
      el("td", {}, o.label),
      el("td", { class: "num" }, `${fmt(o.parameters_after)} (${pctMore(o.parameters_after, o.parameters_before)})`),
      el("td", {}, o.preservation === "exact" ? "exact" : el("span", { title: "A norm's epsilon is divided over more dimensions: outputs change by about one part in a million" }, "within rounding")),
      el("td", { class: "num", style: "white-space: nowrap", title: `Model state (weights, gradients, optimizer) plus activations for a batch of ${m.rows} × ${m.window} tokens (the Reading settings)` }, `+${bytes(m.added)} (total ${bytes(m.total)})`),
      el("td", {}, button));
  });
  const history = g.history.map((r, i) => el("tr", { class: r.rolledBack ? "nb-later" : "" },
    el("td", {}, new Date(r.time * 1000).toLocaleString()),
    el("td", {}, r.label + (r.automatic ? " (automatic)" : "")),
    el("td", { class: "num" }, `${fmt(r.parametersBefore)} → ${fmt(r.parametersAfter)}`),
    el("td", { class: "num", title: "Largest change of any output when the grown network was checked against the original" }, r.check.max_logit_change.toExponential(1)),
    el("td", {}, r.rolledBack ? "rolled back" : (i === g.history.map(x => !x.rolledBack).lastIndexOf(true) ? "latest" : ""))));
  const rollback = el("button", { type: "button", class: "quiet", disabled: !g.canRollBack || teaching ? "" : null }, "Roll back the latest growth…");
  rollback.addEventListener("click", safe(async () => {
    const ok = await askYes(`Restore ${active.name} to its size before the latest growth? Training done since that growth is not lost: the current model is kept as a checkpoint in the model's growth folder.`);
    if (!ok) return;
    await invoke("roll_back_growth");
    showActive((await invoke("status")).active);
    await refreshModels();
    say(`${active.name} restored to ${fmt(active.parameters)} parameters.`);
  }, "Roll back growth"));
  box.replaceChildren(
    el("h4", {}, "Network growth"),
    el("div", { class: "line-form" }, el("label", { for: "growthMode" }, "Growth"), mode),
    el("table", { class: "figures" },
      el("tr", {}, el("th", {}, "Original size"), el("td", {}, `${fmt(g.originalParameters)} parameters`),
        el("th", {}, "Current size"), el("td", {}, `${fmt(g.currentParameters)} parameters${grown ? ` (${pctMore(g.currentParameters, g.originalParameters)})` : ""}`))),
    g.mode === "off" ? el("p", { class: "note" }, "Choose Manual or Automatic to grow this model. Growth keeps every trained weight: the grown network gives the same outputs, and its new capacity is trained by further reading and lessons.") : "",
    g.mode === "automatic" ? el("p", { class: "note" }, "Automatic growth decides between training runs, from how training is going and within the memory limits; until it decides, you can still expand manually below.") : "",
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Growth step"), el("th", { class: "num" }, "Parameters after"), el("th", {}, "Outputs kept"), el("th", { class: "num" }, "Training memory"), el("th", {}, ""))),
      el("tbody", {}, ...rows)),
    g.notOffered.length ? el("p", { class: "note" }, "Not offered (would change what the network computes): ",
      ...g.notOffered.flatMap((n, i) => [i ? "; " : "", el("b", {}, n.part), ` — ${n.reason}`])) : "",
    g.history.length ? el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "When"), el("th", {}, "Growth"), el("th", { class: "num" }, "Parameters"), el("th", { class: "num" }, "Output change"), el("th", {}, ""))),
      el("tbody", {}, ...history)) : "",
    g.history.length ? el("div", { class: "actions" }, rollback) : "");
}
window.showGrowth = () => renderGrowth().catch(e => log("ERROR", e));
