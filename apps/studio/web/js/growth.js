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

// The Reading settings' batch: memory fits are estimated for it.
const growthBatch = () => ({ rows: Number($("readBatch")?.value) || 16, tokens: Number($("readWindow")?.value) || 128 });

// Memory a model of this size needs for training, how it would run, and whether that is within
// the Training memory limits.
const fitAmount = f => f.device === "cpu" ? `${bytes(f.ramBytes)} RAM` : `${bytes(f.gpuBytes)} GPU${f.ramBytes ? ` + ${bytes(f.ramBytes)} RAM` : ""}`;
function fitCell(f) {
  const amount = fitAmount(f);
  const verdict = f.fits === false ? el("span", { class: "bad" }, " · over the limits")
    : f.fits === null ? el("span", { class: "note" }, " · GPU size not given") : el("span", { class: "ok" }, " · fits");
  return el("td", { class: "num", style: "white-space: nowrap", title: f.plan }, amount, verdict);
}

const RULE_FIELDS = [
  ["maxParameters", "Largest size (parameters)", "no limit", v => v === "" ? null : Math.round(Number(v)), v => v ?? ""],
  ["minStepsBetween", "Training steps between growths", "", v => Math.round(Number(v)), v => v],
  ["plateauChecks", "Held-out checks that make a plateau", "", v => Math.round(Number(v)), v => v],
  ["minGain", "Plateau: improvement below (%)", "", v => Number(v) / 100, v => +(100 * v).toFixed(2)],
  ["maxGap", "Not memorising: held-out above training by at most (%)", "", v => Number(v) / 100, v => +(100 * v).toFixed(1)],
];

async function renderGrowth() {
  const box = $("growthPanel");
  if (!box) return;
  if (!active) { box.replaceChildren(); return; }
  let g;
  try { g = await invoke("growth_view", growthBatch()); } catch (e) { box.replaceChildren(el("p", { class: "note" }, String(e))); return; }
  const mode = el("select", { id: "growthMode", "aria-label": "Growth" },
    ...Object.entries(GROWTH_MODES).map(([v, t]) => el("option", { value: v, selected: g.mode === v ? "" : null }, t)));
  mode.addEventListener("change", safe(async () => {
    await invoke("set_growth_mode", { mode: mode.value, ...growthBatch() });
    await renderGrowth();
    say(`Growth for ${active.name}: ${mode.value}.`);
  }, "Growth setting"));
  const grown = g.currentParameters !== g.originalParameters;
  const rows = g.options.map(o => {
    const button = el("button", { type: "button", class: "quiet", disabled: g.mode === "off" || teaching || o.fit.fits === false ? "" : null,
      title: o.fit.fits === false ? "Over the Training memory limits for the Reading settings' batch" : null }, "Expand");
    button.addEventListener("click", safe(async () => {
      const ok = await askYes(`Grow ${active.name}: ${o.label}? ${fmt(o.parameters_before)} → ${fmt(o.parameters_after)} parameters (${pctMore(o.parameters_after, o.parameters_before)}). The model as it is now is kept as a recovery checkpoint; the grown network is checked to give the same outputs before it replaces the original. New capacity is used only after further training.`);
      if (!ok) return;
      button.disabled = true;
      await invoke("expand_model", { option: o.id, ...growthBatch() });
      showActive((await invoke("status")).active);
      await refreshModels();
      say(`${active.name} grew to ${fmt(active.parameters)} parameters; its outputs were checked to be unchanged.`);
    }, "Expand"));
    return el("tr", {},
      el("td", {}, o.label),
      el("td", { class: "num" }, `${fmt(o.parameters_after)} (${pctMore(o.parameters_after, o.parameters_before)})`),
      el("td", {}, o.preservation === "exact" ? "exact" : el("span", { title: "A norm's epsilon is divided over more dimensions: outputs change by about one part in a million" }, "within rounding")),
      fitCell(o.fit),
      g.mode === "automatic" ? el("td", {}, o.withinLimit ? "" : el("span", { class: "note" }, "over the size limit")) : "",
      el("td", {}, button));
  });
  const next = g.options.find(o => o.id === "balanced") || g.options[0];
  window.nextGrowth = next ? { label: next.label, fit: next.fit, parameters: next.parameters_after } : null;
  window.showMemoryNext?.();
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
    el("p", { class: "note" }, `Training it now, with the Reading settings' batches of ${g.rows} × ${g.tokens} tokens: `, fitAmount(g.fit), " — ", g.fit.plan, g.fit.fits === false ? el("b", { class: "bad" }, " (over the Training memory limits)") : ""),
    g.mode === "automatic" ? automaticRules(g) : "",
    el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "Growth step"), el("th", { class: "num" }, "Parameters after"), el("th", {}, "Outputs kept"), el("th", { class: "num" }, "Training memory"), g.mode === "automatic" ? el("th", {}, "") : "", el("th", {}, ""))),
      el("tbody", {}, ...rows)),
    g.notOffered.length ? el("p", { class: "note" }, "Not offered (would change what the network computes): ",
      ...g.notOffered.flatMap((n, i) => [i ? "; " : "", el("b", {}, n.part), ` — ${n.reason}`])) : "",
    g.history.length ? el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "When"), el("th", {}, "Growth"), el("th", { class: "num" }, "Parameters"), el("th", { class: "num" }, "Output change"), el("th", {}, ""))),
      el("tbody", {}, ...history)) : "",
    g.history.length ? el("div", { class: "actions" }, rollback) : "");
}
// Automatic growth: its limits and thresholds, and its latest decisions with their reasons.
function automaticRules(g) {
  const inputs = RULE_FIELDS.map(([key, label, placeholder, , show]) => [key, label,
    el("input", { id: `rule-${key}`, value: show(g.rules[key]), placeholder, size: 8, inputmode: "decimal" })]);
  const save = el("button", { type: "button", class: "quiet", disabled: teaching ? "" : null }, "Save limits");
  save.addEventListener("click", safe(async () => {
    const rules = {};
    for (const [key, , , parse] of RULE_FIELDS) {
      const v = parse($(`rule-${key}`).value.trim());
      if (v !== null && !Number.isFinite(v)) throw new Error("Check the growth limits: numbers only");
      rules[key] = v;
    }
    await invoke("set_growth_rules", { rules, ...growthBatch() });
    await renderGrowth();
    say("Automatic growth limits saved.");
  }, "Growth limits"));
  return el("div", {},
    el("p", { class: "note" }, "Automatic growth decides at the end of each reading run (nothing else is training then), from the run's held-out checks, never from a token count: it grows when the held-out loss has levelled off while the training loss is about as high (the network is short of capacity), and not when the training loss is well below it (the network is memorising). It takes the balanced step, or the largest step within the size limit and the Training memory limits. The grown network gives the same outputs, and reading keeps a checkpoint only when the held-out loss improves, so growth cannot make the kept model worse."),
    el("div", { class: "line-form" }, ...inputs.flatMap(([key, label, input]) => [el("label", { for: `rule-${key}` }, label), input]), save),
    g.decisions.length ? el("table", { class: "ledger compact" },
      el("thead", {}, el("tr", {}, el("th", {}, "When"), el("th", { class: "num" }, "Step"), el("th", {}, "Decision"))),
      el("tbody", {}, ...g.decisions.slice(-5).reverse().map(d => el("tr", {},
        el("td", {}, new Date(d.time * 1000).toLocaleString()), el("td", { class: "num" }, fmt(d.step)),
        el("td", {}, d.grown ? el("b", {}, "Grown. ") : "", d.reason))))) : el("p", { class: "note" }, "No decision yet: the first comes at the end of the next reading run."));
}

window.showGrowth = () => renderGrowth().catch(e => log("ERROR", e));
// After a reading run: automatic growth's decision, and the model's new size.
if (typeof onReadFinished === "function") {
  const beforeGrowth = onReadFinished;
  onReadFinished = f => {
    beforeGrowth(f);
    if (f.growth) {
      say(f.growth.reason);
      log("INFO", `Automatic growth: ${f.growth.reason}`);
      if (f.growth.grown && f.model) showActive(f.model);
    }
    window.showGrowth();
  };
}
