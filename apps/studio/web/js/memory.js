"use strict";
// Training memory (docs/GROWTH.md, section 3): what GPU training may use on the GPU and in
// system RAM, and what the running training really uses (shown here and under the Reading,
// Classroom and Planner progress cards while training runs).

const MEMORY_MODES = {
  gpu_only: "GPU only: everything stays on the GPU",
  automatic: "Automatic offloading: smaller micro-batches, then the optimizer state in system RAM, only when needed",
  manual: "Manual (advanced): you choose the micro-batch size and where the optimizer state lives",
};
const GB = 1 << 30;
const gbValue = b => b == null ? "" : String(Math.round(10 * b / GB) / 10);
const gbBytes = v => v.trim() === "" ? null : Math.round(Number(v) * GB);

function memoryReportLine(r, budget) {
  if (!r || !r.steps) return "";
  const parts = [r.strategy];
  parts.push(`GPU ${bytes(r.gpuBytes)}${budget ? ` of ${bytes(budget)} allowed` : ""}${r.gpuAllocatedBytes ? ` (driver: ${bytes(r.gpuAllocatedBytes)})` : ""}`);
  if (r.optimizerInRam) {
    const share = r.stepSeconds > 0 ? Math.round(100 * r.transferSeconds / r.stepSeconds) : 0;
    parts.push(`system RAM for offloading ${bytes(r.ramOffloadBytes)}`, `GPU ↔ RAM ${bytes(r.transferBytes)} moved (${share}% of the time)`);
  }
  if (r.stepSeconds > 0) parts.push(`${fmt(Math.round(r.tokens / r.stepSeconds))} tokens/s`);
  if (r.refits) parts.push(`${r.refits} step(s) refitted`);
  return parts.join(" · ");
}

function showMemoryLive(v) {
  const r = v?.report;
  const text = teaching && r?.steps ? "Memory: " + memoryReportLine(r, v.gpuBudget) : "";
  const warn = teaching ? (v?.warnings ?? []).join(" ") : "";
  for (const p of document.querySelectorAll("[data-memory-live]")) {
    p.hidden = !text;
    p.replaceChildren(text, warn ? el("span", { class: "bad" }, " " + warn) : "", r?.note ? el("span", {}, ` (${r.note})`) : "");
  }
}

async function renderMemory() {
  const box = $("memoryPanel");
  if (!box) return;
  let v;
  try { v = await invoke("training_memory_view"); } catch (e) { box.replaceChildren(el("p", { class: "note" }, String(e))); return; }
  const s = v.settings;
  const input = (id, value, attrs = {}) => el("input", { id, value, size: 6, inputmode: "decimal", ...attrs });
  const check = (id, on) => el("input", { id, type: "checkbox", checked: on ? "" : null });
  const mode = el("select", { id: "memMode" }, ...Object.entries(MEMORY_MODES).map(([k, t]) => el("option", { value: k, selected: s.mode === k ? "" : null }, t)));
  const manual = el("div", { class: "line-form", hidden: s.mode === "manual" ? null : "" },
    el("label", { for: "memMicro" }, "Rows per micro-batch"), input("memMicro", s.microRows ?? "", { placeholder: "whole batch", inputmode: "numeric" }),
    el("label", {}, check("memOptRam", s.optimizerInRam), " Optimizer state in system RAM"));
  mode.addEventListener("change", () => { manual.hidden = mode.value !== "manual"; });
  const save = el("button", { type: "button", disabled: teaching ? "" : null }, "Save memory settings");
  save.addEventListener("click", safe(async () => {
    const micro = $("memMicro").value.trim();
    const settings = {
      mode: mode.value,
      gpuMemoryBytes: gbBytes($("memGpu").value),
      gpuMarginPercent: Number($("memMargin").value),
      optimize: $("memOptimize").checked,
      ramLimitBytes: gbBytes($("memRamLimit").value),
      ramReserveBytes: gbBytes($("memReserve").value) ?? 0,
      warnSlowdown: $("memWarn").checked,
      microRows: micro === "" ? null : Number(micro),
      optimizerInRam: $("memOptRam").checked,
    };
    for (const [k, x] of Object.entries(settings)) if (typeof x === "number" && !Number.isFinite(x)) throw new Error(`Check the number for ${k}`);
    await invoke("set_training_memory", { settings });
    await renderMemory();
    say("Training memory settings saved; they apply from the next training run.");
  }, "Training memory"));
  const r = v.report;
  const gpuName = deviceInfo?.gpu || deviceInfo?.activeName || "your graphics card";
  box.replaceChildren(
    el("h4", {}, "Training memory"),
    el("div", { class: "line-form" }, el("label", { for: "memMode" }, "Memory"), mode),
    el("div", { class: "line-form" },
      el("label", { for: "memGpu" }, "GPU memory (GB)"), input("memGpu", gbValue(s.gpuMemoryBytes), { placeholder: "unknown", title: `The dedicated memory of ${gpuName}, e.g. 8 for an 8 GB card` }),
      el("label", { for: "memMargin" }, "Safety margin (%)"), input("memMargin", s.gpuMarginPercent, { inputmode: "numeric" }),
      el("label", {}, check("memOptimize", s.optimize), " Reduce memory automatically (micro-batches)")),
    el("div", { class: "line-form" },
      el("label", { for: "memRamLimit" }, "System RAM for offloading, at most (GB)"), input("memRamLimit", gbValue(s.ramLimitBytes), { placeholder: "half" }),
      el("label", { for: "memReserve" }, "Always keep free (GB)"), input("memReserve", gbValue(s.ramReserveBytes)),
      el("label", {}, check("memWarn", s.warnSlowdown), " Warn when offloading slows training")),
    manual,
    el("div", { class: "actions" }, save),
    el("p", { class: "note" },
      `System RAM: ${bytes(v.ramTotal)}, ${bytes(v.ramAvailable)} free; offloading may use ${bytes(v.ramAllowance)} now. `,
      v.gpuBudget ? `Training plans with and never allocates more than ${bytes(v.gpuBudget)} of GPU memory (the GPU's memory less the margin); a step that would need more is refused before it runs and retried with less. ` : el("b", {}, "Give the GPU's memory (e.g. 8 for an 8 GB card) so training stays within it: without it, training uses whole batches and adapts only when the GPU reports that memory ran out, and Windows may move memory into slower shared memory first. "),
      "Micro-batches give exactly the same training step; the optimizer state in RAM gives the same step with the GPU computing the gradient and the CPU applying it."),
    r?.steps ? el("p", { class: "note" }, "Last GPU training: " + memoryReportLine(r, v.gpuBudget), r.note ? ` (${r.note})` : "") : "",
    el("p", { class: "note", id: "memoryNext", hidden: "" }),
    ...v.warnings.map(w => el("p", { class: "note bad" }, w)));
  showMemoryNext();
}

// The next growth step's training memory (set by the growth panel for the open model).
function showMemoryNext() {
  const p = $("memoryNext");
  if (!p) return;
  const n = window.nextGrowth;
  p.hidden = !n;
  if (!n) return;
  const f = n.fit;
  p.replaceChildren(`Next growth of the open model (${n.label}, ${fmt(n.parameters)} parameters): about `,
    f.device === "cpu" ? `${bytes(f.ramBytes)} of RAM on the CPU` : `${bytes(f.gpuBytes)} of GPU memory${f.ramBytes ? ` and ${bytes(f.ramBytes)} of RAM` : ""} (${f.plan})`,
    f.fits === false ? el("span", { class: "bad" }, " — over these limits") : f.fits ? " — within these limits" : "", ".");
}
window.showMemoryNext = showMemoryNext;

let memoryTimer = null;
function pollMemory() {
  if (memoryTimer) return;
  memoryTimer = setInterval(async () => {
    if (!teaching) { showMemoryLive(null); return; }
    try { showMemoryLive(await invoke("training_memory_view")); } catch (_) { /* the next poll tries again */ }
  }, 3000);
}
window.showMemory = () => { renderMemory().catch(e => log("ERROR", e)); pollMemory(); };
document.addEventListener("DOMContentLoaded", () => window.showMemory());
