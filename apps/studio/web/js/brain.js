"use strict";
// Brain visualizer. Nova Bicameral: the hemisphere shapes and dot layout are Neural Nova's
// original. Nova Neuro: one brain with its networks as regions (only the model's networks drawn
// solid). Colours come from the notebook palette (styles.css); every value comes from Engine
// measurements of a real forward pass.
window.Brain = (() => {
  const svgNS = "http://www.w3.org/2000/svg";
  const $ = id => document.getElementById(id);
  const colour = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function neuron(x, y, magnitude, hue, detail) {
    const c = document.createElementNS(svgNS, "circle");
    const level = Math.min(1, Math.max(0, magnitude));
    c.setAttribute("cx", x.toFixed(1));
    c.setAttribute("cy", y.toFixed(1));
    c.setAttribute("r", (1.7 + level * 4.5).toFixed(1));
    c.setAttribute("fill", hue);
    c.setAttribute("opacity", (.16 + level * .82).toFixed(2));
    const title = document.createElementNS(svgNS, "title");
    title.textContent = detail;
    c.append(title);
    c.addEventListener("click", () => { $("brainDetail").textContent = detail; });
    return c;
  }
  function nodeLocation(i, n, centerX) {
    const angle = i * 2.399963229728653;
    const radius = Math.sqrt((i + .5) / (n + .3));
    return { x: centerX + Math.cos(angle) * 122 * radius, y: 207 + Math.sin(angle) * 119 * radius };
  }
  function stageLabel(s) {
    return s.round === 0 ? `Block ${s.block + 1} (lower stage)` : `Block ${s.block + 1}, thinking round ${s.round}`;
  }
  // Mean over heads of how much the newest token attends to each token.
  function headMean(stage) {
    const heads = stage.attention;
    const n = heads[0]?.length || 0;
    const out = new Array(n).fill(0);
    for (const h of heads) h.forEach((w, i) => { out[i] += w / heads.length; });
    return out;
  }

  function clear() {
    $("leftActivity").replaceChildren();
    $("rightActivity").replaceChildren();
    $("bridgeFibers").classList.remove("live");
    $("bridgeFibers").querySelectorAll("path").forEach(p => { p.style.opacity = ""; p.style.strokeWidth = ""; });
    for (const id of ["nBusDots", "nCoreDots", "nControlDots", "nAssocDots", "nMemoryDots"]) $(id)?.replaceChildren();
  }

  function show(m, stageIndex) {
    const left = $("leftActivity"), right = $("rightActivity");
    left.replaceChildren();
    right.replaceChildren();
    const stages = m.trace.stages;
    const stage = stages[Math.min(stageIndex, stages.length - 1)];
    const attention = stage ? headMean(stage) : [];
    const strongest = Math.max(1e-9, ...attention);
    attention.forEach((w, i) => {
      const p = nodeLocation(i, attention.length, 245);
      const detail = `${stageLabel(stage)} · attention to token ${i} (${m.tokens[i]}) · ${(w * 100).toFixed(3)}% (mean of ${stage.attention.length} heads)`;
      left.append(neuron(p.x, p.y, w / strongest, colour("--brain-left"), detail));
    });

    // Right hemisphere: a deterministic subsample of the actual activations.
    const layers = [{ name: "bottom layer", values: m.trace.right_trunk, max: 40 }];
    m.trace.right_upper.forEach((z, r) => layers.push({ name: `upper layer, round ${r + 1}`, values: z, max: 52 }));
    const dots = [];
    layers.forEach(layer => {
      const peak = Math.max(1e-7, ...layer.values.map(Math.abs));
      const stride = Math.max(1, Math.ceil(layer.values.length / layer.max));
      for (let i = 0; i < layer.values.length; i += stride) {
        dots.push({ layer: layer.name, index: i, value: layer.values[i], level: Math.abs(layer.values[i]) / peak });
      }
    });
    dots.forEach((d, i) => {
      const p = nodeLocation(i, dots.length, 555);
      right.append(neuron(p.x, p.y, d.level, colour("--brain-right"), `Dense pathway · ${d.layer} · neuron ${d.index} · measured activation ${d.value.toFixed(6)}`));
    });

    // Interconnect fibres: gate openings of every right->left injection, in five groups.
    const gates = m.trace.gates.flatMap(g => g.gate);
    const maximum = Math.max(1e-8, ...gates.map(Math.abs));
    // The fibres turn "live" green only when cutting the right->left path measurably changes the output.
    const verified = m.ablation.without_right_to_left > 1e-5;
    $("bridgeFibers").classList.toggle("live", verified);
    $("bridgeFibers").querySelectorAll("path").forEach((path, i) => {
      const start = Math.floor(i * gates.length / 5), end = Math.floor((i + 1) * gates.length / 5);
      const mean = gates.slice(start, end).reduce((s, v) => s + Math.abs(v), 0) / Math.max(1, end - start);
      const intensity = Math.min(1, mean / maximum);
      path.style.opacity = (.15 + .85 * intensity).toFixed(2);
      path.style.strokeWidth = (1.4 + 2.2 * intensity).toFixed(1);
    });
    return { attention, stage };
  }

  // ---- Nova Neuro -------------------------------------------------------------------------
  // One brain with the model's networks as regions. Regions the model does not have are drawn
  // dashed and labelled as absent; every dot and fibre is a measured value.
  const NEURO_REGIONS = {
    core: { cx: 405, cy: 180, rx: 116, ry: 88 },
    controller: { cx: 190, cy: 176, rx: 66, ry: 66 },
    associative: { cx: 612, cy: 296, rx: 58, ry: 30 },
    memory: { cx: 380, cy: 318, rx: 118, ry: 32 },
    relational: { cx: 590, cy: 98, rx: 52, ry: 30 },
    experts: { cx: 668, cy: 186, rx: 46, ry: 50 },
    consolidation: { cx: 196, cy: 300, rx: 56, ry: 28 },
  };
  const NEURO_NAMES = {
    core: "Language core", memory: "Working memory", associative: "Associative memory",
    relational: "Relational network", experts: "Experts", consolidation: "Consolidation",
  };
  // Network keys (as in the design) and their names in the Engine's trace.
  const NEURO_MODULES = [["memory", "working memory"], ["associative", "associative memory"], ["relational", "relational network"], ["experts", "experts"], ["consolidation", "consolidation"]];
  function regionLocation(i, n, r) {
    const angle = i * 2.399963229728653;
    const radius = Math.sqrt((i + .5) / (n + .3)) * .9;
    return { x: r.cx + Math.cos(angle) * r.rx * radius, y: r.cy + Math.sin(angle) * r.ry * radius };
  }
  // At most `max` evenly spaced values of `values`, with their indices.
  function sample(values, max) {
    const stride = Math.max(1, Math.ceil(values.length / max));
    const out = [];
    for (let i = 0; i < values.length; i += stride) out.push([i, values[i]]);
    return out;
  }
  const neuroGroups = ["nBusDots", "nCoreDots", "nControlDots", "nAssocDots", "nMemoryDots", "nRelDots", "nExpertDots", "nCodeDots"];

  // Which networks the model has (from its design), drawn before anything is measured.
  function neuroSchematic(design) {
    const svg = $("neuroBrain");
    const has = { core: design?.blocks !== 0, controller: true };
    for (const [key] of NEURO_MODULES) has[key] = !!design?.[key];
    has.controller = NEURO_MODULES.some(([key]) => has[key]);
    svg.querySelectorAll("[data-region]").forEach(e => e.classList.toggle("absent", !has[e.dataset.region]));
    svg.querySelectorAll("[data-label]").forEach(e => {
      const k = e.dataset.label;
      e.classList.toggle("absent", !has[k]);
      e.textContent = has[k] ? NEURO_NAMES[k] : `${NEURO_NAMES[k]} (not in this brain)`;
    });
    svg.querySelectorAll("[data-fiber]").forEach(f => {
      const k = f.dataset.fiber.split("-")[1];
      f.classList.toggle("absent", !!k && !has[k]);
      f.style.opacity = ""; f.style.strokeWidth = "";
    });
    neuroGroups.forEach(id => $(id).replaceChildren());
    $("nBusRing").style.strokeWidth = "";
    return has;
  }

  // The measured activity of the inspected token at a stage: the language core's attention at
  // that stage, and the networks' activity in its round (round 1 for the lower blocks).
  function showNeuro(m, stageIndex, design) {
    const n = m.trace.neuro;
    const inferred = { blocks: m.trace.stages.length ? undefined : 0 };
    for (const [key, name] of NEURO_MODULES) inferred[key] = n.modules.includes(name) ? {} : null;
    const has = neuroSchematic(design || inferred);
    const stages = m.trace.stages;
    const stage = stages[Math.min(stageIndex, stages.length - 1)];
    const roundIndex = Math.max(0, Math.min(n.rounds.length - 1, (stage?.round || 1) - 1));
    const round = n.rounds[roundIndex];
    const roundName = `round ${roundIndex + 1} of ${n.rounds.length}`;
    const R = NEURO_REGIONS;

    // Language core: the newest token's attention (mean over heads).
    const attention = stage ? headMean(stage) : [];
    const strongest = Math.max(1e-9, ...attention);
    attention.forEach((w, i) => {
      const p = regionLocation(i, attention.length, R.core);
      $("nCoreDots").append(neuron(p.x, p.y, w / strongest, colour("--brain-left"),
        `Language core · ${stageLabel(stage)} · attention to token ${i} (${m.tokens[i]}) · ${(w * 100).toFixed(3)}% (mean of ${stage.attention.length} heads)`));
    });

    // Executive Controller: its hidden activations in this round.
    const ctl = round?.controller || [];
    const ctlPeak = Math.max(1e-7, ...ctl.map(Math.abs));
    sample(ctl, 44).forEach(([i, v], k, all) => {
      const p = regionLocation(k, all.length, R.controller);
      $("nControlDots").append(neuron(p.x, p.y, Math.abs(v) / ctlPeak, colour("--neuro-control"),
        `Executive Controller · ${roundName} · hidden unit ${i} · activation ${v.toFixed(6)}`));
    });

    // Working memory: every layer's state after this token.
    if (has.memory && n.memory_state?.length) {
      const units = n.memory_state.flatMap((layer, l) => layer.map((v, j) => ({ l, j, v })));
      const peak = Math.max(1e-7, ...units.map(u => Math.abs(u.v)));
      const shown = sample(units, 64).map(([, u]) => u);
      shown.forEach((u, k) => {
        const p = regionLocation(k, shown.length, R.memory);
        $("nMemoryDots").append(neuron(p.x, p.y, Math.abs(u.v) / peak, colour("--neuro-memory"),
          `Working memory · layer ${u.l + 1}, unit ${u.j} · state ${u.v.toFixed(6)} (after this token; the update gate opened ${(100 * (n.memory_update[u.l] ?? 0)).toFixed(1)}% on average)`));
      });
    }

    // Associative memory: the patterns recalled most strongly in this round.
    if (has.associative && round?.patterns?.length) {
      const top = round.patterns[0][1] || 1e-9;
      round.patterns.forEach(([i, w], k) => {
        const p = regionLocation(k, round.patterns.length, R.associative);
        $("nAssocDots").append(neuron(p.x, p.y, w / top, colour("--brain-right"),
          `Associative memory · ${roundName} · pattern #${i} recalled with weight ${(100 * w).toFixed(2)}%`));
      });
    }

    // Relational network: the slots read in this round (each hop's weights, summed), and the
    // slots this token was written into.
    if (has.relational && round?.slot_reads?.length) {
      const slots = round.slot_reads[0].length;
      const read = Array.from({ length: slots }, (_, i) => round.slot_reads.reduce((a, p) => a + p[i], 0));
      const peak = Math.max(1e-9, ...read);
      read.forEach((w, i) => {
        const p = regionLocation(i, slots, R.relational);
        const written = n.slot_write?.[i] ?? 0;
        $("nRelDots").append(neuron(p.x, p.y, w / peak, colour("--neuro-relational"),
          `Relational network · ${roundName} · slot ${i + 1} · read ${(100 * w / round.slot_reads.length).toFixed(1)}% on average over ${round.slot_reads.length} hop(s) · this token wrote ${(100 * written).toFixed(1)}% of itself here · slot size ${(n.slot_norms?.[i] ?? 0).toFixed(3)}`));
      });
    }

    // Experts: the router's weight for each expert in this round.
    if (has.experts && round?.experts?.length) {
      const peak = Math.max(1e-9, ...round.experts);
      round.experts.forEach((w, i) => {
        const p = regionLocation(i, round.experts.length, R.experts);
        $("nExpertDots").append(neuron(p.x, p.y, w / peak, colour("--neuro-experts"),
          `Experts · ${roundName} · expert ${i + 1} · router weight ${(100 * w).toFixed(1)}%`));
      });
    }

    // Consolidation: the compressed code in this round.
    if (has.consolidation && round?.code?.length) {
      const peak = Math.max(1e-7, ...round.code.map(Math.abs));
      const shown = sample(round.code, 40);
      shown.forEach(([i, v], k) => {
        const p = regionLocation(k, shown.length, R.consolidation);
        $("nCodeDots").append(neuron(p.x, p.y, Math.abs(v) / peak, colour("--neuro-code"),
          `Consolidation · ${roundName} · code value ${i} · ${v.toFixed(6)}`));
      });
    }

    // Communication bus: the state the networks read this round, as dots along the ring.
    const bus = round?.bus || [];
    const busPeak = Math.max(1e-7, ...bus.map(Math.abs));
    const busShown = sample(bus, 56);
    busShown.forEach(([i, v], k) => {
      const a = 2 * Math.PI * k / busShown.length - Math.PI / 2;
      const x = R.core.cx + Math.cos(a) * 134, y = R.core.cy + Math.sin(a) * 104;
      $("nBusDots").append(neuron(x, y, .25 + .75 * Math.abs(v) / busPeak, colour("--brain-bridge"),
        `Communication bus · ${roundName} · dimension ${i} · value ${v.toFixed(6)}`));
    });

    // Fibres: the controller's gate for each network, and each network's gated output.
    const gateOf = name => round?.gates?.[n.modules.indexOf(name)] ?? 0;
    const outOf = name => (round?.outputs?.[n.modules.indexOf(name)] ?? 0) * gateOf(name);
    const outPeak = Math.max(1e-9, ...n.modules.map(outOf));
    const set = (key, level, detail) => {
      const f = $("neuroBrain").querySelector(`[data-fiber="${key}"]`);
      if (!f || f.classList.contains("absent")) return;
      const v = Math.max(0, Math.min(1, level));
      f.style.opacity = (.15 + .85 * v).toFixed(2);
      f.style.strokeWidth = (1.4 + 4 * v).toFixed(1);
      f.replaceChildren(Object.assign(document.createElementNS(svgNS, "title"), { textContent: detail }));
    };
    for (const [key, name] of NEURO_MODULES) {
      if (!has[key]) continue;
      set(`gate-${key}`, gateOf(name), `Executive Controller → ${name} · gate ${(100 * gateOf(name)).toFixed(1)}% open (${roundName})`);
      set(`out-${key}`, outOf(name) / outPeak, `${name} → bus · gated output size ${outOf(name).toFixed(4)} (${roundName})`);
    }
    const write = m.trace.gates[roundIndex]?.gate || [];
    const writeMean = write.reduce((s, v) => s + v, 0) / Math.max(1, write.length);
    set("read", writeMean, `Bus ↔ language core · write-back gate ${(100 * writeMean).toFixed(1)}% open on average (${roundName}); bus state size ${(round?.bus_norm ?? 0).toFixed(3)}`);
    $("nBusRing").style.strokeWidth = (2 + 5 * writeMean).toFixed(1);
    return { attention, stage, roundIndex };
  }

  // Shows the plate of the open model's architecture.
  function setArchitecture(neuro, design) {
    // SVG elements have no `hidden` property: the attribute (and the stylesheet) hides them.
    $("hemisphereBrain").toggleAttribute("hidden", neuro);
    $("neuroBrain").toggleAttribute("hidden", !neuro);
    $("bicaLegend").hidden = neuro;
    $("neuroLegend").hidden = !neuro;
    if (neuro) neuroSchematic(design);
  }

  return { show, clear, stageLabel, headMean, showNeuro, setArchitecture, neuroSchematic };
})();
