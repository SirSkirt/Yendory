"use strict";
// Brain visualizer. The hemisphere shapes and dot layout are Neural Nova's original; colours come
// from the notebook palette (styles.css); the data comes from Engine measurements of a real
// forward pass.
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

  return { show, clear, stageLabel, headMean };
})();
