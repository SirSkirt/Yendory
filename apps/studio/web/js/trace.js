"use strict";
// § 11 Brain, per-network trace: what each of Nova's networks did at every step of a reply.
// Everything shown comes from the Engine's trace (yendory_engine::model::inspect); this page
// only draws it. Nothing here names a concept or writes a sentence about what Nova thought.
(() => {
  const ROLE = {
    context: "Context text", memory: "Study Mode memory", user: "You (earlier)", nova: "Nova (earlier)",
    message: "Your message (answered)", reply: "Nova's reply",
  };
  const NET_COLOR = {
    transformer: "var(--brain-left)", memory: "var(--neuro-memory)", associative: "var(--brain-right)",
    relational: "var(--neuro-relational)", experts: "var(--neuro-experts)", consolidation: "var(--neuro-code)",
    controller: "var(--neuro-control)", bus: "var(--brain-bridge)", dense: "var(--brain-right)",
  };
  const MODULE_NET = {
    "working memory": "memory", "associative memory": "associative", "relational network": "relational",
    experts: "experts", consolidation: "consolidation",
  };
  const GROUP_COLOR = { message: "var(--margin)", earlier: "var(--accent)", reply: "var(--neuro-memory)", context: "var(--ink-soft)" };
  const GROUP_NAME = { message: "your message", earlier: "earlier turns", reply: "the reply so far", context: "context and memories" };

  let trace = null;          // the trace shown
  let origin = null;         // { kind: "question" | "replay" | "recorded" | "live", chat, index, name }
  let live = false;          // steps are still arriving
  let stepIdx = 0;
  let net = "all";
  let turns = [];
  let frame = 0;

  const f2 = v => (v ?? 0).toFixed(2);
  const f3 = v => (v ?? 0).toFixed(3);
  const p100 = v => (100 * (v ?? 0)).toFixed(v >= 0.995 || v < 0.0005 ? 0 : 1) + "%";
  const tag = kind => el("span", { class: "trace-tag " + kind }, kind);
  const roleOf = l => trace.lines[l]?.role || "context";
  const groupOf = l => {
    const r = roleOf(l);
    return r === "message" ? "message" : r === "reply" ? "reply" : (r === "user" || r === "nova") ? "earlier" : "context";
  };
  const tokenAt = p => trace.tokens[p];
  const lineName = l => `${ROLE[roleOf(l)] || roleOf(l)} · line ${l + 1}`;
  const steps = () => trace?.steps || [];
  const step = () => steps()[stepIdx];
  const hasNet = id => trace.networks.some(n => n.id === id);
  const netName = id => trace.networks.find(n => n.id === id)?.name || id;
  const moduleIndex = id => trace.modules.findIndex(m => MODULE_NET[m] === id);

  // ---- small drawing helpers -------------------------------------------------------------
  function bar(v, max, color, title) {
    const b = el("span", { class: "trace-bar", title: title || "" });
    const i = el("i");
    i.style.width = (100 * Math.max(0, Math.min(1, max > 0 ? v / max : 0))).toFixed(1) + "%";
    i.style.background = color;
    b.append(i);
    return b;
  }
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}) => {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };
  // A line chart over the steps; clicking picks a step.
  function chart(title, unit, series, { max = null, kind = "measured" } = {}) {
    const n = steps().length;
    const box = el("figure", { class: "trace-fig" });
    const head = el("figcaption", {}, title, " ", tag(kind), unit ? el("span", { class: "trace-unit" }, ` (${unit})`) : "");
    const W = 720, H = 150, L = 38, R = 8, T = 8, B = 20;
    const all = series.flatMap(s => s.values.filter(v => Number.isFinite(v)));
    const top = max ?? Math.max(1e-9, ...all) * 1.08;
    const x = i => L + (n <= 1 ? (W - L - R) / 2 : i * (W - L - R) / (n - 1));
    const y = v => T + (H - T - B) * (1 - Math.max(0, Math.min(1, v / top)));
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "trace-svg", role: "img", "aria-label": title });
    for (const f of [0, 0.5, 1]) {
      g.append(svg("line", { x1: L, x2: W - R, y1: y(top * f), y2: y(top * f), class: "trace-grid" }));
      const t = svg("text", { x: L - 4, y: y(top * f) + 4, "text-anchor": "end", class: "trace-axis" });
      t.textContent = top * f >= 10 ? (top * f).toFixed(0) : (top * f).toFixed(2);
      g.append(t);
    }
    for (const s of series) {
      const pts = s.values.map((v, i) => Number.isFinite(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : null).filter(Boolean);
      if (!pts.length) continue;
      const pl = svg("polyline", { points: pts.join(" "), fill: "none", "stroke-width": 1.8 });
      pl.setAttribute("style", `stroke:${s.color}`);
      if (s.dashed) pl.setAttribute("stroke-dasharray", "5 3");
      g.append(pl);
    }
    if (n) {
      g.append(svg("line", { x1: x(stepIdx), x2: x(stepIdx), y1: T, y2: H - B, class: "trace-cursor" }));
      const lab = svg("text", { x: x(stepIdx), y: H - 5, "text-anchor": "middle", class: "trace-axis" });
      lab.textContent = `step ${stepIdx + 1}`;
      g.append(lab);
    }
    g.addEventListener("click", ev => {
      const r = g.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width * W;
      pick(Math.round((px - L) / Math.max(1e-9, (W - L - R) / Math.max(1, n - 1))));
    });
    const legend = el("div", { class: "trace-legend" }, ...series.map(s => {
      const sw = el("i", { class: "trace-swatch" + (s.dashed ? " dashed" : "") });
      sw.style.borderColor = s.color;
      const v = s.values[stepIdx];
      return el("span", {}, sw, `${s.name}`, Number.isFinite(v) ? el("b", {}, ` ${s.fmt ? s.fmt(v) : f3(v)}`) : "");
    }));
    box.append(head, g, legend);
    return box;
  }
  // The conversation's tokens as a strip of cells, shaded by a per-token value.
  function strip(title, values, color, kind = "measured") {
    const n = trace.tokens.length;
    const g = svg("svg", { viewBox: `0 0 ${n} 10`, preserveAspectRatio: "none", class: "trace-strip", role: "img", "aria-label": title });
    const top = Math.max(1e-9, ...values.filter(Number.isFinite));
    values.forEach((v, i) => {
      if (!Number.isFinite(v) || v <= 0) return;
      const r = svg("rect", { x: i, y: 0, width: 1.02, height: 10 });
      r.setAttribute("style", `fill:${color};opacity:${(0.12 + 0.88 * v / top).toFixed(3)}`);
      const t = svg("title");
      t.textContent = `${tokenAt(i)?.text ?? ""} · ${lineName(tokenAt(i)?.line ?? 0)} · ${f3(v)}`;
      r.append(t);
      g.append(r);
    });
    for (const l of trace.lines.slice(1)) g.append(svg("line", { x1: l.start, x2: l.start, y1: 0, y2: 10, class: "trace-strip-line" }));
    const s = step();
    if (s) g.append(svg("rect", { x: s.position - 0.5, y: 0, width: 2, height: 10, class: "trace-strip-now" }));
    return el("figure", { class: "trace-fig" }, el("figcaption", {}, title, " ", tag(kind)), g,
      el("div", { class: "trace-legend" }, el("span", {}, "each cell is a token, left to right through the conversation; thin rules mark line breaks; the dark mark is this step")));
  }
  // The conversation, line by line, with one or more measured bars per line.
  function linesTable(cols, note) {
    const t = el("table", { class: "ledger compact trace-lines" },
      el("thead", {}, el("tr", {}, el("th", {}, "Line"), el("th", {}, "Text"), ...cols.map(c => el("th", {}, c.name, " ", tag(c.kind || "measured"))))));
    const body = el("tbody");
    const s = step();
    for (const l of trace.lines) {
      if (s && l.start >= s.position) break;
      const row = el("tr", { class: "role-" + l.role },
        el("td", {}, el("span", { class: "trace-role role-" + l.role }, ROLE[l.role] || l.role), el("div", { class: "trace-unit" }, `line ${l.index + 1}`)),
        el("td", { class: "trace-linetext" }, l.role === "reply" && s ? replySoFar(s.position) : l.text),
        ...cols.map(c => {
          const v = c.value(l.index);
          return el("td", { class: "num" }, v == null ? "—" : el("span", {}, bar(v, c.max, c.color, f3(v)), " ", c.fmt ? c.fmt(v) : f3(v)));
        }));
      body.append(row);
    }
    t.append(body);
    return el("div", { class: "trace-lines-wrap" }, note ? el("p", { class: "note" }, note) : "", t);
  }
  const replySoFar = pos => trace.tokens.slice(0, pos).filter(t => t.line === trace.tokens[pos - 1]?.line).map(t => t.text.replace("␠", " ").replace("↵", "")).join("");
  const table = rows => el("table", { class: "figures narrow trace-table" }, ...rows.filter(Boolean).map(([k, v, kind]) => el("tr", {}, el("th", {}, k, ...(kind ? [" ", tag(kind)] : [])), el("td", {}, v))));
  const tokChip = (pos, w) => {
    const t = tokenAt(pos);
    return el("span", { class: "tok trace-poschip role-" + (trace.lines[t?.line]?.role || ""), title: `position ${pos} · ${lineName(t?.line ?? 0)}` }, t ? t.text : "?", w != null ? el("small", {}, ` ${p100(w)}`) : "");
  };
  function influenceTable(filter) {
    const s = step();
    if (!s?.influence?.length) {
      return el("p", { class: "note" }, trace.influence?.length ? "—" : "Influence not measured for this trace. Measure it with the button above: the turn is read again once per network with that network switched off.");
    }
    const t = el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {},
      el("th", {}, "Switched off"), el("th", {}, "Change of the next-token distribution"), el("th", {}, `P("${s.chosen.token}") without it`), el("th", {}, "Its most likely token"))));
    const body = el("tbody");
    for (const i of s.influence.filter(i => !filter || filter(i.network))) {
      body.append(el("tr", {},
        el("td", {}, INFLUENCE_NAME[i.network] || i.network),
        el("td", { class: "num" }, bar(i.change, 1, "var(--accent)"), ` ${f3(i.change)}`),
        el("td", { class: "num" }, `${p100(i.chosen_probability)} (with it ${p100(s.chosen.probability)})`),
        el("td", {}, `${i.top.token} ${p100(i.top.probability)}`, i.flips ? el("span", { class: "warn" }, " · the top choice changes") : "")));
    }
    t.append(body);
    return el("div", {}, t, el("p", { class: "note" }, "Measured: the whole turn read again with that network switched off (the controller: every gate fully open; the bus: its write-back into the transformer, or the transformer's input to it, cut). Change is the total-variation distance, 0 = no difference, 1 = completely different."));
  }
  const INFLUENCE_NAME = {
    memory: "Working memory", associative: "Associative memory", relational: "Relational network", experts: "Experts",
    consolidation: "Consolidation", controller: "Executive Controller (gates forced open)", "bus write-back": "Bus → transformer write-back",
    "bus input": "Transformer → bus", dense: "Dense → transformer", "dense input": "Transformer → dense",
  };

  // ---- per-network measures ----------------------------------------------------------------
  // Attention mass per group of lines (mean over every block execution and head).
  function groupMass(s, stages = null) {
    const out = { message: 0, earlier: 0, reply: 0, context: 0 };
    const list = stages ? stages.flatMap(st => st.lines.map(([l, w]) => [l, w / stages.length])) : s.transformer.lines;
    for (const [l, w] of list) out[groupOf(l)] += w;
    return out;
  }
  const meanRounds = (s, f) => s.rounds.length ? s.rounds.reduce((a, r) => a + f(r), 0) / s.rounds.length : NaN;
  // Working memory: share of the contribution lengths from lines before the message.
  function memoryEarlierShare(layer) {
    const msg = trace.lines.find(l => l.role === "message")?.index ?? Infinity;
    const total = layer.lines.reduce((a, [, v]) => a + v, 0);
    const before = layer.lines.filter(([l]) => l < msg).reduce((a, [, v]) => a + v, 0);
    return total > 0 ? before / total : 0;
  }

  const VIEWS = {
    all: {
      charts() {
        const out = [];
        const series = [{ name: "P(chosen token)", values: steps().map(s => s.chosen.probability), color: "var(--ink)" }];
        trace.modules.forEach((m, k) => series.push({
          name: `gate · ${m}`, color: NET_COLOR[MODULE_NET[m]] || "var(--neuro-control)",
          values: steps().map(s => meanRounds(s, r => r.modules[k]?.gate ?? NaN)),
        }));
        out.push(chart("Chosen token's probability and the Executive Controller's gates (mean over rounds)", "0 to 1", series, { max: 1 }));
        if (trace.influence?.length) {
          out.push(chart("Influence: change of the next-token distribution with each network switched off", "total variation, 0 to 1",
            trace.influence.map((name, k) => ({ name: INFLUENCE_NAME[name] || name, color: NET_COLOR[name.split(" ")[0]] || "var(--accent)", dashed: name.includes("input"), values: steps().map(s => s.influence?.[k]?.change ?? NaN) })), { max: 1 }));
        }
        return out;
      },
      detail(s) {
        const g = groupMass(s);
        const rows = [
          ["Transformer · attention", `${p100(g.message)} on your message · ${p100(g.earlier)} on earlier turns · ${p100(g.reply)} on the reply so far · ${p100(g.context)} on context`, "measured"],
        ];
        s.memory.forEach((l, i) => rows.push([`Working memory · layer ${i + 1}`, `update gate ${f2(l.update)} · ${p100(l.overwrite)} of channels mostly overwritten · ${p100(memoryEarlierShare(l))} of contribution length from lines before your message`, "measured"]));
        const a = moduleIndex("associative");
        if (a >= 0) s.rounds.forEach(r => rows.push([`Associative memory · round ${r.round}`, `gate ${f2(r.modules[a]?.gate)} · strongest pattern ${p100(r.patterns[0]?.[1])} · entropy ${f2(r.retrieval_entropy)} bits`, "measured"]));
        s.rounds.forEach(r => rows.push([`Bus · round ${r.round}`, `write-back ${f3(r.injection_norm)} on a state of ${f3(r.state_norm)} (${p100(r.injection_norm / Math.max(1e-9, r.state_norm))})`, "measured"]));
        return [table(rows), el("h5", {}, "Influence"), influenceTable(), contextLines(s)];
      },
    },
    transformer: {
      charts() {
        return [
          chart("Where the attention went (mean over all blocks, rounds and heads)", "share of attention", Object.keys(GROUP_NAME).map(k => ({
            name: GROUP_NAME[k], color: GROUP_COLOR[k], values: steps().map(s => groupMass(s)[k]),
          })), { max: 1 }),
          chart("Uncertainty of the next token", "bits", [{ name: "entropy", color: "var(--brain-left)", values: steps().map(s => s.entropy) }]),
        ];
      },
      detail(s) {
        const out = [];
        out.push(el("h5", {}, "What the model would say at each stage ", tag("measured")),
          el("p", { class: "note" }, "The output layer applied to the state after the lower blocks and after each round; the last is the real output."),
          table(s.transformer.readouts.map(r => [`After ${r.stage}`, r.top.map(c => `${c.token} ${p100(c.probability)}`).join(" · ")])));
        const t = el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {},
          el("th", {}, "Block"), el("th", {}, "State length"), el("th", {}, "Block's change"), el("th", {}, "Head entropy (mean, bits)"),
          el("th", {}, "On message"), el("th", {}, "On earlier turns"), el("th", {}, "Mean distance back"), el("th", {}, "Strongest positions per head"))));
        const body = el("tbody");
        for (const st of s.transformer.stages) {
          const g = groupMass(s, [st]);
          const ent = st.heads.reduce((a, h) => a + h.entropy, 0) / Math.max(1, st.heads.length);
          body.append(el("tr", {},
            el("td", {}, st.round === 0 ? `${st.block + 1} (lower)` : `${st.block + 1}, round ${st.round}`),
            el("td", { class: "num" }, f2(st.state_norm)), el("td", { class: "num" }, f2(st.update_norm)), el("td", { class: "num" }, f2(ent)),
            el("td", { class: "num" }, bar(g.message, 1, GROUP_COLOR.message), ` ${p100(g.message)}`),
            el("td", { class: "num" }, bar(g.earlier, 1, GROUP_COLOR.earlier), ` ${p100(g.earlier)}`),
            el("td", { class: "num" }, `${f2(st.distance)} tokens`),
            el("td", {}, el("details", {}, el("summary", {}, `${st.heads.length} heads`),
              ...st.heads.map((h, k) => el("div", { class: "trace-head" }, el("b", {}, `head ${k + 1}`), ` ${f2(h.entropy)} bits: `, ...h.top.map(([p, w]) => tokChip(p, w))))))));
        }
        t.append(body);
        out.push(el("h5", {}, "Every block execution for this token ", tag("measured")), t,
          el("p", { class: "note" }, "A position in a later block holds what earlier blocks mixed into it, so attention to a position is attention to that position's state, not only to its word."));
        out.push(linesTable([{ name: "Attention (mean of all blocks and heads)", color: "var(--brain-left)", max: 1, fmt: p100, value: l => s.transformer.lines.find(([x]) => x === l)?.[1] ?? 0 }]));
        return out;
      },
    },
    memory: {
      charts() {
        const layers = steps()[0]?.memory.length || 0;
        const series1 = [], series2 = [];
        for (let i = 0; i < layers; i++) {
          series1.push({ name: `layer ${i + 1} · mean update gate`, color: i ? "var(--neuro-code)" : "var(--neuro-memory)", values: steps().map(s => s.memory[i]?.update) });
          series1.push({ name: `layer ${i + 1} · channels mostly overwritten`, dashed: true, color: i ? "var(--neuro-code)" : "var(--neuro-memory)", values: steps().map(s => s.memory[i]?.overwrite) });
          series2.push({ name: `layer ${i + 1}`, color: i ? "var(--neuro-code)" : "var(--neuro-memory)", fmt: p100, values: steps().map(s => s.memory[i] ? memoryEarlierShare(s.memory[i]) : NaN) });
        }
        return [
          chart("Update gate (0 keeps the state, 1 overwrites it)", "0 to 1", series1, { max: 1 }),
          chart("Share of the state's contributions written by lines before your message", "share of contribution lengths", series2, { max: 1, kind: "inferred" }),
        ];
      },
      detail(s) {
        const out = [table(s.memory.flatMap((l, i) => [
          [`Layer ${i + 1} · update gate (mean)`, f3(l.update), "measured"],
          [`Layer ${i + 1} · channels mostly overwritten`, p100(l.overwrite), "measured"],
          [`Layer ${i + 1} · change of the state / state length`, `${f3(l.change)} / ${f3(l.norm)}`, "measured"],
        ]))];
        out.push(linesTable(s.memory.map((l, i) => ({
          name: `Still in layer ${i + 1} (length)`, color: i ? "var(--neuro-code)" : "var(--neuro-memory)",
          max: Math.max(1e-9, ...l.lines.map(([, v]) => v)), value: x => l.lines.find(([k]) => k === x)?.[1] ?? 0,
        })), "The working memory's state split exactly into what each line wrote and what of it is left: m = Σ over tokens of z·candidate·Π(1 − z) of the later tokens. A short bar means that line's writing has been mostly overwritten since. The memory reads each token once, after the lower blocks; its output is the same in every round."));
        const layers = s.memory.length;
        for (let i = 0; i < layers; i++) {
          out.push(strip(`Layer ${i + 1}: update gate at every token of the conversation`, trace.profile.map(p => p.memory_update?.[i]), i ? "var(--neuro-code)" : "var(--neuro-memory)"));
        }
        out.push(el("h5", {}, "Influence"), influenceTable(n => n === "memory"));
        return out;
      },
    },
    associative: {
      charts() {
        const a = moduleIndex("associative");
        const rounds = steps()[0]?.rounds.length || 0;
        const s1 = [], s2 = [];
        for (let r = 0; r < rounds; r++) {
          s1.push({ name: `round ${r + 1} · strongest pattern's weight`, dashed: r > 0, color: "var(--brain-right)", values: steps().map(s => s.rounds[r]?.patterns[0]?.[1]) });
          s1.push({ name: `round ${r + 1} · controller gate`, dashed: r > 0, color: "var(--neuro-control)", values: steps().map(s => s.rounds[r]?.modules[a]?.gate) });
          s2.push({ name: `round ${r + 1}`, dashed: r > 0, color: "var(--brain-right)", values: steps().map(s => s.rounds[r]?.retrieval_entropy) });
        }
        return [chart("Retrieval: the strongest pattern's weight and the gate on the memory's output", "0 to 1", s1, { max: 1 }),
          chart("Retrieval entropy (0 = one pattern; higher = spread over many)", "bits", s2)];
      },
      detail(s) {
        const a = moduleIndex("associative");
        const out = [el("p", { class: "note" }, "This network's patterns are learned in training and fixed while Nova talks: it does not store anything said in this conversation. What changes is which patterns the current state recalls. Pattern numbers are only indices; they carry no meaning by themselves.")];
        for (const r of s.rounds) {
          const strongest = r.patterns.slice(0, 8);
          const t = el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, "Pattern"), el("th", {}, "Weight"), el("th", {}, "Also recalled most strongly at (this conversation) ", tag("measured")))));
          const body = el("tbody");
          for (const [p, w] of strongest) {
            const at = [];
            trace.profile.forEach((pr, pos) => {
              const hit = pr.patterns?.[r.round - 1]?.find(([q]) => q === p);
              if (hit && pos < s.position) at.push([pos, hit[1]]);
            });
            at.sort((x, y) => y[1] - x[1]);
            body.append(el("tr", {}, el("td", {}, `#${p}`), el("td", { class: "num" }, bar(w, 1, "var(--brain-right)"), ` ${p100(w)}`),
              el("td", {}, ...at.slice(0, 4).map(([pos, ww]) => tokChip(pos, ww)), at.length ? "" : "— (not among the strongest at any earlier token)")));
          }
          t.append(body);
          out.push(el("h5", {}, `Round ${r.round}`),
            table([["Controller gate on its output", f3(r.modules[a]?.gate), "measured"], ["Output length (before the gate)", f3(r.modules[a]?.output_norm), "measured"],
              ["Retrieval entropy", `${f2(r.retrieval_entropy)} bits`, "measured"]]), t);
        }
        out.push(linesTable(s.rounds.map(r => ({
          name: `Similar recall, round ${r.round}`, kind: "inferred", color: "var(--brain-right)", max: 1,
          value: l => r.similar_lines.find(([x]) => x === l)?.[1] ?? null,
        })), "Cosine similarity between the patterns recalled at this step and those recalled (on average) at each line's tokens, from the strongest patterns kept per token. Similar recall shows the memory is in a similar state, not that it retrieved that line: it holds no copy of what was said."));
        out.push(strip("Retrieval entropy at every token (round 1)", trace.profile.map(p => p.retrieval_entropy?.[0]), "var(--brain-right)"));
        out.push(el("h5", {}, "Influence"), influenceTable(n => n === "associative"));
        return out;
      },
    },
    controller: {
      charts() {
        const rounds = steps()[0]?.rounds.length || 0;
        const series = [];
        trace.modules.forEach((m, k) => {
          for (let r = 0; r < rounds; r++) series.push({ name: `${m} · round ${r + 1}`, dashed: r > 0, color: NET_COLOR[MODULE_NET[m]] || "var(--neuro-control)", values: steps().map(s => s.rounds[r]?.modules[k]?.gate) });
        });
        return [chart("Gates the Executive Controller set (0 closed, 1 fully open)", "0 to 1", series, { max: 1 })];
      },
      detail(s) {
        const out = [];
        const t = el("table", { class: "ledger compact" }, el("thead", {}, el("tr", {}, el("th", {}, "Round"), el("th", {}, "Network"),
          el("th", {}, "Gate ", tag("measured")), el("th", {}, "Output length ", tag("measured")), el("th", {}, "Gate × length ", tag("measured")), el("th", {}, "Share of the gated lengths ", tag("inferred")))));
        const body = el("tbody");
        for (const r of s.rounds) {
          const gated = r.modules.map(m => m.gate * m.output_norm);
          const sum = gated.reduce((a, b) => a + b, 0);
          r.modules.forEach((m, k) => body.append(el("tr", {}, el("td", {}, `${r.round}`), el("td", {}, m.name),
            el("td", { class: "num" }, bar(m.gate, 1, "var(--neuro-control)"), ` ${f3(m.gate)}`), el("td", { class: "num" }, f3(m.output_norm)),
            el("td", { class: "num" }, f3(gated[k])), el("td", { class: "num" }, sum > 0 ? p100(gated[k] / sum) : "—"))));
        }
        t.append(body);
        out.push(t, el("p", { class: "note" }, "The controller reads the bus and sets one gate per network per round. The share of gated lengths is a rough guide to how much each network's output weighs in the sum (the vectors can point in different directions, so lengths do not add exactly)."));
        for (const r of s.rounds) {
          if (!r.controller?.length) continue;
          const peak = Math.max(1e-9, ...r.controller.map(Math.abs));
          const g = svg("svg", { viewBox: `0 0 ${r.controller.length} 10`, preserveAspectRatio: "none", class: "trace-strip" });
          r.controller.forEach((v, i) => {
            const rect = svg("rect", { x: i, y: 0, width: 1.02, height: 10 });
            rect.setAttribute("style", `fill:${v >= 0 ? "var(--neuro-control)" : "var(--margin)"};opacity:${(0.1 + 0.9 * Math.abs(v) / peak).toFixed(3)}`);
            g.append(rect);
          });
          out.push(el("figure", { class: "trace-fig" }, el("figcaption", {}, `Round ${r.round}: the controller's ${r.controller.length} hidden activations `, tag("measured")), g,
            el("div", { class: "trace-legend" }, el("span", {}, "purple positive, red negative; shade is the size"))));
        }
        trace.modules.forEach((m, k) => out.push(strip(`Gate on ${m} at every token (round 1)`, trace.profile.map(p => p.gates?.[0]?.[k]), NET_COLOR[MODULE_NET[m]] || "var(--neuro-control)")));
        out.push(el("h5", {}, "Influence"), influenceTable(n => n === "controller"));
        return out;
      },
    },
    bus: {
      charts() {
        const rounds = steps()[0]?.rounds.length || 0;
        const s1 = [], s2 = [];
        for (let r = 0; r < rounds; r++) {
          const d = r > 0;
          s1.push({ name: `r${r + 1} transformer → bus`, dashed: d, color: "var(--brain-left)", values: steps().map(s => s.rounds[r]?.read_norm) });
          if (hasNet("memory")) s1.push({ name: `r${r + 1} working memory → bus`, dashed: d, color: "var(--neuro-memory)", values: steps().map(s => s.rounds[r]?.memory_in_norm) });
          s1.push({ name: `r${r + 1} networks → bus`, dashed: d, color: "var(--neuro-control)", values: steps().map(s => s.rounds[r]?.write_norm) });
          s2.push({ name: `round ${r + 1}`, dashed: d, color: "var(--brain-bridge)", fmt: p100, values: steps().map(s => s.rounds[r] ? s.rounds[r].injection_norm / Math.max(1e-9, s.rounds[r].state_norm) : NaN) });
        }
        return [chart("What the bus carried each round", "vector length", s1),
          chart("Bus → transformer: the write-back's length relative to the transformer's state", "ratio", s2)];
      },
      detail(s) {
        const out = [el("p", { class: "note" }, "The bus is a connection: each round the transformer's state is projected onto it (with the working memory's output added), the controller and the networks read it, and their gated outputs are summed and written back into the transformer's state through a learned gate before the upper blocks run again.")];
        for (const r of s.rounds) {
          const flow = el("div", { class: "trace-flow" },
            el("span", { class: "trace-node t" }, `Transformer state`, el("small", {}, `length ${f2(r.state_norm)}`)),
            el("span", { class: "trace-arrow" }, `→ ${f2(r.read_norm)}`),
            hasNet("memory") ? el("span", { class: "trace-node m" }, `+ working memory`, el("small", {}, `${f2(r.memory_in_norm)}`)) : "",
            el("span", { class: "trace-arrow" }, "="),
            el("span", { class: "trace-node b" }, "Bus", el("small", {}, `length ${f2(r.bus_norm)}`)),
            el("span", { class: "trace-arrow" }, "→"),
            el("span", { class: "trace-node c" }, "Networks (gated)", el("small", {}, r.modules.map(m => `${m.name} ${f2(m.gate)}×${f2(m.output_norm)}`).join(" · "))),
            el("span", { class: "trace-arrow" }, `→ sum ${f2(r.write_norm)} →`),
            el("span", { class: "trace-node t" }, "Back into the transformer", el("small", {}, `${f3(r.injection_norm)} (${p100(r.injection_norm / Math.max(1e-9, r.state_norm))} of the state) · gate ${f2(r.write_gate)}`)));
          out.push(el("h5", {}, `Round ${r.round} (before the upper blocks run for the ${r.round === 1 ? "first" : "next"} time) `, tag("measured")), flow);
          if (r.bus?.length) {
            const peak = Math.max(1e-9, ...r.bus.map(Math.abs));
            const g = svg("svg", { viewBox: `0 0 ${r.bus.length} 10`, preserveAspectRatio: "none", class: "trace-strip" });
            r.bus.forEach((v, i) => {
              const rect = svg("rect", { x: i, y: 0, width: 1.02, height: 10 });
              rect.setAttribute("style", `fill:${v >= 0 ? "var(--brain-bridge)" : "var(--margin)"};opacity:${(0.1 + 0.9 * Math.abs(v) / peak).toFixed(3)}`);
              g.append(rect);
            });
            out.push(el("figure", { class: "trace-fig" }, el("figcaption", {}, `The bus's ${r.bus.length} values this round `, tag("measured")), g));
          }
        }
        out.push(el("h5", {}, "Whether it was taken up downstream"),
          el("p", { class: "note" }, "The prediction after the lower blocks and after each round (before and after each write-back), and the change when the write-back or the transformer's input to the bus is cut."),
          table(s.transformer.readouts.map(r => [`After ${r.stage}`, r.top.map(c => `${c.token} ${p100(c.probability)}`).join(" · "), "measured"])),
          influenceTable(n => n.startsWith("bus")));
        out.push(strip("Write-back relative to the state at every token (round 1)", trace.profile.map(p => p.injection?.[0]), "var(--brain-bridge)"));
        return out;
      },
    },
    dense: {
      charts() {
        const n = steps()[0]?.rounds.length || 0;
        return [chart("Dense → transformer: injection relative to the state", "ratio", Array.from({ length: n }, (_, r) => ({
          name: r === 0 ? "before the upper blocks" : `round ${r}`, dashed: r > 1, color: "var(--brain-right)", fmt: p100,
          values: steps().map(s => s.rounds[r] ? s.rounds[r].injection_norm / Math.max(1e-9, s.rounds[r].state_norm) : NaN),
        })))];
      },
      detail(s) {
        return [table(s.rounds.map((r, i) => [i === 0 ? "Exchange before the upper blocks" : `Round ${i}`, `injection ${f3(r.injection_norm)} on a state of ${f3(r.state_norm)} · gate ${f2(r.write_gate)}`, "measured"])),
          el("h5", {}, "Influence"), influenceTable(n => n.startsWith("dense"))];
      },
    },
  };
  // Other networks (relational, experts, consolidation): their gates and outputs.
  for (const id of ["relational", "experts", "consolidation"]) {
    VIEWS[id] = {
      charts() {
        const k = moduleIndex(id);
        const rounds = steps()[0]?.rounds.length || 0;
        return [chart("Controller gate and output length", "", Array.from({ length: rounds }, (_, r) => [
          { name: `round ${r + 1} gate`, dashed: r > 0, color: "var(--neuro-control)", values: steps().map(s => s.rounds[r]?.modules[k]?.gate) },
          { name: `round ${r + 1} output length`, dashed: r > 0, color: NET_COLOR[id], values: steps().map(s => s.rounds[r]?.modules[k]?.output_norm) },
        ]).flat())];
      },
      detail(s) {
        const k = moduleIndex(id);
        return [table(s.rounds.flatMap(r => [
          [`Round ${r.round} · gate / output length`, `${f3(r.modules[k]?.gate)} / ${f3(r.modules[k]?.output_norm)}`, "measured"],
          id === "relational" ? [`Round ${r.round} · slot reads per hop`, r.slot_reads.map(h => h.map(f2).join(" ")).join(" | "), "measured"] : null,
          id === "experts" ? [`Round ${r.round} · router weights`, r.experts.map(f2).join(" "), "measured"] : null,
          id === "consolidation" ? [`Round ${r.round} · code length`, f3(r.code_norm), "measured"] : null,
        ])), el("h5", {}, "Influence"), influenceTable(n => n === id)];
      },
    };
  }

  // Context diagnostics: every earlier line with the three measures side by side.
  function contextLines(s) {
    const cols = [{ name: "Attention", color: "var(--brain-left)", max: 1, fmt: p100, value: l => s.transformer.lines.find(([x]) => x === l)?.[1] ?? 0 }];
    s.memory.forEach((layer, i) => cols.push({
      name: `Working memory ${i + 1}`, color: i ? "var(--neuro-code)" : "var(--neuro-memory)", max: Math.max(1e-9, ...layer.lines.map(([, v]) => v)),
      value: l => layer.lines.find(([x]) => x === l)?.[1] ?? 0,
    }));
    if (s.rounds[0]?.similar_lines?.length) cols.push({ name: "Similar recall (round 1)", kind: "inferred", color: "var(--brain-right)", max: 1, value: l => s.rounds[0].similar_lines.find(([x]) => x === l)?.[1] ?? null });
    return el("div", {}, el("h5", {}, "Context: what of each earlier line is in use at this step"),
      linesTable(cols, "Attention: how much this step's attention (all blocks and heads) went to the line's positions. Working memory: how much of that line's writing is still in the memory's state (length). Similar recall: how close the associative memory's recall is to its recall at that line. A specific fact can only reach the reply through attention or the working memory; the associative memory's patterns are learned, not written during the conversation."));
  }

  // ---- rendering -------------------------------------------------------------------------
  function render() {
    frame = 0;
    if (!trace) { $("traceView").hidden = true; return; }
    $("traceView").hidden = false;
    const n = steps().length;
    if (stepIdx >= n) stepIdx = Math.max(0, n - 1);
    if (live && $("traceFollow").checked) stepIdx = Math.max(0, n - 1);
    const how = { question: "The question above, answered now", replay: "Replay of an earlier reply", recorded: "Recorded while Nova wrote the reply", live: "Live: Nova is writing" }[origin?.kind] || "";
    $("traceHeader").replaceChildren(
      el("b", {}, how), ` · model step ${fmt(trace.model_step)} · ${trace.architecture === "neuro" ? "Nova Neuro" : "Nova Bicameral"} · prompt ${fmt(trace.prompt_tokens)} tokens in ${trace.lines.length} lines · ${n} step(s)`,
      live ? el("span", { class: "trace-live" }, " ● writing") : (trace.millis ? ` · traced in ${(trace.millis / 1000).toFixed(1)} s` : ""));
    $("traceNotes").replaceChildren(...(trace.notes || []).map(t => el("li", {}, t)));
    // Networks: the ones this model has.
    const nets = [{ id: "all", name: "All networks" }, ...trace.networks];
    if (!nets.some(x => x.id === net)) net = "all";
    $("traceNets").replaceChildren(...nets.map(x => {
      const b = el("button", { type: "button", role: "tab", class: "trace-net" + (x.id === net ? " on" : ""), "aria-selected": String(x.id === net), title: x.detail || "", onclick: () => { net = x.id; render(); } }, x.name);
      b.style.setProperty("--net", NET_COLOR[x.id] || "var(--accent)");
      return b;
    }));
    // The reply, one chip per step.
    $("traceReply").replaceChildren(el("span", { class: "trace-reply-label" }, "N:"), ...steps().map((s, i) => {
      const c = el("button", { type: "button", class: "trace-tokbtn" + (i === stepIdx ? " on" : ""), title: `step ${i + 1}: ${p100(s.chosen.probability)}`, onclick: () => pick(i) }, s.chosen.token);
      c.style.setProperty("--p", s.chosen.probability.toFixed(3));
      return c;
    }), live ? el("span", { class: "trace-caret" }, "▍") : "");
    const range = $("traceStep");
    range.max = String(Math.max(0, n - 1));
    range.value = String(stepIdx);
    $("traceFollowWrap").hidden = !live;
    const s = step();
    $("traceStepLabel").textContent = s ? `Step ${stepIdx + 1} of ${n}: "${s.chosen.token}" (${p100(s.chosen.probability)}; ${f2(s.entropy)} bits) · also weighed ${s.alternatives.slice(0, 3).map(a => `${a.token} ${p100(a.probability)}`).join(", ")}` : "No step yet.";
    const view = VIEWS[net] || VIEWS.all;
    const network = trace.networks.find(x => x.id === net);
    $("traceChart").replaceChildren(network ? el("p", { class: "note" }, network.detail) : "", ...(n ? view.charts() : []));
    $("traceDetail").replaceChildren(...(s ? view.detail(s) : [el("p", { class: "empty" }, "Waiting for the first step…")]));
  }
  const soon = () => { if (!frame) frame = requestAnimationFrame(render); };
  function pick(i) {
    const n = steps().length;
    if (!n) return;
    stepIdx = Math.max(0, Math.min(n - 1, i));
    if (live && stepIdx < n - 1) $("traceFollow").checked = false;
    render();
  }
  function show(t, how) {
    trace = t;
    origin = how;
    stepIdx = 0;
    live = false;
    render();
  }

  // ---- live events -----------------------------------------------------------------------
  if (tauri?.event?.listen) {
    tauri.event.listen("brain-trace", ev => {
      const e = ev.payload;
      if (e.kind === "read") {
        // A reply being written in a chat is shown unless a trace started here is running.
        if (e.source === "chat" && busy) return;
        trace = e.trace;
        origin = { kind: e.source === "chat" ? "live" : e.source, chat: e.chat };
        live = true;
        stepIdx = 0;
        soon();
      } else if (e.kind === "step" && live && trace && (origin.chat ?? null) === (e.chat ?? null)) {
        trace.steps.push(e.step);
        trace.tokens.push(e.token);
        soon();
      } else if (e.kind === "done" && live && (origin?.chat ?? null) === (e.chat ?? null)) {
        live = false;
        if (e.source === "chat") {
          origin = { kind: "recorded", chat: e.chat, name: e.name };
          if (e.name) safe(async () => { trace = await invoke("trace_load", { chat: e.chat, name: e.name }); render(); refreshTurns(); storage(); }, "Trace")();
        }
        soon();
      }
    });
  }

  // ---- choosing what to trace -------------------------------------------------------------
  let busy = false;
  async function refreshChats() {
    const sel = $("traceChat");
    const keep = sel.value;
    const list = active ? await invoke("chats") : [];
    sel.replaceChildren(el("option", { value: "" }, "The question above (on its own)"),
      ...list.map(c => el("option", { value: c.id }, `Chat: ${c.title}`)));
    if (list.some(c => c.id === keep)) sel.value = keep;
    await refreshTurns();
  }
  async function refreshTurns() {
    const chat = $("traceChat").value;
    $("traceTurn").hidden = $("traceTurnLabel").hidden = !chat;
    turns = chat ? await invoke("trace_turns", { chat }) : [];
    const keep = $("traceTurn").value;
    $("traceTurn").replaceChildren(...turns.map(t => el("option", { value: t.index }, `${t.trace ? "● " : ""}“${t.message.slice(0, 50)}” → “${t.reply.slice(0, 50)}”`)));
    if (turns.some(t => String(t.index) === keep)) $("traceTurn").value = keep;
    else if (turns.length) $("traceTurn").value = String(turns[turns.length - 1].index);
    turnChanged();
  }
  function turnChanged() {
    const t = turns.find(x => String(x.index) === $("traceTurn").value);
    const chat = $("traceChat").value;
    $("traceOpen").hidden = !(chat && t?.trace);
    $("traceRun").textContent = chat ? "Replay this reply" : "Trace the question above";
    $("traceInfluenceRun").hidden = !(trace && !live && (origin?.kind === "recorded") && !trace.influence?.length);
    $("traceHint").textContent = !chat
      ? "Nova answers the question in the box above (the most likely token each time) while every network is traced."
      : t?.trace
        ? "● A trace was recorded while Nova wrote this reply: open it to see exactly what happened. Replaying reads the saved reply through the current model."
        : "No trace was recorded for this reply. Replaying reads the saved reply through the current model: the page says where that can differ from what happened. To record replies as they are written, turn on \"Record a brain trace of each reply\" in the chat's settings (§ 13 Chat).";
  }
  async function storage() {
    if (!active) { $("traceStorage").textContent = ""; return; }
    const s = await invoke("trace_storage");
    $("traceStorage").replaceChildren(s.traces ? `Saved traces of this model: ${fmt(s.traces)} (${(s.bytes / 1e6).toFixed(1)} MB). ` : "No saved traces for this model.",
      s.traces ? el("button", { type: "button", class: "quiet", onclick: safe(async () => {
        if (!confirm(`Delete all ${s.traces} saved traces of this model? The chats are kept.`)) return;
        await invoke("trace_delete", {});
        await refreshTurns();
        await storage();
      }, "Delete traces") }, "Delete all") : "");
  }
  async function run(fn) {
    if (busy) return;
    busy = true;
    $("traceRun").disabled = $("traceOpen").disabled = $("traceInfluenceRun").disabled = true;
    try { await fn(); } finally {
      busy = false;
      $("traceRun").disabled = $("traceOpen").disabled = $("traceInfluenceRun").disabled = false;
      turnChanged();
    }
  }
  $("traceChat").addEventListener("change", safe(refreshTurns, "Trace"));
  $("traceTurn").addEventListener("change", turnChanged);
  $("traceRun").addEventListener("click", safe(() => run(async () => {
    if (!active) throw new Error("Open a model first");
    const chat = $("traceChat").value;
    const influence = $("traceInfluence").checked;
    if (chat) {
      const index = Number($("traceTurn").value);
      say("Replaying the reply through the model with every network traced…");
      const t = await invoke("trace_replay", { chat, index, influence });
      show(t, { kind: "replay", chat, index });
    } else {
      say("Nova is answering the question with every network traced…");
      const t = await invoke("trace_question", { prompt: $("probe").value, influence });
      show(t, { kind: "question" });
    }
    say(`Traced ${trace.steps.length} step(s).`);
  }), "Trace"));
  $("traceOpen").addEventListener("click", safe(() => run(async () => {
    const chat = $("traceChat").value;
    const t = turns.find(x => String(x.index) === $("traceTurn").value);
    if (!t?.trace) return;
    show(await invoke("trace_load", { chat, name: t.trace }), { kind: "recorded", chat, index: t.index, name: t.trace });
  }), "Trace"));
  $("traceInfluenceRun").addEventListener("click", safe(() => run(async () => {
    if (origin?.kind !== "recorded") return;
    say("Measuring each network's influence: the turn is read again once per network…");
    const t = await invoke("trace_influence", { chat: origin.chat, name: origin.name });
    trace = t;
    render();
    say("Influence measured.");
  }), "Influence"));
  $("traceStep").addEventListener("input", () => pick(Number($("traceStep").value)));
  $("tracePrev").addEventListener("click", () => pick(stepIdx - 1));
  $("traceNext").addEventListener("click", () => pick(stepIdx + 1));
  $("traceFollow").addEventListener("change", () => { if ($("traceFollow").checked) pick(steps().length - 1); });
  $("traceView").addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") { pick(stepIdx - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { pick(stepIdx + 1); e.preventDefault(); }
  });

  // From § 13 Chat: a reply's "Brain trace" button.
  window.openTrace = safe(async (chat, index) => {
    go("brain");
    await refreshChats();
    $("traceChat").value = chat;
    await refreshTurns();
    $("traceTurn").value = String(index);
    turnChanged();
    const t = turns.find(x => x.index === index);
    if (t?.trace) $("traceOpen").click();
    $("traceSection").scrollIntoView({ behavior: "smooth", block: "start" });
  }, "Trace");

  let shownModel = null;
  const prevPage = window.onPage;
  window.onPage = page => {
    prevPage?.(page);
    if (page !== "brain") return;
    if (shownModel !== (active?.id ?? null)) {
      shownModel = active?.id ?? null;
      if (!live) { trace = null; render(); }
    }
    safe(async () => { await refreshChats(); await storage(); }, "Trace")();
  };
})();
