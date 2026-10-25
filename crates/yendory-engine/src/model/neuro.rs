//! Nova Neuro (experimental): a modular brain in which specialised networks exchange
//! information through a learned communication bus. Optional; Nova Bicameral is unchanged.
//!
//! The model shares Bicameral's container (flat weights, named parameter slots, tokenizer,
//! optimizer state, parameter groups) and its transformer core (embeddings tied with the output
//! layer, blocks, rotary positions, thinking rounds). Instead of the Dense hemisphere it has
//! the modules a [`NeuroSpec`] lists. Neuro-Core, the first milestone:
//!
//! * **Working memory**: a minGRU stack (`m_t = (1 - z_t) m_(t-1) + z_t c_t`, gates from the
//!   current input only, so a sequence is one parallel scan). Its state belongs to the current
//!   context and is never stored with the model.
//! * **Associative memory**: learned patterns read by attention (a modern Hopfield retrieval:
//!   softmax over pattern keys, a weighted sum of pattern values). The patterns are parameters
//!   learned in training; Study Mode's external text memory is a separate thing.
//! * **Executive Controller**: a small network that reads the bus and gives every module a gate
//!   between 0 and 1 for each token and round. Nothing in it looks at words: it is trained like
//!   every other weight.
//! * **Communication bus**: each round, the transformer state is projected onto the bus, the
//!   working memory adds its state, the controller and modules read the bus, the modules'
//!   gated outputs are summed and written back into the transformer's residual stream through
//!   a learned gate, and the upper blocks run again, so information moves both ways.
//!
//! Stage 4 adds three more networks, each optional:
//!
//! * **Relational network**: entity slots. Each token writes itself into a few of `slots`
//!   vectors (a learned address, softmax over the slots, and a learned per-channel write gate),
//!   so the slots hold what has been read about each entity, causally: a slot at position `t`
//!   holds only tokens up to `t` (the write is a gated scan, like working memory, over
//!   `slots x width` channels). Every round the bus queries the slots by content and follows
//!   the result for `hops` steps (each hop's query comes from the previous read), which is how
//!   it can connect "who has the apple" to "where that person went".
//! * **Experts**: a mixture of small SwiGLU networks on the bus with a learned soft router
//!   (softmax over the experts, per token and round). Specialisation is learned, never assigned.
//! * **Consolidation**: an autoencoder-shaped bottleneck. The bus is compressed into a short code
//!   and expanded again; with associative memory present, the memory is queried with that code.
//!
//! Any combination is allowed, including no transformer blocks (the networks alone, reading the
//! word embeddings) and no networks (the transformer alone).
//!
//! Parameter groups: the transformer and embeddings are `Left` (language core), the modules are
//! `Right`, the bus and controller are `Interconnect`, so Foundation candidates and the
//! bicameral education stages carry over ("modules", "language core", "communication").
use super::tape::Graph;
use super::{Ablation, Bicameral, Config, Focus, Init};
use serde::{Deserialize, Serialize};

/// Current version of the Neuro design format. Version 1 designs (Neuro-Core) are still read;
/// a version 1 design without networks keeps its unused bus, as it was saved.
pub const VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// minGRU layers (1-4).
    pub layers: usize,
    /// State width of each layer.
    pub width: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Associative {
    /// Learned patterns.
    pub patterns: usize,
    /// Width of the pattern keys (the query is projected to it).
    pub key: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Relational {
    /// Entity slots (1-64).
    pub slots: usize,
    /// Width of each slot (1-512).
    pub width: usize,
    /// Hops each read follows through the slots (1-4).
    pub hops: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experts {
    /// Expert networks (2-16).
    pub count: usize,
    /// Hidden width of each expert (1-4096).
    pub width: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Consolidation {
    /// Width of the compressed code (1-4096).
    pub code: usize,
    /// Hidden width of the encoder and decoder (1-4096).
    pub hidden: usize,
}

/// The design of a Nova Neuro model: which modules it has and their sizes. Saved with the
/// model. The transformer core is described by the model's [`Config`] (width, blocks, heads,
/// feed-forward width, lower blocks `split`, rounds `loops`, context, vocabulary).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuroSpec {
    pub version: u32,
    /// Width of the communication bus.
    pub bus: usize,
    #[serde(default)]
    pub memory: Option<WorkingMemory>,
    #[serde(default)]
    pub associative: Option<Associative>,
    /// Hidden width of the Executive Controller.
    pub controller: usize,
    #[serde(default)]
    pub relational: Option<Relational>,
    #[serde(default)]
    pub experts: Option<Experts>,
    #[serde(default)]
    pub consolidation: Option<Consolidation>,
}

/// The modules a design has, in the order of the controller's gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Module {
    Memory,
    Associative,
    Relational,
    Experts,
    Consolidation,
}
impl Module {
    pub fn name(self) -> &'static str {
        match self {
            Module::Memory => "working memory",
            Module::Associative => "associative memory",
            Module::Relational => "relational network",
            Module::Experts => "experts",
            Module::Consolidation => "consolidation",
        }
    }
}

impl NeuroSpec {
    /// The modules present, in gate order.
    pub fn modules(&self) -> Vec<Module> {
        let mut m = Vec::new();
        if self.memory.is_some() {
            m.push(Module::Memory);
        }
        if self.associative.is_some() {
            m.push(Module::Associative);
        }
        if self.relational.is_some() {
            m.push(Module::Relational);
        }
        if self.experts.is_some() {
            m.push(Module::Experts);
        }
        if self.consolidation.is_some() {
            m.push(Module::Consolidation);
        }
        m
    }
    /// The communication bus and its write-back exist: the design has networks (a version 1
    /// design always has them, as it was saved).
    pub fn has_bus(&self) -> bool {
        self.version == 1 || !self.modules().is_empty()
    }
    pub fn validate(&self, c: &Config) -> Result<(), String> {
        let ok = |x: usize, hi: usize| (1..=hi).contains(&x);
        if !(1..=VERSION).contains(&self.version) {
            return Err(format!("Unsupported Nova Neuro design version {}", self.version));
        }
        if self.version == 1
            && (self.relational.is_some() || self.experts.is_some() || self.consolidation.is_some())
        {
            return Err(
                "Nova Neuro: a version 1 design has no relational network, experts or consolidation".into()
            );
        }
        if !ok(self.bus, 4096) || !ok(self.controller, 4096) {
            return Err("Nova Neuro: bus and controller widths must be 1-4096".into());
        }
        if let Some(m) = &self.memory {
            if !ok(m.layers, 4) || !ok(m.width, 4096) {
                return Err("Nova Neuro: working memory needs 1-4 layers of width 1-4096".into());
            }
        }
        if let Some(a) = &self.associative {
            if !ok(a.patterns, 16384) || !ok(a.key, 4096) {
                return Err("Nova Neuro: associative memory needs 1-16384 patterns, key width 1-4096".into());
            }
        }
        if let Some(r) = &self.relational {
            if !ok(r.slots, 64) || !ok(r.width, 512) || !ok(r.hops, 4) {
                return Err(
                    "Nova Neuro: the relational network needs 1-64 slots of width 1-512 and 1-4 hops".into(),
                );
            }
        }
        if let Some(e) = &self.experts {
            if !(2..=16).contains(&e.count) || !ok(e.width, 4096) {
                return Err("Nova Neuro: experts need 2-16 experts of width 1-4096".into());
            }
        }
        if let Some(k) = &self.consolidation {
            if !ok(k.code, 4096) || !ok(k.hidden, 4096) {
                return Err("Nova Neuro: consolidation needs a code and hidden width of 1-4096".into());
            }
        }
        if c.loops == 0 {
            return Err("Nova Neuro: needs at least one round".into());
        }
        if c.blocks == 0 && self.modules().is_empty() {
            return Err("Nova Neuro: choose at least one network (the transformer or a module)".into());
        }
        if c.blocks > 0 && c.split >= c.blocks {
            return Err("Nova Neuro: with transformer blocks, at least one must be an upper block".into());
        }
        if c.blocks == 0 && c.split != 0 {
            return Err("Nova Neuro: without transformer blocks there are no lower blocks".into());
        }
        Ok(())
    }
}

/// The parameter slots of a Neuro design, after the shared transformer core.
pub(crate) fn slots(c: &Config, n: &NeuroSpec, put: &mut dyn FnMut(String, usize, usize, Focus, Init, bool)) {
    let scale = |width: usize| 0.5 / (width as f32).sqrt();
    let (d, b) = (c.d, n.bus);
    if !n.has_bus() {
        return;
    }
    // Communication bus (Interconnect).
    put("n.read_norm".into(), 1, d, Focus::Interconnect, Init::One, false);
    put("n.read".into(), b, d, Focus::Interconnect, Init::Normal(scale(d)), true);
    put("n.bus_norm".into(), 1, b, Focus::Interconnect, Init::One, false);
    put("n.gate_w".into(), d, d, Focus::Interconnect, Init::Normal(scale(d)), true);
    put("n.gate_b".into(), 1, d, Focus::Interconnect, Init::Constant(c.gate_bias), false);
    put("n.write".into(), d, b, Focus::Interconnect, Init::Normal(scale(b)), true);
    // Executive Controller (Interconnect): one gate per module.
    let k = n.modules().len();
    if k > 0 {
        put("n.ctl1".into(), n.controller, b, Focus::Interconnect, Init::Normal(scale(b)), true);
        put("n.ctl1b".into(), 1, n.controller, Focus::Interconnect, Init::Zero, false);
        put("n.ctl2".into(), k, n.controller, Focus::Interconnect, Init::Normal(scale(n.controller)), true);
        put("n.ctl2b".into(), 1, k, Focus::Interconnect, Init::Zero, false);
    }
    // Working memory (Right).
    if let Some(m) = &n.memory {
        let mut input = d;
        for l in 0..m.layers {
            put(format!("n.wm{l}.norm"), 1, input, Focus::Right, Init::One, false);
            put(format!("n.wm{l}.z"), m.width, input, Focus::Right, Init::Normal(scale(input)), true);
            // A negative update-gate bias: the memory starts by keeping most of its state.
            put(format!("n.wm{l}.zb"), 1, m.width, Focus::Right, Init::Constant(-1.), false);
            put(format!("n.wm{l}.c"), m.width, input, Focus::Right, Init::Normal(scale(input) * 2.), true);
            put(format!("n.wm{l}.cb"), 1, m.width, Focus::Right, Init::Zero, false);
            input = m.width;
        }
        put("n.wm_out".into(), b, m.width, Focus::Right, Init::Normal(scale(m.width)), true);
    }
    // Associative memory (Right). With consolidation it is queried with the compressed code.
    if let Some(a) = &n.associative {
        let input = n.consolidation.as_ref().map_or(b, |k| k.code);
        put("n.as_norm".into(), 1, input, Focus::Right, Init::One, false);
        put("n.as_q".into(), a.key, input, Focus::Right, Init::Normal(1. / (input as f32).sqrt()), true);
        put(
            "n.as_keys".into(),
            a.patterns,
            a.key,
            Focus::Right,
            Init::Normal(1. / (a.key as f32).sqrt()),
            true,
        );
        put("n.as_vals".into(), b, a.patterns, Focus::Right, Init::Normal(scale(a.patterns) * 4.), true);
    }
    // Relational network (Right): written from the lower blocks' output, read from the bus.
    if let Some(r) = &n.relational {
        let w = r.width;
        put("n.rel.norm".into(), 1, d, Focus::Right, Init::One, false);
        put("n.rel.addr".into(), r.slots, d, Focus::Right, Init::Normal(1. / (d as f32).sqrt()), true);
        put("n.rel.addr_b".into(), 1, r.slots, Focus::Right, Init::Zero, false);
        put("n.rel.gate".into(), w, d, Focus::Right, Init::Normal(scale(d)), true);
        // A negative write-gate bias: slots start by keeping what they hold.
        put("n.rel.gate_b".into(), 1, w, Focus::Right, Init::Constant(-1.), false);
        put("n.rel.val".into(), w, d, Focus::Right, Init::Normal(scale(d) * 2.), true);
        put("n.rel.qnorm".into(), 1, b, Focus::Right, Init::One, false);
        put("n.rel.q0".into(), w, b, Focus::Right, Init::Normal(1. / (b as f32).sqrt()), true);
        for h in 1..r.hops {
            put(format!("n.rel.hop{h}"), w, w, Focus::Right, Init::Normal(1. / (w as f32).sqrt()), true);
        }
        put("n.rel.out".into(), b, w, Focus::Right, Init::Normal(scale(w)), true);
    }
    // Experts (Right): a soft router and `count` SwiGLU networks on the bus.
    if let Some(e) = &n.experts {
        put("n.ex.norm".into(), 1, b, Focus::Right, Init::One, false);
        put("n.ex.router".into(), e.count, b, Focus::Right, Init::Normal(scale(b)), true);
        put("n.ex.router_b".into(), 1, e.count, Focus::Right, Init::Zero, false);
        for i in 0..e.count {
            put(format!("n.ex{i}.gate"), e.width, b, Focus::Right, Init::Normal(scale(b)), true);
            put(format!("n.ex{i}.up"), e.width, b, Focus::Right, Init::Normal(scale(b)), true);
            put(format!("n.ex{i}.down"), b, e.width, Focus::Right, Init::Normal(scale(e.width)), true);
        }
    }
    // Consolidation (Right): bus -> hidden -> code -> hidden -> bus.
    if let Some(k) = &n.consolidation {
        put("n.co.norm".into(), 1, b, Focus::Right, Init::One, false);
        put("n.co.enc1".into(), k.hidden, b, Focus::Right, Init::Normal(scale(b)), true);
        put("n.co.enc1b".into(), 1, k.hidden, Focus::Right, Init::Zero, false);
        put("n.co.enc2".into(), k.code, k.hidden, Focus::Right, Init::Normal(scale(k.hidden)), true);
        put("n.co.enc2b".into(), 1, k.code, Focus::Right, Init::Zero, false);
        put("n.co.dec1".into(), k.hidden, k.code, Focus::Right, Init::Normal(scale(k.code)), true);
        put("n.co.dec1b".into(), 1, k.hidden, Focus::Right, Init::Zero, false);
        put("n.co.dec2".into(), b, k.hidden, Focus::Right, Init::Normal(scale(k.hidden)), true);
    }
}

/// Which module a parameter belongs to, for the parameter breakdown.
pub fn part_of(name: &str) -> &'static str {
    if name.starts_with("n.wm") {
        "working memory"
    } else if name.starts_with("n.as") {
        "associative memory"
    } else if name.starts_with("n.rel") {
        "relational network"
    } else if name.starts_with("n.ex") {
        "experts"
    } else if name.starts_with("n.co") {
        "consolidation"
    } else if name.starts_with("n.ctl") {
        "executive controller"
    } else if name.starts_with("n.") {
        "communication bus"
    } else if name == "embed" || name == "out_bias" || name == "ln_f" {
        // The final norm belongs to the output layer (it exists without transformer blocks).
        "embeddings (tied with the output layer)"
    } else {
        "transformer"
    }
}

/// Values stored per token by the training forward pass (activations kept for the backward
/// pass), counted from the graph below; multiply by 4 bytes and the tokens of a row.
pub fn activation_floats_per_token(c: &Config, n: &NeuroSpec) -> usize {
    let executions = c.split + (c.blocks - c.split) * c.loops;
    let b = n.bus;
    let mut f = c.d + executions * (12 * c.d + 4 * c.ff);
    if let Some(m) = &n.memory {
        f += c.d + m.layers * (6 * m.width) + b;
    }
    if let Some(r) = &n.relational {
        f += c.d + 3 * r.slots + 3 * r.width + 3 * r.slots * r.width;
    }
    let k = n.modules().len();
    let mut round = 0;
    if n.has_bus() {
        round += 3 * c.d + 3 * b + 2 * c.d; // read, bus, norms, gate, write, inject
    }
    if k > 0 {
        round += 2 * n.controller + 2 * k;
    }
    if n.memory.is_some() {
        round += 2 * b;
    }
    if let Some(a) = &n.associative {
        let input = n.consolidation.as_ref().map_or(b, |k| k.code);
        round += input + a.key + 2 * a.patterns + 2 * b;
    }
    if let Some(r) = &n.relational {
        round += b + r.width + r.hops * (2 * r.slots + 2 * r.width) + 2 * b;
    }
    if let Some(e) = &n.experts {
        round += b + 2 * e.count + e.count * (4 * e.width + 2 * b) + 2 * b;
    }
    if let Some(k) = &n.consolidation {
        round += b + 4 * k.hidden + k.code + 2 * b;
    }
    f + c.loops * round + c.d
}

impl Bicameral {
    fn nparam<G: Graph>(&self, t: &mut G, name: &str) -> usize {
        t.parameter(name, self.slot(name).range.clone())
    }
    fn nlin<G: Graph>(&self, t: &mut G, x: usize, name: &str, bias: Option<&str>) -> usize {
        let slot = self.slot(name);
        let (cols, rows) = (slot.cols, slot.rows);
        let w = self.nparam(t, name);
        let b = bias.map(|n| self.nparam(t, n));
        t.linear(x, w, b, cols, rows)
    }
    fn nnorm<G: Graph>(&self, t: &mut G, x: usize, name: &str) -> usize {
        let cols = self.slot(name).cols;
        let g = self.nparam(t, name);
        t.norm(x, g, cols)
    }

    /// Nova Neuro's forward pass up to the output layer (see the module documentation).
    pub(crate) fn neuro_hidden<G: Graph>(
        &self,
        t: &mut G,
        ids: &[usize],
        seq: usize,
        abl: Ablation,
        drops: &mut super::DropSites,
    ) -> usize {
        let c = &self.config;
        let n = self.neuro.as_ref().expect("a Nova Neuro model");
        let embed = self.nparam(t, "embed");
        let mut h = t.embed(ids, embed, c.d);
        for i in 0..c.split {
            h = self.block(t, i, h, seq, drops);
        }
        // Working memory: computed once from the lower blocks' output.
        let memory = n.memory.as_ref().map(|m| {
            let mut x = h;
            for l in 0..m.layers {
                let xn = self.nnorm(t, x, &format!("n.wm{l}.norm"));
                let z = self.nlin(t, xn, &format!("n.wm{l}.z"), Some(&format!("n.wm{l}.zb")));
                let z = t.sigmoid(z);
                let cand = self.nlin(t, xn, &format!("n.wm{l}.c"), Some(&format!("n.wm{l}.cb")));
                x = t.scan(z, cand, seq, m.width);
            }
            let out = self.nlin(t, x, "n.wm_out", None);
            if abl.no_memory || abl.no_right {
                t.zeros_like(out)
            } else {
                out
            }
        });
        // Relational network: the slots at every position, written once from the lower blocks'
        // output (a gated scan over `slots x width` channels, so a position sees only itself
        // and earlier tokens).
        let slot_memory = n.relational.as_ref().map(|r| {
            let hn = self.nnorm(t, h, "n.rel.norm");
            let addr = self.nlin(t, hn, "n.rel.addr", Some("n.rel.addr_b"));
            let addr = t.softmax_rows(addr, r.slots);
            let gate = self.nlin(t, hn, "n.rel.gate", Some("n.rel.gate_b"));
            let gate = t.sigmoid(gate);
            let z = t.outer(addr, gate, r.slots, r.width);
            let value = self.nlin(t, hn, "n.rel.val", None);
            let ones = t.input(vec![1.; ids.len() * r.slots]);
            let cand = t.outer(ones, value, r.slots, r.width);
            t.scan(z, cand, seq, r.slots * r.width)
        });
        let rounds = if abl.one_round { 1 } else { c.loops };
        for _ in 0..rounds {
            if n.has_bus() {
                h = self.neuro_round(t, h, n, abl, memory, slot_memory);
            }
            for i in c.split..c.blocks {
                h = self.block(t, i, h, seq, drops);
            }
        }
        self.norm(t, h, "ln_f")
    }

    /// One round of the bus: the transformer state and working memory onto the bus, the
    /// controller's gates, every network's gated output, and the write-back into `h`.
    #[allow(clippy::too_many_arguments)]
    fn neuro_round<G: Graph>(
        &self,
        t: &mut G,
        h: usize,
        n: &NeuroSpec,
        abl: Ablation,
        memory: Option<usize>,
        slot_memory: Option<usize>,
    ) -> usize {
        let modules = n.modules();
        let k = modules.len();
        let cut =
            |t: &mut G, out: usize, off: bool| if off || abl.no_right { t.zeros_like(out) } else { out };
        let hn = self.nnorm(t, h, "n.read_norm");
        let mut bus = self.nlin(t, hn, "n.read", None);
        if abl.no_efferent {
            bus = t.zeros_like(bus);
        }
        if let Some(m) = memory {
            bus = t.add(bus, m);
        }
        let bn = self.nnorm(t, bus, "n.bus_norm");
        let gates = (k > 0).then(|| {
            let hid = self.nlin(t, bn, "n.ctl1", Some("n.ctl1b"));
            let hid = t.silu(hid);
            let g = self.nlin(t, hid, "n.ctl2", Some("n.ctl2b"));
            let g = t.sigmoid(g);
            if abl.no_controller {
                // Every gate fully open: the controller's choices are removed.
                let zero = t.zeros_like(g);
                let half = t.sigmoid(zero);
                t.add(half, half)
            } else {
                g
            }
        });
        // Consolidation's code (also the associative memory's query input).
        let code = n.consolidation.as_ref().map(|_| {
            let cn = self.nnorm(t, bn, "n.co.norm");
            let e = self.nlin(t, cn, "n.co.enc1", Some("n.co.enc1b"));
            let e = t.silu(e);
            self.nlin(t, e, "n.co.enc2", Some("n.co.enc2b"))
        });
        let mut write: Option<usize> = None;
        for (gi, module) in modules.iter().enumerate() {
            let out = match module {
                Module::Memory => memory.expect("memory module"),
                Module::Associative => {
                    let a = n.associative.as_ref().unwrap();
                    let an = self.nnorm(t, code.unwrap_or(bn), "n.as_norm");
                    let q = self.nlin(t, an, "n.as_q", None);
                    let scores = self.nlin(t, q, "n.as_keys", None);
                    let p = t.softmax_rows(scores, a.patterns);
                    let out = self.nlin(t, p, "n.as_vals", None);
                    cut(t, out, abl.no_associative)
                }
                Module::Relational => {
                    let r = n.relational.as_ref().unwrap();
                    let m = slot_memory.expect("relational slots");
                    let scale = 1. / (r.width as f32).sqrt();
                    let qn = self.nnorm(t, bn, "n.rel.qnorm");
                    let mut q = self.nlin(t, qn, "n.rel.q0", None);
                    let mut total: Option<usize> = None;
                    for hop in 0..r.hops {
                        if hop > 0 {
                            q = self.nlin(
                                t,
                                total.expect("an earlier hop"),
                                &format!("n.rel.hop{hop}"),
                                None,
                            );
                        }
                        let scores = t.slot_scores(q, m, r.slots, r.width, scale);
                        let p = t.softmax_rows(scores, r.slots);
                        let read = t.slot_read(p, m, r.slots, r.width);
                        total = Some(match total {
                            Some(x) => t.add(x, read),
                            None => read,
                        });
                    }
                    let out = self.nlin(t, total.expect("one hop"), "n.rel.out", None);
                    cut(t, out, abl.no_relational)
                }
                Module::Experts => {
                    let e = n.experts.as_ref().unwrap();
                    let en = self.nnorm(t, bn, "n.ex.norm");
                    let route = self.nlin(t, en, "n.ex.router", Some("n.ex.router_b"));
                    let route = t.softmax_rows(route, e.count);
                    let mut sum: Option<usize> = None;
                    for i in 0..e.count {
                        let g = self.nlin(t, en, &format!("n.ex{i}.gate"), None);
                        let g = t.silu(g);
                        let u = self.nlin(t, en, &format!("n.ex{i}.up"), None);
                        let hid = t.mul(g, u);
                        let o = self.nlin(t, hid, &format!("n.ex{i}.down"), None);
                        let o = t.gate_col(o, route, i, e.count, n.bus);
                        sum = Some(match sum {
                            Some(x) => t.add(x, o),
                            None => o,
                        });
                    }
                    cut(t, sum.expect("two experts or more"), abl.no_experts)
                }
                Module::Consolidation => {
                    let d1 = self.nlin(t, code.expect("code"), "n.co.dec1", Some("n.co.dec1b"));
                    let d1 = t.silu(d1);
                    let out = self.nlin(t, d1, "n.co.dec2", None);
                    cut(t, out, abl.no_consolidation)
                }
            };
            let gated = t.gate_col(out, gates.expect("gates"), gi, k, n.bus);
            write = Some(match write {
                Some(w) => t.add(w, gated),
                None => gated,
            });
        }
        match (write, abl.no_afferent) {
            (Some(w), false) => {
                let gate = self.nlin(t, hn, "n.gate_w", Some("n.gate_b"));
                let gate = t.sigmoid(gate);
                let z = self.nlin(t, w, "n.write", None);
                let fused = t.mul(gate, z);
                t.add(h, fused)
            }
            _ => h,
        }
    }
}

/// Named designs offered when creating a model. Each keeps the transformer core of the
/// Bicameral preset of the same size, so the two architectures can be compared at equal size;
/// the parameters Bicameral gives the Dense hemisphere go to the modules and the bus.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuroPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub config: Config,
    pub design: NeuroSpec,
}

pub fn presets() -> Vec<NeuroPreset> {
    let core = |vocab, d, heads, ff, blocks| Config {
        vocab,
        d,
        heads,
        ff,
        blocks,
        split: blocks / 2,
        loops: 2,
        // Not used by Nova Neuro (no Dense hemisphere); kept minimal and valid.
        window_chars: 1,
        r1: 1,
        r2: 1,
        ..Config::default()
    };
    let design = |bus, layers, width, patterns, key, controller| NeuroSpec {
        version: VERSION,
        bus,
        memory: Some(WorkingMemory { layers, width }),
        associative: Some(Associative { patterns, key }),
        controller,
        relational: None,
        experts: None,
        consolidation: None,
    };
    // Every network: working memory (2 layers), associative memory queried with the
    // consolidation code, relational slots, experts and consolidation, at the same size as the
    // Neuro-Core preset (working memory's width takes up the difference).
    let full = |bus,
                memory_width,
                (patterns, key),
                (slots, width, hops),
                (count, expert),
                (code, hidden),
                controller| {
        NeuroSpec {
            version: VERSION,
            bus,
            memory: Some(WorkingMemory { layers: 2, width: memory_width }),
            associative: Some(Associative { patterns, key }),
            controller,
            relational: Some(Relational { slots, width, hops }),
            experts: Some(Experts { count, width: expert }),
            consolidation: Some(Consolidation { code, hidden }),
        }
    };
    vec![
        NeuroPreset {
            id: "neuro-core-1m",
            name: "Neuro-Core, Nova-2 size (1.2M)",
            config: core(384, 128, 4, 288, 4),
            design: design(128, 2, 224, 896, 64, 64),
        },
        NeuroPreset {
            id: "neuro-core-4m",
            name: "Neuro-Core 4M",
            config: core(384, 192, 6, 448, 6),
            design: design(192, 2, 448, 2176, 96, 96),
        },
        NeuroPreset {
            id: "neuro-core-8m",
            name: "Neuro-Core 8M",
            config: core(384, 256, 8, 736, 6),
            design: design(256, 2, 640, 3392, 128, 128),
        },
        NeuroPreset {
            id: "neuro-core-15m",
            name: "Neuro-Core 15M",
            config: core(384, 320, 10, 768, 8),
            design: design(256, 2, 912, 6496, 128, 128),
        },
        NeuroPreset {
            id: "neuro-full-1m",
            name: "Every network, Nova-2 size (1.2M)",
            config: core(384, 128, 4, 288, 4),
            design: full(128, 216, (384, 32), (16, 32, 2), (4, 48), (32, 96), 64),
        },
        NeuroPreset {
            id: "neuro-full-4m",
            name: "Every network 4M",
            config: core(384, 192, 6, 448, 6),
            design: full(192, 464, (1024, 48), (16, 48, 2), (4, 96), (48, 160), 96),
        },
        NeuroPreset {
            id: "neuro-full-8m",
            name: "Every network 8M",
            config: core(384, 256, 8, 736, 6),
            design: full(256, 672, (1536, 64), (24, 64, 2), (4, 160), (64, 224), 128),
        },
        NeuroPreset {
            id: "neuro-full-15m",
            name: "Every network 15M",
            config: core(384, 320, 10, 768, 8),
            design: full(256, 1048, (2560, 64), (32, 64, 3), (4, 256), (64, 256), 128),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{KvCache, TokenId};

    fn tiny() -> (Config, NeuroSpec) {
        let c = Config {
            vocab: 40,
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 3,
            split: 1,
            loops: 2,
            window_chars: 1,
            r1: 1,
            r2: 1,
            max_tokens: 12,
            ..Config::default()
        };
        let n = NeuroSpec {
            version: VERSION,
            bus: 8,
            memory: Some(WorkingMemory { layers: 2, width: 6 }),
            associative: Some(Associative { patterns: 10, key: 5 }),
            controller: 7,
            relational: Some(Relational { slots: 3, width: 4, hops: 2 }),
            experts: Some(Experts { count: 3, width: 5 }),
            consolidation: Some(Consolidation { code: 3, hidden: 6 }),
        };
        (c, n)
    }
    /// Designs with one network, or with the networks and no transformer.
    fn singles() -> Vec<(&'static str, Config, NeuroSpec)> {
        let (c, full) = tiny();
        let none = NeuroSpec {
            memory: None,
            associative: None,
            relational: None,
            experts: None,
            consolidation: None,
            ..full.clone()
        };
        let no_blocks = Config { blocks: 0, split: 0, ..c.clone() };
        vec![
            ("transformer only", c.clone(), none.clone()),
            (
                "working memory only",
                no_blocks.clone(),
                NeuroSpec { memory: full.memory.clone(), ..none.clone() },
            ),
            (
                "associative only",
                no_blocks.clone(),
                NeuroSpec { associative: full.associative.clone(), ..none.clone() },
            ),
            (
                "relational only",
                no_blocks.clone(),
                NeuroSpec { relational: full.relational.clone(), ..none.clone() },
            ),
            ("experts only", no_blocks.clone(), NeuroSpec { experts: full.experts.clone(), ..none.clone() }),
            (
                "consolidation only",
                no_blocks.clone(),
                NeuroSpec { consolidation: full.consolidation.clone(), ..none.clone() },
            ),
            ("every network, no transformer", no_blocks, full.clone()),
            ("transformer and relational", c, NeuroSpec { relational: full.relational.clone(), ..none }),
        ]
    }
    fn jitter(m: &mut Bicameral, seed: u64) {
        // Move gates and biases off their initial constants so every path carries gradient.
        let mut x = seed;
        for w in m.weights.iter_mut() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *w += ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2;
        }
    }
    fn model(seed: u64) -> Bicameral {
        let (c, n) = tiny();
        let mut m = Bicameral::with_design(c, Some(n), seed);
        jitter(&mut m, seed);
        m
    }
    /// Analytic gradients of `m` against finite differences at a few positions of every Neuro
    /// tensor; returns how many were checked.
    fn check_gradients(m: &Bicameral, abl: Ablation) -> usize {
        let mask = [1., 0.5, 1., 1., 0.2, 1., 1., 1., 1.];
        let (_, g) = m.loss_and_gradients(&IDS, &TARGETS, &mask, abl).unwrap();
        let loss = |w: &[f32]| {
            let mut mm = m.clone();
            mm.weights.copy_from_slice(w);
            mm.loss(&IDS, &TARGETS, &mask, abl).unwrap() as f64
        };
        let biggest = g.iter().fold(0f32, |a, x| a.max(x.abs()));
        let mut checked = 0;
        for slot in &m.slots {
            if !(slot.name.starts_with("n.") || slot.name == "embed" || slot.name.starts_with("b2.")) {
                continue;
            }
            for k in [0, slot.range.len() / 2, slot.range.len() - 1] {
                let i = slot.range.start + k;
                let eps = 1e-2f32;
                let mut w = m.weights.clone();
                w[i] += eps;
                let up = loss(&w);
                w[i] -= 2. * eps;
                let down = loss(&w);
                let numeric = ((up - down) / (2. * eps as f64)) as f32;
                let err = (numeric - g[i]).abs();
                assert!(
                    err <= 2e-3 * biggest.max(1e-3) + 0.05 * g[i].abs(),
                    "{} [{k}]: analytic {} numeric {numeric} (largest gradient {biggest})",
                    slot.name,
                    g[i]
                );
                checked += 1;
            }
        }
        checked
    }
    const IDS: [TokenId; 9] = [3, 10, 17, 0, 7, 14, 21, 4, 33];
    const TARGETS: [TokenId; 9] = [10, 17, 0, 7, 14, 21, 4, 33, 2];

    #[test]
    fn gradients_match_finite_differences_through_every_module() {
        let m = model(11);
        let checked = check_gradients(&m, Ablation::default());
        assert!(checked > 100, "{checked}");
        // Every module receives gradient.
        let mask = [1., 0.5, 1., 1., 0.2, 1., 1., 1., 1.];
        let (_, g) = m.loss_and_gradients(&IDS, &TARGETS, &mask, Ablation::default()).unwrap();
        for name in [
            "n.wm0.z",
            "n.wm1.c",
            "n.wm_out",
            "n.as_q",
            "n.as_keys",
            "n.as_vals",
            "n.ctl1",
            "n.ctl2",
            "n.read",
            "n.write",
            "n.gate_w",
            "n.rel.addr",
            "n.rel.gate",
            "n.rel.val",
            "n.rel.q0",
            "n.rel.hop1",
            "n.rel.out",
            "n.ex.router",
            "n.ex0.gate",
            "n.ex2.down",
            "n.co.enc1",
            "n.co.enc2",
            "n.co.dec1",
            "n.co.dec2",
        ] {
            let r = m.slot(name).range.clone();
            assert!(g[r].iter().any(|x| x.abs() > 1e-7), "{name} gets no gradient");
        }
        // The designs with a single network, and the networks without a transformer.
        for (name, c, n) in singles() {
            let mut m = Bicameral::with_design(c, Some(n), 4);
            jitter(&mut m, 4);
            let checked = check_gradients(&m, Ablation::default());
            println!("{name}: {checked} gradients checked, {} parameters", m.parameter_count());
        }
    }

    #[test]
    fn cached_generation_matches_the_full_pass_with_every_ablation() {
        let m = model(5);
        for abl in [
            Ablation::default(),
            Ablation { no_memory: true, ..Ablation::default() },
            Ablation { no_associative: true, ..Ablation::default() },
            Ablation { no_controller: true, ..Ablation::default() },
            Ablation { one_round: true, ..Ablation::default() },
            Ablation { no_afferent: true, ..Ablation::default() },
            Ablation { no_efferent: true, ..Ablation::default() },
            Ablation { no_right: true, ..Ablation::default() },
            Ablation { no_relational: true, ..Ablation::default() },
            Ablation { no_experts: true, ..Ablation::default() },
            Ablation { no_consolidation: true, ..Ablation::default() },
        ] {
            let mut cache = KvCache::new(&m, abl);
            let mut prefix = Vec::new();
            // Longer than the context: the window is rebuilt (positions and memory restart).
            for &id in IDS.iter().chain(&IDS) {
                prefix.push(id);
                let got = cache.append(&m, id).unwrap();
                let start = prefix.len().saturating_sub(m.config.max_tokens);
                let full = m.logits(&prefix[start..], abl).unwrap();
                let want = &full[full.len() - m.config.vocab..];
                for (x, y) in got.iter().zip(want) {
                    assert!((x - y).abs() < 2e-4, "{abl:?} at {}: {x} vs {y}", prefix.len());
                }
            }
        }
        // The ablations really change the computation.
        let base = m.logits(&IDS, Ablation::default()).unwrap();
        for abl in [
            Ablation { no_memory: true, ..Ablation::default() },
            Ablation { no_associative: true, ..Ablation::default() },
            Ablation { no_controller: true, ..Ablation::default() },
            Ablation { one_round: true, ..Ablation::default() },
            Ablation { no_relational: true, ..Ablation::default() },
            Ablation { no_experts: true, ..Ablation::default() },
            Ablation { no_consolidation: true, ..Ablation::default() },
        ] {
            let cut = m.logits(&IDS, abl).unwrap();
            assert!(base.iter().zip(&cut).any(|(a, b)| (a - b).abs() > 1e-4), "{abl:?} has no effect");
        }
    }

    #[test]
    fn working_memory_never_sees_later_tokens() {
        // Changing a later token must not change any earlier position's output.
        let m = model(9);
        let a = m.logits(&IDS, Ablation::default()).unwrap();
        let mut ids = IDS;
        ids[6] = 30;
        let b = m.logits(&ids, Ablation::default()).unwrap();
        let v = m.config.vocab;
        assert_eq!(a[..6 * v], b[..6 * v], "positions before the change are identical");
        assert_ne!(a[6 * v..7 * v], b[6 * v..7 * v]);
    }

    #[test]
    fn checkpoints_round_trip_in_their_own_format() {
        let mut m = model(3);
        m.step = 17;
        let bytes = m.to_bytes().unwrap();
        assert_eq!(&bytes[..8], b"NNEURO1\0");
        let back = Bicameral::from_bytes(&bytes).unwrap();
        assert_eq!(back.neuro, m.neuro);
        assert_eq!(back.weights, m.weights);
        assert_eq!(back.step, 17);
        assert_eq!(back.to_bytes().unwrap(), bytes);
        for cut in [0, 7, 30, bytes.len() / 2, bytes.len() - 1] {
            assert!(Bicameral::from_bytes(&bytes[..cut]).is_err());
        }
        // A Bicameral model still writes `.nnova2` and has no Neuro design.
        let b = Bicameral::new();
        let bb = b.to_bytes().unwrap();
        assert_eq!(&bb[..6], b"NNOVA2");
        assert!(Bicameral::from_bytes(&bb).unwrap().neuro.is_none());
        // Growing the vocabulary keeps the design.
        let grown = m.with_word_capacity(m.word_capacity() + 4, 1).unwrap();
        assert_eq!(grown.neuro, m.neuro);
    }

    #[test]
    fn training_lowers_the_loss_and_the_modules_matter() {
        let (c, n) = tiny();
        let mut m = Bicameral::with_design(c, Some(n), 2);
        let mask = [1.; 9];
        let before = m.loss(&IDS, &TARGETS, &mask, Ablation::default()).unwrap();
        for _ in 0..60 {
            m.train_sequence(&IDS, &TARGETS, &mask, 3e-3, Focus::Joint).unwrap();
        }
        let after = m.loss(&IDS, &TARGETS, &mask, Ablation::default()).unwrap();
        assert!(after < before * 0.5, "{before} -> {after}");
        // Freezing the language core: only modules, bus and controller change.
        let core = m.group_values(Focus::Left);
        let modules = m.group_values(Focus::Right);
        let (_, g) = m.loss_and_gradients(&IDS, &TARGETS, &mask, Ablation::default()).unwrap();
        m.adam_step(&g, 1e-3, Focus::Right).unwrap();
        assert_eq!(m.group_values(Focus::Left), core);
        assert_ne!(m.group_values(Focus::Right), modules);
    }

    #[test]
    fn every_single_network_design_generates_like_its_full_pass_and_learns() {
        for (name, c, n) in singles() {
            n.validate(&c).unwrap();
            let mut m = Bicameral::with_design(c.clone(), Some(n.clone()), 6);
            jitter(&mut m, 6);
            // Generation with the cache equals the full pass, past the context window.
            let mut cache = KvCache::new(&m, Ablation::default());
            let mut prefix = Vec::new();
            for &id in IDS.iter().chain(&IDS) {
                prefix.push(id);
                let got = cache.append(&m, id).unwrap();
                let start = prefix.len().saturating_sub(m.config.max_tokens);
                let full = m.logits(&prefix[start..], Ablation::default()).unwrap();
                let want = &full[full.len() - m.config.vocab..];
                for (x, y) in got.iter().zip(want) {
                    assert!((x - y).abs() < 2e-4, "{name} at {}: {x} vs {y}", prefix.len());
                }
            }
            // A later token never changes an earlier position.
            let a = m.logits(&IDS, Ablation::default()).unwrap();
            let mut ids = IDS;
            ids[6] = 30;
            let b = m.logits(&ids, Ablation::default()).unwrap();
            let v = m.config.vocab;
            assert_eq!(a[..6 * v], b[..6 * v], "{name}");
            // It learns.
            let mut m = Bicameral::with_design(c, Some(n), 2);
            let mask = [1.; 9];
            let before = m.loss(&IDS, &TARGETS, &mask, Ablation::default()).unwrap();
            for _ in 0..80 {
                m.train_sequence(&IDS, &TARGETS, &mask, 3e-3, Focus::Joint).unwrap();
            }
            let after = m.loss(&IDS, &TARGETS, &mask, Ablation::default()).unwrap();
            println!("{name}: {before} -> {after}");
            assert!(after < before * 0.7, "{name}: {before} -> {after}");
            // The checkpoint keeps the design.
            let back = Bicameral::from_bytes(&m.to_bytes().unwrap()).unwrap();
            assert_eq!(back.neuro, m.neuro);
            assert_eq!(back.config.blocks, m.config.blocks);
        }
        // Designs without any network at all are refused.
        let (c, full) = tiny();
        let none = NeuroSpec {
            memory: None,
            associative: None,
            relational: None,
            experts: None,
            consolidation: None,
            ..full
        };
        assert!(none.validate(&Config { blocks: 0, split: 0, ..c }).is_err());
    }

    #[test]
    fn a_version_1_design_keeps_its_parameters() {
        // Neuro-Core models saved before Stage 4 load with the same slots, including a design
        // without networks (its bus is kept, unused, as it was saved).
        let (c, full) = tiny();
        let core = NeuroSpec { version: 1, relational: None, experts: None, consolidation: None, ..full };
        let bare = NeuroSpec { memory: None, associative: None, ..core.clone() };
        for n in [core, bare] {
            let m = Bicameral::with_design(c.clone(), Some(n.clone()), 1);
            assert!(m.slots.iter().any(|s| s.name == "n.read"));
            let back = Bicameral::from_bytes(&m.to_bytes().unwrap()).unwrap();
            assert_eq!(back.weights, m.weights);
            assert_eq!(back.neuro.unwrap().version, 1);
        }
        let v2 = NeuroSpec {
            version: VERSION,
            memory: None,
            associative: None,
            relational: None,
            experts: None,
            consolidation: None,
            ..tiny().1
        };
        let m = Bicameral::with_design(c, Some(v2), 1);
        assert!(!m.slots.iter().any(|s| s.name.starts_with("n.")), "no networks, no bus");
    }

    #[test]
    fn presets_are_exactly_counted_and_close_to_bicameral() {
        let bicameral: Vec<usize> =
            crate::model::size_presets().iter().map(|s| Bicameral::parameters_for(&s.config)).collect();
        for p in presets() {
            p.design.validate(&p.config).unwrap();
            let tensors = Bicameral::tensors_for(&p.config, Some(&p.design));
            let total: usize = tensors.iter().map(|(_, r, c)| r * c).sum();
            let m = Bicameral::with_design(p.config.clone(), Some(p.design.clone()), 1);
            assert_eq!(m.parameter_count(), total, "{}", p.id);
            // The Bicameral preset with the same transformer core.
            let twin = crate::model::size_presets()
                .iter()
                .position(|s| s.config.d == p.config.d && s.config.blocks == p.config.blocks)
                .map(|i| bicameral[i])
                .unwrap();
            let ratio = total as f64 / twin as f64;
            println!("{}: {total} parameters, Bicameral twin {twin} ({ratio:.3})", p.id);
            assert!((0.99..=1.01).contains(&ratio), "{}: {total} vs {twin}", p.id);
        }
    }
}
