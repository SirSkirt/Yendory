//! The Bicameral model (architecture "Nova-2 Bicameral"). Faithful Rust implementation of
//! Claude's NumPy reference (`reference/nova2_ref.py`), with three independently
//! parameterized pathways and one shared objective.
mod cache;
mod checkpoint;
#[cfg(test)]
mod golden;
pub mod grow;
pub mod inspect;
mod linalg;
mod measure;
pub mod neuro;
mod tape;
mod think;
pub use crate::tokenizer::TokenId;
use crate::tokenizer::Tokenizer;
pub use cache::{
    argmax, CacheStats, GateTrace, KvCache, NeuroRound, NeuroTrace, StageTrace, Trace, TRACE_PATTERNS,
};
pub use grow::{compare_outputs, Growth, GrowthOption, NotOffered, OutputComparison, Preservation};
pub use measure::{AblationEffects, Measurements, NeuroEffects};
pub use neuro::NeuroSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
pub(crate) use tape::Graph;
pub use think::{Branch, Choice, Readout, ThinkOptions, ThinkingTree, ThoughtStep};

/// Largest model the Engine creates or loads (1 billion parameters: 12 GB with the optimizer
/// state). A large vocabulary counts too: every word slot is one embedding row (`d` values)
/// plus an output bias. Checkpoints describing a bigger model are refused before anything is
/// allocated.
pub const MAX_PARAMETERS: usize = 1_000_000_000;
/// Largest vocabulary (token ids, including the 128 characters) a configuration may declare.
pub const MAX_VOCAB: usize = 1 << 28;
/// Longest context (tokens) a configuration may declare.
pub const MAX_CONTEXT: usize = 1 << 16;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub heads: usize,
    pub ff: usize,
    pub blocks: usize,
    pub split: usize,
    pub loops: usize,
    pub window_chars: usize,
    pub r1: usize,
    pub r2: usize,
    pub max_tokens: usize,
    pub rope_base: f32,
    pub gate_bias: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            vocab: 384,
            d: 128,
            heads: 4,
            ff: 288,
            blocks: 4,
            split: 2,
            loops: 2,
            window_chars: 64,
            r1: 256,
            r2: 192,
            max_tokens: 256,
            rope_base: 10_000.,
            gate_bias: -2.,
        }
    }
}
/// A model size offered when creating a model. Every preset keeps Nova-2's division of
/// parameters (about 64% transformer, 28.5% dense, 7.7% interconnect) and its tokenizer,
/// context length, thinking rounds and right-hemisphere character window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SizePreset {
    pub id: &'static str,
    pub name: &'static str,
    pub config: Config,
}

pub fn size_presets() -> Vec<SizePreset> {
    let shape = |blocks: usize, d: usize, ff: usize, r1: usize, r2: usize| Config {
        d,
        heads: d / 32,
        ff,
        blocks,
        split: blocks / 2,
        r1,
        r2,
        ..Config::default()
    };
    vec![
        SizePreset { id: "nova2", name: "Nova-2 size (1.2M)", config: Config::default() },
        SizePreset { id: "4m", name: "4M", config: shape(6, 192, 448, 672, 512) },
        SizePreset { id: "5m", name: "5M", config: shape(8, 192, 416, 608, 992) },
        SizePreset { id: "8m", name: "8M", config: shape(6, 256, 736, 1152, 768) },
        SizePreset { id: "15m", name: "15M", config: shape(8, 320, 768, 1664, 1248) },
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Focus {
    Joint,
    Left,
    Right,
    Interconnect,
    /// The dense hemisphere and the interconnect together (the transformer frozen): Dense
    /// learns, and the cable learns to carry what Dense sends. A training focus only; no
    /// parameter belongs to this group.
    DenseAndInterconnect,
}
impl Focus {
    fn matches(self, g: Focus) -> bool {
        match self {
            Focus::Joint => true,
            Focus::DenseAndInterconnect => g == Focus::Right || g == Focus::Interconnect,
            _ => self == g,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Ablation {
    pub no_afferent: bool,
    pub no_efferent: bool,
    pub no_right: bool,
    /// Nova Neuro: the working memory's output is cut.
    pub no_memory: bool,
    /// Nova Neuro: the associative memory's output is cut.
    pub no_associative: bool,
    /// Nova Neuro: every controller gate fully open (the controller's choices removed).
    pub no_controller: bool,
    /// Nova Neuro: one computation round instead of the model's rounds.
    pub one_round: bool,
    /// Nova Neuro: the relational network's output is cut.
    pub no_relational: bool,
    /// Nova Neuro: the experts' output is cut.
    pub no_experts: bool,
    /// Nova Neuro: consolidation's output is cut (the associative memory still reads its code).
    pub no_consolidation: bool,
    /// Training only: dropout on the transformer blocks' attention and feed-forward outputs.
    /// Never set when evaluating, measuring or generating.
    pub dropout: Option<Dropout>,
}

/// Dropout while training: each value of a transformer block's attention output and
/// feed-forward output is zeroed with probability `rate`, the others scaled by 1 / (1 - rate).
/// Whether a value is kept is a hash of (seed, site, row, position, column), so the CPU (one
/// row at a time), the GPU (a padded batch) and hybrid training drop the same values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dropout {
    pub rate: f32,
    pub seed: u64,
    /// The batch index of the first row of this forward pass.
    pub row: u64,
}

impl Ablation {
    /// The same ablation for the forward pass of the batch's row `offset` rows further on.
    pub fn at_row(mut self, offset: usize) -> Ablation {
        if let Some(d) = self.dropout.as_mut() {
            d.row += offset as u64;
        }
        self
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The dropout sites of one forward pass, numbered in the order the pass reaches them (the
/// same order on every device).
pub(crate) struct DropSites {
    dropout: Option<Dropout>,
    /// Tokens in this forward pass (rows x positions) and positions per row.
    tokens: usize,
    seq: usize,
    next: u64,
}

impl DropSites {
    pub(crate) fn new(abl: Ablation, tokens: usize, seq: usize) -> Self {
        DropSites { dropout: abl.dropout.filter(|d| d.rate > 0.), tokens, seq: seq.max(1), next: 0 }
    }
    /// `x` (tokens x `d`) with dropout applied (unchanged without dropout).
    pub(crate) fn apply<G: Graph>(&mut self, t: &mut G, x: usize, d: usize) -> usize {
        let site = self.next;
        self.next += 1;
        let Some(dp) = self.dropout else { return x };
        let scale = 1. / (1. - dp.rate);
        let threshold = (dp.rate as f64 * (1u64 << 32) as f64) as u64;
        let base = splitmix(dp.seed ^ splitmix(site.wrapping_add(0xD1B5_4A32_D192_ED03)));
        let mut mask = vec![0f32; self.tokens * d];
        for (tok, chunk) in mask.chunks_exact_mut(d).enumerate() {
            let row = dp.row + (tok / self.seq) as u64;
            let pos = (tok % self.seq) as u64;
            let key = splitmix(base ^ splitmix(row.wrapping_mul(0x1_0000_0001).wrapping_add(pos)));
            for (col, m) in chunk.iter_mut().enumerate() {
                let u = splitmix(key ^ col as u64) >> 32;
                *m = if u >= threshold { scale } else { 0. };
            }
        }
        let m = t.input(mask);
        t.mul(x, m)
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Slot {
    pub name: String,
    pub range: Range<usize>,
    pub rows: usize,
    pub cols: usize,
    pub group: Focus,
    pub decay: bool,
    pub init: Init,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Init {
    Normal(f32),
    Zero,
    One,
    Constant(f32),
}
fn spec(c: &Config, neuro: Option<&NeuroSpec>) -> Vec<Slot> {
    let mut out = Vec::new();
    let mut next = 0;
    let mut put = |name: String, rows: usize, cols: usize, group: Focus, init: Init, decay: bool| {
        let a = next;
        next += rows * cols;
        out.push(Slot { name, rows, cols, group, init, decay, range: a..next });
    };
    let scale = |width: usize| 0.5 / (width as f32).sqrt();
    let res = 1. / (2. * c.blocks as f32).sqrt();
    put("embed".into(), c.vocab, c.d, Focus::Left, Init::Normal(0.04), false);
    put("out_bias".into(), 1, c.vocab, Focus::Left, Init::Zero, false);
    for i in 0..c.blocks {
        let n = |name: &str| format!("b{i}.{name}");
        put(n("ln1"), 1, c.d, Focus::Left, Init::One, false);
        for suffix in ["wq", "wk", "wv"] {
            put(n(suffix), c.d, c.d, Focus::Left, Init::Normal(scale(c.d)), true);
        }
        put(n("wo"), c.d, c.d, Focus::Left, Init::Normal(scale(c.d) * res), true);
        put(n("ln2"), 1, c.d, Focus::Left, Init::One, false);
        for suffix in ["w_gate", "w_up"] {
            put(n(suffix), c.ff, c.d, Focus::Left, Init::Normal(scale(c.d)), true);
        }
        put(n("w_down"), c.d, c.ff, Focus::Left, Init::Normal(scale(c.ff) * res), true);
    }
    put("ln_f".into(), 1, c.d, Focus::Left, Init::One, false);
    if let Some(n) = neuro {
        let mut add =
            |name: String, rows, cols, group, init, decay| put(name, rows, cols, group, init, decay);
        neuro::slots(c, n, &mut add);
        return out;
    }
    put(
        "r.w1".into(),
        c.r1,
        c.window_chars * 16,
        Focus::Right,
        Init::Normal((2. / (c.window_chars * 16) as f32).sqrt() * 0.5),
        true,
    );
    put("r.b1".into(), 1, c.r1, Focus::Right, Init::Zero, false);
    put("r.w2".into(), c.r2, c.r1, Focus::Right, Init::Normal(scale(c.r1)), true);
    put("r.wl".into(), c.r2, c.d, Focus::Right, Init::Normal(scale(c.d)), true);
    put("r.b2".into(), 1, c.r2, Focus::Right, Init::Zero, false);
    put("c.n1".into(), 1, c.d, Focus::Interconnect, Init::One, false);
    put("c.a1".into(), c.d, c.r1, Focus::Interconnect, Init::Normal(scale(c.r1)), true);
    put("c.g1w".into(), c.d, c.d, Focus::Interconnect, Init::Normal(scale(c.d)), true);
    put("c.g1b".into(), 1, c.d, Focus::Interconnect, Init::Constant(c.gate_bias), false);
    put("c.n2".into(), 1, c.d, Focus::Interconnect, Init::One, false);
    put("c.a2".into(), c.d, c.r2, Focus::Interconnect, Init::Normal(scale(c.r2)), true);
    put("c.g2w".into(), c.d, c.d, Focus::Interconnect, Init::Normal(scale(c.d)), true);
    put("c.g2b".into(), 1, c.d, Focus::Interconnect, Init::Constant(c.gate_bias), false);
    out
}
#[derive(Clone)]
pub struct Bicameral {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub weights: Vec<f32>,
    pub momentum: Vec<f32>,
    pub variance: Vec<f32>,
    pub step: u64,
    pub(crate) slots: Vec<Slot>,
    pub(crate) offsets: HashMap<String, usize>,
    /// Config JSON exactly as read from a checkpoint, so an unchanged model saves
    /// byte-identically (the numpy reference and serde format the JSON differently).
    pub(crate) stored_config_json: Option<Vec<u8>>,
    /// Nova Neuro (experimental): the modules that replace the Dense hemisphere. `None` for
    /// Nova Bicameral, whose behaviour and checkpoints are unchanged.
    pub neuro: Option<NeuroSpec>,
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (((self.0 >> 11) as f64) / (1u64 << 53) as f64).clamp(1e-12, 1. - 1e-12) as f32
    }
    fn normal(&mut self) -> f32 {
        let a = self.next();
        let b = self.next();
        (-2. * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}
impl Bicameral {
    pub fn new() -> Self {
        Self::with_config(Config::default(), 0x4E_4F_56_41)
    }
    pub fn with_config(config: Config, seed: u64) -> Self {
        Self::with_design(config, None, seed)
    }
    /// A new model: Nova Bicameral (`neuro` None) or Nova Neuro with the given design.
    pub fn with_design(config: Config, neuro: Option<NeuroSpec>, seed: u64) -> Self {
        assert!(config.d % config.heads == 0 && config.d / config.heads % 2 == 0);
        assert!(config.loops > 0 && config.window_chars > 0);
        match &neuro {
            Some(n) => n.validate(&config).expect("a valid Nova Neuro design"),
            None => assert!(config.split < config.blocks),
        }
        let slots = spec(&config, neuro.as_ref());
        let n = slots.last().map(|x| x.range.end).unwrap_or(0);
        let mut weights = vec![0.; n];
        let mut rng = Rng(seed.max(1));
        for slot in &slots {
            for x in &mut weights[slot.range.clone()] {
                *x = match slot.init {
                    Init::Normal(std) => rng.normal() * std,
                    Init::Zero => 0.,
                    Init::One => 1.,
                    Init::Constant(v) => v,
                };
            }
        }
        let offsets = slots.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
        Self {
            config,
            tokenizer: Tokenizer::default(),
            weights,
            momentum: vec![0.; n],
            variance: vec![0.; n],
            step: 0,
            slots,
            offsets,
            stored_config_json: None,
            neuro,
        }
    }
    /// Parameters a configuration would have, without building the model.
    pub fn parameters_for(config: &Config) -> usize {
        spec(config, None).last().map(|s| s.range.end).unwrap_or(0)
    }
    /// Every parameter tensor a design would have, `(name, rows, columns)`, without building it
    /// (the exact parameter calculator).
    pub fn tensors_for(config: &Config, neuro: Option<&NeuroSpec>) -> Vec<(String, usize, usize)> {
        spec(config, neuro).into_iter().map(|s| (s.name, s.rows, s.cols)).collect()
    }
    /// The architecture: "bicameral" or "neuro".
    pub fn architecture(&self) -> &'static str {
        if self.neuro.is_some() {
            "neuro"
        } else {
            "bicameral"
        }
    }
    /// Whether a configuration is valid and within the Engine's limits (the same check a
    /// checkpoint passes before it is loaded). Call it before `with_config` for user choices.
    pub fn check_config(config: &Config) -> Result<(), String> {
        checkpoint::validate_config(config)
    }
    /// The same model with a different context length (`max_tokens`). Positions are encoded by
    /// rotation (RoPE), so no parameter depends on the context: every weight is kept. A model
    /// uses a longer context well only after training on text that long.
    pub fn with_context(&self, max_tokens: usize) -> Result<Bicameral, String> {
        let config = Config { max_tokens, ..self.config.clone() };
        checkpoint::validate_config(&config)?;
        let mut out = self.clone();
        out.config = config;
        out.stored_config_json = None;
        Ok(out)
    }
    /// Values the training forward pass stores for one token (activations kept for the
    /// backward pass), counted from the graph: per block execution (the lower blocks once, the
    /// upper blocks once per thinking round) 12·d + 4·ff, plus the right hemisphere's input
    /// and layers and the exchanges. Multiply by 4 bytes and the tokens in a training row; the
    /// output layer and attention weights are recomputed instead of stored.
    pub fn activation_floats_per_token(config: &Config) -> usize {
        Self::activation_floats_for(config, None)
    }
    /// [`Bicameral::activation_floats_per_token`] for either architecture.
    pub fn activation_floats_for(config: &Config, neuro: Option<&NeuroSpec>) -> usize {
        if let Some(n) = neuro {
            return neuro::activation_floats_per_token(config, n);
        }
        let c = config;
        let executions = c.split + (c.blocks - c.split) * c.loops;
        c.d + executions * (12 * c.d + 4 * c.ff)
            + c.window_chars * 16
            + 2 * c.r1
            + (c.loops + 1) * 6 * c.d
            + c.loops * (4 * c.r2 + c.d)
            + c.d
    }
    /// How many learned word units this model's vocabulary holds (`vocab` minus the 128
    /// character ids).
    pub fn word_capacity(&self) -> usize {
        self.config.vocab.saturating_sub(crate::tokenizer::BASE)
    }
    /// The same model with room for `words` learned word units: existing weights, optimizer
    /// state, step and words are kept; new word slots get fresh embedding rows (initialised like
    /// a new model's) and a zero output bias. Shrinking is allowed down to the words in use.
    pub fn with_word_capacity(&self, words: usize, seed: u64) -> Result<Bicameral, String> {
        if words < self.tokenizer.words().len() {
            return Err(format!(
                "The model already uses {} word units; the vocabulary cannot be smaller than that",
                self.tokenizer.words().len()
            ));
        }
        let config = Config { vocab: crate::tokenizer::BASE + words, ..self.config.clone() };
        checkpoint::validate_config(&config)?;
        let mut out = Bicameral::with_design(config, self.neuro.clone(), seed);
        out.momentum.fill(0.);
        out.variance.fill(0.);
        for slot in &self.slots {
            let new = out.slot(&slot.name).range.clone();
            // Rows of `embed` and entries of `out_bias` are word ids: copy the ids both have.
            let n = slot.range.len().min(new.len());
            let (src, dst) = (slot.range.start..slot.range.start + n, new.start..new.start + n);
            out.weights[dst.clone()].copy_from_slice(&self.weights[src.clone()]);
            out.momentum[dst.clone()].copy_from_slice(&self.momentum[src.clone()]);
            out.variance[dst].copy_from_slice(&self.variance[src]);
        }
        out.step = self.step;
        out.tokenizer = self.tokenizer.clone();
        Ok(out)
    }
    pub fn parameter_count(&self) -> usize {
        self.weights.len()
    }
    /// The current values of one parameter group (Left, Right or Interconnect), in layout order.
    pub fn group_values(&self, group: Focus) -> Vec<f32> {
        self.slots
            .iter()
            .filter(|s| s.group == group)
            .flat_map(|s| self.weights[s.range.clone()].to_vec())
            .collect()
    }
    pub fn group_parameters(&self, focus: Focus) -> usize {
        self.slots.iter().filter(|s| s.group == focus).map(|s| s.range.len()).sum()
    }
    fn slot(&self, name: &str) -> &Slot {
        &self.slots[*self.offsets.get(name).expect("missing Nova-2 parameter")]
    }
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    /// Where a named parameter matrix lies in the flat weight array.
    pub fn slot_range(&self, name: &str) -> Range<usize> {
        self.slot(name).range.clone()
    }
    pub(crate) fn val(&self, name: &str) -> &[f32] {
        &self.weights[self.slot(name).range.clone()]
    }
    fn param<G: Graph>(&self, t: &mut G, name: &str) -> usize {
        t.parameter(name, self.slot(name).range.clone())
    }
    fn norm<G: Graph>(&self, t: &mut G, h: usize, name: &str) -> usize {
        let g = self.param(t, name);
        t.norm(h, g, self.config.d)
    }
    fn lin<G: Graph>(&self, t: &mut G, h: usize, name: &str, bias: Option<&str>) -> usize {
        let slot = self.slot(name);
        let w = self.param(t, name);
        let b = bias.map(|n| self.param(t, n));
        t.linear(h, w, b, slot.cols, slot.rows)
    }
    fn block<G: Graph>(&self, t: &mut G, i: usize, h: usize, tokens: usize, drops: &mut DropSites) -> usize {
        let c = &self.config;
        let p = |n: &str| format!("b{i}.{n}");
        let x = self.norm(t, h, &p("ln1"));
        let q = self.lin(t, x, &p("wq"), None);
        let k = self.lin(t, x, &p("wk"), None);
        let v = self.lin(t, x, &p("wv"), None);
        let q = t.rope_dim(q, c.heads, c.d, c.rope_base);
        let k = t.rope_dim(k, c.heads, c.d, c.rope_base);
        let a = t.attention(q, k, v, tokens, c.d, c.heads);
        let a = self.lin(t, a, &p("wo"), None);
        let a = drops.apply(t, a, c.d);
        let h = t.add(h, a);
        let x = self.norm(t, h, &p("ln2"));
        let gate = self.lin(t, x, &p("w_gate"), None);
        let gate = t.silu(gate);
        let up = self.lin(t, x, &p("w_up"), None);
        let f = t.mul(gate, up);
        let down = self.lin(t, f, &p("w_down"), None);
        let down = drops.apply(t, down, c.d);
        t.add(h, down)
    }
    fn efferent<G: Graph>(&self, t: &mut G, z1: usize, h: usize, abl: Ablation) -> usize {
        let pre = self.lin(t, z1, "r.w2", Some("r.b2"));
        let pre = if abl.no_efferent {
            pre
        } else {
            let hn = self.norm(t, h, "c.n1");
            let from_left = self.lin(t, hn, "r.wl", None);
            t.add(pre, from_left)
        };
        let z = t.silu(pre);
        if abl.no_right {
            t.zeros_like(z)
        } else {
            z
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn inject<G: Graph>(&self, t: &mut G, h: usize, z: usize, n: &str, w: &str, b: &str, a: &str) -> usize {
        let hn = self.norm(t, h, n);
        let gate = self.lin(t, hn, w, Some(b));
        let gate = t.sigmoid(gate);
        let zproj = self.lin(t, z, a, None);
        let fused = t.mul(gate, zproj);
        t.add(h, fused)
    }
    fn forward(&self, t: &mut tape::Tape, ids: &[TokenId], feats: Vec<f32>, abl: Ablation) -> usize {
        let h = self.forward_hidden(t, ids, feats, ids.len(), abl);
        let embed = self.param(t, "embed");
        let bias = self.param(t, "out_bias");
        t.linear(h, embed, Some(bias), self.config.d, self.config.vocab)
    }
    /// Everything up to the output layer: the final normalised left state of every position.
    /// `ids` may hold several sequences of `seq` tokens one after another (the GPU trains a
    /// batch at once); attention and positions are per sequence.
    pub(crate) fn forward_hidden<G: Graph>(
        &self,
        t: &mut G,
        ids: &[TokenId],
        feats: Vec<f32>,
        seq: usize,
        abl: Ablation,
    ) -> usize {
        let c = &self.config;
        let tok = ids.iter().map(|&v| v as usize).collect::<Vec<_>>();
        let mut drops = DropSites::new(abl, tok.len(), seq);
        if self.neuro.is_some() {
            return self.neuro_hidden(t, &tok, seq, abl, &mut drops);
        }
        let embed = self.param(t, "embed");
        let mut h = t.embed(&tok, embed, c.d);
        for i in 0..c.split {
            h = self.block(t, i, h, seq, &mut drops);
        }
        let feat = t.input(feats);
        let z1 = self.lin(t, feat, "r.w1", Some("r.b1"));
        let z1 = t.silu(z1);
        let z1 = if abl.no_right { t.zeros_like(z1) } else { z1 };
        if !abl.no_afferent {
            h = self.inject(t, h, z1, "c.n1", "c.g1w", "c.g1b", "c.a1");
        }
        let mut z2 = self.efferent(t, z1, h, abl);
        for round in 0..c.loops {
            if !abl.no_afferent {
                h = self.inject(t, h, z2, "c.n2", "c.g2w", "c.g2b", "c.a2");
            }
            for i in c.split..c.blocks {
                h = self.block(t, i, h, seq, &mut drops);
            }
            if round + 1 < c.loops {
                z2 = self.efferent(t, z1, h, abl);
            }
        }
        self.norm(t, h, "ln_f")
    }
    pub fn features(&self, ids: &[TokenId]) -> Result<Vec<f32>, String> {
        let c = &self.config;
        let mut decoded = Vec::new();
        let mut out = Vec::with_capacity(ids.len() * c.window_chars * 16);
        for &id in ids {
            if (id as usize) < 128 {
                decoded.push(id as u8);
            } else {
                let w = self.tokenizer.words().get(id as usize - 128).ok_or("Unknown word token")?;
                decoded.extend_from_slice(w.as_bytes());
            }
            let recent = &decoded[decoded.len().saturating_sub(c.window_chars)..];
            let mut f = vec![0f32; c.window_chars * 16];
            let start = c.window_chars - recent.len();
            for (i, &ch) in recent.iter().enumerate() {
                let idx = (start + i) * 16;
                for bit in 0..7 {
                    f[idx + bit] = if (ch >> bit) & 1 == 1 { 1. } else { -1. };
                }
                f[idx + 7] = f32::from(ch.is_ascii_uppercase());
                f[idx + 8] = f32::from(ch.is_ascii_lowercase());
                f[idx + 9] = f32::from(ch.is_ascii_digit());
                f[idx + 10] = f32::from(matches!(ch, b' ' | b'\t' | b'\r'));
                f[idx + 11] = f32::from(ch.is_ascii_punctuation());
                f[idx + 12] = f32::from(b"aeiouAEIOU".contains(&ch));
                f[idx + 13] = f32::from(matches!(ch, b'.' | b'!' | b'?' | b'\n'));
                f[idx + 14] = f32::from(ch == b'\n');
                f[idx + 15] = 1.;
            }
            out.extend(f);
        }
        Ok(out)
    }
    pub fn logits(&self, ids: &[TokenId], abl: Ablation) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("Nova-2 expects at least one token".into());
        }
        if ids.len() > self.config.max_tokens {
            return Err("Context exceeds Nova-2 window".into());
        }
        let f = self.features(ids)?;
        self.logits_with_features(ids, &f, abl)
    }
    pub fn logits_with_features(
        &self,
        ids: &[TokenId],
        features: &[f32],
        abl: Ablation,
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty()
            || ids.len() > self.config.max_tokens
            || features.len() != ids.len() * self.config.window_chars * 16
        {
            return Err("Invalid Nova-2 inference features".into());
        }
        let mut t = tape::Tape::new(&self.weights, None);
        let node = self.forward(&mut t, ids, features.to_vec(), abl);
        Ok(t.values(node).to_vec())
    }
    pub fn predict(&self, ids: &[TokenId], abl: Ablation) -> Result<Vec<f32>, String> {
        let z = self.logits(ids, abl)?;
        let last = &z[z.len() - self.config.vocab..];
        let mut p = last.to_vec();
        let valid = (128 + self.tokenizer.words().len()).min(self.config.vocab);
        for item in p.iter_mut().skip(valid) {
            *item = f32::NEG_INFINITY;
        }
        let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for x in &mut p {
            *x = (*x - max).exp();
            sum += *x;
        }
        for x in &mut p {
            *x /= sum;
        }
        Ok(p)
    }
    pub fn loss_and_gradients(
        &self,
        ids: &[TokenId],
        targets: &[TokenId],
        mask: &[f32],
        abl: Ablation,
    ) -> Result<(f32, Vec<f32>), String> {
        if ids.len() != targets.len()
            || ids.len() != mask.len()
            || ids.is_empty()
            || ids.len() > self.config.max_tokens
        {
            return Err("Nova-2 training dimensions or context invalid".into());
        }
        if targets.iter().any(|&x| x as usize >= self.config.vocab) {
            return Err("Invalid target token".into());
        }
        let feats = self.features(ids)?;
        self.loss_and_gradients_with_features(ids, &feats, targets, mask, abl)
    }
    pub fn loss_and_gradients_with_features(
        &self,
        ids: &[TokenId],
        features: &[f32],
        targets: &[TokenId],
        mask: &[f32],
        abl: Ablation,
    ) -> Result<(f32, Vec<f32>), String> {
        if ids.is_empty()
            || ids.len() != targets.len()
            || ids.len() != mask.len()
            || ids.len() > self.config.max_tokens
            || features.len() != ids.len() * self.config.window_chars * 16
        {
            return Err("Invalid Nova-2 training dimensions".into());
        }
        let mut grads = vec![0.; self.weights.len()];
        let value = self.add_gradients(ids, features.to_vec(), targets, mask, abl, &mut grads, 1.);
        Ok((value, grads))
    }
    /// The row's loss as [`Bicameral::loss_and_gradients`] computes it, without the backward
    /// pass (held-out checks and scores need only the loss: about a third of the work).
    pub fn loss(
        &self,
        ids: &[TokenId],
        targets: &[TokenId],
        mask: &[f32],
        abl: Ablation,
    ) -> Result<f32, String> {
        if ids.len() != targets.len()
            || ids.len() != mask.len()
            || ids.is_empty()
            || ids.len() > self.config.max_tokens
        {
            return Err("Nova-2 training dimensions or context invalid".into());
        }
        if targets.iter().any(|&x| x as usize >= self.config.vocab) {
            return Err("Invalid target token".into());
        }
        let feats = self.features(ids)?;
        let mut t = tape::Tape::new(&self.weights, None);
        let h = self.forward_hidden(&mut t, ids, feats, ids.len(), abl);
        let targets = targets.iter().map(|&v| v as usize).collect::<Vec<_>>();
        let (embed, bias) = (self.param(&mut t, "embed"), self.param(&mut t, "out_bias"));
        let loss = t.head_xent(h, embed, bias, self.config.d, self.config.vocab, &targets, mask);
        Ok(t.values(loss)[0])
    }
    /// Like [`Bicameral::loss_and_gradients`], but ADDS `scale ×` the gradient into `grads`
    /// (one buffer reused across rows) and returns the row's loss.
    pub fn accumulate_loss_and_gradients(
        &self,
        ids: &[TokenId],
        targets: &[TokenId],
        mask: &[f32],
        abl: Ablation,
        grads: &mut [f32],
        scale: f32,
    ) -> Result<f32, String> {
        if ids.len() != targets.len()
            || ids.len() != mask.len()
            || ids.is_empty()
            || ids.len() > self.config.max_tokens
            || grads.len() != self.weights.len()
        {
            return Err("Nova-2 training dimensions or context invalid".into());
        }
        if targets.iter().any(|&x| x as usize >= self.config.vocab) {
            return Err("Invalid target token".into());
        }
        let feats = self.features(ids)?;
        Ok(self.add_gradients(ids, feats, targets, mask, abl, grads, scale))
    }
    fn add_gradients(
        &self,
        ids: &[TokenId],
        features: Vec<f32>,
        targets: &[TokenId],
        mask: &[f32],
        abl: Ablation,
        grads: &mut [f32],
        scale: f32,
    ) -> f32 {
        let mut t = tape::Tape::new(&self.weights, Some(grads));
        let h = self.forward_hidden(&mut t, ids, features, ids.len(), abl);
        let targets = targets.iter().map(|&v| v as usize).collect::<Vec<_>>();
        let (embed, bias) = (self.param(&mut t, "embed"), self.param(&mut t, "out_bias"));
        let loss = t.head_xent(h, embed, bias, self.config.d, self.config.vocab, &targets, mask);
        let value = t.values(loss)[0];
        t.backward(loss, scale);
        value
    }
    /// A model `alpha` of the way from `self` to `other` (0 = `self`, 1 = `other`): weights and
    /// the AdamW moments interpolated alike (both moments stay valid: the second is an average
    /// of non-negative values), the later step count, and `other`'s tokenizer and design. The
    /// two must be the same network: configuration, design, vocabulary and parameter layout.
    pub fn blend_toward(&self, other: &Bicameral, alpha: f32) -> Result<Bicameral, String> {
        if !(0. ..=1.).contains(&alpha) {
            return Err("The blend share must be between 0 and 1".into());
        }
        let same_layout = self.slots.len() == other.slots.len()
            && self.slots.iter().zip(&other.slots).all(|(a, b)| a.name == b.name && a.range == b.range);
        if self.config != other.config
            || self.neuro != other.neuro
            || self.tokenizer.words() != other.tokenizer.words()
            || !same_layout
            || self.weights.len() != other.weights.len()
        {
            return Err("Only two versions of the same network can be blended".into());
        }
        let mix = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(x, y)| (1. - alpha) * x + alpha * y).collect()
        };
        let mut out = other.clone();
        out.weights = mix(&self.weights, &other.weights);
        out.momentum = mix(&self.momentum, &other.momentum);
        out.variance = mix(&self.variance, &other.variance);
        out.step = self.step.max(other.step);
        out.stored_config_json = None;
        Ok(out)
    }
    /// One AdamW step on the parameters `focus` selects. Returns the norm of the gradients it
    /// applied (the focused groups'), by which they were clipped to at most 1.
    pub fn adam_step(&mut self, gradients: &[f32], lr: f32, focus: Focus) -> Result<f32, String> {
        if gradients.len() != self.weights.len() || !lr.is_finite() || lr <= 0. {
            return Err("Invalid Nova-2 optimizer inputs".into());
        }
        // Clipped by the norm of the gradients this step applies: the focused groups only, so
        // a frozen group's gradient cannot shrink the update. With every group (Joint) this is
        // the whole gradient, summed in the same order as always.
        let mut squares = 0f64;
        for slot in &self.slots {
            if focus.matches(slot.group) {
                for &g in &gradients[slot.range.clone()] {
                    squares += (g as f64) * (g as f64);
                }
            }
        }
        let norm = squares.sqrt() as f32;
        let clip = 1. / norm.max(1.);
        self.step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.95f32;
        let bias1 = 1. - beta1.powf(self.step as f32);
        let bias2 = 1. - beta2.powf(self.step as f32);
        for slot in &self.slots {
            if !focus.matches(slot.group) {
                continue;
            }
            for i in slot.range.clone() {
                let g = gradients[i] * clip;
                self.momentum[i] = beta1 * self.momentum[i] + (1. - beta1) * g;
                self.variance[i] = beta2 * self.variance[i] + (1. - beta2) * g * g;
                let m = self.momentum[i] / bias1;
                let v = self.variance[i] / bias2;
                let wd = if slot.decay { 0.01 * self.weights[i] } else { 0. };
                self.weights[i] -= lr * (m / (v.sqrt() + 1e-8) + wd);
            }
        }
        Ok(norm)
    }
    pub fn train_sequence(
        &mut self,
        ids: &[TokenId],
        targets: &[TokenId],
        mask: &[f32],
        rate: f32,
        focus: Focus,
    ) -> Result<f32, String> {
        let (loss, g) = self.loss_and_gradients(ids, targets, mask, Ablation::default())?;
        self.adam_step(&g, rate, focus)?;
        Ok(loss)
    }
    pub fn train_text(&mut self, text: &str, rate: f32, focus: Focus) -> Result<f32, String> {
        let tokens = self.tokenizer.encode(text)?;
        if tokens.len() < 2 {
            return Err("Training text needs at least two tokens".into());
        }
        let seq = &tokens[tokens.len().saturating_sub(self.config.max_tokens + 1)..];
        let mask = vec![1.; seq.len() - 1];
        self.train_sequence(&seq[..seq.len() - 1], &seq[1..], &mask, rate, focus)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Blending checks it is given two versions of one network, interpolates weights and AdamW
    /// state, keeps the later step, and the result trains and survives a checkpoint round trip.
    #[test]
    fn blending_keeps_checkpoint_and_optimizer_state_valid() {
        let neuro = super::neuro::presets().into_iter().next().unwrap();
        for start in [
            Bicameral::with_config(Config::default(), 3),
            Bicameral::with_design(neuro.config, Some(neuro.design), 3),
        ] {
            let mut trained = start.clone();
            for _ in 0..3 {
                trained.train_text("U: Where is the key?\nN: In the box.\n", 3e-3, Focus::Joint).unwrap();
            }
            assert_eq!(start.blend_toward(&trained, 0.).unwrap().weights, start.weights);
            let one = start.blend_toward(&trained, 1.).unwrap();
            assert_eq!(one.weights, trained.weights);
            assert_eq!(one.momentum, trained.momentum);
            assert_eq!(one.variance, trained.variance);
            let half = start.blend_toward(&trained, 0.5).unwrap();
            assert_eq!(half.step, trained.step);
            for i in [0, start.weights.len() / 2, start.weights.len() - 1] {
                assert!((half.weights[i] - (start.weights[i] + trained.weights[i]) / 2.).abs() < 1e-6);
            }
            assert!(half.variance.iter().all(|v| v.is_finite() && *v >= 0.));
            assert!(half.momentum.iter().all(|m| m.is_finite()));
            // A checkpoint round trip keeps everything; training continues from it.
            let back = Bicameral::from_bytes(&half.to_bytes().unwrap()).unwrap();
            assert_eq!(back.weights, half.weights);
            assert_eq!(back.momentum, half.momentum);
            assert_eq!(back.variance, half.variance);
            assert_eq!(back.step, half.step);
            assert_eq!(back.neuro, half.neuro);
            let mut next = back.clone();
            let loss = next.train_text("U: Where is the key?\nN: In the box.\n", 3e-3, Focus::Joint).unwrap();
            assert!(loss.is_finite() && next.step == half.step + 1);
            // Different networks are refused.
            assert!(start
                .blend_toward(&start.with_word_capacity(start.word_capacity() + 8, 1).unwrap(), 0.5)
                .is_err());
            assert!(start.blend_toward(&trained, 1.5).is_err());
        }
        let other = Bicameral::with_config(Config { blocks: 3, split: 1, ..Config::default() }, 3);
        assert!(Bicameral::with_config(Config::default(), 3).blend_toward(&other, 0.5).is_err());
    }

    fn dropout(rate: f32, seed: u64) -> Ablation {
        Ablation { dropout: Some(Dropout { rate, seed, row: 0 }), ..Ablation::default() }
    }

    /// Dropout drops about its rate of values (scaling the rest), changes the loss, is the same
    /// for the same seed, and a rate of 0 is exactly no dropout.
    #[test]
    fn dropout_masks_and_off_switch() {
        let m = Bicameral::with_config(Config::default(), 4);
        let ids = m.tokenizer.encode("U: Where is the key?\nN: In the box.\n").unwrap();
        let (x, y) = (&ids[..ids.len() - 1], &ids[1..]);
        let mask = vec![1.; x.len()];
        let plain = m.loss_and_gradients(x, y, &mask, Ablation::default()).unwrap();
        assert_eq!(m.loss_and_gradients(x, y, &mask, dropout(0., 5)).unwrap(), plain, "rate 0 is off");
        let a = m.loss_and_gradients(x, y, &mask, dropout(0.1, 5)).unwrap();
        assert_ne!(a.0, plain.0);
        assert_eq!(m.loss_and_gradients(x, y, &mask, dropout(0.1, 5)).unwrap(), a, "same seed, same mask");
        assert_ne!(m.loss_and_gradients(x, y, &mask, dropout(0.1, 6)).unwrap().0, a.0);
        // The share of dropped values.
        let mut t = tape::Tape::new(&m.weights, None);
        let mut sites = DropSites::new(dropout(0.1, 9), 4000, 200);
        let input = t.input(vec![1.; 4000 * 16]);
        let out = sites.apply(&mut t, input, 16);
        let v = t.values(out);
        let dropped = v.iter().filter(|&&x| x == 0.).count() as f64 / v.len() as f64;
        assert!((dropped - 0.1).abs() < 0.01, "{dropped}");
        assert!(v.iter().all(|&x| x == 0. || (x - 1. / 0.9).abs() < 1e-6));
    }

    /// With a fixed mask the loss is an ordinary function of the weights: the gradient matches
    /// finite differences.
    #[test]
    fn dropout_gradients_match_finite_differences() {
        let m = Bicameral::with_config(Config::default(), 4);
        let ids = m.tokenizer.encode("U: hi\nN: Hello!\n").unwrap();
        let (x, y) = (&ids[..ids.len() - 1], &ids[1..]);
        let mask = vec![1.; x.len()];
        let abl = dropout(0.2, 3);
        let (_, g) = m.loss_and_gradients(x, y, &mask, abl).unwrap();
        let mut rng = crate::learn::Rng::new(2);
        let mut checked = 0;
        for name in ["b0.wo", "b0.w_down", "b3.wv", "c.a1"] {
            let r = m.slot(name).range.clone();
            for _ in 0..4 {
                let i = r.start + rng.below(r.len());
                let e = 1e-2;
                let (mut p, mut q) = (m.clone(), m.clone());
                p.weights[i] += e;
                q.weights[i] -= e;
                let fd = (p.loss(x, y, &mask, abl).unwrap() as f64
                    - q.loss(x, y, &mask, abl).unwrap() as f64)
                    / (2. * e as f64);
                let scale = fd.abs().max(g[i].abs() as f64).max(1e-3);
                assert!((fd - g[i] as f64).abs() < 0.05 * scale, "{name}[{i}]: {fd} vs {}", g[i]);
                checked += 1;
            }
        }
        assert_eq!(checked, 16);
    }

    /// A row's mask depends on its place in the batch, not on how the batch is computed: one row
    /// alone at offset k sees the values row k of a padded batch sees.
    #[test]
    fn dropout_masks_follow_the_row() {
        let m = Bicameral::with_config(Config::default(), 4);
        let (seq, d, rows) = (7usize, 8usize, 3usize);
        let mut t = tape::Tape::new(&m.weights, None);
        let all = t.input(vec![1.; rows * seq * d]);
        let batch = DropSites::new(dropout(0.3, 11), rows * seq, seq).apply(&mut t, all, d);
        let batch = t.values(batch).to_vec();
        for r in 0..rows {
            let one = t.input(vec![1.; seq * d]);
            let alone = DropSites::new(dropout(0.3, 11).at_row(r), seq, seq).apply(&mut t, one, d);
            assert_eq!(t.values(alone), &batch[r * seq * d..(r + 1) * seq * d]);
        }
    }

    /// Gradient clipping uses the norm of the gradients a step applies. A frozen group's
    /// gradient, however large, does not change a focused step; a whole-brain (Joint) step is
    /// exactly the original whole-gradient clipping.
    #[test]
    fn clipping_uses_only_the_trained_groups() {
        let neuro = super::neuro::presets().into_iter().next().unwrap();
        let models = [
            Bicameral::with_config(Config::default(), 3),
            Bicameral::with_design(neuro.config, Some(neuro.design), 3),
        ];
        for base in models {
            let n = base.parameter_count();
            let mut rng = crate::learn::Rng::new(11);
            // Large enough that clipping is active for every focus.
            let grads: Vec<f32> = (0..n).map(|_| (rng.unit() as f32 - 0.5) * 0.2).collect();
            for focus in [Focus::Left, Focus::Right, Focus::Interconnect, Focus::DenseAndInterconnect] {
                let trained =
                    |i: usize| base.slots.iter().any(|s| s.range.contains(&i) && focus.matches(s.group));
                let applied: Vec<f32> = (0..n).map(|i| if trained(i) { grads[i] } else { 0. }).collect();
                let inflated: Vec<f32> =
                    (0..n).map(|i| if trained(i) { grads[i] } else { grads[i] * 1e3 }).collect();
                let (mut a, mut b) = (base.clone(), base.clone());
                let norm = a.adam_step(&applied, 1e-3, focus).unwrap();
                assert!(norm > 1., "{focus:?}: clipping is active ({norm})");
                assert_eq!(b.adam_step(&inflated, 1e-3, focus).unwrap(), norm);
                assert_eq!(a.weights, b.weights, "{focus:?}: frozen gradients changed the step");
                assert_eq!(a.momentum, b.momentum);
                assert_eq!(a.variance, b.variance);
                for (i, (w, w0)) in a.weights.iter().zip(&base.weights).enumerate() {
                    if !trained(i) {
                        assert_eq!(w, w0, "{focus:?}: a frozen weight changed");
                    }
                }
            }
            // Joint: the original rule (norm of the whole gradient), bit for bit.
            let mut joint = base.clone();
            joint.adam_step(&grads, 1e-3, Focus::Joint).unwrap();
            let mut reference = base.clone();
            let norm = grads.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>().sqrt() as f32;
            assert!(norm > 1.);
            let clip = 1. / norm.max(1.);
            reference.step += 1;
            let (bias1, bias2) =
                (1. - 0.9f32.powf(reference.step as f32), 1. - 0.95f32.powf(reference.step as f32));
            for slot in &base.slots {
                for i in slot.range.clone() {
                    let g = grads[i] * clip;
                    reference.momentum[i] = 0.9 * reference.momentum[i] + (1. - 0.9) * g;
                    reference.variance[i] = 0.95 * reference.variance[i] + (1. - 0.95) * g * g;
                    let (m, v) = (reference.momentum[i] / bias1, reference.variance[i] / bias2);
                    let wd = if slot.decay { 0.01 * reference.weights[i] } else { 0. };
                    reference.weights[i] -= 1e-3 * (m / (v.sqrt() + 1e-8) + wd);
                }
            }
            assert_eq!(joint.weights, reference.weights, "Joint steps are unchanged");
            assert_eq!(joint.momentum, reference.momentum);
            assert_eq!(joint.variance, reference.variance);
        }
    }
    #[test]
    fn fused_output_layer_equals_linear_then_cross_entropy() {
        let mut m = Bicameral::new();
        m.tokenizer.set_words(vec!["the".into(), "key".into(), "box".into()]);
        for _ in 0..2 {
            m.train_text("U: Where is the key?\nN: In the box.\n", 1e-2, Focus::Joint).unwrap();
        }
        let ids =
            m.tokenizer.encode("U: The key is in the box. Where is the key?\nN: In the box.\n").unwrap();
        let (x, y) = (&ids[..ids.len() - 1], &ids[1..]);
        let mask: Vec<f32> = (0..x.len()).map(|i| if i % 3 == 0 { 0.15 } else { 1. }).collect();
        let feats = m.features(x).unwrap();
        let (fused, gf) = m.loss_and_gradients(x, y, &mask, Ablation::default()).unwrap();
        let mut gu = vec![0f32; m.weights.len()];
        let mut t = tape::Tape::new(&m.weights, Some(&mut gu));
        let logits = m.forward(&mut t, x, feats, Ablation::default());
        let targets: Vec<usize> = y.iter().map(|&v| v as usize).collect();
        let loss = t.xent(logits, &targets, &mask, m.config.vocab);
        let unfused = t.values(loss)[0];
        t.backward(loss, 1.);
        assert!((fused - unfused).abs() < 1e-5, "{fused} vs {unfused}");
        let scale = gu.iter().fold(0f32, |a, g| a.max(g.abs()));
        let worst = gf.iter().zip(&gu).fold(0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst <= 1e-5 * scale, "gradient difference {worst} (largest {scale})");
    }
    #[test]
    fn size_and_groups() {
        let m = Bicameral::new();
        assert_eq!(m.parameter_count(), 1_182_144);
        assert_eq!(m.group_parameters(Focus::Left), 755_200);
        assert_eq!(m.group_parameters(Focus::Right), 336_320);
        assert_eq!(m.group_parameters(Focus::Interconnect), 90_624);
    }
    #[test]
    fn small_training_and_inference() {
        let mut c = Config::default();
        c.vocab = 24;
        c.d = 16;
        c.heads = 2;
        c.ff = 24;
        c.blocks = 3;
        c.split = 1;
        c.window_chars = 4;
        c.r1 = 12;
        c.r2 = 10;
        c.max_tokens = 16;
        let mut m = Bicameral::with_config(c, 42);
        assert_eq!(m.parameter_count(), 9046);
        let ids = [3, 10, 17, 0, 7, 14, 21];
        let targets = [1, 6, 11, 16, 21, 2, 7];
        let weights = [1., 0.5, 1., 0., 1., 0.5, 1.];
        let (loss, g) = m.loss_and_gradients(&ids, &targets, &weights, Ablation::default()).unwrap();
        assert!(loss.is_finite() && g.iter().all(|g| g.is_finite()));
        assert_eq!(g.len(), 9046);
        let s = m.step;
        m.adam_step(&g, 0.0001, Focus::Joint).unwrap();
        assert_eq!(m.step, s + 1);
    }
}
