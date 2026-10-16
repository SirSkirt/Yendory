//! The Bicameral model (architecture "Nova-2 Bicameral"). Faithful Rust implementation of
//! Claude's NumPy reference (`reference/nova2_ref.py`), with three independently
//! parameterized pathways and one shared objective.
mod cache;
mod checkpoint;
#[cfg(test)]
mod golden;
mod linalg;
mod measure;
mod tape;
mod think;
use crate::tokenizer::Tokenizer;
pub use cache::{CacheStats, GateTrace, KvCache, StageTrace, Trace};
pub use measure::{AblationEffects, Measurements};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
pub use think::{Branch, Choice, Readout, ThinkOptions, ThinkingTree, ThoughtStep};

/// Largest model the Engine creates or loads (the 15M size preset fits with room to spare).
/// Checkpoints describing a bigger model are refused before anything is allocated.
pub const MAX_PARAMETERS: usize = 16_000_000;
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
}
impl Focus {
    fn matches(self, g: Focus) -> bool {
        self == Focus::Joint || self == g
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Ablation {
    pub no_afferent: bool,
    pub no_efferent: bool,
    pub no_right: bool,
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
fn spec(c: &Config) -> Vec<Slot> {
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
        assert!(config.d % config.heads == 0 && config.d / config.heads % 2 == 0);
        assert!(config.split < config.blocks && config.loops > 0 && config.window_chars > 0);
        let slots = spec(&config);
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
        }
    }
    /// Parameters a configuration would have, without building the model.
    pub fn parameters_for(config: &Config) -> usize {
        spec(config).last().map(|s| s.range.end).unwrap_or(0)
    }
    pub fn parameter_count(&self) -> usize {
        self.weights.len()
    }
    pub fn group_parameters(&self, focus: Focus) -> usize {
        self.slots.iter().filter(|s| s.group == focus).map(|s| s.range.len()).sum()
    }
    fn slot(&self, name: &str) -> &Slot {
        &self.slots[*self.offsets.get(name).expect("missing Nova-2 parameter")]
    }
    pub(crate) fn val(&self, name: &str) -> &[f32] {
        &self.weights[self.slot(name).range.clone()]
    }
    fn param(&self, t: &mut tape::Tape, name: &str) -> usize {
        t.parameter(name, self.slot(name).range.clone())
    }
    fn norm(&self, t: &mut tape::Tape, h: usize, name: &str) -> usize {
        let g = self.param(t, name);
        t.norm(h, g, self.config.d)
    }
    fn lin(&self, t: &mut tape::Tape, h: usize, name: &str, bias: Option<&str>) -> usize {
        let slot = self.slot(name);
        let w = self.param(t, name);
        let b = bias.map(|n| self.param(t, n));
        t.linear(h, w, b, slot.cols, slot.rows)
    }
    fn block(&self, t: &mut tape::Tape, i: usize, h: usize, tokens: usize) -> usize {
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
        let h = t.add(h, a);
        let x = self.norm(t, h, &p("ln2"));
        let gate = self.lin(t, x, &p("w_gate"), None);
        let gate = t.silu(gate);
        let up = self.lin(t, x, &p("w_up"), None);
        let f = t.mul(gate, up);
        let down = self.lin(t, f, &p("w_down"), None);
        t.add(h, down)
    }
    fn efferent(&self, t: &mut tape::Tape, z1: usize, h: usize, abl: Ablation) -> usize {
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
            t.input(vec![0.; t.values(z).len()])
        } else {
            z
        }
    }
    fn inject(&self, t: &mut tape::Tape, h: usize, z: usize, n: &str, w: &str, b: &str, a: &str) -> usize {
        let hn = self.norm(t, h, n);
        let gate = self.lin(t, hn, w, Some(b));
        let gate = t.sigmoid(gate);
        let zproj = self.lin(t, z, a, None);
        let fused = t.mul(gate, zproj);
        t.add(h, fused)
    }
    fn forward(&self, t: &mut tape::Tape, ids: &[u16], feats: Vec<f32>, abl: Ablation) -> usize {
        let c = &self.config;
        let tok = ids.iter().map(|&v| v as usize).collect::<Vec<_>>();
        let embed = self.param(t, "embed");
        let mut h = t.embed(&tok, embed, c.d);
        for i in 0..c.split {
            h = self.block(t, i, h, ids.len());
        }
        let feat = t.input(feats);
        let z1 = self.lin(t, feat, "r.w1", Some("r.b1"));
        let z1 = t.silu(z1);
        let z1 = if abl.no_right { t.input(vec![0.; t.values(z1).len()]) } else { z1 };
        if !abl.no_afferent {
            h = self.inject(t, h, z1, "c.n1", "c.g1w", "c.g1b", "c.a1");
        }
        let mut z2 = self.efferent(t, z1, h, abl);
        for round in 0..c.loops {
            if !abl.no_afferent {
                h = self.inject(t, h, z2, "c.n2", "c.g2w", "c.g2b", "c.a2");
            }
            for i in c.split..c.blocks {
                h = self.block(t, i, h, ids.len());
            }
            if round + 1 < c.loops {
                z2 = self.efferent(t, z1, h, abl);
            }
        }
        let h = self.norm(t, h, "ln_f");
        let bias = self.param(t, "out_bias");
        t.linear(h, embed, Some(bias), c.d, c.vocab)
    }
    pub fn features(&self, ids: &[u16]) -> Result<Vec<f32>, String> {
        let c = &self.config;
        let mut decoded = Vec::new();
        let mut out = Vec::with_capacity(ids.len() * c.window_chars * 16);
        for &id in ids {
            if (id as usize) < 128 {
                decoded.push(id as u8);
            } else {
                let w = self.tokenizer.words.get(id as usize - 128).ok_or("Unknown word token")?;
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
    pub fn logits(&self, ids: &[u16], abl: Ablation) -> Result<Vec<f32>, String> {
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
        ids: &[u16],
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
    pub fn predict(&self, ids: &[u16], abl: Ablation) -> Result<Vec<f32>, String> {
        let z = self.logits(ids, abl)?;
        let last = &z[z.len() - self.config.vocab..];
        let mut p = last.to_vec();
        let valid = (128 + self.tokenizer.words.len()).min(self.config.vocab);
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
        ids: &[u16],
        targets: &[u16],
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
        ids: &[u16],
        features: &[f32],
        targets: &[u16],
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
    /// Like [`Bicameral::loss_and_gradients`], but ADDS `scale ×` the gradient into `grads`
    /// (one buffer reused across rows) and returns the row's loss.
    pub fn accumulate_loss_and_gradients(
        &self,
        ids: &[u16],
        targets: &[u16],
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
        ids: &[u16],
        features: Vec<f32>,
        targets: &[u16],
        mask: &[f32],
        abl: Ablation,
        grads: &mut [f32],
        scale: f32,
    ) -> f32 {
        let mut t = tape::Tape::new(&self.weights, Some(grads));
        let logits = self.forward(&mut t, ids, features, abl);
        let targets = targets.iter().map(|&v| v as usize).collect::<Vec<_>>();
        let loss = t.xent(logits, &targets, mask, self.config.vocab);
        let value = t.values(loss)[0];
        t.backward(loss, scale);
        value
    }
    pub fn adam_step(&mut self, gradients: &[f32], lr: f32, focus: Focus) -> Result<f32, String> {
        if gradients.len() != self.weights.len() || !lr.is_finite() || lr <= 0. {
            return Err("Invalid Nova-2 optimizer inputs".into());
        }
        let norm = gradients.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>().sqrt() as f32;
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
        ids: &[u16],
        targets: &[u16],
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
