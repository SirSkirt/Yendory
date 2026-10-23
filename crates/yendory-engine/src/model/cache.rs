//! Incremental KV cache for faithful Nova-2 generation.
//! Each execution of a shared upper transformer block has its OWN cache:
//! loop 2 reads different input activations than loop 1.
//! Training never uses the cache; no parameters or gradients are changed.
//! Past `max_tokens`, the window is rebuilt from the most recent `max_tokens` tokens with
//! positions restarting at 0, exactly like the reference's uncached `next_token_probs`; the
//! right hemisphere still sees the characters just before the window, as the reference does.
use super::{Ablation, Bicameral};
use crate::tokenizer::TokenId;

#[derive(Clone, Debug, Default)]
struct Stage {
    keys: Vec<f32>,
    values: Vec<f32>,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub cached_tokens: usize,
    pub attention_keys_read: usize,
    pub transformer_evaluations: usize,
}
pub struct KvCache {
    stages: Vec<Stage>,
    tokens: Vec<TokenId>,
    chars: Vec<u8>,
    /// Up to `window_chars` characters that precede the first token of the window.
    before: Vec<u8>,
    sequence: usize,
    appended: usize,
    model_step: u64,
    model_signature: u64,
    abl: Ablation,
    stats: CacheStats,
    trace: Option<Trace>,
    /// Nova Neuro: the working memory's state per layer (belongs to this context only).
    memory: Vec<Vec<f32>>,
}

/// Attention weights of one transformer block execution for the newest token.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StageTrace {
    pub block: usize,
    /// 0 for the lower stage; 1.. for each thinking round of the upper stage.
    pub round: usize,
    /// `[head][position]`: how much the newest token attends to each token in the window.
    pub attention: Vec<Vec<f32>>,
}
/// One right->left injection: `h + gate * A z`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GateTrace {
    /// Sigmoid gate per left-state dimension (0 closed, 1 open).
    pub gate: Vec<f32>,
    /// Length of the injected vector and of the left state it was added to.
    pub injection_norm: f32,
    pub state_norm: f32,
}
/// Internal activity of the newest token's forward pass, measured, never estimated.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Trace {
    pub stages: Vec<StageTrace>,
    /// Right hemisphere bottom layer (SiLU outputs, r1 values).
    pub right_trunk: Vec<f32>,
    /// Right hemisphere upper layer for each thinking round (r2 values).
    pub right_upper: Vec<Vec<f32>>,
    /// Afferent injections in order: exchange 1, then one per thinking round.
    pub gates: Vec<GateTrace>,
    /// Read-outs: the output layer (final norm, embedding, bias) applied to the left state after
    /// the lower stage and after each thinking round, i.e. what the model would predict if it
    /// stopped there. The last read-out is the model's actual output.
    #[serde(default)]
    pub readouts: Vec<Vec<f32>>,
    /// Nova Neuro: the modules' measured activity (None for Nova Bicameral).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neuro: Option<NeuroTrace>,
}

/// One computation round of Nova Neuro for the newest token, measured.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NeuroRound {
    /// The Executive Controller's gate for each module (0 closed, 1 open), in module order.
    pub gates: Vec<f32>,
    /// Length of each module's output before its gate.
    pub outputs: Vec<f32>,
    /// Length of the bus state the modules read, and of the combined write-back.
    pub bus_norm: f32,
    pub write_norm: f32,
    /// Associative memory: the most strongly retrieved patterns (index, weight), strongest
    /// first (up to [`TRACE_PATTERNS`]).
    pub patterns: Vec<(usize, f32)>,
    /// The bus state the networks read this round (before its norm).
    #[serde(default)]
    pub bus: Vec<f32>,
    /// The Executive Controller's hidden activations this round (empty without networks).
    #[serde(default)]
    pub controller: Vec<f32>,
}

/// Recalled associative patterns kept in a trace, per round.
pub const TRACE_PATTERNS: usize = 32;
/// Nova Neuro's activity for the newest token.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NeuroTrace {
    /// Module names, in gate order.
    pub modules: Vec<String>,
    pub rounds: Vec<NeuroRound>,
    /// Working memory: how much each layer's state changed with this token, and its length.
    pub memory_change: Vec<f32>,
    pub memory_norm: Vec<f32>,
    /// The memory's update gate: mean opening per layer (0 keep, 1 overwrite).
    pub memory_update: Vec<f32>,
    /// Working memory: each layer's state after this token.
    #[serde(default)]
    pub memory_state: Vec<Vec<f32>>,
}
impl KvCache {
    pub fn new(model: &Bicameral, abl: Ablation) -> Self {
        let c = &model.config;
        let stages = (0..c.split + (c.blocks - c.split) * c.loops).map(|_| Stage::default()).collect();
        Self {
            stages,
            tokens: Vec::new(),
            chars: Vec::new(),
            before: Vec::new(),
            sequence: 0,
            appended: 0,
            model_step: model.step,
            model_signature: signature(model),
            abl,
            stats: CacheStats::default(),
            trace: None,
            memory: fresh_memory(model),
        }
    }
    /// A cache that also records a [`Trace`] of every token's forward pass (the latest one is
    /// kept). Tracing does not change any computed value.
    pub fn with_trace(model: &Bicameral, abl: Ablation) -> Self {
        Self { trace: Some(Trace::default()), ..Self::new(model, abl) }
    }
    pub fn trace(&self) -> Option<&Trace> {
        self.trace.as_ref()
    }
    pub fn stats(&self) -> CacheStats {
        self.stats
    }
    pub fn token_count(&self) -> usize {
        self.appended
    }
    pub fn reset(&mut self, model: &Bicameral) {
        let tracing = self.trace.is_some();
        *self = Self::new(model, self.abl);
        if tracing {
            self.trace = Some(Trace::default());
        }
    }
    pub fn prefill(&mut self, model: &Bicameral, ids: &[TokenId]) -> Result<Vec<f32>, String> {
        let mut last = Vec::new();
        for &id in ids {
            last = self.append(model, id)?;
        }
        if last.is_empty() {
            return Err("Empty prompt".into());
        }
        Ok(last)
    }
    pub fn append(&mut self, model: &Bicameral, token: TokenId) -> Result<Vec<f32>, String> {
        if model.step != self.model_step || signature(model) != self.model_signature {
            return Err("KV cache is stale; create a new cache after training or loading".into());
        }
        let c = &model.config;
        let id = token as usize;
        if id >= c.vocab || id >= 128 + model.tokenizer.words().len() {
            return Err("Invalid generated or input token ID".into());
        }
        self.appended += 1;
        if self.tokens.len() < c.max_tokens {
            return Ok(self.step(model, token));
        }
        // Window full: drop the oldest token and recompute the window from position 0.
        let evicted = self.tokens.remove(0);
        self.before.extend_from_slice(&token_text(model, evicted));
        if self.before.len() > c.window_chars {
            self.before.drain(..self.before.len() - c.window_chars);
        }
        let mut window = std::mem::take(&mut self.tokens);
        window.push(token);
        for stage in &mut self.stages {
            *stage = Stage::default();
        }
        self.chars = self.before.clone();
        self.sequence = 0;
        self.memory = fresh_memory(model);
        let mut last = Vec::new();
        for t in window {
            last = self.step(model, t);
        }
        Ok(last)
    }
    fn step(&mut self, model: &Bicameral, token: TokenId) -> Vec<f32> {
        if model.neuro.is_some() {
            return self.neuro_step(model, token);
        }
        let c = &model.config;
        let id = token as usize;
        let value = token_text(model, token);
        self.chars.extend_from_slice(&value);
        if self.chars.len() > c.window_chars {
            self.chars.drain(..self.chars.len() - c.window_chars);
        }
        let f = right_features(&self.chars, c.window_chars);
        let tracing = self.trace.is_some();
        let mut trace = Trace::default();
        let mut h = model.val("embed")[id * c.d..(id + 1) * c.d].to_vec();
        let mut stage = 0;
        for i in 0..c.split {
            let mut heads = Vec::new();
            h = block(
                model,
                &mut self.stages[stage],
                &h,
                i,
                self.sequence,
                tracing.then_some(&mut heads),
                &mut self.stats,
                c.max_tokens,
            );
            if tracing {
                trace.stages.push(StageTrace { block: i, round: 0, attention: heads });
            }
            stage += 1;
        }
        // Bias before SiLU, matching the reference's r.w1 + r.b1.
        let mut z1raw = mat(model.val("r.w1"), c.r1, c.window_chars * 16, &f);
        for (x, b) in z1raw.iter_mut().zip(model.val("r.b1")) {
            *x += *b;
        }
        let mut z1 = silu(&z1raw);
        if self.abl.no_right {
            z1.fill(0.);
        }
        if tracing {
            trace.readouts.push(readout(model, &h));
        }
        trace.right_trunk = z1.clone();
        if !self.abl.no_afferent {
            let gate;
            (h, gate) = inject(model, &h, &z1, "c.n1", "c.g1w", "c.g1b", "c.a1");
            trace.gates.push(gate);
        }
        let mut z2 = efferent(model, &z1, &h, self.abl);
        for round in 0..c.loops {
            trace.right_upper.push(z2.clone());
            if !self.abl.no_afferent {
                let gate;
                (h, gate) = inject(model, &h, &z2, "c.n2", "c.g2w", "c.g2b", "c.a2");
                trace.gates.push(gate);
            }
            for i in c.split..c.blocks {
                let mut heads = Vec::new();
                h = block(
                    model,
                    &mut self.stages[stage],
                    &h,
                    i,
                    self.sequence,
                    tracing.then_some(&mut heads),
                    &mut self.stats,
                    c.max_tokens,
                );
                if tracing {
                    trace.stages.push(StageTrace { block: i, round: round + 1, attention: heads });
                }
                stage += 1;
            }
            if round + 1 < c.loops {
                z2 = efferent(model, &z1, &h, self.abl);
            }
            if tracing && round + 1 < c.loops {
                trace.readouts.push(readout(model, &h));
            }
        }
        let logits = readout(model, &h);
        if tracing {
            trace.readouts.push(logits.clone());
        }
        self.tokens.push(token);
        self.sequence += 1;
        self.stats.cached_tokens = self.tokens.len();
        if tracing {
            self.trace = Some(trace);
        }
        logits
    }
}
fn fresh_memory(model: &Bicameral) -> Vec<Vec<f32>> {
    match model.neuro.as_ref().and_then(|n| n.memory.as_ref()) {
        Some(m) => vec![vec![0.; m.width]; m.layers],
        None => Vec::new(),
    }
}
fn affine(model: &Bicameral, w: &str, b: Option<&str>, x: &[f32]) -> Vec<f32> {
    let slot = model.slot(w);
    let mut y = mat(model.val(w), slot.rows, slot.cols, x);
    if let Some(b) = b {
        for (v, b) in y.iter_mut().zip(model.val(b)) {
            *v += *b;
        }
    }
    y
}
fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
impl KvCache {
    /// Nova Neuro, one token: the same computation as `Bicameral::neuro_hidden` for the newest
    /// position, with the transformer's keys and values and the working memory's state kept.
    fn neuro_step(&mut self, model: &Bicameral, token: TokenId) -> Vec<f32> {
        let c = &model.config;
        let n = model.neuro.as_ref().expect("Nova Neuro");
        let id = token as usize;
        let tracing = self.trace.is_some();
        let mut trace = Trace::default();
        let mut nt = NeuroTrace {
            modules: n.modules().iter().map(|m| m.name().to_string()).collect(),
            ..NeuroTrace::default()
        };
        let mut h = model.val("embed")[id * c.d..(id + 1) * c.d].to_vec();
        let mut stage = 0;
        for i in 0..c.split {
            let mut heads = Vec::new();
            h = block(
                model,
                &mut self.stages[stage],
                &h,
                i,
                self.sequence,
                tracing.then_some(&mut heads),
                &mut self.stats,
                c.max_tokens,
            );
            if tracing {
                trace.stages.push(StageTrace { block: i, round: 0, attention: heads });
            }
            stage += 1;
        }
        if tracing {
            trace.readouts.push(readout(model, &h));
        }
        let memory = n.memory.as_ref().map(|m| {
            let mut x = h.clone();
            for l in 0..m.layers {
                let xn = norm(&x, model.val(&format!("n.wm{l}.norm")));
                let z = sigmoid(&affine(model, &format!("n.wm{l}.z"), Some(&format!("n.wm{l}.zb")), &xn));
                let cand = affine(model, &format!("n.wm{l}.c"), Some(&format!("n.wm{l}.cb")), &xn);
                let old = self.memory[l].clone();
                for j in 0..m.width {
                    self.memory[l][j] = (1. - z[j]) * old[j] + z[j] * cand[j];
                }
                let change: Vec<f32> = self.memory[l].iter().zip(&old).map(|(a, b)| a - b).collect();
                nt.memory_change.push(l2(&change));
                nt.memory_norm.push(l2(&self.memory[l]));
                nt.memory_update.push(z.iter().sum::<f32>() / z.len().max(1) as f32);
                if tracing {
                    nt.memory_state.push(self.memory[l].clone());
                }
                x = self.memory[l].clone();
            }
            let mut out = affine(model, "n.wm_out", None, &x);
            if self.abl.no_memory || self.abl.no_right {
                out.fill(0.);
            }
            out
        });
        let modules = n.modules();
        let rounds = if self.abl.one_round { 1 } else { c.loops };
        for round in 0..rounds {
            let hn = norm(&h, model.val("n.read_norm"));
            let mut bus = affine(model, "n.read", None, &hn);
            if self.abl.no_efferent {
                bus.fill(0.);
            }
            if let Some(m) = &memory {
                bus = add(&bus, m);
            }
            let bn = norm(&bus, model.val("n.bus_norm"));
            let mut r = NeuroRound { bus_norm: l2(&bus), ..NeuroRound::default() };
            if tracing {
                r.bus = bus.clone();
            }
            let gates = if modules.is_empty() {
                Vec::new()
            } else if self.abl.no_controller {
                vec![1.; modules.len()]
            } else {
                let hid = silu(&affine(model, "n.ctl1", Some("n.ctl1b"), &bn));
                if tracing {
                    r.controller = hid.clone();
                }
                sigmoid(&affine(model, "n.ctl2", Some("n.ctl2b"), &hid))
            };
            let mut write: Option<Vec<f32>> = None;
            for (gi, module) in modules.iter().enumerate() {
                let out = match module {
                    super::neuro::Module::Memory => memory.clone().expect("memory"),
                    super::neuro::Module::Associative => {
                        let an = norm(&bn, model.val("n.as_norm"));
                        let q = affine(model, "n.as_q", None, &an);
                        let mut p = affine(model, "n.as_keys", None, &q);
                        let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut total = 0.;
                        for v in p.iter_mut() {
                            *v = (*v - max).exp();
                            total += *v;
                        }
                        for v in p.iter_mut() {
                            *v /= total;
                        }
                        if tracing {
                            let mut top: Vec<(usize, f32)> = p.iter().copied().enumerate().collect();
                            top.sort_by(|a, b| b.1.total_cmp(&a.1));
                            r.patterns = top.into_iter().take(TRACE_PATTERNS).collect();
                        }
                        let mut out = affine(model, "n.as_vals", None, &p);
                        if self.abl.no_associative || self.abl.no_right {
                            out.fill(0.);
                        }
                        out
                    }
                };
                r.outputs.push(l2(&out));
                let gated: Vec<f32> = out.iter().map(|v| v * gates[gi]).collect();
                write = Some(match write {
                    Some(w) => add(&w, &gated),
                    None => gated,
                });
            }
            r.gates = gates;
            if let (Some(w), false) = (write, self.abl.no_afferent) {
                r.write_norm = l2(&w);
                let gate = sigmoid(&affine(model, "n.gate_w", Some("n.gate_b"), &hn));
                let z = affine(model, "n.write", None, &w);
                let delta: Vec<f32> = gate.iter().zip(&z).map(|(g, x)| g * x).collect();
                trace.gates.push(GateTrace { injection_norm: l2(&delta), state_norm: l2(&h), gate });
                h = add(&h, &delta);
            }
            nt.rounds.push(r);
            for i in c.split..c.blocks {
                let mut heads = Vec::new();
                h = block(
                    model,
                    &mut self.stages[stage],
                    &h,
                    i,
                    self.sequence,
                    tracing.then_some(&mut heads),
                    &mut self.stats,
                    c.max_tokens,
                );
                if tracing {
                    trace.stages.push(StageTrace { block: i, round: round + 1, attention: heads });
                }
                stage += 1;
            }
            if tracing && round + 1 < rounds {
                trace.readouts.push(readout(model, &h));
            }
        }
        let logits = readout(model, &h);
        if tracing {
            trace.readouts.push(logits.clone());
            trace.neuro = Some(nt);
        }
        self.tokens.push(token);
        self.sequence += 1;
        self.stats.cached_tokens = self.tokens.len();
        if tracing {
            self.trace = Some(trace);
        }
        logits
    }
}

/// The output layer applied to a left state: final norm, tied embedding, output bias.
fn readout(model: &Bicameral, h: &[f32]) -> Vec<f32> {
    let c = &model.config;
    let mut logits = mat(model.val("embed"), c.vocab, c.d, &norm(h, model.val("ln_f")));
    for (x, b) in logits.iter_mut().zip(model.val("out_bias")) {
        *x += *b;
    }
    logits
}
fn token_text(model: &Bicameral, token: TokenId) -> Vec<u8> {
    let id = token as usize;
    if id < 128 {
        vec![id as u8]
    } else {
        model.tokenizer.words()[id - 128].as_bytes().to_vec()
    }
}
fn signature(model: &Bicameral) -> u64 {
    // Deliberately lightweight guard; a cache is always freshly instantiated
    // for a generation request, so this is a defensive stale-state check.
    let len = model.weights.len();
    let mut sig = model.step ^ (len as u64);
    for i in [0, len / 4, len / 2, len * 3 / 4, len.saturating_sub(1)] {
        sig = sig.rotate_left(9) ^ model.weights[i].to_bits() as u64;
    }
    sig
}
fn mat(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    super::linalg::matvec(w, rows, cols, x)
}
fn norm(x: &[f32], gain: &[f32]) -> Vec<f32> {
    let div = (x.iter().map(|x| x * x).sum::<f32>() / x.len() as f32 + 1e-6).sqrt();
    x.iter().zip(gain).map(|(x, g)| x * g / div).collect()
}
fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|x| x / (1. + (-x).exp())).collect()
}
fn sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|x| 1. / (1. + (-x).exp())).collect()
}
fn add(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(x, y)| x + y).collect()
}
fn inject(
    model: &Bicameral,
    h: &[f32],
    z: &[f32],
    ln: &str,
    gate_w: &str,
    gate_b: &str,
    a: &str,
) -> (Vec<f32>, GateTrace) {
    let d = model.config.d;
    let hn = norm(h, model.val(ln));
    let gate = mat(model.val(gate_w), d, d, &hn);
    let gate = gate.iter().zip(model.val(gate_b)).map(|(x, b)| x + b).collect::<Vec<f32>>();
    let gate = sigmoid(&gate);
    let injected = mat(model.val(a), d, z.len(), z);
    let delta: Vec<f32> = gate.iter().zip(&injected).map(|(g, x)| g * x).collect();
    let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let measured = GateTrace { injection_norm: l2(&delta), state_norm: l2(h), gate };
    (h.iter().zip(&delta).map(|(h, x)| h + x).collect(), measured)
}
fn efferent(model: &Bicameral, z1: &[f32], h: &[f32], abl: Ablation) -> Vec<f32> {
    let c = &model.config;
    let mut pre = mat(model.val("r.w2"), c.r2, c.r1, z1);
    for (v, b) in pre.iter_mut().zip(model.val("r.b2")) {
        *v += *b;
    }
    if !abl.no_efferent {
        let hn = norm(h, model.val("c.n1"));
        let back = mat(model.val("r.wl"), c.r2, c.d, &hn);
        for (v, b) in pre.iter_mut().zip(&back) {
            *v += *b;
        }
    }
    let mut z = silu(&pre);
    if abl.no_right {
        z.fill(0.);
    }
    z
}
fn right_features(chars: &[u8], window: usize) -> Vec<f32> {
    let mut f = vec![0f32; window * 16];
    let start = window - chars.len();
    for (i, &ch) in chars.iter().enumerate() {
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
    f
}
fn block(
    model: &Bicameral,
    stage: &mut Stage,
    h: &[f32],
    i: usize,
    pos: usize,
    mut attention: Option<&mut Vec<Vec<f32>>>,
    stats: &mut CacheStats,
    max_context: usize,
) -> Vec<f32> {
    let c = &model.config;
    let d = c.d;
    let hd = d / c.heads;
    let p = |name: &str| format!("b{i}.{name}");
    let x = norm(h, model.val(&p("ln1")));
    let mut q = mat(model.val(&p("wq")), d, d, &x);
    let mut k = mat(model.val(&p("wk")), d, d, &x);
    let v = mat(model.val(&p("wv")), d, d, &x);
    let half = hd / 2;
    for head in 0..c.heads {
        for j in 0..half {
            let angle = pos as f32 * c.rope_base.powf(-2. * j as f32 / hd as f32);
            let (sn, cs) = angle.sin_cos();
            let a = head * hd + j;
            let b = a + half;
            let qa = q[a];
            let qb = q[b];
            q[a] = qa * cs - qb * sn;
            q[b] = qb * cs + qa * sn;
            let ka = k[a];
            let kb = k[b];
            k[a] = ka * cs - kb * sn;
            k[b] = kb * cs + ka * sn;
        }
    }
    stage.keys.extend(&k);
    stage.values.extend(&v);
    let n = stage.keys.len() / d;
    if n > max_context {
        stage.keys.drain(..d);
        stage.values.drain(..d);
    }
    let n = stage.keys.len() / d;
    let mut att = vec![0.; d];
    let scale = 1. / (hd as f32).sqrt();
    for head in 0..c.heads {
        let mut scores = vec![0f32; n];
        let mut top = f32::NEG_INFINITY;
        for t in 0..n {
            let score = super::linalg::dot(
                &q[head * hd..(head + 1) * hd],
                &stage.keys[t * d + head * hd..t * d + (head + 1) * hd],
            );
            scores[t] = score * scale;
            top = top.max(scores[t]);
        }
        let mut total = 0f32;
        for s in &mut scores {
            *s = (*s - top).exp();
            total += *s;
        }
        for t in 0..n {
            let weight = scores[t] / total;
            super::linalg::axpy(
                &mut att[head * hd..(head + 1) * hd],
                weight,
                &stage.values[t * d + head * hd..t * d + (head + 1) * hd],
            );
        }
        if let Some(out) = attention.as_deref_mut() {
            out.push(scores.iter().map(|s| s / total).collect());
        }
    }
    stats.attention_keys_read += n * c.heads;
    stats.transformer_evaluations += 1;
    let a = mat(model.val(&p("wo")), d, d, &att);
    let h = add(h, &a);
    let x = norm(&h, model.val(&p("ln2")));
    let gate = silu(&mat(model.val(&p("w_gate")), c.ff, d, &x));
    let up = mat(model.val(&p("w_up")), c.ff, d, &x);
    let ff = gate.iter().zip(up).map(|(x, y)| x * y).collect::<Vec<f32>>();
    let down = mat(model.val(&p("w_down")), d, c.ff, &ff);
    add(&h, &down)
}
impl Bicameral {
    pub fn predict_cached(&self, cache: &mut KvCache, token: TokenId) -> Result<Vec<f32>, String> {
        let logits = cache.append(self, token)?;
        Ok(self.next_token_probabilities(logits))
    }
    /// Softmax over the valid tokens (learned word ids beyond the vocabulary get 0), exactly as
    /// generation uses it.
    pub(crate) fn next_token_probabilities(&self, logits: Vec<f32>) -> Vec<f32> {
        let mut p = logits;
        let valid = (128 + self.tokenizer.words().len()).min(self.config.vocab);
        for x in p.iter_mut().skip(valid) {
            *x = f32::NEG_INFINITY;
        }
        let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.;
        for x in &mut p {
            *x = (*x - max).exp();
            total += *x;
        }
        for x in &mut p {
            *x /= total;
        }
        p
    }
    pub fn generate_cached(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        let ids = self.tokenizer.encode(prompt)?;
        let output = self.generate_cached_ids(&ids, max_tokens, Ablation::default())?;
        self.tokenizer.decode(&output)
    }
    /// Greedy generation from token ids; stops after emitting a newline token.
    /// Generated ids are kept as emitted (never re-tokenized).
    pub fn generate_cached_ids(
        &self,
        ids: &[TokenId],
        max_tokens: usize,
        abl: Ablation,
    ) -> Result<Vec<TokenId>, String> {
        let mut cache = KvCache::new(self, abl);
        let mut probs = Vec::new();
        for &id in ids {
            probs = self.predict_cached(&mut cache, id)?;
        }
        if probs.is_empty() {
            return Err("Generation needs nonempty prompt".into());
        }
        let mut output = Vec::new();
        for _ in 0..max_tokens {
            let next = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as TokenId;
            output.push(next);
            if next == b'\n' as TokenId {
                break;
            }
            probs = self.predict_cached(&mut cache, next)?;
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Config, Focus};
    #[test]
    fn cached_matches_full_logits_across_both_thinking_rounds() {
        let mut cfg = Config::default();
        cfg.d = 16;
        cfg.heads = 2;
        cfg.ff = 24;
        cfg.blocks = 3;
        cfg.split = 1;
        cfg.loops = 2;
        cfg.r1 = 12;
        cfg.r2 = 10;
        cfg.window_chars = 4;
        cfg.max_tokens = 16;
        let m = Bicameral::with_config(cfg, 23);
        for abl in [
            Ablation::default(),
            Ablation { no_efferent: true, ..Ablation::default() },
            Ablation { no_afferent: true, ..Ablation::default() },
            Ablation { no_right: true, ..Ablation::default() },
        ] {
            let mut cache = KvCache::new(&m, abl);
            let mut prefix = Vec::new();
            for id in [3, 10, 17, 0, 7, 14, 21, 4] {
                prefix.push(id);
                let got = cache.append(&m, id).unwrap();
                let full = m.logits(&prefix, abl).unwrap();
                let target = &full[full.len() - m.config.vocab..];
                for (x, y) in got.iter().zip(target) {
                    assert!((x - y).abs() < 2e-4, "KV mismatch: {x} vs {y}, len {}", prefix.len());
                }
            }
            assert_eq!(cache.stats().cached_tokens, prefix.len());
        }
    }
    #[test]
    fn cache_invalidated_after_gradient_update() {
        let mut m = Bicameral::new();
        let mut cache = KvCache::new(&m, Ablation::default());
        m.adam_step(&vec![0.; m.parameter_count()], 0.0001, Focus::Joint).unwrap();
        assert!(cache.append(&m, b'a' as TokenId).is_err());
    }
    #[test]
    fn full_window_restarts_positions_like_the_reference() {
        let mut cfg = Config::default();
        cfg.d = 16;
        cfg.heads = 2;
        cfg.ff = 24;
        cfg.blocks = 3;
        cfg.split = 1;
        cfg.r1 = 12;
        cfg.r2 = 10;
        cfg.window_chars = 6;
        cfg.max_tokens = 8;
        cfg.vocab = 130;
        let mut m = Bicameral::with_config(cfg, 5);
        m.tokenizer.set_words(vec!["ab".repeat(2), "xyz".into()]);
        let ids: Vec<TokenId> = (0..30).map(|i| [104, 128, 33, 129, 10][i % 5]).collect();
        let row = m.config.window_chars * 16;
        let feats = m.features(&ids).unwrap();
        let mut cache = KvCache::new(&m, Ablation::default());
        for n in 1..=ids.len() {
            let got = cache.append(&m, ids[n - 1]).unwrap();
            // Reference windowing: features over the whole history, last max_tokens positions.
            let start = n.saturating_sub(m.config.max_tokens);
            let full =
                m.logits_with_features(&ids[start..n], &feats[start * row..n * row], Ablation::default());
            let full = full.unwrap();
            for (x, y) in got.iter().zip(&full[full.len() - m.config.vocab..]) {
                assert!((x - y).abs() < 2e-4, "window mismatch at {n}: {x} vs {y}");
            }
        }
        assert_eq!(cache.token_count(), ids.len());
        assert_eq!(cache.stats().cached_tokens, m.config.max_tokens);
    }
}
