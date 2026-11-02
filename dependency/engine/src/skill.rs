//! Game skills: a network that plays a real game from its pictures, carried inside a Nova Neuro
//! model (docs/FORMAT.md, "Game skills"). Yendory Tools trains them (its docs/GAME_TRAINING.md).
//!
//! The network turns the game's pictures into the state of every permitted control, one tick at
//! a time. It is built from the Engine's own blocks, so it belongs to Neuro rather than standing
//! apart:
//!
//! - **One container**: every weight in one flat array with named slots (`conv0.w`, `mem.z.w`,
//!   …), each in a group that maps onto the Engine's training focus (vision, memory and heads are
//!   modules, `Right`; the goal input is `Interconnect`), trained with AdamW.
//! - **The Engine's building blocks**: linear layers, SiLU, sigmoid, softmax, and Neuro's working
//!   memory, a minGRU (`m_t = (1 - z_t) m_(t-1) + z_t c_t`, gates from the current input only),
//!   carrying what was seen across ticks; plus a convolution for pictures, written as the Engine
//!   writes its matrix kernels.
//! - **A bus-shaped goal input** of fixed width, all zeros until Command Mode gives it the
//!   current command; later the model's Neuro bus can write into it.
//!
//! The network: the last `frames` pictures, stacked, go through convolutions (SiLU after each),
//! a linear encoder (SiLU) to `e_t`, the minGRU to `m_t`, and from `[m_t, e_t]` four heads: one
//! yes/no logit per key and per button, and a class over mouse steps for each mouse axis.
//! Nothing in it knows any game: every output comes from the weights.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Conv {
    pub out: usize,
    pub kernel: usize,
    pub stride: usize,
}

/// Which group a weight belongs to, as the Engine's training focus sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Group {
    Vision,
    Memory,
    Heads,
    Interconnect,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Arch {
    pub width: usize,
    pub height: usize,
    /// Channels of one picture (1 grey, 3 colour).
    pub channels: usize,
    /// Pictures stacked as the input (the newest last).
    pub frames: usize,
    pub convs: Vec<Conv>,
    pub hidden: usize,
    /// minGRU width (0: no memory).
    pub memory: usize,
    /// Width of the goal input (the bus).
    pub goal: usize,
    pub keys: usize,
    pub buttons: usize,
    /// Mouse steps (units): each axis chooses one of these every tick. Empty: no mouse.
    pub mouse_bins: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Size {
    Small,
    Medium,
}

impl Arch {
    /// An architecture for pictures of this size: the convolutions that fit, the widths of `size`.
    pub fn new(
        width: usize,
        height: usize,
        channels: usize,
        keys: usize,
        buttons: usize,
        mouse_bins: Vec<f32>,
        size: Size,
    ) -> Arch {
        let (c, hidden, memory) = match size {
            Size::Small => ([16, 32, 32], 128, 128),
            Size::Medium => ([32, 64, 64], 256, 256),
        };
        let wanted = [(c[0], 8, 4), (c[1], 4, 2), (c[2], 3, 1)];
        let mut convs = Vec::new();
        let (mut h, mut w) = (height, width);
        for (out, kernel, stride) in wanted {
            if h < kernel || w < kernel {
                break;
            }
            h = (h - kernel) / stride + 1;
            w = (w - kernel) / stride + 1;
            convs.push(Conv { out, kernel, stride });
        }
        if convs.is_empty() {
            let k = height.min(width).clamp(1, 3);
            convs.push(Conv { out: c[0], kernel: k, stride: 1 });
        }
        Arch {
            width,
            height,
            channels,
            frames: 2,
            convs,
            hidden,
            memory,
            goal: 32,
            keys,
            buttons,
            mouse_bins,
        }
    }

    /// (channels, height, width) entering each convolution, and leaving the last.
    pub fn shapes(&self) -> Vec<(usize, usize, usize)> {
        let mut out = vec![(self.channels * self.frames, self.height, self.width)];
        for c in &self.convs {
            let (_, h, w) = *out.last().unwrap();
            out.push((c.out, (h - c.kernel) / c.stride + 1, (w - c.kernel) / c.stride + 1));
        }
        out
    }

    pub fn features(&self) -> usize {
        let (c, h, w) = *self.shapes().last().unwrap();
        c * h * w
    }

    /// The heads' input: `[m_t, e_t]`.
    fn head_in(&self) -> usize {
        self.memory + self.hidden
    }

    pub fn check(&self) -> Result<(), String> {
        if self.width == 0
            || self.height == 0
            || !(self.channels == 1 || self.channels == 3)
            || self.frames == 0
        {
            return Err("The network's picture is not usable".into());
        }
        for (c, h, w) in self.shapes() {
            if c == 0 || h == 0 || w == 0 {
                return Err("The network's layers don't fit its picture".into());
            }
        }
        if self.keys > 64 || self.buttons > 8 || self.hidden == 0 {
            return Err("The network's sizes are not usable".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Slot {
    pub name: String,
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
    pub group: Group,
}

impl Slot {
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.rows * self.cols
    }
}

/// The layout of every weight: name, place, shape, group.
pub fn layout(a: &Arch) -> Vec<Slot> {
    let mut slots = Vec::new();
    let mut at = 0;
    let mut add = |name: String, rows: usize, cols: usize, group: Group| {
        slots.push(Slot { name, offset: at, rows, cols, group });
        at += rows * cols;
    };
    let shapes = a.shapes();
    for (i, c) in a.convs.iter().enumerate() {
        let cin = shapes[i].0;
        add(format!("conv{i}.w"), c.out, cin * c.kernel * c.kernel, Group::Vision);
        add(format!("conv{i}.b"), c.out, 1, Group::Vision);
    }
    add("enc.w".into(), a.hidden, a.features(), Group::Vision);
    add("enc.b".into(), a.hidden, 1, Group::Vision);
    add("goal.w".into(), a.hidden, a.goal, Group::Interconnect);
    if a.memory > 0 {
        add("mem.z.w".into(), a.memory, a.hidden, Group::Memory);
        add("mem.z.b".into(), a.memory, 1, Group::Memory);
        add("mem.c.w".into(), a.memory, a.hidden, Group::Memory);
        add("mem.c.b".into(), a.memory, 1, Group::Memory);
    }
    let u = a.head_in();
    if a.keys > 0 {
        add("key.w".into(), a.keys, u, Group::Heads);
        add("key.b".into(), a.keys, 1, Group::Heads);
    }
    if a.buttons > 0 {
        add("button.w".into(), a.buttons, u, Group::Heads);
        add("button.b".into(), a.buttons, 1, Group::Heads);
    }
    if !a.mouse_bins.is_empty() {
        let n = a.mouse_bins.len();
        add("mx.w".into(), n, u, Group::Heads);
        add("mx.b".into(), n, 1, Group::Heads);
        add("my.w".into(), n, u, Group::Heads);
        add("my.b".into(), n, 1, Group::Heads);
    }
    slots
}

/// A small seeded random generator (xorshift64*), so a new network repeats from its seed.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Debug)]
pub struct Policy {
    pub arch: Arch,
    pub slots: Vec<Slot>,
    pub w: Vec<f32>,
}

/// The heads' outputs for one tick (logits).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Logits {
    pub keys: Vec<f32>,
    pub buttons: Vec<f32>,
    pub mx: Vec<f32>,
    pub my: Vec<f32>,
}

/// What one tick should have been (a recording's controls).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Target {
    pub keys: u64,
    pub buttons: u8,
    pub mx: usize,
    pub my: usize,
}

/// Loss parts, averaged over ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Loss {
    pub total: f64,
    pub keys: f64,
    pub buttons: f64,
    pub mouse: f64,
}

impl std::ops::AddAssign for Loss {
    fn add_assign(&mut self, o: Loss) {
        self.total += o.total;
        self.keys += o.keys;
        self.buttons += o.buttons;
        self.mouse += o.mouse;
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (yv, xv) in y.iter_mut().zip(x) {
        *yv += a * xv;
    }
}

/// c[m×n] += a[m×k] · b[k×n]
fn mm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let crow = &mut c[i * n..(i + 1) * n];
        for p in 0..k {
            let av = a[i * k + p];
            if av != 0.0 {
                axpy(crow, av, &b[p * n..(p + 1) * n]);
            }
        }
    }
}

/// c[m×n] += a[m×k] · b[n×k]ᵀ
fn mm_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        for j in 0..n {
            c[i * n + j] += dot(arow, &b[j * k..(j + 1) * k]);
        }
    }
}

/// c[m×n] += a[k×m]ᵀ · b[k×n]
fn mm_at(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for p in 0..k {
        let brow = &b[p * n..(p + 1) * n];
        for i in 0..m {
            let av = a[p * m + i];
            if av != 0.0 {
                axpy(&mut c[i * n..(i + 1) * n], av, brow);
            }
        }
    }
}

/// Picture [c, h, w] to columns [c·k·k, ho·wo].
fn im2col(x: &[f32], c: usize, h: usize, w: usize, k: usize, s: usize) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = ((h - k) / s + 1, (w - k) / s + 1);
    let p = ho * wo;
    let mut cols = vec![0f32; c * k * k * p];
    for ch in 0..c {
        for ky in 0..k {
            for kx in 0..k {
                let row = (ch * k + ky) * k + kx;
                let out = &mut cols[row * p..(row + 1) * p];
                for oy in 0..ho {
                    let src = &x[ch * h * w + (oy * s + ky) * w + kx..];
                    for ox in 0..wo {
                        out[oy * wo + ox] = src[ox * s];
                    }
                }
            }
        }
    }
    (cols, ho, wo)
}

/// Columns back to a picture, adding (the reverse of [`im2col`]).
fn col2im(cols: &[f32], c: usize, h: usize, w: usize, k: usize, s: usize) -> Vec<f32> {
    let (ho, wo) = ((h - k) / s + 1, (w - k) / s + 1);
    let p = ho * wo;
    let mut x = vec![0f32; c * h * w];
    for ch in 0..c {
        for ky in 0..k {
            for kx in 0..k {
                let row = (ch * k + ky) * k + kx;
                let src = &cols[row * p..(row + 1) * p];
                for oy in 0..ho {
                    let base = ch * h * w + (oy * s + ky) * w + kx;
                    for ox in 0..wo {
                        x[base + ox * s] += src[oy * wo + ox];
                    }
                }
            }
        }
    }
    x
}

/// What a forward pass keeps for the backward pass, for one tick.
pub struct StepCache {
    cols: Vec<Vec<f32>>,
    pre: Vec<Vec<f32>>,
    feat: Vec<f32>,
    enc_pre: Vec<f32>,
    e: Vec<f32>,
    z: Vec<f32>,
    c: Vec<f32>,
    m_prev: Vec<f32>,
    pub m: Vec<f32>,
    u: Vec<f32>,
    pub logits: Logits,
}

/// Pictures (oldest first, `frames` of them, each width × height × channels bytes, pixels
/// interleaved) to the network's input planes, scaled to -1..1.
pub fn encode(a: &Arch, pictures: &[&[u8]]) -> Vec<f32> {
    let (w, h, c) = (a.width, a.height, a.channels);
    let mut x = vec![0f32; a.frames * c * h * w];
    for (fi, px) in pictures.iter().enumerate().take(a.frames) {
        for ch in 0..c {
            let plane = &mut x[(fi * c + ch) * h * w..(fi * c + ch + 1) * h * w];
            for (i, v) in plane.iter_mut().enumerate() {
                *v = px[i * c + ch] as f32 / 127.5 - 1.0;
            }
        }
    }
    x
}

impl Policy {
    /// A new network with weights drawn from `seed` (He initialisation; heads start small).
    pub fn new(arch: Arch, seed: u64) -> Result<Policy, String> {
        arch.check()?;
        let slots = layout(&arch);
        let n = slots.last().map(|s| s.offset + s.rows * s.cols).unwrap_or(0);
        let mut w = vec![0f32; n];
        let mut rng = Rng::new(seed);
        for s in &slots {
            if s.name.ends_with(".b") {
                continue;
            }
            let head = s.group == Group::Heads;
            let scale = (2.0 / s.cols as f32).sqrt() * if head { 0.1 } else { 1.0 };
            for v in &mut w[s.range()] {
                *v = rng.normal() * scale;
            }
        }
        Ok(Policy { arch, slots, w })
    }

    pub fn parameters(&self) -> usize {
        self.w.len()
    }

    pub fn slot(&self, name: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.name == name)
    }

    fn get(&self, name: &str) -> &[f32] {
        self.slot(name).map(|s| &self.w[s.range()]).unwrap_or(&[])
    }

    /// One tick forward. `m_prev` is the memory before it, `goal` the goal input.
    pub fn forward(&self, input: &[f32], m_prev: &[f32], goal: &[f32]) -> StepCache {
        let a = &self.arch;
        let shapes = a.shapes();
        let mut x = input.to_vec();
        let (mut cols_all, mut pre_all) = (Vec::new(), Vec::new());
        for (i, conv) in a.convs.iter().enumerate() {
            let (c, h, w) = shapes[i];
            let (cols, ho, wo) = im2col(&x, c, h, w, conv.kernel, conv.stride);
            let p = ho * wo;
            let ck = c * conv.kernel * conv.kernel;
            let mut pre = vec![0f32; conv.out * p];
            let b = self.get(&format!("conv{i}.b"));
            for (o, row) in pre.chunks_mut(p).enumerate() {
                row.fill(b[o]);
            }
            mm(self.get(&format!("conv{i}.w")), &cols, &mut pre, conv.out, ck, p);
            x = pre.iter().map(|&v| silu(v)).collect();
            cols_all.push(cols);
            pre_all.push(pre);
        }
        let feat = x;
        let mut enc_pre = self.get("enc.b").to_vec();
        let ew = self.get("enc.w");
        let gw = self.get("goal.w");
        for (r, v) in enc_pre.iter_mut().enumerate() {
            *v += dot(&ew[r * feat.len()..(r + 1) * feat.len()], &feat);
            if a.goal > 0 && goal.len() == a.goal {
                *v += dot(&gw[r * a.goal..(r + 1) * a.goal], goal);
            }
        }
        let e: Vec<f32> = enc_pre.iter().map(|&v| silu(v)).collect();
        let (mut z, mut c, mut m) = (Vec::new(), Vec::new(), Vec::new());
        if a.memory > 0 {
            let (zw, zb, cw, cb) =
                (self.get("mem.z.w"), self.get("mem.z.b"), self.get("mem.c.w"), self.get("mem.c.b"));
            for j in 0..a.memory {
                let zj = sigmoid(zb[j] + dot(&zw[j * a.hidden..(j + 1) * a.hidden], &e));
                let cj = cb[j] + dot(&cw[j * a.hidden..(j + 1) * a.hidden], &e);
                z.push(zj);
                c.push(cj);
                m.push((1.0 - zj) * m_prev[j] + zj * cj);
            }
        }
        let mut u = m.clone();
        u.extend_from_slice(&e);
        let head = |wn: &str, bn: &str, n: usize| -> Vec<f32> {
            let (w, b) = (self.get(wn), self.get(bn));
            (0..n).map(|r| b[r] + dot(&w[r * u.len()..(r + 1) * u.len()], &u)).collect()
        };
        let nb = a.mouse_bins.len();
        let logits = Logits {
            keys: if a.keys > 0 { head("key.w", "key.b", a.keys) } else { vec![] },
            buttons: if a.buttons > 0 { head("button.w", "button.b", a.buttons) } else { vec![] },
            mx: if nb > 0 { head("mx.w", "mx.b", nb) } else { vec![] },
            my: if nb > 0 { head("my.w", "my.b", nb) } else { vec![] },
        };
        StepCache {
            cols: cols_all,
            pre: pre_all,
            feat,
            enc_pre,
            e,
            z,
            c,
            m_prev: m_prev.to_vec(),
            m,
            u,
            logits,
        }
    }

    /// Runs a sequence of ticks from empty memory, measures the loss against `targets`, and
    /// adds the loss's gradient (averaged over the ticks, times `scale`) to `grad`.
    pub fn sequence_loss(
        &self,
        inputs: &[Vec<f32>],
        targets: &[Target],
        grad: &mut [f32],
        scale: f32,
    ) -> Loss {
        let a = &self.arch;
        let goal = vec![0f32; a.goal];
        let mut m = vec![0f32; a.memory];
        let mut caches = Vec::with_capacity(inputs.len());
        for x in inputs {
            let cache = self.forward(x, &m, &goal);
            m = cache.m.clone();
            caches.push(cache);
        }
        let t_n = inputs.len().max(1) as f32;
        let mut loss = Loss::default();
        // Gradient of the loss at each tick's logits.
        let mut dlogits = Vec::with_capacity(caches.len());
        for (cache, t) in caches.iter().zip(targets) {
            let l = &cache.logits;
            let mut d = Logits::default();
            for (k, &z) in l.keys.iter().enumerate() {
                let y = ((t.keys >> k) & 1) as f32;
                let p = sigmoid(z);
                loss.keys += bce(z, y) as f64;
                d.keys.push((p - y) / t_n * scale);
            }
            for (b, &z) in l.buttons.iter().enumerate() {
                let y = ((t.buttons >> b) & 1) as f32;
                let p = sigmoid(z);
                loss.buttons += bce(z, y) as f64;
                d.buttons.push((p - y) / t_n * scale);
            }
            for (zs, target, out) in [(&l.mx, t.mx, &mut d.mx), (&l.my, t.my, &mut d.my)] {
                if zs.is_empty() {
                    continue;
                }
                let p = softmax(zs);
                loss.mouse -= (p[target].max(1e-12) as f64).ln();
                for (i, &pi) in p.iter().enumerate() {
                    out.push((pi - if i == target { 1.0 } else { 0.0 }) / t_n * scale);
                }
            }
            dlogits.push(d);
        }
        let n = caches.len().max(1) as f64;
        loss.keys /= n;
        loss.buttons /= n;
        loss.mouse /= n;
        loss.total = loss.keys + loss.buttons + loss.mouse;
        self.backward(&caches, &dlogits, grad);
        loss
    }

    /// Back through the sequence (the memory carries gradient from later ticks to earlier).
    fn backward(&self, caches: &[StepCache], dlogits: &[Logits], grad: &mut [f32]) {
        let a = &self.arch;
        let slot = |n: &str| self.slot(n).map(|s| s.range());
        let ulen = a.head_in();
        let mut dm_next = vec![0f32; a.memory];
        for t in (0..caches.len()).rev() {
            let c = &caches[t];
            let d = &dlogits[t];
            let mut du = vec![0f32; ulen];
            for (wn, bn, dz) in [
                ("key.w", "key.b", &d.keys),
                ("button.w", "button.b", &d.buttons),
                ("mx.w", "mx.b", &d.mx),
                ("my.w", "my.b", &d.my),
            ] {
                if dz.is_empty() {
                    continue;
                }
                let (wr, br) = (slot(wn).unwrap(), slot(bn).unwrap());
                let w = &self.w[wr.clone()];
                for (r, &g) in dz.iter().enumerate() {
                    grad[br.start + r] += g;
                    axpy(&mut grad[wr.start + r * ulen..wr.start + (r + 1) * ulen], g, &c.u);
                    axpy(&mut du, g, &w[r * ulen..(r + 1) * ulen]);
                }
            }
            let (dm_own, de_direct) = du.split_at(a.memory);
            let mut de = de_direct.to_vec();
            if a.memory > 0 {
                let dm: Vec<f32> = dm_own.iter().zip(&dm_next).map(|(x, y)| x + y).collect();
                let (zw, zb, cw, cb) = (
                    slot("mem.z.w").unwrap(),
                    slot("mem.z.b").unwrap(),
                    slot("mem.c.w").unwrap(),
                    slot("mem.c.b").unwrap(),
                );
                for j in 0..a.memory {
                    let (z, cj, mp) = (c.z[j], c.c[j], c.m_prev[j]);
                    let dz_pre = dm[j] * (cj - mp) * z * (1.0 - z);
                    let dc = dm[j] * z;
                    dm_next[j] = dm[j] * (1.0 - z);
                    grad[zb.start + j] += dz_pre;
                    grad[cb.start + j] += dc;
                    axpy(&mut grad[zw.start + j * a.hidden..zw.start + (j + 1) * a.hidden], dz_pre, &c.e);
                    axpy(&mut grad[cw.start + j * a.hidden..cw.start + (j + 1) * a.hidden], dc, &c.e);
                    axpy(&mut de, dz_pre, &self.w[zw.start + j * a.hidden..zw.start + (j + 1) * a.hidden]);
                    axpy(&mut de, dc, &self.w[cw.start + j * a.hidden..cw.start + (j + 1) * a.hidden]);
                }
            }
            // Encoder (the goal input is zero here, so goal.w gets no gradient from it).
            let nf = c.feat.len();
            let (ew, eb) = (slot("enc.w").unwrap(), slot("enc.b").unwrap());
            let mut dfeat = vec![0f32; nf];
            for r in 0..a.hidden {
                let g = de[r] * silu_grad(c.enc_pre[r]);
                if g == 0.0 {
                    continue;
                }
                grad[eb.start + r] += g;
                axpy(&mut grad[ew.start + r * nf..ew.start + (r + 1) * nf], g, &c.feat);
                axpy(&mut dfeat, g, &self.w[ew.start + r * nf..ew.start + (r + 1) * nf]);
            }
            // Convolutions, last first.
            let shapes = a.shapes();
            let mut dout = dfeat;
            for i in (0..a.convs.len()).rev() {
                let conv = &a.convs[i];
                let (cin, h, w) = shapes[i];
                let p = shapes[i + 1].1 * shapes[i + 1].2;
                let ck = cin * conv.kernel * conv.kernel;
                let dpre: Vec<f32> = dout.iter().zip(&c.pre[i]).map(|(g, &x)| g * silu_grad(x)).collect();
                let (wr, br) = (slot(&format!("conv{i}.w")).unwrap(), slot(&format!("conv{i}.b")).unwrap());
                for (o, row) in dpre.chunks(p).enumerate() {
                    grad[br.start + o] += row.iter().sum::<f32>();
                }
                mm_bt(&dpre, &c.cols[i], &mut grad[wr.clone()], conv.out, p, ck);
                if i > 0 {
                    let mut dcols = vec![0f32; ck * p];
                    mm_at(&self.w[wr], &dpre, &mut dcols, ck, conv.out, p);
                    dout = col2im(&dcols, cin, h, w, conv.kernel, conv.stride);
                }
            }
        }
    }

    /// The network's weights as bytes (little-endian f32), for `weights.bin`.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub fn from_bytes(arch: Arch, bytes: &[u8]) -> Result<Policy, String> {
        let mut p = Policy::new(arch, 0)?;
        if bytes.len() != p.w.len() * 4 {
            return Err(format!(
                "The weights file has {} values; this network has {}",
                bytes.len() / 4,
                p.w.len()
            ));
        }
        for (v, b) in p.w.iter_mut().zip(bytes.chunks_exact(4)) {
            *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        if p.w.iter().any(|v| !v.is_finite()) {
            return Err("The weights file holds values that are not numbers".into());
        }
        Ok(p)
    }
}

/// Binary cross-entropy of a logit against 0 or 1, computed stably.
pub fn bce(z: f32, y: f32) -> f32 {
    z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln()
}

/// AdamW, with weight decay on the matrices (not the biases).
#[derive(Clone, Debug, Default)]
pub struct Adam {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: u64,
}

impl Adam {
    pub fn new(n: usize) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    pub fn step(&mut self, p: &mut Policy, g: &[f32], lr: f32, weight_decay: f32) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        self.t += 1;
        let c1 = 1.0 - b1.powi(self.t.min(i32::MAX as u64) as i32);
        let c2 = 1.0 - b2.powi(self.t.min(i32::MAX as u64) as i32);
        for s in &p.slots {
            let decay = if s.name.ends_with(".w") { weight_decay } else { 0.0 };
            for i in s.range() {
                self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
                self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
                let update = (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + eps);
                p.w[i] -= lr * (update + decay * p.w[i]);
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.t.to_le_bytes().to_vec();
        for v in self.m.iter().chain(&self.v) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8], n: usize) -> Option<Adam> {
        if bytes.len() != 8 + n * 8 {
            return None;
        }
        let t = u64::from_le_bytes(bytes[..8].try_into().ok()?);
        let vals: Vec<f32> =
            bytes[8..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        Some(Adam { m: vals[..n].to_vec(), v: vals[n..].to_vec(), t })
    }
}

/// A small network for checking the maths (every kind of layer, tiny sizes).
pub fn check_arch() -> Arch {
    Arch {
        width: 12,
        height: 9,
        channels: 3,
        frames: 2,
        convs: vec![Conv { out: 3, kernel: 3, stride: 2 }, Conv { out: 2, kernel: 2, stride: 1 }],
        hidden: 5,
        memory: 4,
        goal: 3,
        keys: 2,
        buttons: 1,
        mouse_bins: vec![-3.0, 0.0, 3.0],
    }
}

/// Random pictures and targets for a sequence of `n` ticks.
pub fn random_sequence(a: &Arch, seed: u64, n: usize) -> (Vec<Vec<f32>>, Vec<Target>) {
    let mut rng = Rng::new(seed);
    let inputs = (0..n)
        .map(|_| {
            let px: Vec<Vec<u8>> = (0..a.frames)
                .map(|_| (0..a.width * a.height * a.channels).map(|_| rng.below(256) as u8).collect())
                .collect();
            encode(a, &px.iter().map(|v| v.as_slice()).collect::<Vec<_>>())
        })
        .collect();
    let targets = (0..n)
        .map(|_| Target {
            keys: rng.below(4) as u64,
            buttons: rng.below(2) as u8,
            mx: rng.below(3),
            my: rng.below(3),
        })
        .collect();
    (inputs, targets)
}

/// Compares the backward pass with finite differences on a small network: how many weights were
/// checked, and the worst error (relative to the size of the gradient, with a small floor).
pub fn gradient_check(seed: u64) -> (usize, f64) {
    let a = check_arch();
    let p = Policy::new(a.clone(), seed).expect("the check network");
    let (inputs, targets) = random_sequence(&a, seed ^ 11, 3);
    let mut g = vec![0f32; p.parameters()];
    p.sequence_loss(&inputs, &targets, &mut g, 1.0);
    let loss_at = |w: &[f32]| {
        let q = Policy { w: w.to_vec(), ..p.clone() };
        q.sequence_loss(&inputs, &targets, &mut vec![0f32; q.parameters()], 1.0).total
    };
    let mut rng = Rng::new(seed ^ 5);
    let (mut checked, mut worst) = (0, 0f64);
    // Every slot except goal.w (its input is zero until Command Mode, so its gradient is zero).
    for s in p.slots.iter().filter(|s| s.name != "goal.w") {
        for _ in 0..4 {
            let i = s.offset + rng.below(s.rows * s.cols);
            let eps = 1e-2f32;
            let mut w = p.w.clone();
            w[i] += eps;
            let up = loss_at(&w);
            w[i] -= 2.0 * eps;
            let down = loss_at(&w);
            let numeric = (up - down) / (2.0 * eps as f64);
            let analytic = g[i] as f64;
            let err = (numeric - analytic).abs() / (0.1 + numeric.abs().max(analytic.abs()));
            worst = worst.max(err);
            checked += 1;
        }
    }
    (checked, worst)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn tiny() -> Arch {
        Arch {
            width: 12,
            height: 9,
            channels: 3,
            frames: 2,
            convs: vec![Conv { out: 3, kernel: 3, stride: 2 }, Conv { out: 2, kernel: 2, stride: 1 }],
            hidden: 5,
            memory: 4,
            goal: 3,
            keys: 2,
            buttons: 1,
            mouse_bins: vec![-3.0, 0.0, 3.0],
        }
    }

    fn sequence(a: &Arch, seed: u64, n: usize) -> (Vec<Vec<f32>>, Vec<Target>) {
        let mut rng = Rng::new(seed);
        let inputs = (0..n)
            .map(|_| {
                let px: Vec<Vec<u8>> = (0..a.frames)
                    .map(|_| (0..a.width * a.height * a.channels).map(|_| rng.below(256) as u8).collect())
                    .collect();
                encode(a, &px.iter().map(|v| v.as_slice()).collect::<Vec<_>>())
            })
            .collect();
        let targets = (0..n)
            .map(|_| Target {
                keys: rng.below(4) as u64,
                buttons: rng.below(2) as u8,
                mx: rng.below(3),
                my: rng.below(3),
            })
            .collect();
        (inputs, targets)
    }

    #[test]
    fn shapes_and_layout_fit() {
        let a = Arch::new(128, 72, 3, 5, 1, vec![-1.0, 0.0, 1.0], Size::Small);
        assert_eq!(a.shapes().last().unwrap(), &(32, 5, 12));
        let p = Policy::new(a.clone(), 1).unwrap();
        let last = p.slots.last().unwrap();
        assert_eq!(last.offset + last.rows * last.cols, p.parameters());
        assert!(p.slot("goal.w").unwrap().group == Group::Interconnect);
        // Small pictures get the layers that fit.
        let s = Arch::new(16, 9, 1, 2, 0, vec![], Size::Small);
        assert_eq!(s.convs.len(), 1);
        assert!(s.check().is_ok());
        assert_eq!(Policy::new(a, 1).unwrap().w, p.w, "the same seed gives the same network");
    }

    #[test]
    fn convolution_columns_round_trip_as_adjoints() {
        // <im2col(x), y> == <x, col2im(y)> for any x, y: col2im is im2col's adjoint.
        let mut rng = Rng::new(3);
        let (c, h, w, k, s) = (2, 7, 9, 3, 2);
        let x: Vec<f32> = (0..c * h * w).map(|_| rng.normal()).collect();
        let (cols, _, _) = im2col(&x, c, h, w, k, s);
        let y: Vec<f32> = (0..cols.len()).map(|_| rng.normal()).collect();
        let lhs = dot(&cols, &y);
        let rhs = dot(&x, &col2im(&y, c, h, w, k, s));
        assert!((lhs - rhs).abs() < 1e-3 * lhs.abs().max(1.0), "{lhs} {rhs}");
    }

    #[test]
    fn gradients_match_finite_differences() {
        for seed in [7, 8, 9] {
            let (checked, worst) = gradient_check(seed);
            assert!(checked > 40);
            assert!(worst < 2e-2, "worst relative error {worst}");
        }
    }

    #[test]
    fn training_lowers_the_loss_and_weights_round_trip() {
        let a = tiny();
        let mut p = Policy::new(a.clone(), 9).unwrap();
        let (inputs, targets) = sequence(&a, 2, 6);
        let mut adam = Adam::new(p.parameters());
        let first = p.sequence_loss(&inputs, &targets, &mut vec![0f32; p.parameters()], 1.0).total;
        for _ in 0..200 {
            let mut g = vec![0f32; p.parameters()];
            p.sequence_loss(&inputs, &targets, &mut g, 1.0);
            adam.step(&mut p, &g, 3e-3, 0.0);
        }
        let last = p.sequence_loss(&inputs, &targets, &mut vec![0f32; p.parameters()], 1.0).total;
        assert!(last < first * 0.3, "{first} -> {last}");
        let q = Policy::from_bytes(a.clone(), &p.to_bytes()).unwrap();
        assert_eq!(q.w, p.w);
        let r = Adam::from_bytes(&adam.to_bytes(), p.parameters()).unwrap();
        assert_eq!((r.t, r.m.len()), (adam.t, adam.m.len()));
        assert!(Policy::from_bytes(a, &[0u8; 8]).is_err());
    }
}

// ---------------------------------------------------------------------------------------------
// Skills carried by a model
// ---------------------------------------------------------------------------------------------

/// Where a skill came from (generations build on this).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lineage {
    pub generation: u32,
    /// The skill (or model) this one started from; none for a new network.
    pub parent: Option<String>,
    /// How its weights came to be: `new`, then each training method in order.
    pub method: String,
}

/// What a model's file says about one of its skills (the weights follow in the file).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillSpec {
    /// The game it plays (Yendory Tools' profile id); one skill per game.
    pub game: String,
    /// The game's name when the skill was made.
    pub name: String,
    pub size: Size,
    pub seed: u64,
    pub arch: Arch,
    /// The controls the heads stand for, in order.
    pub keys: Vec<String>,
    pub buttons: Vec<String>,
    pub lineage: Lineage,
    /// Training steps taken in all, and the step the accepted weights were kept at.
    pub steps: u64,
    pub accepted_step: u64,
    pub created: u64,
    /// The trainer's own record (Yendory Tools: every training run and its measurements),
    /// kept as written.
    #[serde(default)]
    pub notes: serde_json::Value,
}

/// A skill: its description, the accepted weights (what plays), and the latest weights with
/// the optimizer's state (to continue training from where it stopped).
#[derive(Clone, Debug)]
pub struct Skill {
    pub spec: SkillSpec,
    pub accepted: Vec<f32>,
    pub latest: Vec<f32>,
    pub adam: Adam,
}

impl Skill {
    /// A new, untrained skill.
    pub fn new(spec: SkillSpec) -> Result<Skill, String> {
        let p = Policy::new(spec.arch.clone(), spec.seed)?;
        let n = p.parameters();
        Ok(Skill { accepted: p.w.clone(), latest: p.w, adam: Adam::new(n), spec })
    }

    /// Weights in this skill's layout.
    pub fn parameters(&self) -> usize {
        layout_len(&self.spec.arch)
    }

    /// The network with the accepted weights (what plays).
    pub fn policy(&self) -> Result<Policy, String> {
        let mut p = Policy::new(self.spec.arch.clone(), 0)?;
        p.w.clone_from(&self.accepted);
        Ok(p)
    }

    /// The network with the latest weights, and the optimizer (to continue training).
    pub fn latest_policy(&self) -> Result<(Policy, Adam), String> {
        let mut p = Policy::new(self.spec.arch.clone(), 0)?;
        p.w.clone_from(&self.latest);
        Ok((p, self.adam.clone()))
    }

    /// Checks a skill read from somewhere else (a file, a program) before it is kept.
    pub fn check(&self) -> Result<(), String> {
        self.spec.arch.check()?;
        if self.spec.game.is_empty() || self.spec.game.len() > 64 {
            return Err("A game skill needs the game's id".into());
        }
        if self.spec.keys.len() != self.spec.arch.keys || self.spec.buttons.len() != self.spec.arch.buttons {
            return Err("A game skill's controls don't match its network".into());
        }
        let n = self.parameters();
        if n > MAX_SKILL_PARAMETERS {
            return Err(format!("A game skill of {n} weights is larger than the Engine allows"));
        }
        for v in [&self.accepted, &self.latest, &self.adam.m, &self.adam.v] {
            if v.len() != n {
                return Err(format!("A game skill's weights hold {} values; its network has {n}", v.len()));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err("A game skill holds values that are not numbers".into());
            }
        }
        Ok(())
    }
}

/// The most weights one skill may have.
pub const MAX_SKILL_PARAMETERS: usize = 64_000_000;

/// Weights in an architecture's layout.
pub fn layout_len(a: &Arch) -> usize {
    layout(a).last().map(|s| s.offset + s.rows * s.cols).unwrap_or(0)
}
