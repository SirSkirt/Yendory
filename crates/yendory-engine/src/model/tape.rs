//! Reverse-mode tensor primitives for Claude's Nova-2 reference architecture.
//! Row-major matrices; no external neural framework.
use std::collections::HashMap;

#[derive(Clone)]
enum Op {
    Input,
    Embed {
        table: usize,
        ids: Vec<usize>,
        d: usize,
    },
    Linear {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        rows: usize,
        cols: usize,
        out: usize,
    },
    Add {
        a: usize,
        b: usize,
    },
    Mul {
        a: usize,
        b: usize,
    },
    Silu {
        src: usize,
    },
    Sigmoid {
        src: usize,
    },
    Norm {
        src: usize,
        gain: usize,
        cols: usize,
    },
    Rope {
        src: usize,
        heads: usize,
        dim: usize,
        base: f32,
    },
    /// Causal attention. Only each row's softmax maximum and normaliser are kept
    /// (`[head][row]`); the backward pass recomputes the weights from them exactly as the
    /// forward pass made them, so memory grows with the context, not with its square.
    Attention {
        q: usize,
        k: usize,
        v: usize,
        t: usize,
        d: usize,
        heads: usize,
        max: Vec<f32>,
        total: Vec<f32>,
    },
    /// Unfused cross-entropy, kept to test [`Op::HeadXent`] against.
    #[cfg(test)]
    Xent {
        logits: usize,
        targets: Vec<usize>,
        mask: Vec<f32>,
        vocab: usize,
        probs: Vec<f32>,
        denom: f32,
    },
    /// Output layer + cross-entropy in one step: logits are computed a few rows at a time and
    /// recomputed in the backward pass, so memory does not grow with rows × vocabulary.
    HeadXent {
        h: usize,
        embed: usize,
        bias: usize,
        d: usize,
        vocab: usize,
        targets: Vec<usize>,
        mask: Vec<f32>,
        denom: f32,
    },
}
#[derive(Clone)]
struct Node {
    values: Vec<f32>,
    grad: Vec<f32>,
    op: Op,
    /// Parameter nodes read the model's weights in place (this range of them) and add their
    /// gradient straight into the caller's gradient buffer, so no weight is copied per row.
    param: Option<(usize, usize)>,
}
/// The operations the model's forward pass is written in. The CPU [`Tape`] and the GPU graph
/// (`gpu::GpuGraph`) both implement them, so there is one definition of the network.
pub(crate) trait Graph {
    fn parameter(&mut self, name: &str, range: std::ops::Range<usize>) -> usize;
    fn input(&mut self, x: Vec<f32>) -> usize;
    /// A constant zero node shaped like `like` (ablations).
    fn zeros_like(&mut self, like: usize) -> usize;
    fn embed(&mut self, ids: &[usize], table: usize, d: usize) -> usize;
    fn linear(&mut self, x: usize, w: usize, b: Option<usize>, in_dim: usize, out_dim: usize) -> usize;
    fn add(&mut self, a: usize, b: usize) -> usize;
    fn mul(&mut self, a: usize, b: usize) -> usize;
    fn silu(&mut self, src: usize) -> usize;
    fn sigmoid(&mut self, src: usize) -> usize;
    fn norm(&mut self, src: usize, gain: usize, d: usize) -> usize;
    fn rope_dim(&mut self, src: usize, heads: usize, d: usize, base: f32) -> usize;
    /// Causal attention over sequences of `t` tokens.
    fn attention(&mut self, q: usize, k: usize, v: usize, t: usize, d: usize, heads: usize) -> usize;
}

impl Graph for Tape<'_> {
    fn parameter(&mut self, name: &str, range: std::ops::Range<usize>) -> usize {
        Tape::parameter(self, name, range)
    }
    fn input(&mut self, x: Vec<f32>) -> usize {
        Tape::input(self, x)
    }
    fn zeros_like(&mut self, like: usize) -> usize {
        let n = self.v(like).len();
        Tape::input(self, vec![0.; n])
    }
    fn embed(&mut self, ids: &[usize], table: usize, d: usize) -> usize {
        Tape::embed(self, ids, table, d)
    }
    fn linear(&mut self, x: usize, w: usize, b: Option<usize>, in_dim: usize, out_dim: usize) -> usize {
        Tape::linear(self, x, w, b, in_dim, out_dim)
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        Tape::add(self, a, b)
    }
    fn mul(&mut self, a: usize, b: usize) -> usize {
        Tape::mul(self, a, b)
    }
    fn silu(&mut self, src: usize) -> usize {
        Tape::silu(self, src)
    }
    fn sigmoid(&mut self, src: usize) -> usize {
        Tape::sigmoid(self, src)
    }
    fn norm(&mut self, src: usize, gain: usize, d: usize) -> usize {
        Tape::norm(self, src, gain, d)
    }
    fn rope_dim(&mut self, src: usize, heads: usize, d: usize, base: f32) -> usize {
        Tape::rope_dim(self, src, heads, d, base)
    }
    fn attention(&mut self, q: usize, k: usize, v: usize, t: usize, d: usize, heads: usize) -> usize {
        Tape::attention(self, q, k, v, t, d, heads)
    }
}

pub(crate) struct Tape<'a> {
    nodes: Vec<Node>,
    params: HashMap<String, usize>,
    weights: &'a [f32],
    /// Where parameter gradients are added (`None` for forward-only tapes).
    wgrad: Option<&'a mut [f32]>,
}
impl<'a> Tape<'a> {
    pub(crate) fn new(weights: &'a [f32], wgrad: Option<&'a mut [f32]>) -> Self {
        Self { nodes: Vec::new(), params: HashMap::new(), weights, wgrad }
    }
    fn node(&mut self, v: Vec<f32>, op: Op) -> usize {
        let id = self.nodes.len();
        // The gradient is allocated when the backward pass first reaches this node, so the
        // forward pass holds only activations.
        self.nodes.push(Node { values: v, grad: Vec::new(), op, param: None });
        id
    }
    /// The values of a node (for a parameter, the model's weights).
    fn v(&self, id: usize) -> &[f32] {
        match self.nodes[id].param {
            Some((a, b)) => &self.weights[a..b],
            None => &self.nodes[id].values,
        }
    }
    pub(crate) fn input(&mut self, x: Vec<f32>) -> usize {
        self.node(x, Op::Input)
    }
    pub(crate) fn parameter(&mut self, name: &str, range: std::ops::Range<usize>) -> usize {
        if let Some(&id) = self.params.get(name) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            values: Vec::new(),
            grad: Vec::new(),
            op: Op::Input,
            param: Some((range.start, range.end)),
        });
        self.params.insert(name.to_owned(), id);
        id
    }
    pub(crate) fn values(&self, id: usize) -> &[f32] {
        self.v(id)
    }
    pub(crate) fn embed(&mut self, ids: &[usize], table: usize, d: usize) -> usize {
        let src = self.v(table);
        let mut out = vec![0.; ids.len() * d];
        for (i, &token) in ids.iter().enumerate() {
            out[i * d..(i + 1) * d].copy_from_slice(&src[token * d..(token + 1) * d]);
        }
        self.node(out, Op::Embed { table, ids: ids.to_vec(), d })
    }
    pub(crate) fn linear(
        &mut self,
        x: usize,
        w: usize,
        b: Option<usize>,
        in_dim: usize,
        out_dim: usize,
    ) -> usize {
        let input = self.v(x);
        let mat = self.v(w);
        assert_eq!(input.len() % in_dim, 0);
        assert_eq!(mat.len(), out_dim * in_dim);
        let rows = input.len() / in_dim;
        let bias = b.map(|id| self.v(id));
        let out = super::linalg::linear(input, mat, bias, rows, in_dim, out_dim);
        self.node(out, Op::Linear { input: x, weight: w, bias: b, rows, cols: in_dim, out: out_dim })
    }
    pub(crate) fn add(&mut self, a: usize, b: usize) -> usize {
        assert_eq!(self.nodes[a].values.len(), self.nodes[b].values.len());
        let v = self.nodes[a].values.iter().zip(&self.nodes[b].values).map(|(x, y)| x + y).collect();
        self.node(v, Op::Add { a, b })
    }
    pub(crate) fn mul(&mut self, a: usize, b: usize) -> usize {
        assert_eq!(self.nodes[a].values.len(), self.nodes[b].values.len());
        let v = self.nodes[a].values.iter().zip(&self.nodes[b].values).map(|(x, y)| x * y).collect();
        self.node(v, Op::Mul { a, b })
    }
    pub(crate) fn silu(&mut self, src: usize) -> usize {
        let v = self.nodes[src].values.iter().map(|&x| x / (1. + (-x).exp())).collect();
        self.node(v, Op::Silu { src })
    }
    pub(crate) fn sigmoid(&mut self, src: usize) -> usize {
        let v = self.nodes[src].values.iter().map(|&x| 1. / (1. + (-x).exp())).collect();
        self.node(v, Op::Sigmoid { src })
    }
    pub(crate) fn norm(&mut self, src: usize, gain: usize, d: usize) -> usize {
        let x = self.v(src);
        let g = self.v(gain);
        let mut out = vec![0.; x.len()];
        assert_eq!(x.len() % d, 0);
        assert_eq!(g.len(), d);
        for row in 0..x.len() / d {
            let s = &x[row * d..(row + 1) * d];
            let inv = 1. / ((s.iter().map(|z| z * z).sum::<f32>() / d as f32) + 1e-6).sqrt();
            for j in 0..d {
                out[row * d + j] = s[j] * inv * g[j];
            }
        }
        self.node(out, Op::Norm { src, gain, cols: d })
    }
    pub(crate) fn rope_dim(&mut self, src: usize, heads: usize, d: usize, base: f32) -> usize {
        let x = &self.nodes[src].values;
        assert_eq!(x.len() % d, 0);
        let t = x.len() / d;
        let hd = d / heads;
        assert_eq!(hd % 2, 0);
        let half = hd / 2;
        let mut out = x.clone();
        for row in 0..t {
            for h in 0..heads {
                for j in 0..half {
                    let ang = row as f32 * base.powf(-2. * j as f32 / hd as f32);
                    let (sn, cs) = ang.sin_cos();
                    let a = row * d + h * hd + j;
                    let b = a + half;
                    out[a] = x[a] * cs - x[b] * sn;
                    out[b] = x[b] * cs + x[a] * sn;
                }
            }
        }
        self.node(out, Op::Rope { src, heads, dim: d, base })
    }
    /// Scaled scores of one query row against keys `0..=row` of one head.
    fn scores(
        qv: &[f32],
        kv: &[f32],
        row: usize,
        h0: usize,
        hd: usize,
        d: usize,
        scale: f32,
        out: &mut Vec<f32>,
    ) {
        let qr = &qv[row * d + h0..row * d + h0 + hd];
        out.clear();
        out.extend((0..=row).map(|col| super::linalg::dot(qr, &kv[col * d + h0..col * d + h0 + hd]) * scale));
    }
    pub(crate) fn attention(
        &mut self,
        q: usize,
        k: usize,
        v: usize,
        t: usize,
        d: usize,
        heads: usize,
    ) -> usize {
        assert_eq!(self.nodes[q].values.len(), t * d);
        let hd = d / heads;
        let scale = 1. / (hd as f32).sqrt();
        let qv = &self.nodes[q].values;
        let kv = &self.nodes[k].values;
        let vv = &self.nodes[v].values;
        let mut output = vec![0.; t * d];
        let (mut maxs, mut totals) = (vec![0f32; heads * t], vec![0f32; heads * t]);
        let mut p = Vec::with_capacity(t);
        for head in 0..heads {
            let h0 = head * hd;
            for row in 0..t {
                Self::scores(qv, kv, row, h0, hd, d, scale, &mut p);
                let maximum = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0f32;
                for x in p.iter_mut() {
                    *x = (*x - maximum).exp();
                    total += *x;
                }
                let out = &mut output[row * d + h0..row * d + h0 + hd];
                for (col, x) in p.iter().enumerate() {
                    super::linalg::axpy(out, x / total, &vv[col * d + h0..col * d + h0 + hd]);
                }
                maxs[head * t + row] = maximum;
                totals[head * t + row] = total;
            }
        }
        self.node(output, Op::Attention { q, k, v, t, d, heads, max: maxs, total: totals })
    }
    #[cfg(test)]
    pub(crate) fn xent(&mut self, logits: usize, targets: &[usize], mask: &[f32], vocab: usize) -> usize {
        let vals = &self.nodes[logits].values;
        assert_eq!(vals.len(), targets.len() * vocab);
        let mut p = vec![0.; vals.len()];
        let denom = mask.iter().sum::<f32>().max(1.);
        let mut loss = 0f32;
        for row in 0..targets.len() {
            let off = row * vocab;
            let max = vals[off..off + vocab].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.;
            for j in 0..vocab {
                p[off + j] = (vals[off + j] - max).exp();
                sum += p[off + j];
            }
            for j in 0..vocab {
                p[off + j] /= sum;
            }
            loss -= mask[row] * p[off + targets[row]].max(1e-30).ln() / denom;
        }
        self.node(
            vec![loss],
            Op::Xent { logits, targets: targets.to_vec(), mask: mask.to_vec(), vocab, probs: p, denom },
        )
    }
    /// Rows of logits computed at once by [`Tape::head_xent`] (about 4M floats per chunk).
    fn head_chunk(vocab: usize) -> usize {
        ((4 << 20) / vocab.max(1)).clamp(1, 64)
    }
    /// Softmax of one row of logits in place (the same steps as [`Tape::xent`]).
    fn softmax_row(z: &mut [f32]) {
        let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.;
        for v in z.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in z.iter_mut() {
            *v /= sum;
        }
    }
    /// Mean cross-entropy of `h · embedᵀ + bias` (the tied output layer), without keeping the
    /// rows × vocabulary logits: equal to `linear` followed by `xent`.
    pub(crate) fn head_xent(
        &mut self,
        h: usize,
        embed: usize,
        bias: usize,
        d: usize,
        vocab: usize,
        targets: &[usize],
        mask: &[f32],
    ) -> usize {
        let hv = self.v(h);
        assert_eq!(hv.len(), targets.len() * d);
        let (e, b) = (self.v(embed), self.v(bias));
        let denom = mask.iter().sum::<f32>().max(1.);
        let chunk = Self::head_chunk(vocab);
        let mut loss = 0f32;
        for r0 in (0..targets.len()).step_by(chunk) {
            let rows = chunk.min(targets.len() - r0);
            let mut z = super::linalg::linear(&hv[r0 * d..(r0 + rows) * d], e, Some(b), rows, d, vocab);
            for (r, zr) in z.chunks_exact_mut(vocab).enumerate() {
                Self::softmax_row(zr);
                loss -= mask[r0 + r] * zr[targets[r0 + r]].max(1e-30).ln() / denom;
            }
        }
        self.node(
            vec![loss],
            Op::HeadXent { h, embed, bias, d, vocab, targets: targets.to_vec(), mask: mask.to_vec(), denom },
        )
    }
    fn accum(&mut self, id: usize, delta: &[f32]) {
        let target = match self.nodes[id].param {
            Some((a, b)) => match self.wgrad.as_deref_mut() {
                Some(g) => &mut g[a..b],
                None => return,
            },
            None => {
                let node = &mut self.nodes[id];
                if node.grad.is_empty() {
                    node.grad = vec![0.; node.values.len()];
                }
                &mut node.grad[..]
            }
        };
        for (a, b) in target.iter_mut().zip(delta) {
            *a += *b;
        }
    }
    /// Back-propagates from `root`, whose gradient is `seed` (1 for the plain loss; a row weight
    /// to accumulate a weighted sum of rows). Parameter gradients are added to the buffer given
    /// to [`Tape::new`].
    pub(crate) fn backward(&mut self, root: usize, seed: f32) {
        self.nodes[root].grad = vec![seed];
        for id in (0..self.nodes.len()).rev() {
            if matches!(self.nodes[id].op, Op::Input) {
                continue;
            }
            // Each node is visited once, after everything that uses it: its op and gradient can
            // be moved out instead of copied.
            let op = std::mem::replace(&mut self.nodes[id].op, Op::Input);
            let upstream = std::mem::take(&mut self.nodes[id].grad);
            if upstream.is_empty() {
                // Nothing downstream depends on this node: its gradient is zero.
                continue;
            }
            match op {
                Op::Input => {}
                Op::Add { a, b } => {
                    self.accum(a, &upstream);
                    self.accum(b, &upstream);
                }
                Op::Mul { a, b } => {
                    let av = self.nodes[a].values.clone();
                    let bv = self.nodes[b].values.clone();
                    self.accum(a, &upstream.iter().zip(&bv).map(|(g, v)| g * v).collect::<Vec<_>>());
                    self.accum(b, &upstream.iter().zip(&av).map(|(g, v)| g * v).collect::<Vec<_>>());
                }
                Op::Sigmoid { src } => {
                    let v = &self.nodes[id].values;
                    let g = upstream.iter().zip(v).map(|(g, s)| g * s * (1. - s)).collect::<Vec<_>>();
                    self.accum(src, &g);
                }
                Op::Silu { src } => {
                    let x = &self.nodes[src].values;
                    let g = upstream
                        .iter()
                        .zip(x)
                        .map(|(g, x)| {
                            let s = 1. / (1. + (-x).exp());
                            g * s * (1. + x * (1. - s))
                        })
                        .collect::<Vec<_>>();
                    self.accum(src, &g);
                }
                Op::Embed { table, ids, d } => {
                    let (a, _) = self.nodes[table].param.expect("embedding table is a parameter");
                    if let Some(g) = self.wgrad.as_deref_mut() {
                        for (row, tok) in ids.iter().enumerate() {
                            super::linalg::axpy(
                                &mut g[a + tok * d..a + (tok + 1) * d],
                                1.,
                                &upstream[row * d..(row + 1) * d],
                            );
                        }
                    }
                }
                Op::Linear { input, weight, bias, rows, cols, out } => {
                    let (a, b) = self.nodes[weight].param.expect("linear weight is a parameter");
                    let weights = self.weights;
                    let mut scratch = Vec::new();
                    let dw = match self.wgrad.as_deref_mut() {
                        Some(g) => &mut g[a..b],
                        None => {
                            scratch.resize(b - a, 0.);
                            &mut scratch[..]
                        }
                    };
                    let dx = super::linalg::linear_backward(
                        &upstream,
                        &self.nodes[input].values,
                        &weights[a..b],
                        rows,
                        cols,
                        out,
                        dw,
                        None,
                    );
                    self.accum(input, &dx);
                    if let Some(bias) = bias {
                        let mut db = vec![0f32; out];
                        for grow in upstream.chunks_exact(out) {
                            super::linalg::axpy(&mut db, 1., grow);
                        }
                        self.accum(bias, &db);
                    }
                }
                Op::Norm { src, gain, cols } => {
                    let x = self.nodes[src].values.clone();
                    let gamma = self.v(gain).to_vec();
                    let mut dx = vec![0.; x.len()];
                    let mut dg = vec![0.; cols];
                    for row in 0..x.len() / cols {
                        let start = row * cols;
                        let xr = &x[start..start + cols];
                        let inv = 1. / (xr.iter().map(|z| z * z).sum::<f32>() / cols as f32 + 1e-6).sqrt();
                        let mut dot = 0.;
                        for j in 0..cols {
                            dot += upstream[start + j] * gamma[j] * xr[j];
                        }
                        for j in 0..cols {
                            dg[j] += upstream[start + j] * xr[j] * inv;
                            dx[start + j] = upstream[start + j] * gamma[j] * inv
                                - xr[j] * inv * inv * inv * dot / cols as f32;
                        }
                    }
                    self.accum(src, &dx);
                    self.accum(gain, &dg);
                }
                Op::Rope { src, heads, dim: d, base } => {
                    let t = upstream.len() / d;
                    let hd = d / heads;
                    let half = hd / 2;
                    let mut dx = upstream.clone();
                    for row in 0..t {
                        for h in 0..heads {
                            for j in 0..half {
                                let angle = row as f32 * base.powf(-2. * j as f32 / hd as f32);
                                let (sn, cs) = angle.sin_cos();
                                let a = row * d + h * hd + j;
                                let b = a + half;
                                dx[a] = upstream[a] * cs + upstream[b] * sn;
                                dx[b] = upstream[b] * cs - upstream[a] * sn;
                            }
                        }
                    }
                    self.accum(src, &dx);
                }
                Op::Attention { q, k, v, t, d, heads, max, total } => {
                    let hd = d / heads;
                    let sc = 1. / (hd as f32).sqrt();
                    let (qv, kv, vv) = (&self.nodes[q].values, &self.nodes[k].values, &self.nodes[v].values);
                    let mut dq = vec![0f32; t * d];
                    let mut dk = vec![0f32; t * d];
                    let mut dv = vec![0f32; t * d];
                    let (mut p, mut dp) = (Vec::with_capacity(t), Vec::with_capacity(t));
                    for head in 0..heads {
                        let h0 = head * hd;
                        for row in 0..t {
                            // The same weights as the forward pass: exp(s - max) / total.
                            Self::scores(qv, kv, row, h0, hd, d, sc, &mut p);
                            let (m, tot) = (max[head * t + row], total[head * t + row]);
                            for x in p.iter_mut() {
                                *x = (*x - m).exp() / tot;
                            }
                            let up = &upstream[row * d + h0..row * d + h0 + hd];
                            dp.clear();
                            let mut weighted = 0f32;
                            for (col, &w) in p.iter().enumerate() {
                                let g = super::linalg::dot(up, &vv[col * d + h0..col * d + h0 + hd]);
                                super::linalg::axpy(&mut dv[col * d + h0..col * d + h0 + hd], w, up);
                                weighted += g * w;
                                dp.push(g);
                            }
                            for (col, (&w, &g)) in p.iter().zip(&dp).enumerate() {
                                let g = w * (g - weighted) * sc;
                                super::linalg::axpy(
                                    &mut dq[row * d + h0..row * d + h0 + hd],
                                    g,
                                    &kv[col * d + h0..col * d + h0 + hd],
                                );
                                super::linalg::axpy(
                                    &mut dk[col * d + h0..col * d + h0 + hd],
                                    g,
                                    &qv[row * d + h0..row * d + h0 + hd],
                                );
                            }
                        }
                    }
                    self.accum(q, &dq);
                    self.accum(k, &dk);
                    self.accum(v, &dv);
                }
                Op::HeadXent { h, embed, bias, d, vocab, targets, mask, denom } => {
                    let (ea, eb) = self.nodes[embed].param.expect("output embedding is a parameter");
                    let (ba, bb) = self.nodes[bias].param.expect("output bias is a parameter");
                    let weights = self.weights;
                    let (e, bv) = (&weights[ea..eb], &weights[ba..bb]);
                    let mut scratch = Vec::new();
                    let wgrad: &mut [f32] = match self.wgrad.as_deref_mut() {
                        Some(g) => g,
                        None => {
                            scratch.resize(weights.len(), 0.);
                            &mut scratch
                        }
                    };
                    // The embedding and bias ranges never overlap: split the buffer between them.
                    let (de, db) = if eb <= ba {
                        let (lo, hi) = wgrad.split_at_mut(ba);
                        (&mut lo[ea..eb], &mut hi[..bb - ba])
                    } else {
                        let (lo, hi) = wgrad.split_at_mut(ea);
                        (&mut hi[..eb - ea], &mut lo[ba..bb])
                    };
                    let hv = &self.nodes[h].values;
                    let mut dh = vec![0f32; hv.len()];
                    let chunk = Self::head_chunk(vocab);
                    for r0 in (0..targets.len()).step_by(chunk) {
                        let rows = chunk.min(targets.len() - r0);
                        let hc = &hv[r0 * d..(r0 + rows) * d];
                        let mut g = super::linalg::linear(hc, e, Some(bv), rows, d, vocab);
                        for (r, gr) in g.chunks_exact_mut(vocab).enumerate() {
                            Self::softmax_row(gr);
                            gr[targets[r0 + r]] -= 1.;
                            let scale = mask[r0 + r] * upstream[0] / denom;
                            for v in gr.iter_mut() {
                                *v *= scale;
                            }
                        }
                        let dhc =
                            super::linalg::linear_backward(&g, hc, e, rows, d, vocab, de, Some(&mut *db));
                        dh[r0 * d..(r0 + rows) * d].copy_from_slice(&dhc);
                    }
                    self.accum(h, &dh);
                }
                #[cfg(test)]
                Op::Xent { logits, targets, mask, vocab, mut probs, denom } => {
                    for row in 0..targets.len() {
                        probs[row * vocab + targets[row]] -= 1.;
                        for j in 0..vocab {
                            probs[row * vocab + j] *= mask[row] * upstream[0] / denom;
                        }
                    }
                    self.accum(logits, &probs);
                }
            }
            // Everything that used this node's output was processed before it (the backward
            // pass runs in reverse order), so its activations are no longer needed.
            self.nodes[id].values = Vec::new();
        }
    }
}
