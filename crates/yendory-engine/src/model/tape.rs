//! Reverse-mode tensor primitives for Claude's Nova-2 reference architecture.
//! Row-major matrices; no external neural framework.
use std::collections::HashMap;

#[derive(Clone)]
enum Op {
    Input,
    Embed { table: usize, ids: Vec<usize>, d: usize },
    Linear { input: usize, weight: usize, bias: Option<usize>, rows: usize, cols: usize, out: usize },
    Add { a: usize, b: usize },
    Mul { a: usize, b: usize },
    Silu { src: usize },
    Sigmoid { src: usize },
    Norm { src: usize, gain: usize, cols: usize },
    Rope { src: usize, heads: usize, dim: usize, base: f32 },
    Attention { q: usize, k: usize, v: usize, t: usize, d: usize, heads: usize, probs: Vec<f32> },
    Xent { logits: usize, targets: Vec<usize>, mask: Vec<f32>, vocab: usize, probs: Vec<f32>, denom: f32 },
}
#[derive(Clone)]
struct Node {
    values: Vec<f32>,
    grad: Vec<f32>,
    op: Op,
}
pub(crate) struct Tape {
    nodes: Vec<Node>,
    params: HashMap<String, usize>,
}
impl Tape {
    pub(crate) fn new() -> Self {
        Self { nodes: Vec::new(), params: HashMap::new() }
    }
    fn node(&mut self, v: Vec<f32>, op: Op) -> usize {
        let id = self.nodes.len();
        let grad = vec![0.; v.len()];
        self.nodes.push(Node { values: v, grad, op });
        id
    }
    pub(crate) fn input(&mut self, x: Vec<f32>) -> usize {
        self.node(x, Op::Input)
    }
    pub(crate) fn parameter(&mut self, name: &str, weights: &[f32]) -> usize {
        if let Some(&id) = self.params.get(name) {
            return id;
        }
        let id = self.input(weights.to_vec());
        self.params.insert(name.to_owned(), id);
        id
    }
    pub(crate) fn values(&self, id: usize) -> &[f32] {
        &self.nodes[id].values
    }
    pub(crate) fn param_grad(&self, name: &str) -> Option<&[f32]> {
        self.params.get(name).map(|&id| self.nodes[id].grad.as_slice())
    }
    pub(crate) fn embed(&mut self, ids: &[usize], table: usize, d: usize) -> usize {
        let src = &self.nodes[table].values;
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
        let input = &self.nodes[x].values;
        let mat = &self.nodes[w].values;
        assert_eq!(input.len() % in_dim, 0);
        assert_eq!(mat.len(), out_dim * in_dim);
        let rows = input.len() / in_dim;
        let mut out = vec![0.; rows * out_dim];
        for row in 0..rows {
            for o in 0..out_dim {
                let wr = &mat[o * in_dim..(o + 1) * in_dim];
                let xr = &input[row * in_dim..(row + 1) * in_dim];
                let mut sum = 0f32;
                for j in 0..in_dim {
                    sum += wr[j] * xr[j];
                }
                out[row * out_dim + o] = sum + b.map(|id| self.nodes[id].values[o]).unwrap_or(0.);
            }
        }
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
        let x = &self.nodes[src].values;
        let g = &self.nodes[gain].values;
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
        let mut p = vec![0f32; heads * t * t];
        let mut output = vec![0.; t * d];
        for head in 0..heads {
            for row in 0..t {
                let idx = (head * t + row) * t;
                let mut maximum = f32::NEG_INFINITY;
                for col in 0..=row {
                    let mut s = 0f32;
                    for j in 0..hd {
                        s += qv[row * d + head * hd + j] * kv[col * d + head * hd + j];
                    }
                    p[idx + col] = s * scale;
                    maximum = maximum.max(p[idx + col]);
                }
                let mut total = 0f32;
                for col in 0..=row {
                    p[idx + col] = (p[idx + col] - maximum).exp();
                    total += p[idx + col];
                }
                for col in 0..=row {
                    p[idx + col] /= total;
                    for j in 0..hd {
                        output[row * d + head * hd + j] += p[idx + col] * vv[col * d + head * hd + j];
                    }
                }
            }
        }
        self.node(output, Op::Attention { q, k, v, t, d, heads, probs: p })
    }
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
    fn accum(&mut self, id: usize, delta: &[f32]) {
        for (a, b) in self.nodes[id].grad.iter_mut().zip(delta) {
            *a += *b;
        }
    }
    pub(crate) fn backward(&mut self, root: usize) {
        self.nodes[root].grad[0] = 1.;
        for id in (0..self.nodes.len()).rev() {
            let op = self.nodes[id].op.clone();
            let upstream = self.nodes[id].grad.clone();
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
                    for (row, tok) in ids.iter().enumerate() {
                        for j in 0..d {
                            self.nodes[table].grad[tok * d + j] += upstream[row * d + j];
                        }
                    }
                }
                Op::Linear { input, weight, bias, rows, cols, out } => {
                    let xx = self.nodes[input].values.clone();
                    let w = self.nodes[weight].values.clone();
                    let mut dx = vec![0.; xx.len()];
                    let mut dw = vec![0.; w.len()];
                    let mut db = vec![0.; out];
                    for row in 0..rows {
                        for o in 0..out {
                            let g = upstream[row * out + o];
                            db[o] += g;
                            for j in 0..cols {
                                dx[row * cols + j] += g * w[o * cols + j];
                                dw[o * cols + j] += g * xx[row * cols + j];
                            }
                        }
                    }
                    self.accum(input, &dx);
                    self.accum(weight, &dw);
                    if let Some(b) = bias {
                        self.accum(b, &db);
                    }
                }
                Op::Norm { src, gain, cols } => {
                    let x = self.nodes[src].values.clone();
                    let gamma = self.nodes[gain].values.clone();
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
                Op::Attention { q, k, v, t, d, heads, probs } => {
                    let hd = d / heads;
                    let sc = 1. / (hd as f32).sqrt();
                    let qv = self.nodes[q].values.clone();
                    let kv = self.nodes[k].values.clone();
                    let vv = self.nodes[v].values.clone();
                    let mut dq = vec![0f32; t * d];
                    let mut dk = vec![0f32; t * d];
                    let mut dv = vec![0f32; t * d];
                    for head in 0..heads {
                        for row in 0..t {
                            let idx = (head * t + row) * t;
                            let mut dp = vec![0f32; row + 1];
                            let mut weighted = 0f32;
                            for col in 0..=row {
                                for j in 0..hd {
                                    dp[col] +=
                                        upstream[row * d + head * hd + j] * vv[col * d + head * hd + j];
                                    dv[col * d + head * hd + j] +=
                                        probs[idx + col] * upstream[row * d + head * hd + j];
                                }
                                weighted += dp[col] * probs[idx + col];
                            }
                            for col in 0..=row {
                                let g = probs[idx + col] * (dp[col] - weighted) * sc;
                                for j in 0..hd {
                                    dq[row * d + head * hd + j] += g * kv[col * d + head * hd + j];
                                    dk[col * d + head * hd + j] += g * qv[row * d + head * hd + j];
                                }
                            }
                        }
                    }
                    self.accum(q, &dq);
                    self.accum(k, &dk);
                    self.accum(v, &dv);
                }
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
        }
    }
}
