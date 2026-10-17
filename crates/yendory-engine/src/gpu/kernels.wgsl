// Yendory Engine GPU kernels: the training step of the Bicameral network (forward, backward
// and AdamW) as compute shaders. Every kernel computes the same values as the CPU tape
// (src/model/tape.rs); only the order in which floating-point numbers are added differs.
//
// All kernels share one binding layout: a uniform block of 16 parameters (four vec4<u32>, read
// as p0..p15; floats are passed as their bits), seven f32 storage buffers and one u32 buffer.
// Offsets are element indices into whole buffers.

struct Params {
    a: vec4<u32>,
    b: vec4<u32>,
    c: vec4<u32>,
    d: vec4<u32>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> f0: array<f32>;
@group(0) @binding(2) var<storage, read_write> f1: array<f32>;
@group(0) @binding(3) var<storage, read_write> f2: array<f32>;
@group(0) @binding(4) var<storage, read_write> f3: array<f32>;
@group(0) @binding(5) var<storage, read_write> f4: array<f32>;
@group(0) @binding(6) var<storage, read_write> f5: array<f32>;
@group(0) @binding(7) var<storage, read_write> f6: array<f32>;
@group(0) @binding(8) var<storage, read_write> u0: array<u32>;

fn id(wg: vec3<u32>, nwg: vec3<u32>, li: u32) -> u32 {
    return (wg.y * nwg.x + wg.x) * 256u + li;
}
fn fl(x: u32) -> f32 {
    return bitcast<f32>(x);
}
fn sigm(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// ---- elementwise --------------------------------------------------------------------------

// f0[off + i] = value, i < n.   p: n, off, value
@compute @workgroup_size(256)
fn fill(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i < p.a.x) { f0[p.a.y + i] = fl(p.a.z); }
}

// f1[off1 + i] += f0[off0 + i], i < n.   p: n, off0, off1
@compute @workgroup_size(256)
fn add_into(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i < p.a.x) { f1[p.a.z + i] = f1[p.a.z + i] + f0[p.a.y + i]; }
}

// f2 = f0 + f1 (op 0), f0 * f1 (op 1).   p: n, op
@compute @workgroup_size(256)
fn binary(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    if (p.a.y == 0u) { f2[i] = f0[i] + f1[i]; } else { f2[i] = f0[i] * f1[i]; }
}

// f1 = silu(f0) (op 0), sigmoid(f0) (op 1).   p: n, op
@compute @workgroup_size(256)
fn unary(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let x = f0[i];
    if (p.a.y == 0u) { f1[i] = x / (1.0 + exp(-x)); } else { f1[i] = 1.0 / (1.0 + exp(-x)); }
}

// Backward of silu (op 0: f1 = the input x) or sigmoid (op 1: f1 = the output s):
// f2 = f0 (upstream) * derivative.   p: n, op
@compute @workgroup_size(256)
fn unary_back(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let g = f0[i];
    if (p.a.y == 0u) {
        let x = f1[i];
        let s = sigm(x);
        f2[i] = g * s * (1.0 + x * (1.0 - s));
    } else {
        let s = f1[i];
        f2[i] = g * s * (1.0 - s);
    }
}

// ---- embedding ----------------------------------------------------------------------------

// f1[r * d + j] = f0[off + u0[r] * d + j].   p: n = rows * d, d, off
@compute @workgroup_size(256)
fn embed(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let d = p.a.y;
    let r = i / d;
    let j = i % d;
    f1[i] = f0[p.a.z + u0[r] * d + j];
}

// One thread per column: f1[off + u0[r] * d + j] += f0[r * d + j] for every row in order (no
// two threads write the same element).   p: d, rows, off
@compute @workgroup_size(256)
fn embed_back(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let j = id(wg, nwg, li);
    let d = p.a.x;
    if (j >= d) { return; }
    for (var r = 0u; r < p.a.y; r = r + 1u) {
        let k = p.a.z + u0[r] * d + j;
        f1[k] = f1[k] + f0[r * d + j];
    }
}

// ---- matrix products ----------------------------------------------------------------------

// C (f2) = (or +=) A (f0) x B (f1), with strided views:
// A(i, k) = f0[aoff + i * ars + k * acs], B(k, j) = f1[boff + k * brs + j * bcs],
// C(i, j) = f2[coff + i * n + j].
// p: m, n, k, acc | aoff, ars, acs, boff | brs, bcs, coff
// 64 x 64 tiles, 256 threads, each thread 4 x 4 outputs.
var<workgroup> ta: array<f32, 1024>; // 64 rows x 16
var<workgroup> tb: array<f32, 1024>; // 16 x 64 columns
@compute @workgroup_size(256)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let m = p.a.x;
    let n = p.a.y;
    let kk = p.a.z;
    let aoff = p.b.x;
    let ars = p.b.y;
    let acs = p.b.z;
    let boff = p.b.w;
    let brs = p.c.x;
    let bcs = p.c.y;
    let coff = p.c.z;
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    let tr = (li / 16u) * 4u;
    let tc = (li % 16u) * 4u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }
    for (var k0 = 0u; k0 < kk; k0 = k0 + 16u) {
        // Each thread loads 4 elements of each tile.
        for (var e = 0u; e < 4u; e = e + 1u) {
            let flat = li * 4u + e; // 0..1024
            let r = flat / 16u;
            let c = flat % 16u;
            let gi = row0 + r;
            let gk = k0 + c;
            var va = 0.0;
            if (gi < m && gk < kk) { va = f0[aoff + gi * ars + gk * acs]; }
            ta[r * 16u + c] = va;
            let br = flat / 64u;
            let bc = flat % 64u;
            let bk = k0 + br;
            let bj = col0 + bc;
            var vb = 0.0;
            if (bk < kk && bj < n) { vb = f1[boff + bk * brs + bj * bcs]; }
            tb[br * 64u + bc] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let b0 = tb[k * 64u + tc];
            let b1 = tb[k * 64u + tc + 1u];
            let b2 = tb[k * 64u + tc + 2u];
            let b3 = tb[k * 64u + tc + 3u];
            for (var r = 0u; r < 4u; r = r + 1u) {
                let a = ta[(tr + r) * 16u + k];
                acc[r * 4u] = acc[r * 4u] + a * b0;
                acc[r * 4u + 1u] = acc[r * 4u + 1u] + a * b1;
                acc[r * 4u + 2u] = acc[r * 4u + 2u] + a * b2;
                acc[r * 4u + 3u] = acc[r * 4u + 3u] + a * b3;
            }
        }
        workgroupBarrier();
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let gi = row0 + tr + r;
        if (gi >= m) { continue; }
        for (var c = 0u; c < 4u; c = c + 1u) {
            let gj = col0 + tc + c;
            if (gj >= n) { continue; }
            let o = coff + gi * n + gj;
            if (p.a.w == 1u) { f2[o] = f2[o] + acc[r * 4u + c]; } else { f2[o] = acc[r * 4u + c]; }
        }
    }
}

// f1[r * n + j] += f0[boff + j] (bias).   p: total = rows * n, n, boff
@compute @workgroup_size(256)
fn add_bias(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    f1[i] = f1[i] + f0[p.a.z + i % p.a.y];
}

// Column sums: f1[off + j] += sum over rows of f0[r * n + j].   p: n, rows, off
@compute @workgroup_size(256)
fn colsum(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let j = id(wg, nwg, li);
    if (j >= p.a.x) { return; }
    var s = 0.0;
    for (var r = 0u; r < p.a.y; r = r + 1u) { s = s + f0[r * p.a.x + j]; }
    f1[p.a.z + j] = f1[p.a.z + j] + s;
}

// ---- RMS normalisation --------------------------------------------------------------------

// f3[r] = 1 / sqrt(mean(x^2) + 1e-6) of row r of f0.   p: d, rows
@compute @workgroup_size(256)
fn row_inv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = id(wg, nwg, li);
    if (r >= p.a.y) { return; }
    let d = p.a.x;
    var s = 0.0;
    for (var j = 0u; j < d; j = j + 1u) { let x = f0[r * d + j]; s = s + x * x; }
    f3[r] = 1.0 / sqrt(s / f32(d) + 1e-6);
}

// f2[i] = f0[i] * inv(row) * gain[j]; gain = f1[goff + j], inv = f3[row].   p: n, d, goff
@compute @workgroup_size(256)
fn norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let d = p.a.y;
    f2[i] = f0[i] * f3[i / d] * f1[p.a.z + i % d];
}

// Backward, one thread per row: dx (f4) from upstream (f0), x (f1), gain (f2 at goff) and
// inv (f3).   p: d, rows, goff
@compute @workgroup_size(256)
fn norm_back_x(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = id(wg, nwg, li);
    if (r >= p.a.y) { return; }
    let d = p.a.x;
    let inv = f3[r];
    var dot = 0.0;
    for (var j = 0u; j < d; j = j + 1u) { dot = dot + f0[r * d + j] * f2[p.a.z + j] * f1[r * d + j]; }
    for (var j = 0u; j < d; j = j + 1u) {
        let x = f1[r * d + j];
        f4[r * d + j] = f0[r * d + j] * f2[p.a.z + j] * inv - x * inv * inv * inv * dot / f32(d);
    }
}

// Backward, one thread per column: the gain's gradient, added at f4[goff + j].
// p: d, rows, goff
@compute @workgroup_size(256)
fn norm_back_g(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let j = id(wg, nwg, li);
    let d = p.a.x;
    if (j >= d) { return; }
    var s = 0.0;
    for (var r = 0u; r < p.a.y; r = r + 1u) { s = s + f0[r * d + j] * f1[r * d + j] * f3[r]; }
    f4[p.a.z + j] = f4[p.a.z + j] + s;
}

// ---- rotary positions ---------------------------------------------------------------------

// One thread per (row, head, pair); position = row % t. sign 0: forward, 1: backward (the
// rotation by minus the angle). f1 from f0.   p: pairs, d, heads, t | base bits, sign
@compute @workgroup_size(256)
fn rope(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let d = p.a.y;
    let heads = p.a.z;
    let hd = d / heads;
    let half = hd / 2u;
    let row = i / (heads * half);
    let rest = i % (heads * half);
    let h = rest / half;
    let j = rest % half;
    let pos = f32(row % p.a.w);
    let ang = pos * pow(fl(p.b.x), -2.0 * f32(j) / f32(hd));
    var sn = sin(ang);
    let cs = cos(ang);
    if (p.b.y == 1u) { sn = -sn; }
    let a = row * d + h * hd + j;
    let b = a + half;
    let xa = f0[a];
    let xb = f0[b];
    f1[a] = xa * cs - xb * sn;
    f1[b] = xb * cs + xa * sn;
}

// ---- attention ----------------------------------------------------------------------------
// q, k, v: rows x d (several sequences of t rows); head dimension hd <= 64.
// stats (f4): max at [0, bht), total at [bht, 2 bht), weighted at [2 bht, 3 bht), indexed
// (sequence * heads + head) * t + row.

// One thread per (sequence, head, row): f3 = output.   p: count = b * heads * t, t, d, heads | scale bits
@compute @workgroup_size(256)
fn attention(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let hd = d / heads;
    let sc = fl(p.b.x);
    let row = i % t;
    let h = (i / t) % heads;
    let s = i / (t * heads);
    let qb = (s * t + row) * d + h * hd;
    var mx = -3.4e38;
    for (var c = 0u; c <= row; c = c + 1u) {
        let kb = (s * t + c) * d + h * hd;
        var dot = 0.0;
        for (var e = 0u; e < hd; e = e + 1u) { dot = dot + f0[qb + e] * f1[kb + e]; }
        mx = max(mx, dot * sc);
    }
    var acc: array<f32, 64>;
    for (var e = 0u; e < hd; e = e + 1u) { acc[e] = 0.0; }
    var total = 0.0;
    for (var c = 0u; c <= row; c = c + 1u) {
        let kb = (s * t + c) * d + h * hd;
        var dot = 0.0;
        for (var e = 0u; e < hd; e = e + 1u) { dot = dot + f0[qb + e] * f1[kb + e]; }
        let w = exp(dot * sc - mx);
        total = total + w;
        for (var e = 0u; e < hd; e = e + 1u) { acc[e] = acc[e] + w * f2[kb + e]; }
    }
    for (var e = 0u; e < hd; e = e + 1u) { f3[qb + e] = acc[e] / total; }
    f4[i] = mx;
    f4[p.a.x + i] = total;
}

// Backward, one thread per query row: dq (f5) and the row's weighted sum (stats).
// f0 upstream, f1 q, f2 k, f3 v.   p: count, t, d, heads | scale bits
@compute @workgroup_size(256)
fn attention_back_q(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let hd = d / heads;
    let sc = fl(p.b.x);
    let row = i % t;
    let h = (i / t) % heads;
    let s = i / (t * heads);
    let qb = (s * t + row) * d + h * hd;
    let mx = f4[i];
    let total = f4[p.a.x + i];
    var weighted = 0.0;
    for (var c = 0u; c <= row; c = c + 1u) {
        let kb = (s * t + c) * d + h * hd;
        var dot = 0.0;
        var g = 0.0;
        for (var e = 0u; e < hd; e = e + 1u) { dot = dot + f1[qb + e] * f2[kb + e]; g = g + f0[qb + e] * f3[kb + e]; }
        weighted = weighted + g * (exp(dot * sc - mx) / total);
    }
    var dq: array<f32, 64>;
    for (var e = 0u; e < hd; e = e + 1u) { dq[e] = 0.0; }
    for (var c = 0u; c <= row; c = c + 1u) {
        let kb = (s * t + c) * d + h * hd;
        var dot = 0.0;
        var g = 0.0;
        for (var e = 0u; e < hd; e = e + 1u) { dot = dot + f1[qb + e] * f2[kb + e]; g = g + f0[qb + e] * f3[kb + e]; }
        let w = exp(dot * sc - mx) / total;
        let gg = w * (g - weighted) * sc;
        for (var e = 0u; e < hd; e = e + 1u) { dq[e] = dq[e] + gg * f2[kb + e]; }
    }
    for (var e = 0u; e < hd; e = e + 1u) { f5[qb + e] = dq[e]; }
    f4[2u * p.a.x + i] = weighted;
}

// Backward, one thread per key column: dk (f5) and dv (f6).   p: count, t, d, heads | scale bits
@compute @workgroup_size(256)
fn attention_back_kv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let hd = d / heads;
    let sc = fl(p.b.x);
    let col = i % t;
    let h = (i / t) % heads;
    let s = i / (t * heads);
    let kb = (s * t + col) * d + h * hd;
    var dk: array<f32, 64>;
    var dv: array<f32, 64>;
    for (var e = 0u; e < hd; e = e + 1u) { dk[e] = 0.0; dv[e] = 0.0; }
    for (var row = col; row < t; row = row + 1u) {
        let qb = (s * t + row) * d + h * hd;
        let si = (s * heads + h) * t + row;
        var dot = 0.0;
        var g = 0.0;
        for (var e = 0u; e < hd; e = e + 1u) { dot = dot + f1[qb + e] * f2[kb + e]; g = g + f0[qb + e] * f3[kb + e]; }
        let w = exp(dot * sc - f4[si]) / f4[p.a.x + si];
        let gg = w * (g - f4[2u * p.a.x + si]) * sc;
        for (var e = 0u; e < hd; e = e + 1u) {
            dv[e] = dv[e] + w * f0[qb + e];
            dk[e] = dk[e] + gg * f1[qb + e];
        }
    }
    for (var e = 0u; e < hd; e = e + 1u) { f5[kb + e] = dk[e]; f6[kb + e] = dv[e]; }
}

// ---- output layer and cross-entropy -------------------------------------------------------

// One thread per row of a chunk of logits (f0, rows x vocab). Targets u0, mask f1 and the row
// losses f2 are indexed from r0. mode 0: the row's loss mask * -ln p(target) / denom into f2.
// mode 1: f0 becomes the logits' gradient (softmax - one-hot) * mask * seed / denom.
// p: vocab, rows, r0, mode | denom bits, seed bits
@compute @workgroup_size(256)
fn xent(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = id(wg, nwg, li);
    if (r >= p.a.y) { return; }
    let v = p.a.x;
    let base = r * v;
    let g = p.a.z + r;
    var mx = -3.4e38;
    for (var j = 0u; j < v; j = j + 1u) { mx = max(mx, f0[base + j]); }
    var sum = 0.0;
    for (var j = 0u; j < v; j = j + 1u) { sum = sum + exp(f0[base + j] - mx); }
    let denom = fl(p.b.x);
    let tg = u0[g];
    if (p.a.w == 0u) {
        let pt = exp(f0[base + tg] - mx) / sum;
        f2[g] = -f1[g] * log(max(pt, 1e-30)) / denom;
    } else {
        let scale = f1[g] * fl(p.b.y) / denom;
        for (var j = 0u; j < v; j = j + 1u) {
            var q = exp(f0[base + j] - mx) / sum;
            if (j == tg) { q = q - 1.0; }
            f0[base + j] = q * scale;
        }
    }
}

// ---- AdamW --------------------------------------------------------------------------------

var<workgroup> part: array<f32, 256>;

// Partial sums of squares of f0[0..n) per workgroup into f1[group].   p: n
@compute @workgroup_size(256)
fn sumsq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    var x = 0.0;
    if (i < p.a.x) { x = f0[i] * f0[i]; }
    part[li] = x;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { part[li] = part[li] + part[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) { f1[wg.y * nwg.x + wg.x] = part[0]; }
}

// One workgroup: f1[n] = sqrt(sum of f1[0..n)).   p: n
@compute @workgroup_size(256)
fn norm_finish(@builtin(local_invocation_index) li: u32) {
    var x = 0.0;
    for (var i = li; i < p.a.x; i = i + 256u) { x = x + f1[i]; }
    part[li] = x;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { part[li] = part[li] + part[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) { f1[p.a.x] = sqrt(part[0]); }
}

// AdamW update of the parameters whose group is in the focus mask. f0 weights, f1 gradients,
// f2 momentum, f3 variance, f4[norm_at] the gradient norm; u0 flags (group bits 1/2/4, decay 8).
// p: n, focus mask, norm_at | lr, bias1, bias2, beta1 bits | beta2 bits
@compute @workgroup_size(256)
fn adam(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    let flags = u0[i];
    if ((flags & p.a.y) == 0u) { return; }
    let clip = 1.0 / max(f4[p.a.z], 1.0);
    let g = f1[i] * clip;
    // The same constants as the CPU: beta1 0.9, beta2 0.95, (1 - beta) computed in f32.
    let b1 = fl(p.b.w);
    let b2 = fl(p.c.x);
    let m = b1 * f2[i] + (1.0 - b1) * g;
    let v = b2 * f3[i] + (1.0 - b2) * g * g;
    f2[i] = m;
    f3[i] = v;
    let mh = m / fl(p.b.y);
    let vh = v / fl(p.b.z);
    var wd = 0.0;
    if ((flags & 8u) != 0u) { wd = 0.01 * f0[i]; }
    f0[i] = f0[i] - fl(p.b.x) * (mh / (sqrt(vh) + 1e-8) + wd);
}
