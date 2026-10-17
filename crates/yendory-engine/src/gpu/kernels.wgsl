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

// Embedding gradient by token: rows sorted by token id on the CPU, one thread per (token,
// column). u0 = [rows in token order (rows) | segment starts (segs + 1) | segment tokens (segs)].
// f1[off + token * d + j] += sum of f0[row * d + j] over the token's rows (no two threads write
// the same element).   p: d, segs, off, rows
@compute @workgroup_size(256)
fn embed_back(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    let d = p.a.x;
    let segs = p.a.y;
    if (i >= segs * d) { return; }
    let sg = i / d;
    let j = i % d;
    let rows = p.a.w;
    let a = u0[rows + sg];
    let b = u0[rows + sg + 1u];
    let tok = u0[rows + segs + 1u + sg];
    var acc = 0.0;
    for (var k = a; k < b; k = k + 1u) { acc = acc + f0[u0[k] * d + j]; }
    let o = p.a.z + tok * d + j;
    f1[o] = f1[o] + acc;
}

// ---- matrix products ----------------------------------------------------------------------

// C (f2) = (or +=) A (f0) x B (f1), with strided views:
// A(i, k) = f0[aoff + i * ars + k * acs], B(k, j) = f1[boff + k * brs + j * bcs],
// C(i, j) = f2[coff + i * n + j].
// p: m, n, k, acc | aoff, ars, acs, boff | brs, bcs, coff
// 64 x 64 tiles, 256 threads, each thread 4 x 4 outputs.
// Each thread keeps its 4 x 4 outputs in four vec4 registers (no indexed per-thread arrays,
// which some shader compilers place in slow memory).
var<workgroup> ta: array<f32, 1024>; // 64 rows x 16
var<workgroup> tb: array<vec4<f32>, 256>; // 16 x 16 groups of 4 columns
fn gemm_store(o: u32, gi: u32, gj: u32, m: u32, n: u32, v: vec4<f32>) {
    if (gi >= m) { return; }
    let acc = p.a.w == 1u;
    for (var c = 0u; c < 4u; c = c + 1u) {
        if (gj + c < n) {
            var x = v.x;
            if (c == 1u) { x = v.y; } else if (c == 2u) { x = v.z; } else if (c == 3u) { x = v.w; }
            if (acc) { f2[o + c] = f2[o + c] + x; } else { f2[o + c] = x; }
        }
    }
}
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
    let tc = li % 16u;
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < kk; k0 = k0 + 16u) {
        // A: each thread loads 4 elements; B: each thread loads one group of 4 columns.
        for (var e = 0u; e < 4u; e = e + 1u) {
            let flat = li * 4u + e;
            let r = flat / 16u;
            let c = flat % 16u;
            let gi = row0 + r;
            let gk = k0 + c;
            var va = 0.0;
            if (gi < m && gk < kk) { va = f0[aoff + gi * ars + gk * acs]; }
            ta[r * 16u + c] = va;
        }
        let br = li / 16u;
        let bk = k0 + br;
        let bj = col0 + tc * 4u;
        var vb = vec4<f32>(0.0);
        if (bk < kk) {
            let o = boff + bk * brs + bj * bcs;
            if (bj < n) { vb.x = f1[o]; }
            if (bj + 1u < n) { vb.y = f1[o + bcs]; }
            if (bj + 2u < n) { vb.z = f1[o + 2u * bcs]; }
            if (bj + 3u < n) { vb.w = f1[o + 3u * bcs]; }
        }
        tb[br * 16u + tc] = vb;
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let b = tb[k * 16u + tc];
            c0 = c0 + ta[tr * 16u + k] * b;
            c1 = c1 + ta[(tr + 1u) * 16u + k] * b;
            c2 = c2 + ta[(tr + 2u) * 16u + k] * b;
            c3 = c3 + ta[(tr + 3u) * 16u + k] * b;
        }
        workgroupBarrier();
    }
    let gj = col0 + tc * 4u;
    let gi = row0 + tr;
    gemm_store(coff + gi * n + gj, gi, gj, m, n, c0);
    gemm_store(coff + (gi + 1u) * n + gj, gi + 1u, gj, m, n, c1);
    gemm_store(coff + (gi + 2u) * n + gj, gi + 2u, gj, m, n, c2);
    gemm_store(coff + (gi + 3u) * n + gj, gi + 3u, gj, m, n, c3);
}

// f1[r * n + j] += f0[boff + j] (bias).   p: total = rows * n, n, boff
@compute @workgroup_size(256)
fn add_bias(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id(wg, nwg, li);
    if (i >= p.a.x) { return; }
    f1[i] = f1[i] + f0[p.a.z + i % p.a.y];
}

// Column sums in two stages, so thousands of rows are summed in parallel.
// Stage 1, one thread per (column, chunk of rows): f2[c * n + j] = sum over the chunk's rows of
// f0[r * n + j] (mode 0) or f0[r * n + j] * f1[r * n + j] * f3[r] (mode 1: an RMS norm's gain
// gradient, with f1 the input and f3 the rows' inverse RMS).   p: n, rows, chunk rows, mode
@compute @workgroup_size(256)
fn colsum_part(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p.a.x;
    let j = wg.x * 256u + li;
    if (j >= n) { return; }
    let r0 = wg.y * p.a.z;
    let r1 = min(p.a.y, r0 + p.a.z);
    var s = 0.0;
    if (p.a.w == 0u) {
        for (var r = r0; r < r1; r = r + 1u) { s = s + f0[r * n + j]; }
    } else {
        for (var r = r0; r < r1; r = r + 1u) { s = s + f0[r * n + j] * f1[r * n + j] * f3[r]; }
    }
    f2[wg.y * n + j] = s;
}

// Stage 2: f1[off + j] += sum over chunks of f0[c * n + j].   p: n, chunks, off
@compute @workgroup_size(256)
fn colsum_finish(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let j = id(wg, nwg, li);
    let n = p.a.x;
    if (j >= n) { return; }
    var s = 0.0;
    for (var c = 0u; c < p.a.y; c = c + 1u) { s = s + f0[c * n + j]; }
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
// Each workgroup takes 64 rows of one (sequence, head); keys and values (or queries) are read in
// tiles into workgroup memory, which all 64 rows share.

var<workgroup> ak: array<f32, 2048>; // 32 keys x 64
var<workgroup> av: array<f32, 2048>; // 32 values x 64

// Loads keys (f1 -> ak) and values (f2 -> av) k0..k0+32 of (s, h) into workgroup memory.
fn load_kv(kb: u32, vb: u32, s: u32, h: u32, k0: u32, li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    let hd = d / p.a.w;
    for (var x = li; x < 32u * hd; x = x + 64u) {
        let kr = x / hd;
        let e = x % hd;
        let c = k0 + kr;
        var kv = 0.0;
        var vv = 0.0;
        if (c < t) {
            let o = (s * t + c) * d + h * hd + e;
            if (kb == 1u) { kv = f1[o]; } else { kv = f2[o]; }
            if (vb == 2u) { vv = f2[o]; } else { vv = f3[o]; }
        }
        ak[kr * 64u + e] = kv;
        av[kr * 64u + e] = vv;
    }
}

// Forward, one thread per query row, single pass (the running maximum rescales the sums):
// f3 = output, stats max and total.   p: count = b * heads * t, t, d, heads | scale bits
@compute @workgroup_size(64)
fn attention(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let hd = d / heads;
    let sc = fl(p.b.x);
    let sh = wg.y;
    let s = sh / heads;
    let h = sh % heads;
    let row = wg.x * 64u + li;
    let valid = row < t;
    let qb = (s * t + row) * d + h * hd;
    var q: array<f32, 64>;
    var acc: array<f32, 64>;
    for (var e = 0u; e < hd; e = e + 1u) {
        acc[e] = 0.0;
        if (valid) { q[e] = f0[qb + e]; } else { q[e] = 0.0; }
    }
    var m = -3.4e38;
    var l = 0.0;
    let last = min(t, wg.x * 64u + 64u);
    for (var k0 = 0u; k0 < last; k0 = k0 + 32u) {
        load_kv(1u, 2u, s, h, k0, li);
        workgroupBarrier();
        if (valid && k0 <= row) {
            let nk = min(32u, row + 1u - k0);
            var sv: array<f32, 32>;
            var tmax = -3.4e38;
            for (var kk = 0u; kk < nk; kk = kk + 1u) {
                var dot = 0.0;
                for (var e = 0u; e < hd; e = e + 1u) { dot = dot + q[e] * ak[kk * 64u + e]; }
                sv[kk] = dot * sc;
                tmax = max(tmax, sv[kk]);
            }
            if (tmax > m) {
                let corr = exp(m - tmax);
                l = l * corr;
                for (var e = 0u; e < hd; e = e + 1u) { acc[e] = acc[e] * corr; }
                m = tmax;
            }
            for (var kk = 0u; kk < nk; kk = kk + 1u) {
                let w = exp(sv[kk] - m);
                l = l + w;
                for (var e = 0u; e < hd; e = e + 1u) { acc[e] = acc[e] + w * av[kk * 64u + e]; }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        for (var e = 0u; e < hd; e = e + 1u) { f3[qb + e] = acc[e] / l; }
        let i = sh * t + row;
        f4[i] = m;
        f4[p.a.x + i] = l;
    }
}

// Backward, one thread per query row: dq (f5) and the row's weighted sum (stats).
// f0 upstream, f1 q, f2 k, f3 v.   p: count, t, d, heads | scale bits
@compute @workgroup_size(64)
fn attention_back_q(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let hd = d / heads;
    let sc = fl(p.b.x);
    let sh = wg.y;
    let s = sh / heads;
    let h = sh % heads;
    let row = wg.x * 64u + li;
    let valid = row < t;
    let qb = (s * t + row) * d + h * hd;
    let i = sh * t + row;
    var q: array<f32, 64>;
    var up: array<f32, 64>;
    var dq: array<f32, 64>;
    var mx = 0.0;
    var total = 1.0;
    for (var e = 0u; e < hd; e = e + 1u) {
        dq[e] = 0.0;
        if (valid) { q[e] = f1[qb + e]; up[e] = f0[qb + e]; } else { q[e] = 0.0; up[e] = 0.0; }
    }
    if (valid) {
        mx = f4[i];
        total = f4[p.a.x + i];
    }
    let last = min(t, wg.x * 64u + 64u);
    // Pass 1: the weighted sum of the upstream gradient's projections on the values.
    var weighted = 0.0;
    for (var k0 = 0u; k0 < last; k0 = k0 + 32u) {
        load_kv(0u, 3u, s, h, k0, li);
        workgroupBarrier();
        if (valid && k0 <= row) {
            let nk = min(32u, row + 1u - k0);
            for (var kk = 0u; kk < nk; kk = kk + 1u) {
                var dot = 0.0;
                var g = 0.0;
                for (var e = 0u; e < hd; e = e + 1u) {
                    dot = dot + q[e] * ak[kk * 64u + e];
                    g = g + up[e] * av[kk * 64u + e];
                }
                weighted = weighted + g * (exp(dot * sc - mx) / total);
            }
        }
        workgroupBarrier();
    }
    // Pass 2: dq.
    for (var k0 = 0u; k0 < last; k0 = k0 + 32u) {
        load_kv(0u, 3u, s, h, k0, li);
        workgroupBarrier();
        if (valid && k0 <= row) {
            let nk = min(32u, row + 1u - k0);
            for (var kk = 0u; kk < nk; kk = kk + 1u) {
                var dot = 0.0;
                var g = 0.0;
                for (var e = 0u; e < hd; e = e + 1u) {
                    dot = dot + q[e] * ak[kk * 64u + e];
                    g = g + up[e] * av[kk * 64u + e];
                }
                let w = exp(dot * sc - mx) / total;
                let gg = w * (g - weighted) * sc;
                for (var e = 0u; e < hd; e = e + 1u) { dq[e] = dq[e] + gg * ak[kk * 64u + e]; }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        for (var e = 0u; e < hd; e = e + 1u) { f5[qb + e] = dq[e]; }
        f4[2u * p.a.x + i] = weighted;
    }
}

var<workgroup> bq: array<f32, 1024>;  // 16 query rows x 64
var<workgroup> bu: array<f32, 1024>;  // 16 upstream rows x 64
var<workgroup> bs: array<f32, 48>;    // their max, total, weighted

// Backward, one thread per key row: dk (f5) and dv (f6); query rows are read in tiles.
// f0 upstream, f1 q, f2 k, f3 v, f4 stats.   p: count, t, d, heads | scale bits
@compute @workgroup_size(64)
fn attention_back_kv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let hd = d / heads;
    let sc = fl(p.b.x);
    let sh = wg.y;
    let s = sh / heads;
    let h = sh % heads;
    let col = wg.x * 64u + li;
    let valid = col < t;
    let kb = (s * t + col) * d + h * hd;
    var k: array<f32, 64>;
    var v: array<f32, 64>;
    var dk: array<f32, 64>;
    var dv: array<f32, 64>;
    for (var e = 0u; e < hd; e = e + 1u) {
        dk[e] = 0.0;
        dv[e] = 0.0;
        if (valid) { k[e] = f2[kb + e]; v[e] = f3[kb + e]; } else { k[e] = 0.0; v[e] = 0.0; }
    }
    for (var r0 = wg.x * 64u; r0 < t; r0 = r0 + 16u) {
        for (var x = li; x < 16u * hd; x = x + 64u) {
            let rr = x / hd;
            let e = x % hd;
            let r = r0 + rr;
            var qv = 0.0;
            var uv = 0.0;
            if (r < t) {
                let o = (s * t + r) * d + h * hd + e;
                qv = f1[o];
                uv = f0[o];
            }
            bq[rr * 64u + e] = qv;
            bu[rr * 64u + e] = uv;
        }
        if (li < 16u) {
            let r = r0 + li;
            if (r < t) {
                let si = sh * t + r;
                bs[li] = f4[si];
                bs[16u + li] = f4[p.a.x + si];
                bs[32u + li] = f4[2u * p.a.x + si];
            } else {
                bs[li] = 0.0;
                bs[16u + li] = 1.0;
                bs[32u + li] = 0.0;
            }
        }
        workgroupBarrier();
        if (valid) {
            for (var rr = 0u; rr < 16u; rr = rr + 1u) {
                let r = r0 + rr;
                if (r < col || r >= t) { continue; }
                var dot = 0.0;
                var g = 0.0;
                for (var e = 0u; e < hd; e = e + 1u) {
                    dot = dot + bq[rr * 64u + e] * k[e];
                    g = g + bu[rr * 64u + e] * v[e];
                }
                let w = exp(dot * sc - bs[rr]) / bs[16u + rr];
                let gg = w * (g - bs[32u + rr]) * sc;
                for (var e = 0u; e < hd; e = e + 1u) {
                    dv[e] = dv[e] + w * bu[rr * 64u + e];
                    dk[e] = dk[e] + gg * bq[rr * 64u + e];
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        for (var e = 0u; e < hd; e = e + 1u) { f5[kb + e] = dk[e]; f6[kb + e] = dv[e]; }
    }
}

// ---- attention, head size 32 ---------------------------------------------------------------
// The same computations as the kernels above for the head size every Yendory size uses (32),
// with each head's 32 values held in eight vec4 registers: no indexed per-thread arrays.

var<workgroup> ak32: array<vec4<f32>, 256>; // 32 keys x 8
var<workgroup> av32: array<vec4<f32>, 256>;

fn load_kv32(kb: u32, vb: u32, s: u32, h: u32, k0: u32, li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    for (var x = li; x < 256u; x = x + 64u) {
        let kr = x / 8u;
        let e = (x % 8u) * 4u;
        let c = k0 + kr;
        var kv = vec4<f32>(0.0);
        var vv = vec4<f32>(0.0);
        if (c < t) {
            let o = (s * t + c) * d + h * 32u + e;
            if (kb == 1u) { kv = vec4<f32>(f1[o], f1[o + 1u], f1[o + 2u], f1[o + 3u]); }
            else { kv = vec4<f32>(f2[o], f2[o + 1u], f2[o + 2u], f2[o + 3u]); }
            if (vb == 2u) { vv = vec4<f32>(f2[o], f2[o + 1u], f2[o + 2u], f2[o + 3u]); }
            else { vv = vec4<f32>(f3[o], f3[o + 1u], f3[o + 2u], f3[o + 3u]); }
        }
        ak32[x] = kv;
        av32[x] = vv;
    }
}

@compute @workgroup_size(64)
fn attention32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let sc = fl(p.b.x);
    let sh = wg.y;
    let s = sh / heads;
    let h = sh % heads;
    let row = wg.x * 64u + li;
    let valid = row < t;
    let qb = (s * t + row) * d + h * 32u;
    var q0 = vec4<f32>(0.0);
    var q1 = vec4<f32>(0.0);
    var q2 = vec4<f32>(0.0);
    var q3 = vec4<f32>(0.0);
    var q4 = vec4<f32>(0.0);
    var q5 = vec4<f32>(0.0);
    var q6 = vec4<f32>(0.0);
    var q7 = vec4<f32>(0.0);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);
    q0 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 0u], f0[qb + 1u], f0[qb + 2u], f0[qb + 3u]), valid);
    q1 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 4u], f0[qb + 5u], f0[qb + 6u], f0[qb + 7u]), valid);
    q2 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 8u], f0[qb + 9u], f0[qb + 10u], f0[qb + 11u]), valid);
    q3 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 12u], f0[qb + 13u], f0[qb + 14u], f0[qb + 15u]), valid);
    q4 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 16u], f0[qb + 17u], f0[qb + 18u], f0[qb + 19u]), valid);
    q5 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 20u], f0[qb + 21u], f0[qb + 22u], f0[qb + 23u]), valid);
    q6 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 24u], f0[qb + 25u], f0[qb + 26u], f0[qb + 27u]), valid);
    q7 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 28u], f0[qb + 29u], f0[qb + 30u], f0[qb + 31u]), valid);
    var m = -3.4e38;
    var l = 0.0;
    let last = min(t, wg.x * 64u + 64u);
    for (var k0 = 0u; k0 < last; k0 = k0 + 32u) {
        load_kv32(1u, 2u, s, h, k0, li);
        workgroupBarrier();
        if (valid && k0 <= row) {
            let nk = min(32u, row + 1u - k0);
            for (var kk = 0u; kk < nk; kk = kk + 1u) {
                let sv = (dot(q0, ak32[kk * 8u + 0u]) + dot(q1, ak32[kk * 8u + 1u]) + dot(q2, ak32[kk * 8u + 2u]) + dot(q3, ak32[kk * 8u + 3u]) + dot(q4, ak32[kk * 8u + 4u]) + dot(q5, ak32[kk * 8u + 5u]) + dot(q6, ak32[kk * 8u + 6u]) + dot(q7, ak32[kk * 8u + 7u])) * sc;
                if (sv > m) {
                    let corr = exp(m - sv);
                    l = l * corr;
                    a0 = a0 * corr;
                    a1 = a1 * corr;
                    a2 = a2 * corr;
                    a3 = a3 * corr;
                    a4 = a4 * corr;
                    a5 = a5 * corr;
                    a6 = a6 * corr;
                    a7 = a7 * corr;
                    m = sv;
                }
                let w = exp(sv - m);
                l = l + w;
                a0 = a0 + w * av32[kk * 8u + 0u];
                a1 = a1 + w * av32[kk * 8u + 1u];
                a2 = a2 + w * av32[kk * 8u + 2u];
                a3 = a3 + w * av32[kk * 8u + 3u];
                a4 = a4 + w * av32[kk * 8u + 4u];
                a5 = a5 + w * av32[kk * 8u + 5u];
                a6 = a6 + w * av32[kk * 8u + 6u];
                a7 = a7 + w * av32[kk * 8u + 7u];
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        f3[qb + 0u] = a0.x / l; f3[qb + 1u] = a0.y / l; f3[qb + 2u] = a0.z / l; f3[qb + 3u] = a0.w / l;
        f3[qb + 4u] = a1.x / l; f3[qb + 5u] = a1.y / l; f3[qb + 6u] = a1.z / l; f3[qb + 7u] = a1.w / l;
        f3[qb + 8u] = a2.x / l; f3[qb + 9u] = a2.y / l; f3[qb + 10u] = a2.z / l; f3[qb + 11u] = a2.w / l;
        f3[qb + 12u] = a3.x / l; f3[qb + 13u] = a3.y / l; f3[qb + 14u] = a3.z / l; f3[qb + 15u] = a3.w / l;
        f3[qb + 16u] = a4.x / l; f3[qb + 17u] = a4.y / l; f3[qb + 18u] = a4.z / l; f3[qb + 19u] = a4.w / l;
        f3[qb + 20u] = a5.x / l; f3[qb + 21u] = a5.y / l; f3[qb + 22u] = a5.z / l; f3[qb + 23u] = a5.w / l;
        f3[qb + 24u] = a6.x / l; f3[qb + 25u] = a6.y / l; f3[qb + 26u] = a6.z / l; f3[qb + 27u] = a6.w / l;
        f3[qb + 28u] = a7.x / l; f3[qb + 29u] = a7.y / l; f3[qb + 30u] = a7.z / l; f3[qb + 31u] = a7.w / l;
        let i = sh * t + row;
        f4[i] = m;
        f4[p.a.x + i] = l;
    }
}

@compute @workgroup_size(64)
fn attention_back_q32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let sc = fl(p.b.x);
    let sh = wg.y;
    let s = sh / heads;
    let h = sh % heads;
    let row = wg.x * 64u + li;
    let valid = row < t;
    let qb = (s * t + row) * d + h * 32u;
    let i = sh * t + row;
    var q0 = vec4<f32>(0.0);
    var q1 = vec4<f32>(0.0);
    var q2 = vec4<f32>(0.0);
    var q3 = vec4<f32>(0.0);
    var q4 = vec4<f32>(0.0);
    var q5 = vec4<f32>(0.0);
    var q6 = vec4<f32>(0.0);
    var q7 = vec4<f32>(0.0);
    var u0 = vec4<f32>(0.0);
    var u1 = vec4<f32>(0.0);
    var u2 = vec4<f32>(0.0);
    var u3 = vec4<f32>(0.0);
    var u4 = vec4<f32>(0.0);
    var u5 = vec4<f32>(0.0);
    var u6 = vec4<f32>(0.0);
    var u7 = vec4<f32>(0.0);
    var g0 = vec4<f32>(0.0);
    var g1 = vec4<f32>(0.0);
    var g2 = vec4<f32>(0.0);
    var g3 = vec4<f32>(0.0);
    var g4 = vec4<f32>(0.0);
    var g5 = vec4<f32>(0.0);
    var g6 = vec4<f32>(0.0);
    var g7 = vec4<f32>(0.0);
    q0 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 0u], f1[qb + 1u], f1[qb + 2u], f1[qb + 3u]), valid);
    q1 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 4u], f1[qb + 5u], f1[qb + 6u], f1[qb + 7u]), valid);
    q2 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 8u], f1[qb + 9u], f1[qb + 10u], f1[qb + 11u]), valid);
    q3 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 12u], f1[qb + 13u], f1[qb + 14u], f1[qb + 15u]), valid);
    q4 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 16u], f1[qb + 17u], f1[qb + 18u], f1[qb + 19u]), valid);
    q5 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 20u], f1[qb + 21u], f1[qb + 22u], f1[qb + 23u]), valid);
    q6 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 24u], f1[qb + 25u], f1[qb + 26u], f1[qb + 27u]), valid);
    q7 = select(vec4<f32>(0.0), vec4<f32>(f1[qb + 28u], f1[qb + 29u], f1[qb + 30u], f1[qb + 31u]), valid);
    u0 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 0u], f0[qb + 1u], f0[qb + 2u], f0[qb + 3u]), valid);
    u1 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 4u], f0[qb + 5u], f0[qb + 6u], f0[qb + 7u]), valid);
    u2 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 8u], f0[qb + 9u], f0[qb + 10u], f0[qb + 11u]), valid);
    u3 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 12u], f0[qb + 13u], f0[qb + 14u], f0[qb + 15u]), valid);
    u4 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 16u], f0[qb + 17u], f0[qb + 18u], f0[qb + 19u]), valid);
    u5 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 20u], f0[qb + 21u], f0[qb + 22u], f0[qb + 23u]), valid);
    u6 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 24u], f0[qb + 25u], f0[qb + 26u], f0[qb + 27u]), valid);
    u7 = select(vec4<f32>(0.0), vec4<f32>(f0[qb + 28u], f0[qb + 29u], f0[qb + 30u], f0[qb + 31u]), valid);
    var mx = 0.0;
    var total = 1.0;
    if (valid) {
        mx = f4[i];
        total = f4[p.a.x + i];
    }
    let last = min(t, wg.x * 64u + 64u);
    var weighted = 0.0;
    for (var k0 = 0u; k0 < last; k0 = k0 + 32u) {
        load_kv32(0u, 3u, s, h, k0, li);
        workgroupBarrier();
        if (valid && k0 <= row) {
            let nk = min(32u, row + 1u - k0);
            for (var kk = 0u; kk < nk; kk = kk + 1u) {
                let dt = dot(q0, ak32[kk * 8u + 0u]) + dot(q1, ak32[kk * 8u + 1u]) + dot(q2, ak32[kk * 8u + 2u]) + dot(q3, ak32[kk * 8u + 3u]) + dot(q4, ak32[kk * 8u + 4u]) + dot(q5, ak32[kk * 8u + 5u]) + dot(q6, ak32[kk * 8u + 6u]) + dot(q7, ak32[kk * 8u + 7u]);
                let gv = dot(u0, av32[kk * 8u + 0u]) + dot(u1, av32[kk * 8u + 1u]) + dot(u2, av32[kk * 8u + 2u]) + dot(u3, av32[kk * 8u + 3u]) + dot(u4, av32[kk * 8u + 4u]) + dot(u5, av32[kk * 8u + 5u]) + dot(u6, av32[kk * 8u + 6u]) + dot(u7, av32[kk * 8u + 7u]);
                weighted = weighted + gv * (exp(dt * sc - mx) / total);
            }
        }
        workgroupBarrier();
    }
    for (var k0 = 0u; k0 < last; k0 = k0 + 32u) {
        load_kv32(0u, 3u, s, h, k0, li);
        workgroupBarrier();
        if (valid && k0 <= row) {
            let nk = min(32u, row + 1u - k0);
            for (var kk = 0u; kk < nk; kk = kk + 1u) {
                let dt = dot(q0, ak32[kk * 8u + 0u]) + dot(q1, ak32[kk * 8u + 1u]) + dot(q2, ak32[kk * 8u + 2u]) + dot(q3, ak32[kk * 8u + 3u]) + dot(q4, ak32[kk * 8u + 4u]) + dot(q5, ak32[kk * 8u + 5u]) + dot(q6, ak32[kk * 8u + 6u]) + dot(q7, ak32[kk * 8u + 7u]);
                let gv = dot(u0, av32[kk * 8u + 0u]) + dot(u1, av32[kk * 8u + 1u]) + dot(u2, av32[kk * 8u + 2u]) + dot(u3, av32[kk * 8u + 3u]) + dot(u4, av32[kk * 8u + 4u]) + dot(u5, av32[kk * 8u + 5u]) + dot(u6, av32[kk * 8u + 6u]) + dot(u7, av32[kk * 8u + 7u]);
                let gg = (exp(dt * sc - mx) / total) * (gv - weighted) * sc;
                g0 = g0 + gg * ak32[kk * 8u + 0u];
                g1 = g1 + gg * ak32[kk * 8u + 1u];
                g2 = g2 + gg * ak32[kk * 8u + 2u];
                g3 = g3 + gg * ak32[kk * 8u + 3u];
                g4 = g4 + gg * ak32[kk * 8u + 4u];
                g5 = g5 + gg * ak32[kk * 8u + 5u];
                g6 = g6 + gg * ak32[kk * 8u + 6u];
                g7 = g7 + gg * ak32[kk * 8u + 7u];
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        f5[qb + 0u] = g0.x; f5[qb + 1u] = g0.y; f5[qb + 2u] = g0.z; f5[qb + 3u] = g0.w;
        f5[qb + 4u] = g1.x; f5[qb + 5u] = g1.y; f5[qb + 6u] = g1.z; f5[qb + 7u] = g1.w;
        f5[qb + 8u] = g2.x; f5[qb + 9u] = g2.y; f5[qb + 10u] = g2.z; f5[qb + 11u] = g2.w;
        f5[qb + 12u] = g3.x; f5[qb + 13u] = g3.y; f5[qb + 14u] = g3.z; f5[qb + 15u] = g3.w;
        f5[qb + 16u] = g4.x; f5[qb + 17u] = g4.y; f5[qb + 18u] = g4.z; f5[qb + 19u] = g4.w;
        f5[qb + 20u] = g5.x; f5[qb + 21u] = g5.y; f5[qb + 22u] = g5.z; f5[qb + 23u] = g5.w;
        f5[qb + 24u] = g6.x; f5[qb + 25u] = g6.y; f5[qb + 26u] = g6.z; f5[qb + 27u] = g6.w;
        f5[qb + 28u] = g7.x; f5[qb + 29u] = g7.y; f5[qb + 30u] = g7.z; f5[qb + 31u] = g7.w;
        f4[2u * p.a.x + i] = weighted;
    }
}

var<workgroup> bq32: array<vec4<f32>, 128>; // 16 query rows x 8
var<workgroup> bu32: array<vec4<f32>, 128>; // 16 upstream rows x 8
var<workgroup> bs32: array<f32, 48>;

@compute @workgroup_size(64)
fn attention_back_kv32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = p.a.y;
    let d = p.a.z;
    let heads = p.a.w;
    let sc = fl(p.b.x);
    let sh = wg.y;
    let s = sh / heads;
    let h = sh % heads;
    let col = wg.x * 64u + li;
    let valid = col < t;
    let kb = (s * t + col) * d + h * 32u;
    var k0 = vec4<f32>(0.0);
    var k1 = vec4<f32>(0.0);
    var k2 = vec4<f32>(0.0);
    var k3 = vec4<f32>(0.0);
    var k4 = vec4<f32>(0.0);
    var k5 = vec4<f32>(0.0);
    var k6 = vec4<f32>(0.0);
    var k7 = vec4<f32>(0.0);
    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0);
    var v4 = vec4<f32>(0.0);
    var v5 = vec4<f32>(0.0);
    var v6 = vec4<f32>(0.0);
    var v7 = vec4<f32>(0.0);
    var dk0 = vec4<f32>(0.0);
    var dk1 = vec4<f32>(0.0);
    var dk2 = vec4<f32>(0.0);
    var dk3 = vec4<f32>(0.0);
    var dk4 = vec4<f32>(0.0);
    var dk5 = vec4<f32>(0.0);
    var dk6 = vec4<f32>(0.0);
    var dk7 = vec4<f32>(0.0);
    var dv0 = vec4<f32>(0.0);
    var dv1 = vec4<f32>(0.0);
    var dv2 = vec4<f32>(0.0);
    var dv3 = vec4<f32>(0.0);
    var dv4 = vec4<f32>(0.0);
    var dv5 = vec4<f32>(0.0);
    var dv6 = vec4<f32>(0.0);
    var dv7 = vec4<f32>(0.0);
    k0 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 0u], f2[kb + 1u], f2[kb + 2u], f2[kb + 3u]), valid);
    k1 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 4u], f2[kb + 5u], f2[kb + 6u], f2[kb + 7u]), valid);
    k2 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 8u], f2[kb + 9u], f2[kb + 10u], f2[kb + 11u]), valid);
    k3 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 12u], f2[kb + 13u], f2[kb + 14u], f2[kb + 15u]), valid);
    k4 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 16u], f2[kb + 17u], f2[kb + 18u], f2[kb + 19u]), valid);
    k5 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 20u], f2[kb + 21u], f2[kb + 22u], f2[kb + 23u]), valid);
    k6 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 24u], f2[kb + 25u], f2[kb + 26u], f2[kb + 27u]), valid);
    k7 = select(vec4<f32>(0.0), vec4<f32>(f2[kb + 28u], f2[kb + 29u], f2[kb + 30u], f2[kb + 31u]), valid);
    v0 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 0u], f3[kb + 1u], f3[kb + 2u], f3[kb + 3u]), valid);
    v1 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 4u], f3[kb + 5u], f3[kb + 6u], f3[kb + 7u]), valid);
    v2 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 8u], f3[kb + 9u], f3[kb + 10u], f3[kb + 11u]), valid);
    v3 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 12u], f3[kb + 13u], f3[kb + 14u], f3[kb + 15u]), valid);
    v4 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 16u], f3[kb + 17u], f3[kb + 18u], f3[kb + 19u]), valid);
    v5 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 20u], f3[kb + 21u], f3[kb + 22u], f3[kb + 23u]), valid);
    v6 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 24u], f3[kb + 25u], f3[kb + 26u], f3[kb + 27u]), valid);
    v7 = select(vec4<f32>(0.0), vec4<f32>(f3[kb + 28u], f3[kb + 29u], f3[kb + 30u], f3[kb + 31u]), valid);
    for (var r0 = wg.x * 64u; r0 < t; r0 = r0 + 16u) {
        for (var x = li; x < 128u; x = x + 64u) {
            let rr = x / 8u;
            let e = (x % 8u) * 4u;
            let r = r0 + rr;
            var qv = vec4<f32>(0.0);
            var uv = vec4<f32>(0.0);
            if (r < t) {
                let o = (s * t + r) * d + h * 32u + e;
                qv = vec4<f32>(f1[o], f1[o + 1u], f1[o + 2u], f1[o + 3u]);
                uv = vec4<f32>(f0[o], f0[o + 1u], f0[o + 2u], f0[o + 3u]);
            }
            bq32[x] = qv;
            bu32[x] = uv;
        }
        if (li < 16u) {
            let r = r0 + li;
            if (r < t) {
                let si = sh * t + r;
                bs32[li] = f4[si];
                bs32[16u + li] = f4[p.a.x + si];
                bs32[32u + li] = f4[2u * p.a.x + si];
            } else {
                bs32[li] = 0.0;
                bs32[16u + li] = 1.0;
                bs32[32u + li] = 0.0;
            }
        }
        workgroupBarrier();
        if (valid) {
            for (var rr = 0u; rr < 16u; rr = rr + 1u) {
                let r = r0 + rr;
                if (r < col || r >= t) { continue; }
                let dt = dot(bq32[rr * 8u + 0u], k0) + dot(bq32[rr * 8u + 1u], k1) + dot(bq32[rr * 8u + 2u], k2) + dot(bq32[rr * 8u + 3u], k3) + dot(bq32[rr * 8u + 4u], k4) + dot(bq32[rr * 8u + 5u], k5) + dot(bq32[rr * 8u + 6u], k6) + dot(bq32[rr * 8u + 7u], k7);
                let gv = dot(bu32[rr * 8u + 0u], v0) + dot(bu32[rr * 8u + 1u], v1) + dot(bu32[rr * 8u + 2u], v2) + dot(bu32[rr * 8u + 3u], v3) + dot(bu32[rr * 8u + 4u], v4) + dot(bu32[rr * 8u + 5u], v5) + dot(bu32[rr * 8u + 6u], v6) + dot(bu32[rr * 8u + 7u], v7);
                let w = exp(dt * sc - bs32[rr]) / bs32[16u + rr];
                let gg = w * (gv - bs32[32u + rr]) * sc;
                dv0 = dv0 + w * bu32[rr * 8u + 0u];
                dk0 = dk0 + gg * bq32[rr * 8u + 0u];
                dv1 = dv1 + w * bu32[rr * 8u + 1u];
                dk1 = dk1 + gg * bq32[rr * 8u + 1u];
                dv2 = dv2 + w * bu32[rr * 8u + 2u];
                dk2 = dk2 + gg * bq32[rr * 8u + 2u];
                dv3 = dv3 + w * bu32[rr * 8u + 3u];
                dk3 = dk3 + gg * bq32[rr * 8u + 3u];
                dv4 = dv4 + w * bu32[rr * 8u + 4u];
                dk4 = dk4 + gg * bq32[rr * 8u + 4u];
                dv5 = dv5 + w * bu32[rr * 8u + 5u];
                dk5 = dk5 + gg * bq32[rr * 8u + 5u];
                dv6 = dv6 + w * bu32[rr * 8u + 6u];
                dk6 = dk6 + gg * bq32[rr * 8u + 6u];
                dv7 = dv7 + w * bu32[rr * 8u + 7u];
                dk7 = dk7 + gg * bq32[rr * 8u + 7u];
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        f5[kb + 0u] = dk0.x; f5[kb + 1u] = dk0.y; f5[kb + 2u] = dk0.z; f5[kb + 3u] = dk0.w;
        f6[kb + 0u] = dv0.x; f6[kb + 1u] = dv0.y; f6[kb + 2u] = dv0.z; f6[kb + 3u] = dv0.w;
        f5[kb + 4u] = dk1.x; f5[kb + 5u] = dk1.y; f5[kb + 6u] = dk1.z; f5[kb + 7u] = dk1.w;
        f6[kb + 4u] = dv1.x; f6[kb + 5u] = dv1.y; f6[kb + 6u] = dv1.z; f6[kb + 7u] = dv1.w;
        f5[kb + 8u] = dk2.x; f5[kb + 9u] = dk2.y; f5[kb + 10u] = dk2.z; f5[kb + 11u] = dk2.w;
        f6[kb + 8u] = dv2.x; f6[kb + 9u] = dv2.y; f6[kb + 10u] = dv2.z; f6[kb + 11u] = dv2.w;
        f5[kb + 12u] = dk3.x; f5[kb + 13u] = dk3.y; f5[kb + 14u] = dk3.z; f5[kb + 15u] = dk3.w;
        f6[kb + 12u] = dv3.x; f6[kb + 13u] = dv3.y; f6[kb + 14u] = dv3.z; f6[kb + 15u] = dv3.w;
        f5[kb + 16u] = dk4.x; f5[kb + 17u] = dk4.y; f5[kb + 18u] = dk4.z; f5[kb + 19u] = dk4.w;
        f6[kb + 16u] = dv4.x; f6[kb + 17u] = dv4.y; f6[kb + 18u] = dv4.z; f6[kb + 19u] = dv4.w;
        f5[kb + 20u] = dk5.x; f5[kb + 21u] = dk5.y; f5[kb + 22u] = dk5.z; f5[kb + 23u] = dk5.w;
        f6[kb + 20u] = dv5.x; f6[kb + 21u] = dv5.y; f6[kb + 22u] = dv5.z; f6[kb + 23u] = dv5.w;
        f5[kb + 24u] = dk6.x; f5[kb + 25u] = dk6.y; f5[kb + 26u] = dk6.z; f5[kb + 27u] = dk6.w;
        f6[kb + 24u] = dv6.x; f6[kb + 25u] = dv6.y; f6[kb + 26u] = dv6.z; f6[kb + 27u] = dv6.w;
        f5[kb + 28u] = dk7.x; f5[kb + 29u] = dk7.y; f5[kb + 30u] = dk7.z; f5[kb + 31u] = dk7.w;
        f6[kb + 28u] = dv7.x; f6[kb + 29u] = dv7.y; f6[kb + 30u] = dv7.z; f6[kb + 31u] = dv7.w;
    }
}

// ---- output layer and cross-entropy -------------------------------------------------------

var<workgroup> red: array<f32, 256>;

// One workgroup (256 threads) per row of a chunk of logits (f0, rows x vocab). Targets u0, mask
// f1 and the row losses f2 are indexed from r0. The row's loss mask * -ln p(target) / denom goes
// to f2, and f0 becomes the logits' gradient (softmax - one-hot) * mask * seed / denom.
// p: vocab, rows, r0 | denom bits, seed bits
@compute @workgroup_size(256)
fn xent(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = wg.y * nwg.x + wg.x;
    if (r >= p.a.y) { return; }
    let v = p.a.x;
    let base = r * v;
    let g = p.a.z + r;
    var mx = -3.4e38;
    for (var j = li; j < v; j = j + 256u) { mx = max(mx, f0[base + j]); }
    red[li] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = max(red[li], red[li + s]); }
        workgroupBarrier();
    }
    mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = li; j < v; j = j + 256u) { sum = sum + exp(f0[base + j] - mx); }
    red[li] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    sum = red[0];
    let denom = fl(p.b.x);
    let tg = u0[g];
    let target_logit = f0[base + tg];
    workgroupBarrier();
    if (li == 0u) {
        let pt = exp(target_logit - mx) / sum;
        f2[g] = -f1[g] * log(max(pt, 1e-30)) / denom;
    }
    let scale = f1[g] * fl(p.b.y) / denom;
    for (var j = li; j < v; j = j + 256u) {
        var q = exp(f0[base + j] - mx) / sum;
        if (j == tg) { q = q - 1.0; }
        f0[base + j] = q * scale;
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
