//! Matrix kernels for the forward and backward passes, in plain safe Rust laid out so the
//! compiler vectorises the inner loops (independent lanes over contiguous memory) and each
//! weight row is read once per block of rows instead of once per row.
//!
//! They compute exactly the same products as straightforward loops; only the order in which
//! floating-point numbers are added differs, so results agree with the numpy reference within
//! rounding (the tolerance of the compatibility tests).

const LANES: usize = 8;

/// Dot product with eight independent partial sums (vectorisable).
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; LANES];
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let mut tail = 0f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail += x * y;
    }
    for (x, y) in ca.zip(cb) {
        for l in 0..LANES {
            acc[l] += x[l] * y[l];
        }
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// `y += a * x`.
#[inline]
pub(crate) fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

/// Matrix-vector product: `w` is rows×cols (row-major), result has `rows` values.
pub(crate) fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), cols);
    assert_eq!(w.len(), rows * cols);
    w.chunks_exact(cols).map(|row| dot(row, x)).collect()
}

/// Columns of `out` handled per micro-kernel tile (rows are handled four at a time).
const TILE: usize = 16;

/// `out[i, c] += Σ_j a(i, j) · b(j, c)` for an m×n row-major `out`, with strided views
/// `a(i, j) = a[i * ars + j * acs]` and `b(j, c) = b[j * brs + c * bcs]`, so transposed operands
/// need no copy.
///
/// For each panel of 16 columns, `b` is packed once into a contiguous kk×16 buffer, and 4×16
/// tiles of `out` stay in registers for the whole sum over `j` (the compiler keeps the
/// fixed-size accumulator arrays in vector registers).
fn gemm_acc(
    out: &mut [f32],
    m: usize,
    n: usize,
    kk: usize,
    a: &[f32],
    ars: usize,
    acs: usize,
    b: &[f32],
    brs: usize,
    bcs: usize,
) {
    debug_assert_eq!(out.len(), m * n);
    let mut panel = vec![0f32; kk * TILE];
    for c0 in (0..n).step_by(TILE) {
        let width = TILE.min(n - c0);
        for (j, p) in panel.chunks_exact_mut(TILE).enumerate() {
            for (c, v) in p.iter_mut().enumerate() {
                *v = if c < width { b[j * brs + (c0 + c) * bcs] } else { 0. };
            }
        }
        let mut i0 = 0;
        while i0 + 4 <= m {
            let mut acc = [[0f32; TILE]; 4];
            for (r, row) in acc.iter_mut().enumerate() {
                row[..width].copy_from_slice(&out[(i0 + r) * n + c0..(i0 + r) * n + c0 + width]);
            }
            for (j, w) in panel.chunks_exact(TILE).enumerate() {
                let w: &[f32; TILE] = w.try_into().unwrap();
                for (r, row) in acc.iter_mut().enumerate() {
                    let x = a[(i0 + r) * ars + j * acs];
                    for c in 0..TILE {
                        row[c] += x * w[c];
                    }
                }
            }
            for (r, row) in acc.iter().enumerate() {
                out[(i0 + r) * n + c0..(i0 + r) * n + c0 + width].copy_from_slice(&row[..width]);
            }
            i0 += 4;
        }
        for i in i0..m {
            let mut row = [0f32; TILE];
            row[..width].copy_from_slice(&out[i * n + c0..i * n + c0 + width]);
            for (j, w) in panel.chunks_exact(TILE).enumerate() {
                let x = a[i * ars + j * acs];
                for c in 0..TILE {
                    row[c] += x * w[c];
                }
            }
            out[i * n + c0..i * n + c0 + width].copy_from_slice(&row[..width]);
        }
    }
}

/// `x · wᵀ + bias`: `x` is rows×k, `w` is n×k (one row per output), result is rows×n.
pub(crate) fn linear(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * k);
    assert_eq!(w.len(), n * k);
    let mut out = vec![0f32; rows * n];
    if rows < 4 {
        for (r, orow) in out.chunks_exact_mut(n).enumerate() {
            let xr = &x[r * k..(r + 1) * k];
            for (o, v) in orow.iter_mut().enumerate() {
                *v = dot(xr, &w[o * k..(o + 1) * k]);
            }
        }
    } else {
        gemm_acc(&mut out, rows, n, k, x, k, 1, w, 1, k);
    }
    if let Some(b) = bias {
        for orow in out.chunks_exact_mut(n) {
            for (o, b) in orow.iter_mut().zip(b) {
                *o += b;
            }
        }
    }
    out
}

/// Gradients of [`linear`] for an upstream gradient `g` (rows×n). Returns `dx` (rows×k) and
/// ADDS the weight and bias gradients into `dw` (n×k) and `db` (n).
pub(crate) fn linear_backward(
    g: &[f32],
    x: &[f32],
    w: &[f32],
    rows: usize,
    k: usize,
    n: usize,
    dw: &mut [f32],
    db: Option<&mut [f32]>,
) -> Vec<f32> {
    let mut dx = vec![0f32; rows * k];
    gemm_acc(&mut dx, rows, k, n, g, n, 1, w, k, 1);
    gemm_acc(dw, n, k, rows, g, 1, n, x, k, 1);
    if let Some(db) = db {
        for grow in g.chunks_exact(n) {
            for (d, g) in db.iter_mut().zip(grow) {
                *d += g;
            }
        }
    }
    dx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; rows * n];
        for r in 0..rows {
            for o in 0..n {
                out[r * n + o] =
                    (0..k).map(|j| x[r * k + j] as f64 * w[o * k + j] as f64).sum::<f64>() as f32;
            }
        }
        out
    }
    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len).map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 2001) as f32 / 1000. - 1.).collect()
    }
    fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut t = vec![0f32; w.len()];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = w[r * cols + c];
            }
        }
        t
    }
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-4 * (1. + y.abs()), "{x} vs {y}");
        }
    }

    #[test]
    fn kernels_match_plain_products() {
        for &(rows, k, n) in &[(1, 7, 5), (3, 33, 9), (4, 16, 8), (5, 16, 17), (13, 70, 29), (64, 128, 288)] {
            let x = data(rows * k, 1);
            let w = data(n * k, 2);
            let b = data(n, 3);
            let mut want = naive(&x, &w, rows, k, n);
            for (i, v) in want.iter_mut().enumerate() {
                *v += b[i % n];
            }
            close(&linear(&x, &w, Some(&b), rows, k, n), &want);
            let g = data(rows * n, 4);
            let mut dw = vec![0f32; n * k];
            let mut db = vec![0f32; n];
            let dx = linear_backward(&g, &x, &w, rows, k, n, &mut dw, Some(&mut db));
            // dx = g · w, dw = gᵀ · x, db = column sums of g.
            let wt = transpose(&w, n, k);
            close(&dx, &naive(&g, &wt, rows, n, k));
            let gt = transpose(&g, rows, n);
            let xt = transpose(&x, rows, k);
            close(&dw, &naive(&gt, &xt, n, rows, k));
            let want_db: Vec<f32> = (0..n).map(|o| (0..rows).map(|r| g[r * n + o]).sum()).collect();
            close(&db, &want_db);
            close(&matvec(&w, n, k, &x[..k]), &naive(&x[..k], &w, 1, k, n));
        }
    }
}
