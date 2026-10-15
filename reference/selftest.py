"""Verification suite for the Nova-2 reference. Run:  python3 selftest.py"""
import sys, time, numpy as np
from nova2_ref import *

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (("  " + detail) if detail else ""))

# --- 1. parameter budget ------------------------------------------------------------
tot, groups = param_report(Config())
check("param budget <= Nova v1 (1,200,144)", tot <= BUDGET, f"{tot:,} ({tot/BUDGET:.1%}) {groups}")
check("Nova2.n matches layout total", Nova2(Config()).n == tot)

# --- 2. tokenizer is a faithful port of the Rust tests -------------------------------
t = Tokenizer(); t.grow_from_corpus("Mira opens a box. Mira opens the box.")
cases = ["Mira opens a box.", "Tavik opens the blue box.", "Mira's cat!", "", "Mira opened a box"]
check("tokenizer lossless (unknown words/punctuation)", all(t.decode(t.encode(s)) == s for s in cases))
check("tokenizer compresses known words", len(t.encode("Mira opens a box.")) < len("Mira opens a box."))
t2 = Tokenizer(); t2.grow_from_corpus("Mira Mira opens opens"); orig = t2.encode("Mira")
t2.grow_from_corpus("Tavik Tavik blue blue")
check("tokenizer ids stable when vocabulary grows", t2.encode("Mira") == orig and t2.decode(t2.encode("Tavik Mira")) == "Tavik Mira")

# --- 3. right-hemisphere features == encode_ascii_context --------------------------------
f = window_features(list(b"Hi."), [3], 64)[0].reshape(64, 16)
check("features: padding is all-zero, real chars have present=1", f[:61].sum() == 0 and (f[61:, 15] == 1).all())
check("features: 'H' uppercase+newline flags", f[61, 7] == 1 and f[61, 8] == 0 and f[63, 11] == 1 and f[63, 13] == 1)

# --- 4. gradient check: float64, every parameter, several configs ---------------------------------
def gradcheck(cfg, label, abl=None, nsamp=None, seed=3):
    """Central differences in float64. Pass iff |numeric-analytic| <= 1e-8 + 1e-5*max(|numeric|,|analytic|).
    (Pure relative error is misleading for ~1e-7 gradients: float64 roundoff alone is ~1e-10 absolute.)"""
    abl = abl or {}
    m = Nova2(cfg, seed=seed, dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    m.P += rng.normal(0, 0.05, m.n)            # make every path (incl. gated ones) active
    B, T = 2, 7
    ids = rng.integers(0, cfg.vocab, (B, T)); tg = rng.integers(0, cfg.vocab, (B, T))
    feats = rng.normal(0, 1, (B, T, cfg.feat)); mask = (rng.random((B, T)) > 0.25).astype(float); mask[0, 0] = 1
    loss, G = m.loss_and_grads(ids, feats, tg, mask, **abl)
    idx = np.arange(m.n) if nsamp is None else np.random.default_rng(0).choice(m.n, nsamp, replace=False)
    worst_abs, worst_rel, bad, eps = 0.0, 0.0, 0, 1e-5
    for i in idx:
        old = m.P[i]
        m.P[i] = old + eps; lp = m.loss_only(ids, feats, tg, mask, **abl)
        m.P[i] = old - eps; lm = m.loss_only(ids, feats, tg, mask, **abl)
        m.P[i] = old
        num = (lp - lm) / (2 * eps); a = abs(num - G[i]); scale = max(abs(num), abs(G[i]))
        worst_abs = max(worst_abs, a); worst_rel = max(worst_rel, a / max(scale, 1e-3))
        bad += a > 1e-8 + 1e-5 * scale
    check(f"gradients vs finite differences [{label}]", bad == 0,
          f"{len(idx):,}/{m.n:,} params, {bad} out of tolerance, worst abs err {worst_abs:.1e}, worst rel err (floor 1e-3) {worst_rel:.1e}")

small = Config(vocab=24, d=16, heads=2, ff=24, blocks=3, split=1, loops=2, window_chars=4, r1=12, r2=10, max_tokens=16)
t0 = time.time()
gradcheck(small, "full model, loops=2, ALL parameters")
gradcheck(Config(**{**small.__dict__, "loops": 1, "split": 2}), "loops=1, split=2", nsamp=2500)
gradcheck(Config(**{**small.__dict__, "loops": 3}), "loops=3 (shared weights, 3 rounds)", nsamp=2500)
gradcheck(small, "no_efferent ablation", {"no_efferent": True}, nsamp=2500)
gradcheck(small, "no_afferent ablation", {"no_afferent": True}, nsamp=2500)
print(f"      (gradient checks took {time.time()-t0:.0f}s)")

# --- 5. causality: nothing at position <= t may depend on tokens/features after t ---------------
m = Nova2(small, seed=5, dtype=np.float64); m.P += np.random.default_rng(9).normal(0, 0.05, m.n)
rng = np.random.default_rng(11)
ids = rng.integers(0, small.vocab, (1, 10)); feats = rng.normal(0, 1, (1, 10, small.feat))
a = m.logits(ids, feats)
ids2, feats2 = ids.copy(), feats.copy(); ids2[0, 6:] = (ids2[0, 6:] + 3) % small.vocab; feats2[0, 6:] += 1.0
b = m.logits(ids2, feats2)
check("causal: logits at positions < 6 unchanged by edits at >= 6", np.abs(a[0, :6] - b[0, :6]).max() < 1e-12)
check("causal: logits at positions >= 6 do change", np.abs(a[0, 6:] - b[0, 6:]).max() > 1e-6)

# --- 6. every piece is load-bearing in the computation graph --------------------------------------
full = m.logits(ids, feats)
check("ablation no_afferent changes output", np.abs(full - m.logits(ids, feats, no_afferent=True)).max() > 1e-9)
check("ablation no_efferent changes output", np.abs(full - m.logits(ids, feats, no_efferent=True)).max() > 1e-9)
check("ablation no_right changes output", np.abs(full - m.logits(ids, feats, no_right=True)).max() > 1e-9)

# --- 7. checkpoint round trip + training focus -----------------------------------------------------------
import tempfile, os
tk = Tokenizer(["hello", "Nova"]); m32 = Nova2(small, seed=2); m32.step = 7
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "x.n2"); m32.save(p, tk); m2, tk2 = Nova2.load(p)
check("checkpoint round-trip (weights, optimizer state, step, vocabulary)",
      np.array_equal(m2.P, m32.P) and m2.step == 7 and tk2.words == tk.words and m2.cfg == m32.cfg)
G = np.random.default_rng(1).normal(0, 1, m32.n).astype(np.float32); before = m32.P.copy()
m32.adam_step(G, 1e-2, focus="right"); moved = m32.P != before
check("training focus 'right' updates only right-hemisphere parameters",
      moved[m32.group_mask["right"] > 0].any() and not moved[m32.group_mask["right"] == 0].any())

print("\nALL PASSED" if ok_all else "\nSOME CHECKS FAILED"); sys.exit(0 if ok_all else 1)
