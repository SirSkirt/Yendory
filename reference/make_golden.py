"""Writes tests/compat/golden/*.txt : deterministic weights/inputs -> loss, logits and gradients from the
VERIFIED numpy reference (float64). The Rust port must reproduce these (crates/yendory-engine/tests/golden.rs)."""
import math, numpy as np
from nova2_ref import *

def golden(name, cfg, abl):
    m = Nova2(cfg, seed=1, dtype=np.float64)
    m.P[:] = [0.2 * math.sin(0.37 * i + 1.0) for i in range(m.n)]       # reproducible in any language
    T = 7
    ids = np.array([[(7 * t + 3) % cfg.vocab for t in range(T)]])
    tg = np.array([[(5 * t + 1) % cfg.vocab for t in range(T)]])
    feats = np.array([[[math.sin(0.11 * (t * cfg.feat + j)) for j in range(cfg.feat)] for t in range(T)]])
    wts = np.array([[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0]])
    loss, G = m.loss_and_grads(ids, feats, tg, wts, **abl)
    logits = m.logits(ids, feats, **abl)[0, -1]
    idx = sorted(set(list(range(0, m.n, max(1, m.n // 60))) + [m.n - 1]))
    L = [f"config vocab={cfg.vocab} d={cfg.d} heads={cfg.heads} ff={cfg.ff} blocks={cfg.blocks} split={cfg.split} "
         f"loops={cfg.loops} window_chars={cfg.window_chars} r1={cfg.r1} r2={cfg.r2} max_tokens={cfg.max_tokens}",
         f"ablation no_afferent={int(abl.get('no_afferent', False))} no_efferent={int(abl.get('no_efferent', False))} no_right={int(abl.get('no_right', False))}",
         "ids " + " ".join(map(str, ids[0])), "targets " + " ".join(map(str, tg[0])), "weights " + " ".join(map(str, wts[0])),
         f"param_count {m.n}", f"loss {float(loss)!r}", "last_logits " + " ".join(repr(float(x)) for x in logits[:12])]
    L += [f"grad {i} {float(G[i])!r}" for i in idx]
    open(f"../tests/compat/golden/{name}.txt", "w").write("\n".join(L) + "\n")
    print(name, "params", m.n, "loss", round(loss, 6), "grad samples", len(idx))

small = Config(vocab=24, d=16, heads=2, ff=24, blocks=3, split=1, loops=2, window_chars=4, r1=12, r2=10, max_tokens=16)
golden("small_full", small, {})
golden("small_no_efferent", small, {"no_efferent": True})
golden("small_loops1", Config(**{**small.__dict__, "loops": 1, "split": 2}), {})
