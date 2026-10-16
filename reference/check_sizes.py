"""Cross-checks the Rust Engine against this numpy reference at the larger size presets.

Usage: cargo run --release --example size_compat -- OUT && python check_sizes.py OUT
For each preset, the reference loads the checkpoint the Engine wrote and recomputes the logits,
loss and gradients the Engine reported for the same prompt.
"""
import json
import pathlib
import sys

import numpy as np

from nova2_ref import Nova2, features_for_ids

out = pathlib.Path(sys.argv[1])
failed = False
for record_path in sorted(out.glob("*.json")):
    r = json.loads(record_path.read_text())
    m, tok = Nova2.load(str(out / f"{r['preset']}.nnova2"))
    assert m.n == r["parameters"], (m.n, r["parameters"])
    ids = np.asarray(r["ids"], dtype=np.int64)
    feats = features_for_ids(tok, list(r["ids"]), m.cfg.window_chars)
    logits = m.logits(ids[None], feats[None])[0, -1].astype(np.float64)
    want = np.asarray(r["last_logits"], dtype=np.float64)
    logit_err = float(np.abs(logits - want).max())
    loss, G = m.loss_and_grads(ids[None], feats[None], np.asarray(r["targets"])[None],
                               np.asarray(r["mask"], dtype=np.float32)[None])
    norm = float(np.sqrt((G.astype(np.float64) ** 2).sum()))
    idx = np.asarray([i for i, _ in r["grad_sample"]])
    vals = np.asarray([v for _, v in r["grad_sample"]], dtype=np.float64)
    scale = float(np.abs(G).max())
    grad_err = float(np.abs(G[idx].astype(np.float64) - vals).max())
    ok = (logit_err <= 1e-4 * max(1.0, float(np.abs(want).max()))
          and abs(loss - r["loss"]) <= 1e-4 * max(1.0, abs(loss))
          and abs(norm - r["grad_norm"]) <= 1e-3 * norm
          and grad_err <= 1e-4 * scale)
    failed |= not ok
    print(f"{'OK  ' if ok else 'FAIL'} {r['preset']:>4}: {m.n:,} parameters; max logit diff {logit_err:.2e}, "
          f"loss {loss:.6f} vs {r['loss']:.6f}, grad norm {norm:.6f} vs {r['grad_norm']:.6f}, "
          f"max sampled grad diff {grad_err:.2e} (largest grad {scale:.2e})")
sys.exit(1 if failed else 0)
