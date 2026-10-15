"""Expected values for the Rust Engine on a REAL trained checkpoint, from the numpy reference.

    python3 make_compat.py <checkpoint.nnova2> <out.json>

The checkpoint is loaded exactly as stored (float32 weights) and then evaluated in float64, so
the JSON holds the reference's best answer for those weights. The Rust test
(crates/yendory-engine/tests/compat.rs) must reproduce, for every case:
  - last-position logits for the prompt (uncached and KV-cached), all four ablation modes
  - greedy generation token ids (uncached and KV-cached), identical
  - loss and sampled gradients for one training sequence (backward pass on real weights)
  - one AdamW step from the stored optimizer state, focus "joint" and "right" (parameter deltas)
`min_margin` records the smallest top-1 vs top-2 logit gap seen during each greedy run, so a
near-tie that float32 could legitimately flip is visible instead of hidden.
Prompts are written here (not taken from any lesson file) so the JSON contains no curriculum.
"""
import json, sys
import numpy as np
from nova2_ref import Nova2, features_for_ids

PROMPTS = [
    "U: hello\nN: ",
    "U: What is your name?\nN: ",
    "U: What color is the ball?\nN: ",
    "U: Doro has 3 apples. Pim has 5 apples. Who has more apples?\nN: ",
    "U: The key is in the box. Where is the key?\nN: ",
    "U: Is the sky green?\nN: ",
    "U: Hi\nN: Hello!\nU: How are you?\nN: ",
    # Long conversations: the 256-token window fills during generation / before it.
    "U: hello\nN: Hello!\n" * 21 + "U: What is your name?\nN: ",
    "U: hello\nN: Hello!\n" * 25 + "U: What is your name?\nN: ",
]
TRAIN_TEXT = "U: The cup is on the table. Where is the cup?\nN: The cup is on the table.\n"
MODES = {"full": {}, "no_afferent": {"no_afferent": True},
         "no_efferent": {"no_efferent": True}, "no_right": {"no_right": True}}
MAX_NEW = 24
LR = 1e-3


def last_logits(m, tok, ids, abl):
    """Same windowing as Nova2.next_token_probs: features over the whole history, then the
    last max_tokens positions (positions restart at 0 inside the window)."""
    feats = features_for_ids(tok, ids, m.cfg.window_chars)[-m.cfg.max_tokens:]
    win = np.asarray(ids[-m.cfg.max_tokens:])
    return m.logits(win[None], feats[None], **abl)[0, -1].astype(np.float64)


def greedy(m, tok, ids, abl):
    """Same rule as Nova2.generate (argmax over valid ids, stop after a newline token),
    but keeps ids and the smallest top-2 margin."""
    ids, out, margin = list(ids), [], float("inf")
    for _ in range(MAX_NEW):
        z = last_logits(m, tok, ids + out, abl)
        z[tok.n_valid():] = -np.inf
        top2 = np.sort(z)[-2:]
        margin = min(margin, float(top2[1] - top2[0]))
        nxt = int(np.argmax(z))
        out.append(nxt)
        if tok.token_text(nxt) == "\n":
            break
    return out, margin


def main(path, out_path):
    m, tok = Nova2.load(path)
    m.dtype = np.float64
    m.P = m.P.astype(np.float64)                   # exact copy of the stored float32 values
    cases = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        for mode, abl in MODES.items():
            gen, margin = greedy(m, tok, ids, abl)
            cases.append({"prompt": prompt, "mode": mode, "ids": ids,
                          "last_logits": [float(x) for x in last_logits(m, tok, ids, abl)],
                          "tokens": len(ids), "greedy_ids": gen, "greedy_text": tok.decode(gen), "min_margin": margin})
            print(f"{mode:12s} {len(ids):3d} tok {prompt[-40:]!r} -> {tok.decode(gen)!r}  (min margin {margin:.4g})")
    seq = tok.encode(TRAIN_TEXT)
    ids, tg = seq[:-1], seq[1:]
    mask = np.ones((1, len(ids)))
    feats = features_for_ids(tok, ids, m.cfg.window_chars)
    loss, G = m.loss_and_grads(np.asarray([ids]), feats[None], np.asarray([tg]), mask)
    idx = sorted(set(list(range(0, m.n, m.n // 97)) + [m.n - 1]))
    big = [int(i) for i in np.argsort(-np.abs(G))[:32]]      # the gradients that matter most
    sample = sorted(set(idx + big))
    steps = {}
    for focus in ("joint", "right"):                 # one AdamW step from the stored state
        c = Nova2.load(path)[0]
        c.dtype = np.float64
        for name in ("P", "M", "V"):
            setattr(c, name, getattr(c, name).astype(np.float64))
        c.decay_mask = c.decay_mask.astype(np.float64)
        c.group_mask = {k: v.astype(np.float64) for k, v in c.group_mask.items()}
        before = c.P.copy()
        c.adam_step(G, LR, focus=focus)
        delta = c.P - before
        steps[focus] = {"lr": LR, "step_after": int(c.step), "max_abs_delta": float(np.abs(delta).max()),
                        "delta": [[i, float(delta[i])] for i in sample],
                        "unchanged_params": int((delta == 0).sum())}
    data = {"checkpoint": path.rsplit("/", 1)[-1], "step": int(m.step), "params": int(m.n),
            "words": len(tok.words), "max_new_tokens": MAX_NEW, "cases": cases,
            "train": {"text": TRAIN_TEXT, "ids": ids, "targets": tg, "loss": float(loss),
                      "grad_norm": float(np.sqrt((G ** 2).sum())),
                      "grads": [[i, float(G[i])] for i in sample]},
            "adam": steps}
    json.dump(data, open(out_path, "w"), indent=0)
    print("loss", loss, "grad samples", len(data["train"]["grads"]), "->", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
