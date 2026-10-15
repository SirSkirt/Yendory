"""Expected values for the Engine's learning pipeline, from lessons.py / train_lessons.py.

    python3 make_learn.py <lesson_dir> <checkpoint.nnova2> <out.json> [--min_count N] [--eval]

Records, for crates/yendory-engine/tests/learn.rs:
  - every parsed item (kind, file, held-out flag, segments, prompt/answer/contradicted, turns)
  - the grown word vocabulary (grow_vocabulary over the training items)
  - encoded training rows for a fixed selection of training items (ids, targets, mask)
  - loss and sampled gradients of ONE padded batch of those rows (float64 reference)
  - with --eval: per-unit greedy replies and the supported-vs-contradicted preference on the
    held-out items, plus the aggregate exactly as train_lessons.evaluate reports it
"""
import argparse, glob, json, os
import numpy as np
from nova2_ref import Nova2, Tokenizer, features_for_ids
import lessons as L
from curriculum import fmt


def rows_for(tok, items, max_tokens):
    out = []
    for it in items:
        ids, w = [], []
        for text, weight in it["segments"]:
            t = tok.encode(text); ids += t; w += [weight] * len(t)
        if 2 <= len(ids) <= max_tokens + 1:
            out.append((ids, w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lesson_dir"); ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--min_count", type=int, default=12); ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.lesson_dir, "*.txt")))
    items = [it for f in files for it in L.parse_file(f)]
    model, tok = Nova2.load(a.ckpt)
    words = L.grow_vocabulary(tok.words, items, min_count=a.min_count)
    grown = Tokenizer(words)
    train = [it for it in items if not it["test"]]
    test = [it for it in items if it["test"]]

    # Fixed selection: every 7th training item that fits the window, at most 8 rows.
    rows = rows_for(grown, train[::7], model.cfg.max_tokens)[:8]
    model.dtype = np.float64; model.P = model.P.astype(np.float64)
    B, T = len(rows), max(len(r[0]) for r in rows) - 1
    ids = np.zeros((B, T), np.int64); tg = np.zeros((B, T), np.int64)
    mk = np.zeros((B, T)); fe = np.zeros((B, T, model.cfg.feat))
    for r, (seq, w) in enumerate(rows):                       # same layout as train_lessons.make_batch
        n = len(seq) - 1
        ids[r, :n] = seq[:-1]; tg[r, :n] = seq[1:]; mk[r, :n] = w[1:]
        fe[r, :n] = features_for_ids(grown, seq, model.cfg.window_chars)[:-1]
    # The vocabulary grew, so new word ids must be valid for this checkpoint's embedding (vocab 384).
    loss, G = model.loss_and_grads(ids, fe, tg, mk)
    idx = sorted(set(list(range(0, model.n, model.n // 97)) + [int(i) for i in np.argsort(-np.abs(G))[:32]]))

    data = {
        "files": [os.path.basename(f) for f in files],
        "items": [{"kind": it["kind"], "source": os.path.basename(it["file"]), "held_out": it["test"],
                   "segments": [[t, w] for t, w in it["segments"]], "prompt": it.get("prompt"),
                   "answer": it.get("answer"), "contradicted": it.get("contradicted"),
                   "users": it.get("users", []), "replies": it.get("novas", [])} for it in items],
        "min_count": a.min_count, "words_before": len(tok.words), "words": words,
        "rows": [{"ids": seq[:-1], "targets": seq[1:], "mask": w[1:]} for seq, w in rows],
        "batch": {"loss": float(loss), "grad_norm": float(np.sqrt((G ** 2).sum())),
                  "grads": [[i, float(G[i])] for i in idx]},
    }
    if a.eval:
        model, tok = Nova2.load(a.ckpt)          # evaluation uses the checkpoint as stored
        units = []
        for it in test:
            if it["kind"] == "conversation":
                hist = ""
                for u, ans in zip(it["users"], it["novas"]):
                    hist += f"U: {u}\nN: "
                    units.append({"kind": "conversation", "prompt": hist, "expected": ans,
                                  "reply": model.generate(tok, hist, max_tokens=24)})
                    hist += ans + "\n"
                continue
            p = fmt(it["prompt"])
            u = {"kind": it["kind"], "prompt": p, "expected": it["answer"],
                 "reply": model.generate(tok, p, max_tokens=24)}
            if it["kind"] == "reasoning":
                u["prefers_supported"] = bool(model.score_continuation(tok, p, it["answer"] + "\n")
                                              > model.score_continuation(tok, p, it["contradicted"] + "\n"))
            units.append(u)
        data["eval"] = units
    json.dump(data, open(a.out, "w"), indent=0)
    print(f"{len(items)} items ({len(test)} held out), {len(words)} words, {B} rows, batch loss {loss:.6f}"
          + (f", {len(data.get('eval', []))} evaluated units" if a.eval else ""))


if __name__ == "__main__":
    main()
