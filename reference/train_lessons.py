"""
Continue training a Nova-2 checkpoint on Unified Teacher Lesson files.
    python3 train_lessons.py <lesson_dir> <start.ckpt> <out.ckpt> --steps 8000
- Grows the word vocabulary from the lessons (stable ids; names stay character-level).
- ~10% of every block is held out (lessons.held_out) and never trained on.
- A small share of the earlier synthetic curriculum is mixed in so old skills are rehearsed.
- Saves out.ckpt every 200 steps (resumable) and out.best.ckpt whenever held-out accuracy improves.
"""
import argparse, glob, json, os, random, time
import numpy as np
from nova2_ref import Nova2, Tokenizer, features_for_ids, lr_schedule
import lessons as L
import curriculum as C

REHEARSAL = 0.10


def encode_item(tok, segments, max_tokens):
    ids, w = [], []
    for text, weight in segments:
        t = tok.encode(text); ids += t; w += [weight] * len(t)
    if len(ids) > max_tokens + 1:                      # long prose: random window
        s = random.randrange(len(ids) - max_tokens - 1); ids, w = ids[s:s + max_tokens + 1], w[s:s + max_tokens + 1]
    return ids, w


def make_batch(tok, rng, train, B, cfg):
    rows = []
    for _ in range(B):
        if rng.random() < REHEARSAL:
            _, q, a = C.lesson(rng, C.TRAIN_NAMES); segs = L.qa(q, a)
        else:
            segs = rng.choice(train)["segments"]
        rows.append(encode_item(tok, segs, cfg.max_tokens))
    T = max(len(r[0]) for r in rows) - 1
    ids = np.zeros((B, T), np.int64); tg = np.zeros((B, T), np.int64)
    mk = np.zeros((B, T), np.float32); fe = np.zeros((B, T, cfg.feat), np.float32)
    for r, (seq, w) in enumerate(rows):
        n = len(seq) - 1
        ids[r, :n] = seq[:-1]; tg[r, :n] = seq[1:]; mk[r, :n] = w[1:]   # weight of the token being predicted
        fe[r, :n] = features_for_ids(tok, seq, cfg.window_chars)[:-1]
    return ids, fe, tg, mk


def evaluate(model, tok, test, limit=None, seed=0):
    """Greedy exact match on held-out units. Conversations: each Nova turn given the real earlier turns."""
    rng = random.Random(seed); items = list(test)
    if limit:
        by = {}
        for it in items:
            by.setdefault(it["kind"], []).append(it)
        items = [x for v in by.values() for x in rng.sample(v, min(limit, len(v)))]
    ok, n, pref, npref = {}, {}, 0, 0
    for it in items:
        k = it["kind"]
        if k == "conversation":
            hist = ""
            for u, a in zip(it["users"], it["novas"]):
                hist += f"U: {u}\nN: "
                got = model.generate(tok, hist, max_tokens=24)
                ok["conv_turn"] = ok.get("conv_turn", 0) + (got == a); n["conv_turn"] = n.get("conv_turn", 0) + 1
                hist += a + "\n"
            continue
        got = model.generate(tok, C.fmt(it["prompt"]), max_tokens=24)
        ok[k] = ok.get(k, 0) + (got == it["answer"]); n[k] = n.get(k, 0) + 1
        if k == "reasoning":   # does Nova prefer the supported answer over the contradicted one?
            p = C.fmt(it["prompt"])
            pref += model.score_continuation(tok, p, it["answer"] + "\n") > model.score_continuation(tok, p, it["contradicted"] + "\n")
            npref += 1
    res = {k: round(ok[k] / n[k], 3) for k in n}
    if npref:
        res["reasoning_prefers_supported"] = round(pref / npref, 3)
    res["overall_exact"] = round(sum(ok.values()) / max(1, sum(n.values())), 3)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lesson_dir"); ap.add_argument("start"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_limit", type=int, default=40); ap.add_argument("--max_seconds", type=float, default=1e12)
    a = ap.parse_args()
    items = [it for f in sorted(glob.glob(a.lesson_dir + "/*.txt")) for it in L.parse_file(f)]
    train = [it for it in items if not it["test"]]; test = [it for it in items if it["test"]]
    state_path = a.out + ".state.json"
    if os.path.exists(a.out):
        model, tok = Nova2.load(a.out); st = json.load(open(state_path))
        print(f"resumed at step {model.step}", flush=True)
    else:
        model, tok = Nova2.load(a.start)
        tok = Tokenizer(L.grow_vocabulary(tok.words, items))
        st = {"start_step": model.step, "best": -1.0}
        base = evaluate(model, tok, test, a.eval_limit)
        print("BASELINE", json.dumps(base), flush=True)
        st["baseline"] = base
    start, total = st["start_step"], st["start_step"] + a.steps
    print(f"train units {len(train)}, held-out {len(test)}, vocab {len(tok.words)}, steps {start}->{total}", flush=True)
    rng = random.Random(model.step * 31 + 7); random.seed(model.step)
    t0 = time.time(); losses = []
    while model.step < total and time.time() - t0 < a.max_seconds:
        ids, fe, tg, mk = make_batch(tok, rng, train, a.batch, model.cfg)
        loss, G = model.loss_and_grads(ids, fe, tg, mk)
        model.adam_step(G, lr_schedule(model.step - start, a.lr, warmup=200, total=a.steps))
        losses.append(loss); k = model.step - start
        if k % 100 == 0:
            print(f"step {model.step} loss {np.mean(losses[-100:]):.4f} {time.time()-t0:6.0f}s", flush=True)
        if k % 200 == 0:
            model.save(a.out + ".tmp", tok); os.replace(a.out + ".tmp", a.out); json.dump(st, open(state_path, "w"))
        if k % a.eval_every == 0 or model.step == total:
            r = evaluate(model, tok, test, a.eval_limit)
            print("EVAL", model.step, json.dumps(r), flush=True)
            if r["overall_exact"] > st["best"]:
                st["best"] = r["overall_exact"]; st["best_step"] = model.step
                model.save(a.out + ".best.tmp", tok); os.replace(a.out + ".best.tmp", a.out.replace(".ckpt", ".best.ckpt"))
            json.dump(st, open(state_path, "w"))
    model.save(a.out + ".tmp", tok); os.replace(a.out + ".tmp", a.out); json.dump(st, open(state_path, "w"))
    print("done at step", model.step, "best held-out", st["best"], "at", st.get("best_step"), flush=True)
