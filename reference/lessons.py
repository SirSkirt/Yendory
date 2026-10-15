"""
Reads Nova's Unified Teacher Lesson files ([Dialogue Lesson], [Understanding Lesson],
[Multi-Turn Conversation Lesson], [Reasoning Lesson]) and turns them into Nova-2 training
sequences. Nothing here runs at inference; answers are only training targets.

Each item is a list of (text, weight) segments, trained with full-sequence loss:
  question / user turn text  -> weight PROMPT (context, lightly trained)
  Nova's answer + newline    -> weight 1.0
  explanatory prose ("T: ")  -> weight PROSE (language exposure, like v1's "T: " context)
About 10% of units in every block are held out by a stable hash of their text, so the same
question never lands in both training and test.
"""
import re, sys, zlib, json
from collections import Counter

PROMPT, PROSE = 0.15, 0.3
BLOCKS = {"Dialogue Lesson": "dialogue", "Understanding Lesson": "understanding",
          "Multi-Turn Conversation Lesson": "conversation", "Reasoning Lesson": "reasoning"}
REASON = re.compile(r"^Premise:\s*(.*?)\s*Question:\s*(.*?)\s*Supported:\s*(.*?)\s*Contradicted:\s*(.*?)\s*$")


def held_out(key, pct=10):
    canon = " ".join(key.lower().rstrip("?.! ").split())
    return zlib.crc32(canon.encode()) % 100 < pct


def qa(q, a):
    return [(f"U: {q}\nN: ", PROMPT), (f"{a}\n", 1.0)]


def parse_file(path):
    text = open(path, encoding="ascii").read()
    items = []   # dicts: kind, file, segments, test(bool), answer, prompt, contradicted
    for m in re.finditer(r"^\[(.+?)\]\n(.*?)^\[/\1\]", text, re.S | re.M):
        kind = BLOCKS.get(m.group(1))
        if kind is None:
            continue
        body = m.group(2)
        explanation, _, rest = body.partition("[examples]\n")
        examples = rest.split("[/examples]")[0]
        if explanation.strip():
            items.append({"kind": "prose", "file": path, "test": False,
                          "segments": [(f"T: {explanation.strip()}\n", PROSE)]})
        if kind in ("dialogue", "understanding"):
            for line in examples.splitlines():
                if "  " not in line:
                    continue
                q, a = [s.strip() for s in line.split("  ", 1)]
                items.append({"kind": kind, "file": path, "test": held_out(q), "prompt": q,
                              "answer": a, "segments": qa(q, a)})
        elif kind == "reasoning":
            for line in examples.splitlines():
                r = REASON.match(line.strip())
                if r:
                    premise, question, sup, con = r.groups()
                    q = f"{premise} {question}"
                    items.append({"kind": kind, "file": path, "test": held_out(q), "prompt": q,
                                  "answer": sup, "contradicted": con, "segments": qa(q, sup)})
        else:  # conversation
            for s in re.findall(r"\[session\]\n(.*?)\[/session\]", examples, re.S):
                turns = [l.split(":", 1) for l in s.strip().splitlines() if ":" in l]
                segs, users, novas = [], [], []
                for who, said in turns:
                    said = said.strip()
                    if who.strip() == "User":
                        segs.append((f"U: {said}\nN: ", PROMPT)); users.append(said)
                    else:
                        segs.append((f"{said}\n", 1.0)); novas.append(said)
                items.append({"kind": kind, "file": path, "test": held_out(s), "segments": segs,
                              "users": users, "novas": novas})
    return items


def is_name_like(word, counts):
    """A capitalized word that is rarely written in lowercase is treated as a name and kept
    character-level, so Nova learns to read and copy names instead of memorizing them."""
    return word[0].isupper() and counts.get(word.lower(), 0) < 0.2 * counts[word]


def grow_vocabulary(existing, items, capacity=256, min_count=12):
    from nova2_ref import is_word
    words = Counter()
    for it in items:
        if it["test"]:
            continue
        for text, _ in it["segments"]:
            for w in re.findall(r"[A-Za-z0-9']+", text):
                if 3 <= len(w) <= 24:
                    words[w] += 1
    have = set(existing)
    cand = [(w, c) for w, c in words.items() if c >= min_count and w not in have
            and not is_name_like(w, words) and all(is_word(ord(ch)) for ch in w)]
    cand.sort(key=lambda x: (-x[1], x[0]))
    return list(existing) + [w for w, _ in cand[:capacity - len(existing)]]


if __name__ == "__main__":
    import glob
    items = [it for f in sorted(glob.glob(sys.argv[1] + "/*.txt")) for it in parse_file(f)]
    c = Counter((it["kind"], it["test"]) for it in items)
    for k in ["prose", "dialogue", "understanding", "conversation", "reasoning"]:
        print(f"{k:14s} train {c[(k, False)]:5d}   held-out {c[(k, True)]:4d}")
