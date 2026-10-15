"""
Nova-2 "Bicameral" -- numpy REFERENCE implementation (the verified spec).

Three pieces, one brain, one loss:
  LEFT  hemisphere : causal multi-head Transformer over tokens (language, order, long-range reads)
  RIGHT hemisphere : dense associative network over the last 64 characters (spelling, local patterns)
  INTERCONNECT     : gated right->left injection (afferent) and left->right conditioning (efferent),
                     run twice, optionally looped ("thinking" rounds) with shared weights.

Everything (tape autograd, Adam, tokenizer port, checkpoint) is plain numpy / stdlib.
Parameters live in ONE flat float array (like Nova v1) so Adam state, checkpoints and
append-only migration stay simple.  See ARCHITECTURE.md for the design rationale.
"""
import json, math, struct
from dataclasses import dataclass, asdict
import numpy as np

# ----------------------------------------------------------------------------
# Tokenizer: exact port of neural-nova/src/tokenizer.rs (ids 0..127 chars, 128.. words)
# ----------------------------------------------------------------------------
BASE = 128
MAX_WORDS = 256
TOKEN_CAPACITY = BASE + MAX_WORDS


def is_word(b):
    return (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 39


class Tokenizer:
    def __init__(self, words=None):
        self.words = list(words or [])

    def grow_from_corpus(self, corpus):
        counts = {}
        cur = bytearray()
        parts = []
        for b in corpus.encode("ascii", "replace") + b"\x00":
            if is_word(b):
                cur.append(b)
            else:
                parts.append(bytes(cur)); cur = bytearray()
        for part in parts:
            if len(part) < 3 or len(part) > 24:
                continue
            w = part.decode("ascii")
            if w in self.words:
                continue
            counts[w] = counts.get(w, 0) + 1
        props = sorted(((w, c) for w, c in counts.items() if c >= 2), key=lambda x: (-x[1], x[0]))
        prev = len(self.words)
        for w, _ in props[: MAX_WORDS - len(self.words)]:
            self.words.append(w)
        return len(self.words) - prev

    def encode(self, text):
        data = text.encode("ascii")  # raises on non-ASCII, like the Rust code
        n = len(data)
        wb = [w.encode("ascii") for w in self.words]
        out, i = [], 0
        while i < n:
            matched = None
            if is_word(data[i]) and (i == 0 or not is_word(data[i - 1])):
                for idx, w in enumerate(wb):
                    L = len(w)
                    if data.startswith(w, i) and (i + L == n or not is_word(data[i + L])) \
                            and (matched is None or L > matched[1]):
                        matched = (idx, L)
            if matched:
                out.append(BASE + matched[0]); i += matched[1]
            else:
                out.append(data[i]); i += 1
        return out

    def token_text(self, t):
        return chr(t) if t < BASE else self.words[t - BASE]

    def decode(self, toks):
        return "".join(self.token_text(t) for t in toks)

    def token_len(self, t):
        return 1 if t < BASE else len(self.words[t - BASE])

    def n_valid(self):
        return BASE + len(self.words)


# ----------------------------------------------------------------------------
# Right-hemisphere input features: exact port of encode_ascii_context (16 features / char)
# ----------------------------------------------------------------------------
CHAR_FEATURES = 16


def _feature_table():
    tab = np.zeros((128, CHAR_FEATURES), np.float32)
    for code in range(128):
        for bit in range(7):
            tab[code, bit] = 1.0 if (code >> bit) & 1 else -1.0
        tab[code, 7] = 65 <= code <= 90
        tab[code, 8] = 97 <= code <= 122
        tab[code, 9] = 48 <= code <= 57
        tab[code, 10] = code in (32, 9, 13)
        tab[code, 11] = (33 <= code <= 47) or (58 <= code <= 64) or (91 <= code <= 96) or (123 <= code <= 126)
        tab[code, 12] = chr(code) in "aeiouAEIOU"
        tab[code, 13] = code in (46, 33, 63, 10)
        tab[code, 14] = code == 10
        tab[code, 15] = 1.0
    return tab


FEATURE_TABLE = _feature_table()


def window_features(codes, ends, window):
    """Features for the `window` characters that end at each token end (left zero-padded)."""
    n = len(codes)
    f = FEATURE_TABLE[np.asarray(codes, dtype=np.int64)] if n else np.zeros((0, CHAR_FEATURES), np.float32)
    pad = np.concatenate([np.zeros((window, CHAR_FEATURES), np.float32), f], 0)
    idx = np.asarray(ends, dtype=np.int64)[:, None] + np.arange(window)[None, :]
    return pad[idx].reshape(len(ends), window * CHAR_FEATURES)


def prepare(tok, text, window=64):
    """text -> (token ids, right-hemisphere features aligned to each token end)."""
    ids = tok.encode(text)
    return ids, features_for_ids(tok, ids, window)


def features_for_ids(tok, ids, window=64):
    lens = [tok.token_len(i) for i in ids]
    ends = np.cumsum(lens) if lens else np.zeros(0, np.int64)
    codes = list(tok.decode(ids).encode("ascii"))
    return window_features(codes, ends, window)


# ----------------------------------------------------------------------------
# Config + parameter layout (one flat array; ranges are fixed, append-only friendly)
# ----------------------------------------------------------------------------
BUDGET = 1_200_144  # Nova v1 combined brain (transformer 347,024 + dense 853,120)


@dataclass
class Config:
    vocab: int = TOKEN_CAPACITY   # 128 chars + 256 learnable word units (tied in/out embedding)
    d: int = 128                  # left model width
    heads: int = 4                # head dim = d/heads = 32
    ff: int = 288                 # SwiGLU hidden width
    blocks: int = 4               # unique left transformer blocks
    split: int = 2                # blocks [0,split) = lower stage, [split,blocks) = upper stage
    loops: int = 2                # times the upper stage runs (shared weights) = thinking rounds
    window_chars: int = 64        # right hemisphere sees last 64 chars (16 features each)
    r1: int = 256                 # right trunk width
    r2: int = 192                 # right upper-layer width (after hearing the left)
    max_tokens: int = 256         # context window (tokens)
    rope_base: float = 10000.0
    gate_bias: float = -2.0       # interconnect starts mostly closed (v1 "neutral start" spirit)

    @property
    def hd(self):
        return self.d // self.heads

    @property
    def feat(self):
        return self.window_chars * CHAR_FEATURES


def layout(c):
    """Ordered list of (name, shape, group, init, decay). group in {left,right,inter}."""
    L = []
    s_lin = lambda fan_in: 0.5 / math.sqrt(fan_in)
    res = 1.0 / math.sqrt(2.0 * c.blocks)  # residual-output scaling
    L.append(("embed", (c.vocab, c.d), "left", ("normal", 0.04), False))
    L.append(("out_bias", (c.vocab,), "left", ("zeros", 0), False))
    for i in range(c.blocks):
        p = f"b{i}."
        L.append((p + "ln1", (c.d,), "left", ("ones", 0), False))
        for n in ("wq", "wk", "wv"):
            L.append((p + n, (c.d, c.d), "left", ("normal", s_lin(c.d)), True))
        L.append((p + "wo", (c.d, c.d), "left", ("normal", s_lin(c.d) * res), True))
        L.append((p + "ln2", (c.d,), "left", ("ones", 0), False))
        L.append((p + "w_gate", (c.ff, c.d), "left", ("normal", s_lin(c.d)), True))
        L.append((p + "w_up", (c.ff, c.d), "left", ("normal", s_lin(c.d)), True))
        L.append((p + "w_down", (c.d, c.ff), "left", ("normal", s_lin(c.ff) * res), True))
    L.append(("ln_f", (c.d,), "left", ("ones", 0), False))
    L.append(("r.w1", (c.r1, c.feat), "right", ("normal", math.sqrt(2.0 / c.feat) * 0.5), True))
    L.append(("r.b1", (c.r1,), "right", ("zeros", 0), False))
    L.append(("r.w2", (c.r2, c.r1), "right", ("normal", s_lin(c.r1)), True))
    L.append(("r.wl", (c.r2, c.d), "right", ("normal", s_lin(c.d)), True))   # efferent: left -> right
    L.append(("r.b2", (c.r2,), "right", ("zeros", 0), False))
    L.append(("c.n1", (c.d,), "inter", ("ones", 0), False))
    L.append(("c.a1", (c.d, c.r1), "inter", ("normal", s_lin(c.r1)), True))  # afferent 1: right -> left
    L.append(("c.g1w", (c.d, c.d), "inter", ("normal", s_lin(c.d)), True))
    L.append(("c.g1b", (c.d,), "inter", ("const", c.gate_bias), False))
    L.append(("c.n2", (c.d,), "inter", ("ones", 0), False))
    L.append(("c.a2", (c.d, c.r2), "inter", ("normal", s_lin(c.r2)), True))  # afferent 2
    L.append(("c.g2w", (c.d, c.d), "inter", ("normal", s_lin(c.d)), True))
    L.append(("c.g2b", (c.d,), "inter", ("const", c.gate_bias), False))
    return L


def param_report(c):
    tot, groups = 0, {"left": 0, "right": 0, "inter": 0}
    for name, shape, g, _, _ in layout(c):
        n = int(np.prod(shape)); tot += n; groups[g] += n
    return tot, groups


# ----------------------------------------------------------------------------
# Tiny reverse-mode autograd over numpy arrays (all ops accept leading batch dims)
# ----------------------------------------------------------------------------
class Node:
    __slots__ = ("v", "g", "back")

    def __init__(self, v, back=None):
        self.v, self.g, self.back = v, None, back


def _acc(n, g):
    n.g = g if n.g is None else n.g + g


class Tape:
    def __init__(self, model=None):
        self.nodes, self.params, self.model = [], {}, model

    def new(self, v, back):
        n = Node(v, back); self.nodes.append(n); return n

    def p(self, name):
        if name not in self.params:
            off, shape = self.model.offsets[name]
            self.params[name] = Node(self.model.P[off:off + int(np.prod(shape))].reshape(shape))
        return self.params[name]

    def backward(self, root):
        root.g = np.ones_like(root.v)
        for n in reversed(self.nodes):
            if n.back is not None and n.g is not None:
                n.back(n.g)

    def flat_grad(self):
        G = np.zeros_like(self.model.P)
        for name, node in self.params.items():
            if node.g is not None:
                off, shape = self.model.offsets[name]
                G[off:off + int(np.prod(shape))] += node.g.reshape(-1)
        return G


def linear(t, x, w, b=None):
    y = x.v @ w.v.T
    if b is not None:
        y = y + b.v

    def back(g):
        g2 = g.reshape(-1, g.shape[-1]); x2 = x.v.reshape(-1, x.v.shape[-1])
        _acc(x, g @ w.v); _acc(w, g2.T @ x2)
        if b is not None:
            _acc(b, g2.sum(0))
    return t.new(y, back)


def add(t, a, b):
    return t.new(a.v + b.v, lambda g: (_acc(a, g), _acc(b, g)))


def mul(t, a, b):
    return t.new(a.v * b.v, lambda g: (_acc(a, g * b.v), _acc(b, g * a.v)))


def _sig(x):
    e = np.exp(-np.abs(x))                      # never overflows
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid(t, x):
    s = _sig(x.v)
    return t.new(s, lambda g: _acc(x, g * s * (1 - s)))


def silu(t, x):
    s = _sig(x.v)
    return t.new(x.v * s, lambda g: _acc(x, g * s * (1 + x.v * (1 - s))))


def reshape(t, x, shape):
    old = x.v.shape
    return t.new(x.v.reshape(shape), lambda g: _acc(x, g.reshape(old)))


def rmsnorm(t, x, gain, eps=1e-6):
    m = (x.v * x.v).mean(-1, keepdims=True)
    r = 1.0 / np.sqrt(m + eps)
    y = x.v * r * gain.v
    n = x.v.shape[-1]

    def back(g):
        u = g * gain.v
        _acc(gain, (g * x.v * r).reshape(-1, n).sum(0))
        _acc(x, u * r - x.v * (r ** 3) * (u * x.v).sum(-1, keepdims=True) / n)
    return t.new(y, back)


def embed(t, ids, table):
    def back(g):
        gt = np.zeros_like(table.v)
        np.add.at(gt, ids, g)
        _acc(table, gt)
    return t.new(table.v[ids], back)


def rope(t, x, base):
    """x: [B,T,H,hd]; rotates (first half, second half) pairs by position-dependent angle."""
    T, hd = x.v.shape[1], x.v.shape[3]
    half = hd // 2
    inv = base ** (-np.arange(half) * 2.0 / hd)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos = np.cos(ang)[None, :, None, :].astype(x.v.dtype)
    sin = np.sin(ang)[None, :, None, :].astype(x.v.dtype)
    x1, x2 = x.v[..., :half], x.v[..., half:]
    y = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def back(g):
        g1, g2 = g[..., :half], g[..., half:]
        _acc(x, np.concatenate([g1 * cos + g2 * sin, g2 * cos - g1 * sin], -1))
    return t.new(y, back)


def attention(t, q, k, v):
    """Causal multi-head attention. q,k,v: [B,T,H,hd] -> [B,T,H,hd]."""
    B, T, H, hd = q.v.shape
    sc = 1.0 / math.sqrt(hd)
    s = np.einsum("bthd,bshd->bhts", q.v, k.v) * sc
    mask = np.triu(np.ones((T, T), bool), 1)
    s = np.where(mask[None, None], -np.inf, s)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s); p = p / p.sum(-1, keepdims=True)
    out = np.einsum("bhts,bshd->bthd", p, v.v)

    def back(g):
        _acc(v, np.einsum("bhts,bthd->bshd", p, g))
        dp = np.einsum("bthd,bshd->bhts", g, v.v)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) * sc
        _acc(q, np.einsum("bhts,bshd->bthd", ds, k.v))
        _acc(k, np.einsum("bhts,bthd->bshd", ds, q.v))
    return t.new(out, back)


def xent(t, logits, targets, mask):
    """Mean next-token cross-entropy over positions where mask==1. targets: [B,T] int."""
    z = logits.v - logits.v.max(-1, keepdims=True)
    lse = np.log(np.exp(z).sum(-1))
    pr = np.exp(z - lse[..., None])
    B, T = targets.shape
    lp = z[np.arange(B)[:, None], np.arange(T)[None, :], targets] - lse
    denom = max(float(mask.sum()), 1.0)
    loss = -(lp * mask).sum() / denom

    def back(g):
        d = pr.copy()
        d[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
        _acc(logits, d * (mask[..., None] / denom) * g)
    return t.new(np.asarray(loss, dtype=logits.v.dtype), back)


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class Nova2:
    def __init__(self, cfg=None, seed=1, dtype=np.float32):
        self.cfg = cfg or Config()
        self.dtype = dtype
        self.spec = layout(self.cfg)
        self.offsets, self.groups, n = {}, {}, 0
        for name, shape, g, init, decay in self.spec:
            self.offsets[name] = (n, shape); self.groups[name] = g; n += int(np.prod(shape))
        self.n = n
        self.P = np.zeros(n, dtype); self.M = np.zeros(n, dtype); self.V = np.zeros(n, dtype)
        self.step = 0
        self.decay_mask = np.zeros(n, dtype); self.group_mask = {}
        rng = np.random.default_rng(seed)
        for g in ("left", "right", "inter"):
            self.group_mask[g] = np.zeros(n, dtype)
        for name, shape, g, init, decay in self.spec:
            off, _ = self.offsets[name]; sz = int(np.prod(shape)); sl = slice(off, off + sz)
            kind, val = init
            if kind == "normal":
                self.P[sl] = rng.normal(0.0, val, sz)
            elif kind == "ones":
                self.P[sl] = 1.0
            elif kind == "const":
                self.P[sl] = val
            if decay:
                self.decay_mask[sl] = 1.0
            self.group_mask[g][sl] = 1.0

    # ---- forward ----------------------------------------------------------
    def _block(self, t, i, h):
        c = self.cfg; p = t.p; B, T, D = h.v.shape
        x = rmsnorm(t, h, p(f"b{i}.ln1"))
        q = reshape(t, linear(t, x, p(f"b{i}.wq")), (B, T, c.heads, c.hd))
        k = reshape(t, linear(t, x, p(f"b{i}.wk")), (B, T, c.heads, c.hd))
        v = reshape(t, linear(t, x, p(f"b{i}.wv")), (B, T, c.heads, c.hd))
        a = attention(t, rope(t, q, c.rope_base), rope(t, k, c.rope_base), v)
        h = add(t, h, linear(t, reshape(t, a, (B, T, D)), p(f"b{i}.wo")))
        x = rmsnorm(t, h, p(f"b{i}.ln2"))
        ff = mul(t, silu(t, linear(t, x, p(f"b{i}.w_gate"))), linear(t, x, p(f"b{i}.w_up")))
        return add(t, h, linear(t, ff, p(f"b{i}.w_down")))

    def forward(self, t, ids, feats, no_afferent=False, no_efferent=False, no_right=False):
        """ids [B,T] int, feats [B,T,feat]. Returns logits node [B,T,vocab].
        Ablations (for diagnostics / variant training):
          no_afferent: cut right->left injection  (left hemisphere runs alone)
          no_efferent: right never hears the left (feed-forward only, no reciprocity)
          no_right   : right hemisphere outputs zero everywhere"""
        c = self.cfg; p = t.p
        E = p("embed")
        h = embed(t, ids, E)
        for i in range(c.split):
            h = self._block(t, i, h)
        # ---- right hemisphere, bottom layer: sees only raw characters ----
        z1 = silu(t, linear(t, Node(feats.astype(self.dtype)), p("r.w1"), p("r.b1")))
        if no_right:
            z1 = Node(np.zeros_like(z1.v))
        # ---- exchange 1 ----
        if not no_afferent:
            hn = rmsnorm(t, h, p("c.n1"))
            gate = sigmoid(t, linear(t, hn, p("c.g1w"), p("c.g1b")))
            h = add(t, h, mul(t, gate, linear(t, z1, p("c.a1"))))
        def efferent(h):
            pre = linear(t, z1, p("r.w2"), p("r.b2"))
            if not no_efferent:
                pre = add(t, pre, linear(t, rmsnorm(t, h, p("c.n1")), p("r.wl")))
            z2 = silu(t, pre)
            return Node(np.zeros_like(z2.v)) if no_right else z2
        z2 = efferent(h)
        for k in range(c.loops):
            if not no_afferent:
                hn2 = rmsnorm(t, h, p("c.n2"))
                gate = sigmoid(t, linear(t, hn2, p("c.g2w"), p("c.g2b")))
                h = add(t, h, mul(t, gate, linear(t, z2, p("c.a2"))))
            for i in range(c.split, c.blocks):
                h = self._block(t, i, h)
            if k < c.loops - 1:
                z2 = efferent(h)
        return linear(t, rmsnorm(t, h, p("ln_f")), E, p("out_bias"))   # tied output head

    # ---- loss / gradients ----------------------------------------------------
    def loss_and_grads(self, ids, feats, targets, mask, **abl):
        t = Tape(self)
        logits = self.forward(t, ids, feats, **abl)
        loss = xent(t, logits, targets, mask.astype(self.dtype))
        t.backward(loss)
        return float(loss.v), t.flat_grad()

    def loss_only(self, ids, feats, targets, mask, **abl):
        t = Tape(self)
        return float(xent(t, self.forward(t, ids, feats, **abl), targets, mask.astype(self.dtype)).v)

    def logits(self, ids, feats, **abl):
        return self.forward(Tape(self), ids, feats, **abl).v

    # ---- optimizer (AdamW + global-norm clip + selective "training focus") ---------
    def adam_step(self, G, lr, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.01, clip=1.0, focus="joint"):
        gn = float(np.sqrt((G.astype(np.float64) ** 2).sum()))
        if clip and gn > clip:
            G = G * (clip / gn)
        active = np.ones(self.n, self.dtype) if focus == "joint" else self.group_mask[focus]
        G = G * active
        self.step += 1
        self.M = np.where(active > 0, beta1 * self.M + (1 - beta1) * G, self.M)
        self.V = np.where(active > 0, beta2 * self.V + (1 - beta2) * G * G, self.V)
        mh = self.M / (1 - beta1 ** self.step); vh = self.V / (1 - beta2 ** self.step)
        self.P -= (lr * active * (mh / (np.sqrt(vh) + eps) + wd * self.decay_mask * self.P)).astype(self.dtype)
        return gn

    # ---- inference -----------------------------------------------------------
    def next_token_probs(self, tok, ids, **abl):
        ids = list(ids)
        feats = features_for_ids(tok, ids, self.cfg.window_chars)[-self.cfg.max_tokens:]
        win = np.asarray(ids[-self.cfg.max_tokens:], dtype=np.int64)
        z = self.logits(win[None], feats[None], **abl)[0, -1].astype(np.float64)
        z[tok.n_valid():] = -np.inf
        z -= z.max(); p = np.exp(z); return p / p.sum()

    def generate(self, tok, prompt, max_tokens=24, stop="\n", **abl):
        """Greedy generation. Generated token ids are kept as emitted (never re-tokenized)."""
        ids = tok.encode(prompt); out = []
        for _ in range(max_tokens):
            nxt = int(np.argmax(self.next_token_probs(tok, ids + out, **abl)))
            out.append(nxt)
            if tok.token_text(nxt) == stop:
                break
        return tok.decode(out).rstrip(stop)

    def score_continuation(self, tok, prompt, answer, **abl):
        """Sum log-prob of `answer` tokens after `prompt` (for preference / ranking losses)."""
        ids = tok.encode(prompt); ans = tok.encode(answer); total = 0.0
        for a in ans:
            total += math.log(max(self.next_token_probs(tok, ids, **abl)[a], 1e-30)); ids = ids + [a]
        return total

    # ---- checkpoint ------------------------------------------------------------
    def save(self, path, tok):
        cj = json.dumps(asdict(self.cfg)).encode()
        with open(path, "wb") as f:
            f.write(b"NNOVA2\x00\x00"); f.write(struct.pack("<I", len(cj))); f.write(cj)
            f.write(struct.pack("<Q", self.step))
            for a in (self.P, self.M, self.V):
                f.write(a.astype("<f4").tobytes())
            f.write(b"WORDUNIT"); f.write(struct.pack("<I", len(tok.words)))
            for w in tok.words:
                f.write(struct.pack("<I", len(w))); f.write(w.encode())

    @staticmethod
    def load(path):
        b = open(path, "rb").read(); assert b[:8] == b"NNOVA2\x00\x00"; o = 8
        (n,) = struct.unpack_from("<I", b, o); o += 4
        cfg = Config(**json.loads(b[o:o + n])); o += n
        m = Nova2(cfg); (m.step,) = struct.unpack_from("<Q", b, o); o += 8
        for name in ("P", "M", "V"):
            setattr(m, name, np.frombuffer(b, "<f4", m.n, o).astype(np.float32).copy()); o += 4 * m.n
        assert b[o:o + 8] == b"WORDUNIT"; o += 8
        (k,) = struct.unpack_from("<I", b, o); o += 4; words = []
        for _ in range(k):
            (s,) = struct.unpack_from("<I", b, o); o += 4; words.append(b[o:o + s].decode()); o += s
        return m, Tokenizer(words)


def lr_schedule(step, peak, warmup=50, total=2000, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    pr = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * pr)))


if __name__ == "__main__":
    tot, g = param_report(Config())
    print("default config params:", tot, g, "budget", BUDGET, "headroom", BUDGET - tot)
