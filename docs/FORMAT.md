# `.nnova2` checkpoint format

Little-endian. Written identically by the Rust Engine and `reference/nova2_ref.py`.

| Field | Type | Notes |
|---|---|---|
| magic | 8 bytes | `NNOVA2\0\0` |
| config length | u32 | at most 8192 |
| config | JSON | `Config` (vocab, d, heads, ff, blocks, split, loops, window_chars, r1, r2, max_tokens, rope_base, gate_bias) |
| step | u64 | optimizer step count |
| weights | f32 × n | flat parameter vector in layout order (see `spec()` / `layout()`) |
| Adam m | f32 × n | first moment |
| Adam v | f32 × n | second moment |
| magic | 8 bytes | `WORDUNIT` |
| word count | u32 | at most 256 |
| words | (u32 length, ASCII bytes) × count | learned word units; id = 128 + index; 3–24 chars of `[A-Za-z0-9']` |

No bytes may follow the word list. `n` is determined by the config (1,182,144 for the default).

**Byte-exact round trip.** The numpy reference writes the config with Python's `json.dumps`
(`{"vocab": 384, ...}`), serde writes `{"vocab":384,...}`. The Engine keeps the config bytes it
read and writes them back unchanged when the config has not changed, so load → save reproduces
a file exactly (checked in `tests/compat.rs`).

**Validation on load.** Truncated files, trailing bytes, non-finite parameters, invalid word
units and impossible configurations (e.g. `d` not divisible by `heads`, absurd sizes) return
errors instead of panicking or allocating.

The model-package format that will wrap `.nnova2` (manifest, checksums, runtime vs archive
exports) is a pending decision (D3/D4 in docs/STATUS.md).
