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
| word count | u32 | at most `vocab − 128` (256 for Nova-2's vocab 384) |
| words | (u32 length, ASCII bytes) × count | learned word units; id = 128 + index; 3–24 chars of `[A-Za-z0-9']` |

No bytes may follow the word list. `n` is determined by the config (1,182,144 for the default).

**Byte-exact round trip.** The numpy reference writes the config with Python's `json.dumps`
(`{"vocab": 384, ...}`), serde writes `{"vocab":384,...}`. The Engine keeps the config bytes it
read and writes them back unchanged when the config has not changed, so load → save reproduces
a file exactly (checked in `tests/compat.rs`).

**Validation on load.** Truncated files, trailing bytes, non-finite parameters, invalid word
units, more words than the vocabulary holds, and impossible configurations (e.g. `d` not
divisible by `heads`, a vocabulary above 2^28 ids, more than 1,000,000,000 parameters) return
errors instead of panicking or allocating.

**Vocabulary size.** `vocab` in the config is the number of token ids: 128 ASCII characters plus
one slot per learned word unit. Nova-2 uses 384 (256 word slots); Yendory models can use any
size, and the file layout is the same (the `embed` matrix has `vocab` rows and `out_bias` has
`vocab` entries). Token ids are 32-bit in the Engine. Files stay readable by the numpy
reference at any vocabulary size (checked at 100,128 ids in CI).

## Nova Neuro and game skills

A Nova Neuro checkpoint has the magic `NNEURO1\0` and, in place of the config, a JSON header
`{"architecture": "nova-neuro", "format": 1, "config": …, "design": …}`; the rest of the layout
(step, weights, Adam state, learned words) is the same.

**Format 2** (Engine 0.9.4): the model carries **game skills** (`skill.rs`): networks that play a
real game from its pictures, trained in Yendory Tools. The header has `"format": 2` and a
`"skills"` list (each skill's `SkillSpec`: the game, its name, size, seed, network architecture,
the controls in order, lineage, steps, the step its accepted weights were kept at, and the
trainer's notes). After the learned words:

| Field | Type | Notes |
|---|---|---|
| magic | 8 bytes | `SKILLSET` |
| per skill, in the header's order: step | u64 | the skill's optimizer step |
| accepted weights | f32 × k | what plays (`k` from the skill's architecture) |
| latest weights | f32 × k | to continue training |
| Adam m, v | f32 × k each | |

No bytes may follow. A Neuro model without skills is written as format 1, byte for byte as
before; a Bicameral model never carries skills. Engines before 0.9.4 refuse format 2 ("Unsupported
Nova Neuro checkpoint (nova-neuro format 2)") instead of opening the model and dropping its skills
when they save it. On load: at most 64 skills, one per game, each architecture checked and at
most 64M weights, every value finite, the step and lengths exact.

The model-package format that will wrap `.nnova2` (manifest, checksums, runtime vs archive
exports) is a pending decision (D3/D4 in docs/STATUS.md).
