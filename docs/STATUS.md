# Engine status

Phase 1 (one shared Rust Engine with compatibility tests) is done; Phase 2 (learning in the Engine) is done for the scope in docs/LEARNING.md. This page says what
has been checked, how, and what has not.

## Verified (by `cargo test --release` and `reference/selftest.py`)

| Check | Result |
|---|---|
| numpy reference self-test: finite-difference gradients on every parameter, causality, ablations, round trip, focus masks | all pass |
| Golden vectors (3 small configs: full, `no_efferent`, `loops=1`): loss, 12 logits, 62 gradients each | pass; worst gradient error ≤ 1e-6 of the largest gradient |
| **Real checkpoint, step 15,000**: last-position logits for 9 prompts × 4 ablation modes, uncached and KV-cached | worst relative error 3e-5 (tolerance 1e-4) |
| Same: greedy generations (up to 24 tokens), uncached and KV-cached | token-identical to the reference |
| Same: prompts longer than the 256-token window (247 and 291 tokens) | identical, cached and uncached |
| Same: loss and 131 gradients (sampled + the 32 largest) on a training sequence | gradient error ≤ 4e-7 of the gradient norm |
| Same: one AdamW step from the stored optimizer state, focus Joint and Right | parameter deltas within 1.1e-4 of the learning rate; with Right focus, the count of untouched parameters matches the reference mask (845,824) up to float32 rounding |
| `.nnova2` load → save | byte-identical to the original file |
| Malformed checkpoints (truncated, trailing bytes, bad config, absurd sizes) | return errors, no panic |
| Lesson pipeline on a synthetic course: parsing, held-out split, vocabulary growth, encoded rows, one batch gradient, held-out greedy replies and preferences | identical to `lessons.py` / `train_lessons.py` |
| **Owner's 27 lessons**: parsing and split (6,015 units, 666 held out); vocabulary growth from step 15,000 | identical; reproduces the step-19,000 word list exactly (local only) |
| **Step 19,000 held-out re-score**, all 666 units | dialogue 155/324, understanding 30/54, reasoning 166/264, conversation turns 65/72, prefers supported 77.3%, overall 58.3%: equal to the published report (local only; 24 s in Rust) |
| Teaching session: four whole-model candidates per epoch from the same accepted model, deterministic for a seed, rollback leaves the model bit-identical, stop discards the epoch in progress | pass |
| Courses: per-file SHA-256, steps, outcomes; duplicates not re-taught; a failing file does not end the course | pass |
| Course Builder: file names give order, stage and role; exam questions, evaluation questions and training never overlap; practice leaks excluded; keyword and sentence grading; fingerprints and restart rules (`tests/program.rs`, `grading.rs`) | pass |
| Reading a mix: batches follow the shares, per-corpus checks, bicameral stages train only their parts; generated practice is deterministic and its answers check out (`tests/corpus.rs`, `generate.rs`) | pass |
| Course runs: pre-test, 70/30 grade, three remediation attempts with different strategies and halved learning rates, best checkpoint returned, stop and resume continue where they were | pass |
| Measurements: tracing leaves outputs bit-identical; attention rows sum to 1; `no_right` equals `no_afferent` | pass |
| **Size presets** 4M, 5M, 8M, 15M (Nova-2's 64 / 28.5 / 7.7% division): a briefly trained model of each, last-position logits, loss and 3,000 sampled gradients recomputed by the numpy reference (`examples/size_compat.rs` + `reference/check_sizes.py`, in CI) | logits within 2e-6, loss identical to 6 decimals, gradients within 3e-7 of the largest |
| **Selectable context**: changing `max_tokens` keeps every weight and every short-prompt logit; a 1,024-token-context model trains on a 900-token row and its KV cache matches the full pass; recomputed attention (no stored weights) passes every compatibility test; a 1,024-token context on an 800-token input recomputed by the numpy reference in CI | logits within 6e-7 |
| **Selectable vocabulary**: 32-bit token ids; hash-indexed tokenizer (equal to the original scanning encoder on 6,000 random strings); output layer + cross-entropy computed a few rows at a time (equal to the unfused path); a Nova-2-size model with a 100,000-word vocabulary and ids up to 70,130 recomputed by the numpy reference in CI; resizing keeps every existing logit bit-identical; vocabulary limits add slots only for learned words | logits within 7e-7, loss identical to 6 decimals |
| Matrix kernels (`model/linalg.rs`: packed panels, 4×16 register tiles, 8-lane dot products; safe Rust, no dependencies): same products as plain loops, only the order of additions differs | all compatibility tests above pass unchanged; one core reaches ~14 GFLOP/s (SSE2) |
| **Nexus** (`tests/nexus.rs`): lessons with `Environment:` / `Action:` / `Result:` lines; marker characters in files and typed text removed; a small model taught a test-only environment looks before answering, reads the real result, changes the environment, and is refused a change without approval | pass; an untaught model takes no action |
| **Flat epochs** (`tests/session.rs`): training continues from the best safe candidate through epochs that improve too little to be accepted; the accepted model then includes those epochs' steps; a candidate that loses an answer restores the best checkpoint | pass |
| **GPU kernels rewritten for speed** (tiled attention with a single-pass softmax, vec4 register matrix products, parallel column sums, per-token embedding gradient, one workgroup per row for the loss): on lavapipe at 8M / 1,024 context, rows of 600, 503 and 406 tokens (`examples/gpu_check.rs`) | loss equal to 7 digits; gradients within 9e-7 of the largest; weights after two steps within 5e-7; reading and Foundation lessons on the GPU accept the same checkpoints and candidates as the CPU; a step 2.7x faster than before on the same software GPU |
| **Hybrid CPU + GPU training** (`tests/gpu.rs`): four steps with whole-brain and Stage A focus | the same losses as CPU training to 7 digits; weights within 2e-7; Stage A leaves the transformer untouched |
| Step 19,000 (owner's lessons) — same checks | pass **locally only** (not published; see tests/compat/README.md) |
| **GPU training** (`--features gpu`, `tests/gpu.rs`, run in CI on a software GPU, lavapipe): loss, every gradient and two AdamW steps (Joint and Dense-only focus) against the CPU; reading and a Foundation lesson on the GPU, with the CPU checking at the same time | loss equal to 6 digits; gradients within 1.7e-6 (largest 2.6); weights within 5e-7; the same checkpoints, candidates and early stop as on the CPU |

## Yendory Studio (apps/studio)

Checked in this development container on Linux (real app under a virtual display): create and
open models, import a ZIP course through the native file dialog, teach with Foundation candidates
and live progress, per-file saving and records, conversation, exam grading with taught-material
warnings, the brain visualizer with live measurements, the Library record; a `.deb` package
builds. Course Builder and Planner: a ZIP of Dense / Transformer / Co-op lessons with a practice
file was read into a course (order, stages, practice and exam pool as expected), a plan taught it,
graded it, started remediation, was stopped mid-remediation and resumed at the next lesson.
Reading sources: GSM8K downloaded and converted in the app; generated practice, a GSM8K and a
prose corpus read together in bicameral mode (held-out loss 5.31 → 3.02, each corpus checked
and recorded with its stage and licence). The other sources are downloaded and converted by
CI (this container cannot reach their hosts). Not checked: the Windows installer and Android APK on real devices (CI builds them).

## Not verified

- GPU and hybrid training have not run on a real graphics card here (only the software GPU in
  CI and in this container). Their speed on a real card, and whether DirectX 12 (DXC) or
  Vulkan is faster there, is for the Studio's training benchmark to measure.
- Nothing here has run on Android or in a Windows app. CI builds and tests the library on
  Ubuntu and Windows only.
- Training dynamics over many steps (a long Rust run vs a long reference run). One gradient and
  one optimizer step on real weights are checked.
- A full teaching session on the owner's lessons has not been run end to end (only the
  synthetic course in tests). Whether Foundation candidates beat Joint-only training on real
  lessons is an experiment still to run, not a verified claim.
- ZIP reading itself is Studio's job (Phase 3); the Engine receives the files.

## Findings while porting

1. **KV cache past 256 tokens (fixed).** The Neural Nova cache kept appending with growing
   RoPE positions after the window was full, while the reference re-reads the last 256 tokens
   with positions starting from 0 (the only positions the model was trained on). Measured logit
   drift was up to 1.1. The Engine now re-windows exactly like the reference.
   *Cost:* once a conversation passes 256 tokens, each new token re-reads the whole window
   (~360 ms per token in the development container used for this work, vs ~1.5 ms inside the
   window; a phone will be slower).
   See decision D10.
2. **Header bytes differ between implementations (fixed).** Python and serde format the config
   JSON differently, so files were interchangeable but not byte-identical. See docs/FORMAT.md.
3. **`no_right` and `no_afferent` are the same ablation for outputs.** Silencing the right
   hemisphere zeroes everything it injects into the left, which is exactly what cutting the
   afferent path does. They always produce identical logits; diagnostics should not present
   them as two independent measurements.
4. **Two lesson parsers existed.** Neural Nova's Bicameral lab trains through the v1
   `trainer::prepare` (character-level examples, a different held-out split), while the
   step-19,000 checkpoint was trained with `lessons.py`. The Engine implements `lessons.py`
   only; v1 code is not carried over. Parity with Nova's Foundation Training is listed in
   docs/LEARNING.md.

## Under investigation

### Vocabulary growth on a model that has already learned text

When a lesson or reading adds word units, words the model had learned as characters become new
tokens whose embedding and output rows start untrained. Measured with `examples/vocab_probe.rs`
(1.2M model; owner's lesson 02A taught for 8 epochs with characters only; then the 35 words that
lesson 02B adds, all of which occur in 02A, were grown; no training in between):

| | 02A held-out, per token | 02A held-out, total (nats) |
|---|---|---|
| before growth | 1.325 (1,609 tokens) | 2,133 |
| right after growth | 2.519 (1,081 tokens) | 2,723 (+28%) |
| +1 epoch on 02A, with growth / without | 2.015 / 1.368 | 2,178 / 2,201 |
| +8 epochs on 02A, with growth / without | 1.436 / 1.115 | 1,552 / 1,795 |

Findings so far:

- Growth costs a model about a quarter of what it knew about text containing the new words, at
  once. Text it trains on again recovers within about one epoch and then does better than
  without growth (fewer, more meaningful tokens). Text it does not see again (earlier lessons
  without rehearsal) keeps the loss until those words are learned elsewhere.
- Losses per token are not comparable across a vocabulary change (the same text has fewer
  tokens afterwards). Within one session they are (the baseline is measured after growth), but
  comparisons across sessions, e.g. a lesson's held-out loss before and after a later lesson
  grew the vocabulary, mix two scales. A per-character measure would be comparable.

Possible next steps (no change made yet; the architecture stays as it is): start a new word
slot from what the model already knows about the word's characters instead of at random (an
initialisation change in the trainer); report held-out loss per character alongside per token;
rehearse earlier text after growth.

### Teaching safeguards: strict, net, relative, both

`examples/guard_compare.rs`: the same start checkpoint (a fresh 1.2M model taught lesson 01),
the same two courses through the Course Builder pipeline (course 1: lessons 02A + 02B, course 2:
03A + 03B), same settings and seeds; only the acceptance safeguards differ. Every lesson's
held-out evaluation questions (never trained) are asked at the start, after each course and at
the end; course grades are the first grade, before any remediation. One seed, one CPU thread.

- **strict** (current default): a candidate is rejected if it loses any held-out answer, or if
  earlier lessons' loss rises more than 0.02.
- **net**: held-out answers may be lost if more are gained (`protect_answers_net`).
- **relative**: earlier lessons' loss may rise up to 3% instead of 0.02, measured on 48 units
  of earlier held-out text instead of 16 (`retention_relative`, `retention_units`).
- **both**: net and relative together.

| | strict | net | relative | both |
|---|---|---|---|---|
| Held-out answers right at the end (of 108; 3 at start) | 31 | **38** | 25 | **38** |
| Gained / lost (answers right at the start that were lost) | 29 / 1 | 35 / **0** | 23 / 1 | 36 / 1 |
| Course 1 lessons: right after course 1 → after course 2 | 24 → 22 | 29 → 23 | 21 → 18 | 29 → 24 |
| Course 2 lessons right at the end (of 62) | 9 | **15** | 7 | 14 |
| Held-out loss at the end, 02A / 02B / 03A / 03B (start 3.05 / 2.64 / 2.45 / 2.28) | 1.17 / 0.94 / 0.89 / 0.87 | **1.16 / 0.84 / 0.79 / 0.72** | 1.55 / 1.20 / 0.90 / 0.82 | 1.23 / 0.86 / 0.86 / 0.78 |
| Course grade, course 1 / course 2 (pass mark 0.70) | 0.55 / 0.15 | **0.65 / 0.22** | 0.51 / 0.11 | 0.64 / 0.21 |
| Lesson passes, course passes | 0, 0 | 0, 0 | 0, 0 | 0, 0 |

Findings:

- **net** ended ahead of strict on every lesson (answers and held-out loss), lost none of the
  answers the model had at the start, and had the best course grades. It also forgets more
  of course 1 during course 2 (29 → 23 against 24 → 22) but still ends above strict on it.
- **relative** did worse than strict on every measure. Why is not understood yet, so it stays
  off. **both** is about net, with one answer lost, so the relative part adds nothing here.
- Nothing passed under any safeguard: no lesson and no course reached its pass mark on the
  first grade. Progress (more answers, lower loss) and passing are separate here, as intended.
- One seed and small question sets (21-33 per lesson): differences of one or two answers are
  noise. The gap between net and strict (7 answers, every lesson in the same direction) is the
  only clear one. Times are not comparable (the runs shared the CPU differently).

No default was changed: strict stays the default until net is confirmed with another seed.

## Decisions still open

| # | Decision | Status |
|---|---|---|
| D1 | License | owner will decide later |
| D2 | Parameter cap | **decided**: size presets up to 15M keep Nova-2's parameter division; `MAX_PARAMETERS = 16,000,000` is enforced when a checkpoint is loaded |
| D3 | Package extension/container wrapping `.nnova2` | open |
| D4 | Runtime exports may drop optimizer state | open |
| D5 | One repository | **decided: monorepo** |
| D6 | Nova consumes the Engine via git tag | open (recommended) |
| D7 | Old v1 brain / `.nhyb` | **decided: dropped** |
| D8 | Studio design language (mockups first) | open |
| D9 | Who reviews Engine changes from the Nova side | open |
| D10 | Context policy past the context length | **decided**: the context is selectable per model (256 to 65,536 tokens); Studio keeps conversations within it by dropping the oldest turns; the Engine's exact re-window (matches the reference) remains for anything longer |
