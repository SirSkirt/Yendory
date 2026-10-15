# Engine status

Phase 1 (one shared Rust Engine with compatibility tests) is done; Phase 2 (learning in the Engine) is under way. This page says what
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
| Step 19,000 (owner's lessons) — same checks | pass **locally only** (not published; see tests/compat/README.md) |

## Not verified

- Nothing here has run on Android or in a Windows app. CI builds and tests the library on
  Ubuntu and Windows only.
- Training dynamics over many steps (a long Rust run vs a long reference run). One gradient and
  one optimizer step on real weights are checked.
- A full teaching session on the owner's lessons has not been run end to end (only the
  synthetic course in tests). Whether Foundation candidates beat Joint-only training on real
  lessons is an experiment still to run, not a verified claim.
- Pathway measurements for the visualizer and ZIP course records — not built yet (Phase 2.4/2.5).

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

## Decisions still open

| # | Decision | Status |
|---|---|---|
| D1 | License | owner will decide later |
| D2 | Parameter cap (`MAX_PARAMETERS = 1,200,144`, currently not enforced anywhere) as a configurable safety limit | open |
| D3 | Package extension/container wrapping `.nnova2` | open |
| D4 | Runtime exports may drop optimizer state | open |
| D5 | One repository | **decided: monorepo** |
| D6 | Nova consumes the Engine via git tag | open (recommended) |
| D7 | Old v1 brain / `.nhyb` | **decided: dropped** |
| D8 | Studio design language (mockups first) | open |
| D9 | Who reviews Engine changes from the Nova side | open |
| D10 | Context policy past 256 tokens: exact re-window (current, matches reference) vs. a faster chunked re-window (re-read once every ~64–128 tokens; outputs differ from the reference past 256 tokens) vs. the app trimming history itself | open |
