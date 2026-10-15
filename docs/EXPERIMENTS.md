# What was verified, and what wasn't

## Verified (run `python3 reference/selftest.py`, about 2 minutes on one CPU core)

| Check | Result |
|---|---|
| Parameter count <= 1,200,144 | 1,182,144 |
| Gradients vs finite differences (float64), **every parameter**, full model | 9,046 / 9,046 within tolerance, worst abs error 2.2e-9 |
| Same check: loops=1, loops=3, no_efferent, no_afferent (2,500 sampled params each) | all within tolerance |
| Causality: editing tokens/features at position >= t never changes outputs before t | pass |
| Every piece affects the output (afferent, efferent, right hemisphere ablations) | pass |
| Tokenizer: lossless round trip, stable ids as vocabulary grows (ports of the Rust tests) | pass |
| Right-hemisphere features match `encode_ascii_context` | pass |
| Checkpoint save/load round trip, training focus updates only the chosen piece | pass |

The gradient check covers the whole model on a small config with the same code paths
(3 blocks, 2 heads, loops 2). `rust/tests/golden/` holds loss, logits and gradients from this
verified reference for the Rust port to match.

## Small learning probe (`reference/experiment_relational.py`)

Prompts mirror `neural-nova/src/bin/cooperative_eval.rs`:
`U: Ada has 2 apples. Ben has 5 apples. Who has more apples?\nN: Ben`.
19 names and 60 number pairs for training; held out: 12 number pairs, 6 names.
Fresh weights, batch 8, peak lr 3e-3, one CPU core (numpy), about 7 minutes.

| Step | Seen: correct | New numbers: correct | New names: correct | Answers with a name from the prompt |
|---|---|---|---|---|
| 1,400 | 0.32 | 0.33 | 0.00 | 0.60 / 0.62 / 0.00 |
| 2,800 | 0.50 | 0.48 | 0.00 | 1.00 / 1.00 / 0.00 |

**Reading it honestly:**
- In 2,800 small steps it learned the answer *format*: it always replies with one of the two
  names from the prompt, including for number pairs it never saw.
- It had **not** yet learned the comparison: 0.50 is exactly chance between the two names.
- New names: 0.00. Those names are word tokens that never appeared anywhere in training, so
  their embeddings are still random and the model cannot copy them. This is a property of
  word tokens, not the hemispheres; a character-level copy would be needed for truly new names.
- The planned ablation comparison (full vs no_efferent vs no_afferent vs loops=1) was **not
  run**, because comparing variants that are all at chance would not show anything. The
  script supports it: `./reference/run_experiments.sh` (set `--steps` much higher, ideally in
  the Rust port where it will be far faster than numpy).
- No comparison against Nova v1 on equal data was run. That is the most important next
  experiment once the port exists.

## Not done here
- **No Rust code was compiled.** This workspace has no Rust toolchain and could not download
  one, so the reference is numpy and the port is a plan (`PORTING.md`) plus golden tests.
- No on-device timing. The compute numbers in `ARCHITECTURE.md` are multiply-add counts.
