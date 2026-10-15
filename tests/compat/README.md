# Compatibility fixtures

| Path | What | Published |
|---|---|---|
| `golden/*.txt` | small-config loss, logits and 62 gradients from the float64 reference (`reference/make_golden.py`) | yes |
| `fixtures/nova2-step15000.nnova2` | Nova-2 trained 15,000 steps on the synthetic curriculum (renamed from `nova2.ckpt`) | yes (owner approved) |
| `fixtures/step15000.expected.json` | reference outputs for it (`reference/make_compat.py`) | yes |
| `fixtures/lessons/*.txt` | a small synthetic course written for the tests (all four block types, edge cases, one file with Windows line endings) | yes |
| `fixtures/learn.expected.json` | lessons.py / train_lessons.py outputs for that course (`reference/make_learn.py --eval`) | yes |
| `private/lessons/*.txt` | the owner's 27 Unified lessons (All_Lessons.zip) | **no** — gitignored |
| `private/learn-all-lessons.expected.json` | lessons.py outputs for them (`reference/make_learn.py`) | **no** — gitignored |
| `private/step19000.eval_report.json` | the published held-out results of step 19,000 (from its training package) | **no** — gitignored |
| `private/nova2-lessons-step19000.nnova2` | step 19,000, trained on the owner's lessons | **no** — gitignored |
| `private/step19000.expected.json` | reference outputs for it | **no** — gitignored |

The step-19,000 checkpoint was trained on the owner's own lessons and is not published. To run
its check locally, put the checkpoint in `private/` and generate its expected values:

```
cd reference
python3 make_compat.py ../tests/compat/private/nova2-lessons-step19000.nnova2 ../tests/compat/private/step19000.expected.json
cd .. && YENDORY_REQUIRE_PRIVATE=1 cargo test --release --test compat
```

For the lesson pipeline, unzip All_Lessons.zip into `private/lessons/`, copy the package's
`eval_report.json` to `private/step19000.eval_report.json`, and run
`python3 make_learn.py ../tests/compat/private/lessons ../tests/compat/fixtures/nova2-step15000.nnova2 ../tests/compat/private/learn-all-lessons.expected.json`
from `reference/`. The Rust test then checks parsing, the held-out split, vocabulary growth
(which must reproduce the step-19,000 word list) and re-scores all 666 held-out units, which must
equal the published report.

Without the files, those tests report that it did not run and pass; CI checks the public fixtures only.

The compat prompts are written in `make_compat.py`, not taken from any lesson file.
