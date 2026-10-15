# Compatibility fixtures

| Path | What | Published |
|---|---|---|
| `golden/*.txt` | small-config loss, logits and 62 gradients from the float64 reference (`reference/make_golden.py`) | yes |
| `fixtures/nova2-step15000.nnova2` | Nova-2 trained 15,000 steps on the synthetic curriculum (renamed from `nova2.ckpt`) | yes (owner approved) |
| `fixtures/step15000.expected.json` | reference outputs for it (`reference/make_compat.py`) | yes |
| `private/nova2-lessons-step19000.nnova2` | step 19,000, trained on the owner's lessons | **no** — gitignored |
| `private/step19000.expected.json` | reference outputs for it | **no** — gitignored |

The step-19,000 checkpoint was trained on the owner's own lessons and is not published. To run
its check locally, put the checkpoint in `private/` and generate its expected values:

```
cd reference
python3 make_compat.py ../tests/compat/private/nova2-lessons-step19000.nnova2 ../tests/compat/private/step19000.expected.json
cd .. && YENDORY_REQUIRE_PRIVATE=1 cargo test --release --test compat
```

Without the files, that test reports that it did not run and passes; CI checks step 15,000 only.

The compat prompts are written in `make_compat.py`, not taken from any lesson file.
