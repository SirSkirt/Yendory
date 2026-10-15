# numpy reference (the specification)

`nova2_ref.py` is the verified specification of the Bicameral network: model, tape autograd,
AdamW with training focus, tokenizer, features and `.nnova2` I/O, in plain numpy. It is not
shipped in any app. This copy is the one from the step-19,000 training package (it includes the
overflow-safe sigmoid); the maths is otherwise identical to the architecture package version.

| Script | Purpose |
|---|---|
| `selftest.py` | finite-difference gradient check on every parameter (float64), causality, ablations, round trips. Run in CI. |
| `make_golden.py` | writes `tests/compat/golden/*.txt` |
| `make_compat.py <ckpt> <out.json>` | expected outputs of a real checkpoint for `crates/yendory-engine/tests/compat.rs` |

Requirements: Python 3.10+ and numpy. Run from this directory, e.g. `python3 selftest.py`.
