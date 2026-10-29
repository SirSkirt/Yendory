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
| Study Mode memory: a sentence the model predicts (loss below 0.5) is not kept, at most half a passage, studying again merges; held-out questions never stored; lessons train with memory copies while grading stays without memory, `with_memory` measured separately (`tests/memory.rs`) | pass |
| A fresh 1.2M model learns to use memory lines: 88% of new random facts with them, 9% without (chance 12%), `examples/memory_probe.rs` | measured |
| Practice, reflection and Review Training on a small model: targets keep their origin (Nova confirmed / teacher correction / lesson answer), nothing unapproved is trained, held-out questions judge every pass, before / after reported (`tests/study.rs`, `learn::review` and `learn::reflect` tests) | pass |
| Lesson Designer: analysis, sources, checks (arithmetic, leaks, duplicates), Markdown to plain text, export read back by the parser (`learn::designer` tests) | pass |
| Rewards, Stage 1 (`learn::reward` tests, `tests/reward.rs`): the grader admits uncertainty and judges meaning (negation, required words, numbers), letters only in spelling lessons; progress milestones credited once (the "Hallway" sequence: +0.10, 0, +0.18, 0, +0.72, 0), reworded copies share a cycle, budget capped; retention 0.6 / 0.2 / 0.1, nothing without training in between, lost answers recorded; never-trained questions earn generalization, not progress; memory answers on their own smaller track; teacher grades and notes recorded; protected exam questions refused | pass |
| Recording rewards leaves training unchanged: the same session with and without the ledger gives bit-identical weights; a course run hands back its ledger and no final-exam question reaches it | pass |
| **Nova Neuro (Neuro-Core)**: finite-difference gradients through every new operation and module (working memory, associative memory, controller, bus); every module gets gradient; the generation cache equals the full pass under 8 ablations and a full-window rebuild; changing a later token never changes an earlier position (no future leakage); `.nneuro` round trip and truncation checks; presets within 0.1% of the Bicameral sizes; reading, a Foundation lesson, measurements, the thinking tree and save/load on a Neuro model (`model::neuro` tests, `tests/neuro.rs`) | pass |
| **Nova Neuro on the GPU** (minGRU scan, softmax over patterns, controller gates; forward and backward kernels) on lavapipe, `tests/gpu_neuro.rs`: rows of different lengths, every ablation (full, no working memory, no associative memory, no controller, one round, no write-back, no bus read), every focus | loss equal to 6-7 digits; gradients within 1.2e-7 of the largest; weights after two AdamW steps within 1.2e-7 (frozen groups unchanged); a reading run accepts the same checkpoints as the CPU (best held-out loss 1.44487 vs 1.44488) |
| **Network growth** (`tests/growth.rs`, Studio `growth` tests): every offered step on trained Bicameral and Neuro models, twice in a row, and on the real step-15,000 checkpoint; recovery checkpoint, reload, training on, rollback | exact steps: largest logit change 0; width steps 1e-5 to 3e-5; the same replies after growing step 15,000 from 1.18M to 2.04M; new units get gradient; grown models save, load, train and match the CPU on the GPU; rollback restores the original weights exactly and keeps the grown model |
| **Training memory** (`tests/gpu_memory.rs` on a software GPU, `learn::memory` unit tests, Studio `memory` tests): micro-batches of 1, 2 and 5 rows and the optimizer state in system RAM, both architectures, with and without dropout and Stage-A focus; a step over the GPU memory budget; the trainer under no budget, a tight budget, a budget needing RAM, RAM refused, and Manual settings | every variant takes the CPU's step (weights within 1.2e-7, moments within 5e-9); a step over the budget is refused before anything runs, leaving weights and moments unchanged, and the same batch fits one row at a time; the optimizer state in RAM frees 2 of the 5 model-sized GPU buffers (net 20% of the model state with a 16 MB read buffer) and can be moved there mid-training; the trainer picks micro-batches, then RAM, then the CPU when RAM is refused, trains the CPU's weights each time and publishes its report; the memory estimate is 1.3x the measured working memory |
| **Growth within limits, automatic growth** (`learn::autogrow` tests, `tests/gpu_memory.rs`, Studio `growth` tests): plateau/memorising/too-soon/too-few-checks/broken-number runs; a real step-15,000 checkpoint in Automatic mode through still-improving, memorising, over-the-size-limit and plateaued runs; memory fits of growth steps; a grown model under a budget it no longer fits whole | grows only on a plateau with training loss within 10% of held-out, never too soon or on NaN; the checkpoint grows once, within its limit, with its outputs kept, and then waits for 3,000 steps; every decision is saved with its reason; fits are reported per step and a 1e9-parameter model is refused; the grown model trains within the budget in micro-batches with the CPU's weights (within 1.7e-5) |
| **Nova Neuro brain plate** (`tests/neuro.rs`; Studio rendered with real measurements of trained and untrained Neuro models, with and without associative memory, light and dark) | the language core's attention, each working-memory layer's state, the controller's hidden activity and gates, recalled patterns and the bus, per round, from the traced forward pass (state and bus lengths match the recorded norms to 1e-4); each system's effect measured by switching it off; networks a model lacks drawn dashed and named absent |
| **Built-in courses** (`learn::builtin` tests; baseline and variants in docs/LEARNING.md) | 16 lessons, 125-186 taught and 10-22 held-out questions each, every course prepared without problems, identical in every copy; a new model learns each lesson on its own, but in sequence the earlier-lesson guards reject every candidate after the first mastered lesson (Basic English: 1 of 6 lessons, 10% of all held-out questions at the end); teaching all of a course's lessons mixed in one session instead: 87% (English), 86% (Reasoning), 47% (Math) vs 10%, 49%, 26% one at a time |
| **Teaching together** (`tests/together.rs`, course and program tests) | Classroom: the ticked lessons in one group (one mixed session, each lesson graded on its own held-out questions: 0/10, 0/17, 0/16 -> 6/10, 10/17, 16/16 in three epochs), the unticked lesson taught alone afterwards, every group lesson in the rehearsal archive; courses: ticked lessons of a stage grouped, an unticked lesson alone, every lesson recorded and graded; teaching a lesson alone changes the course version, the default does not |
| **Saved earlier lessons in the mix** (`tests/together.rs`) | With `train_earlier` a saved lesson outside the group is named in the group's events, judged with the group (its held-out questions in the session's total and `by_lesson`) and scored after each attempt; without it, only guarded. Group lessons are still graded on their own held-out questions only. `recover_earlier` keeps training through epochs rejected only for earlier lessons; on by default for the Classroom and courses |
| **Nova Neuro, every network** (`model::neuro` tests, `tests/gpu_neuro.rs`, `tests/growth.rs`) | Relational slots, experts and consolidation, alone or together, with or without the transformer: gradients match finite differences, generation from the cache equals the full pass, no output reads a later token, every single-network design learns; GPU equals CPU to ~1e-7 for every ablation and focus; growth keeps the outputs; version 1 designs load unchanged |
| **Checkpoints while teaching** (`tests/checkpoints.rs`) | Each accepted epoch is written to the checkpoint folder (one per accepted epoch, newest three kept, README, no temporary file left); every file loads and the newest equals the session's final model (weights, step, words) |
| **Continue, back off, override** (`tests/checkpoints.rs`) | A session continued from the epoch-4 checkpoint runs epochs 5-6 with the same learning rate at every batch as the uninterrupted session; back-off halves the rate after two worse epochs (no epoch accepted in six without it, epochs 5 and 6 accepted with it); the first-epoch override accepts epoch 1's best candidate although ineligible, and only epoch 1 |
| Bicameral unchanged by Nova Neuro: real-checkpoint compatibility, golden vectors, measurements, every engine test | pass |
| **Review Point 1** (`examples/neuro_compare.rs`, 1.2M each, 1.5M tokens, seeds 7 and 101; docs/NEURO.md section 13) | level overall: held-out loss Neuro 0.4389 / 0.4321 vs Bicameral 0.4394 / 0.4496; in-distribution exact 17 / 20 vs 23 / 16 of 200; unseen-name "where" exact 0 for both; Neuro needs working memory (cut: 0 answers), associative memory unused (cut: no change) |
| Rewards, Stage 2: one emphasis plan per epoch, the same number of units as ordinary training (within rounding), every emphasis with its reasons, held-out questions never scheduled, known units kept at the minimum but still trained | pass |

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
CI (this container cannot reach their hosts). Lesson Designer, Study Mode, practice, reflection
and Review Training, in the app: a text dataset imported and proposed as blocks with source lines
(a wrong sum, an answer paired by position and a missing answer flagged), edited, saved into a
new course and into the Classroom list and taught there with the progress card; a text studied
into memory (3 of 6 sentences kept by surprise), memory added, retrieved in a conversation (one
step through a shared concept) and logged; a practice session answered, marked, corrected and
trained (accepted, held-out loss 3.19 → 2.52); a reflection on memories measured and written;
review material generated as a Designer lesson, checked, assessed, trained and compared.
Rewards, in the app: a practice session graded automatically (reasons shown, uncertain where no
rule decides), certain grades accepted, a partial teacher grade (40%) and note given, trained
(accepted); the § 9 Rewards page shows the ledger (credit, concepts, the session, questions with
their emphasis and reasons, the audit log with withheld credit and reasons).
§ 13 Chat (0.5.7): the page was checked in a browser with a stand-in backend (layout at desktop
and phone widths, light and dark; streaming text; attachments; settings; the chat menu). Prompt
building, conversion to the model's alphabet and file attachments are unit-tested in Studio, and
sampling in the Engine; the page has not yet been used in the installed app.
§ 4 Conversation (0.5.8): the training is tested in the Engine (rows train only Nova's lines;
both sides; names; cut at turns; a run keeps only better checkpoints; stop), the chat-memory
extraction and contradiction flag in the Engine, the dataset conversions in Studio. The page and
the chat marks were checked in a browser with a stand-in backend; a real practice run on a
taught model has not been done yet, so how much it helps Nova talk is not measured.
Conversation teaching (0.5.9): the curriculum's decisions (mastery and plateau, stuck and
back-off, stop or move on), the practice pool (floor, cap, recency, retiring learned ones), the
stage windows and the weight of replies that use earlier turns, the listening rows, the reply
checks, the context tests (pairs differ in one fact; graded by meaning; practice and tests share
no names, places or things) and a full run reporting every measure are tested in the Engine; on
the step-15000 checkpoint every candidate that lost Classroom answers it had was refused.
Studio's page, the Planner's reading and conversation steps and the new measures were checked in
a browser with a stand-in backend. A real run on a taught model has not been done yet: whether
these teaching methods help Nova more than plain practice is not measured.
0.6.0: held-out checks on the GPU (Reading and Conversation) are scored in parts that fit the
GPU memory budget, halved when a part does not fit, and run on the CPU (with a notice) when not
even one row fits. Conversation checks score whole conversations up to the full context, and
before this they were sent 8 rows at a time without the memory planner, which ran a 0.5.9
Conversation run out of GPU memory. Tested on a software GPU with no budget, room for two rows
and room for none (the same loss as the CPU each time).
0.6.4: GPU checks (`learn::set_gpu_checks`, on by default). The Classroom's candidate losses run
on the GPU, and the GPU confirms the answers greedy writing certainly gives exactly (the expected
token is the clear choice at every position, by more than 0.001 in logits), so the CPU writes
out only the rest; Reading no longer waits for a check's samples. Tested on a software GPU
(`tests/gpu_checks.rs`, `tests/gpu.rs`): every confirmed answer is exactly what the CPU writes
and the reports are identical (step 15,000: 99 of 99 right answers confirmed; Nova Neuro: 95 of
97); a lesson with GPU checks on and off selects the same candidates with the same scores, and
Reading on the GPU stops at the same check as on the CPU. The speed-up on a real graphics card
has not been measured.
0.7.1: **probation, measured** (`examples/probation_compare.rs`; the step-19,000 checkpoint,
which was taught the owner's 27 lessons, learns the built-in "Making up your mind" course
together with those lessons, with the owner's Classroom settings: learning rate 0.00005, batch 8,
net safeguard, earlier lessons trained and guarded, recover-earlier and back-off on, up to 20
epochs). Every lesson's held-out questions are split: half judge checkpoints, half are never
trained on or used to choose anything (the test). A: the current rules; B: probation 3 epochs.

| Test half (never trained, never judged) | A seed 7 | B seed 7 | A seed 101 | B seed 101 |
|---|---|---|---|---|
| New course right, of 51 (0 before) | 1 | **2** | 0 | **2** |
| Earlier lessons right, of 344 (198 before) | 174 | **180** | 173 | **177** |
| New-course loss (4.99 before) | 0.796 | **0.678** | 0.901 | **0.662** |
| Probation branches | | 2 promoted (after 1 and 3 epochs), 1 dropped | | 2 promoted (after 3 and 2), 1 dropped |

Vocabulary limit 1,024 and reply limits fitted to each answer in all four (with the model's full
vocabulary and 24-token replies, no candidate of either arm was ever accepted, and the new
course could not be answered at all: 102 of its 104 answers were longer than a reply). On both
seeds the rejected updates that were given room recovered, were accepted later, and ended
better on questions never used for training or judging. A third arm with no limit on probation
accepted exactly what B did (seed 7), so more room did not help. Two seeds, one model and one
course; the margins are a few answers. Probation stays off by default until it is confirmed on
the owner's own model.

Also found:
- **The earlier-lessons answer check is 12 questions** (`retention_panel` 8, by kind), and one
  answer lost there rejects a candidate. With the full vocabulary it was the most common
  rejection, often the only one, including of candidates with more answers right overall and
  more right on the unseen test.
- **Growing the vocabulary costs earlier answers that no guard sees.** New word slots start
  untrained and words once spelled letter by letter become new tokens: the earlier-lesson test
  fell from 198 to 155 right at once, and no arm got back above 180. The guards are measured
  from the model after growth, so they never see it.
- **Answer churn**: every epoch loses 15-23 answers the accepted model gave and gains 11-17
  (mostly earlier lessons trained again), so the net safeguard rejects most candidates.
0.7.2: follow-ups to the probation results. **New words now come only from the lessons being
taught** (`words_from_new_lessons`, on by default): on step 19,000, adding the 380 words of every
lesson trained (the new course and the 27 earlier lessons trained with it) cost 67 of its 417
earlier answers before any training; the 156 words of the new course alone cost 9
(`examples/growth_probe.rs`). Starting new words from their letters' embeddings changed nothing
before training. The answers just before and after words are added are reported, and can be
guarded (`guard_before_growth`, experimental). Also: a larger earlier-lesson check (every
earlier lesson, judged by net change; experimental), cut-off replies counted apart from wrong
ones, "Compare settings (experiment)" in the Classroom (A/B/C on the open model with a test half
never trained or judged; `learn::compare`, tested on step 15,000: the model is unchanged, every
epoch is tested, the test half is never judged) and a best balanced Conversation checkpoint
(experimental). Not yet run on the owner's model.
0.7.8: conversation and safeguards (docs/LEARNING.md, "Every turn, following the conversation"
and "The earlier lessons' loss allowance"). Verified by tests: every one of Nova's turns is
trained from its own history (one pass equals each reply alone, to 1e-4, on Bicameral and Nova
Neuro); free-running prompts are Nova's own; the understanding tests are held out from the
practice by words and wordings, and the original context tests are unchanged; 13 kinds × 3 levels
× 3 different example conversations; the download filter's reasons; the earlier-lesson allowance
and judging by answers. Measured: DailyDialog's original copy answers 501 (fixed: final, other
copies first); kept by the filter from 300 KB of each: DailyDialog 672 of 700, EmpatheticDialogues
692 of 700, SODA 358 of 400, Persona-Chat 262 of 300, OpenAssistant 533 of 12,947, UltraChat 200k
269 of 33,800. Not yet measured:
whether the examples and understanding practice raise the understanding tests on the owner's
model (needs a conversation run on a GPU); LMSYS-Chat-1M downloaded with a real token (the
gated rows could not be fetched here; its layout and flags are tested on rows in its published
format).
0.8.1: recovery (docs/RECOVERY.md; the first stage of 0.9, docs/PLAN-0.9.md). Reading and
Conversation save recovery points (the working model with its optimizer state and the run's
position) on an interval and when stopped, in two checksummed slots, with a run record; Studio
asks about an interrupted run at start-up and never continues or discards one by itself.
Verified by tests: a reading or conversation run stopped, or killed after any recovery point, and
continued ends with the same weights, optimizer state and checks as an uninterrupted run, bit for
bit on the CPU (`tests/recovery.rs`, Conversation across a curriculum stage change; Studio's files
read back from disk after a simulated crash); a checkpoint accepted after the last recovery point
is kept; a recovery point is refused for other settings, other data or a changed model; a damaged
recovery point falls back to the one before it, and both damaged is reported with the accepted
model untouched. Not checked here: a real power cut or a Windows restart during training (the
writes are flushed and renamed, but only the owner's PC can show it); continuing a run on a GPU
(agrees to rounding, not bit for bit).
0.8.2: recovery finished for this stage and settings (docs/SETTINGS.md). Verified by tests: an
old settings file is upgraded with a copy kept and doing it again changes nothing; a damaged file
falls back to the backup and is kept, never written over; with neither, defaults and nothing is
written at start-up; a newer version's settings are kept; recovery points are checked against
their checksums for the start-up dialog. Not checked here: closing the window during training (the
stop, the recovery point and the close are wired, but only run in the app).
0.8.3: Hugging Face datasets are read from their Parquet files when the dataset viewer has them
(the 100-row pages remain the fallback). Measured here: the row pages give about 65 rows a
second (a page takes 0.3-0.8 s and requests are a second apart); one 244 MB Parquet file
downloaded in 10 s and its 69,289 rows were read in 1.3 s. UltraChat 200k end to end (download,
conversion, quality filter): all 207,865 rows in 63 s, about 53 minutes through the pages. The
filter keeps 1,686 of them (1.9 MB; 202,371 too long or code), so asking for more cannot be met:
the page now shows rows read of the dataset's rows, documents kept, the filter's share and the
size that share gives, and "Finish now" keeps what was downloaded (Cancel keeps nothing). Tests:
file choice by config and split, a broken download continued with a range request, a server
that ignores ranges restarting the file instead of appending, Parquet rows read as the viewer's
JSON. Not measured: LMSYS-Chat-1M (gated; no token here).
Also in 0.8.3 (the owner's report: unticking the AI-voice box did not change the download filter,
which was always on): conversation downloads keep every usable conversation (the dataset's own
flagged ones in a file apart), and the quality filter is applied when practice starts, reason by
reason as chosen on § 4 (all on by default, so practice uses what it used before), with an
optional "use up to N MB" per dataset and a count of what practice would use. Always left out:
unreadable conversations and ones holding Yendory's tests. Your own datasets are never filtered;
datasets downloaded before 0.8.3 were filtered as they downloaded and are unchanged. Reading a
conversation dataset still uses the full filter. Tests: each reason applies only when chosen,
flagged conversations only with that filter off, the tests always left out, the size cap; network
check: UltraChat now stores all 51 usable conversations of a 300 KB sample (the full filter would
pass none).
0.8.4: the Hugging Face token (docs/HUGGINGFACE.md) is kept in the system's credential store
(Windows Credential Manager, the Secret Service, the Keychain), or in a file only the user can
read where there is none, never in the settings file; an old token is moved at start-up and
scrubbed from the settings file and every copy of it; the page shows the connected account, the
token's name and kind, and can check and disconnect; a 403 on a gated dataset says to accept its
conditions. Tests: migration leaves no copy of the token in any file and keeps the other settings,
the fallback file is owner-only, a store that becomes available takes over. Checked here: an
invalid token gets 401 from /api/whoami-v2. Not checked here: Windows Credential Manager itself
(this container has no credential service; the code path is the `keyring` crate's).
0.8.6: probation takes priority over the flat-epoch rule when it is on (the owner's choice; before,
a safe but flat candidate always existed on the owner's model, so probation never acted and
variant B equalled A). Its limit is kept, the accepted checkpoint is restored when it is reached,
and nothing about acceptance changed; probation off behaves as before. The Classroom shows when
it starts, why (the guards the candidate failed, and the flat candidate it took priority over),
the epochs left, and whether it succeeded or failed. Tested: off and on behave differently with a
safe flat candidate present, the accepted model is unchanged throughout. Experiment B is to be
run again later (the owner asked to finish the 0.9 features first).
0.8.7: the download queue is saved and comes back paused at start-up (the interrupted download
first in line) unless the owner chose to resume it; pause / resume, reorder, remove, retry failed
or cancelled downloads, clear finished ones (docs/HUGGINGFACE.md). Tests: the saved queue round
trip and restore (paused, interrupted item first, nothing starting by itself; resuming when chosen;
a damaged file set aside), pause back to the front, reorder, failed / cancelled / done outcomes,
retry rules.
0.8.8: importing datasets by hand (docs/IMPORTING.md, § 14 Datasets). Tests: format and split
from names, Hugging Face links and revisions; JSON lines (a bad line counted, not fatal), JSON
(arrays and objects holding one), CSV with quoted commas, TSV, text and gzip read to records;
messages / prompt-response / turn-list / one-row-per-turn layouts to conversations, roles named or
left out, every rejection reason counted; ZIP entries read safely and a `../` entry refusing the
archive; scripts' data links found without running them; splits never mixed; the page's mapping
JSON. Network check: UltraChat listed with licence and splits; DailyDialog's script noted, not run,
its http link named but not downloaded; SODA's test split (Parquet conversion) mapped 50 of 50
records to conversations. Not checked here: a ZIP or file picked on Android.
Also measured (0.7.8's comparison, finished here; the owner's model, local only; the half of each
lesson's held-out questions never trained or judged): with the earlier lessons' loss allowed to
rise 0.08 instead of 0.02 (variant D), the accepted model ends with 26 of 51 new-course answers
(22 with 0.02), 73 of 150 earlier answers (72), 13 of 42 context pairs (11), and 4 of 6 epochs
accepted (3). Epoch 5, which 0.02 rejected, was accepted; epoch 6 was rejected because it lost two
held-out answers. One run of 51 and 150 questions: a hint, not proof. The default stays 0.02 (it
is a setting: "Earlier lessons' held-out loss may rise by" in the Classroom).
Not checked: the Windows installer and Android APK on real devices (CI builds them).

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

**Second seed (101), strict against net**, everything else the same:

| | strict | net |
|---|---|---|
| Held-out answers right at the end (of 108; 3 at start) | 32 | **35** |
| Gained / lost | 29 / 0 | 32 / 0 |
| Course 1 lessons: right after course 1 → after course 2 | 24 → 23 | 28 → 24 |
| Course 2 lessons right at the end (of 62) | **9** | 11 |
| Held-out loss at the end, 02A / 02B / 03A / 03B | 1.05 / 0.80 / 0.97 / 0.83 | **1.00 / 0.76 / 0.76 / 0.74** |
| Course grade, course 1 / course 2 | 0.52 / 0.14 | **0.61 / 0.18** |

Net is ahead again on both seeds: more answers at the end (38 vs 31, 35 vs 32), no answer the
model had at the start lost, lower held-out loss on every lesson, better course grades. The
margin is smaller on seed 101 (3 answers), and one lesson (03A) ended one answer behind strict.
Net still forgets a little more of course 1 during course 2 on both seeds, and still ends
level with or above strict on it. **Net is now the default** (owner's decision after both seeds) for the Classroom and new
courses; courses saved earlier keep the setting they were saved with.

### Reward-guided training (Stage 2) against ordinary training

`examples/reward_compare.rs`: the same start checkpoint, courses, settings (strict safeguard) and
seeds as the safeguard comparison. Both arms keep the reward ledger; only `rewarded` schedules
units by emphasis. The ordinary arm reproduced the strict runs above exactly on both seeds,
so recording rewards does not change training.

| | Seed 7: ordinary | Seed 7: reward-guided | Seed 101: ordinary | Seed 101: reward-guided |
|---|---|---|---|---|
| Held-out answers right at the end (of 108; 3 at start) | **31** | 29 | **32** | 30 |
| Answers right at the start that were lost | 1 | 1 | 0 | 0 |
| Answers lost after their own course (forgetting) | **3** | 4 | **3** | 5 |
| Held-out loss at the end, 02A / 02B / 03A / 03B | **1.17 / 0.94 / 0.89 / 0.87** | 1.39 / 1.15 / 0.94 / 0.93 | **1.05 / 0.80** / 0.97 / **0.83** | 1.55 / 1.34 / 0.97 / 0.95 |
| Course grade, course 1 / course 2 | **0.55 / 0.15** | 0.53 / 0.13 | **0.52 / 0.14** | 0.51 / 0.14 |
| Final exam (protected), course 1 / course 2 | **0.61 / 0.14** | 0.59 / 0.10 | 0.53 / **0.14** | **0.59** / 0.10 |
| Candidates rejected as unstable | 52 of 664 | **12 of 600** | 30 of 600 | **11 of 560** |
| Ledger: questions recorded as forgotten | **42** | 52 | **20** | 75 |
| Ledger credit: progress / retention / generalization | 504 / 460 / 32 | 481 / 416 / 33 | 445 / 403 / 35 | 487 / 419 / 32 |

Findings:

- Reward-guided training ended **behind** ordinary training on both seeds: two fewer held-out
  answers, higher held-out loss on three or four of the four lessons, more forgetting. The one
  exception is course 1's final exam on seed 101 (0.59 against 0.53).
- It was more stable (about a quarter of the unstable candidates) but learned less. The likely
  cause is the schedule itself: units Nova answers are trained in only about one epoch in four
  at the minimum emphasis, so they are forgotten (52 / 75 forgotten against 42 / 20), and
  emphasis piles onto questions she cannot yet answer.
- On seed 101 the ledger's progress and retention credit was *higher* for the reward-guided arm
  while its real results were lower: exactly the case where reward totals must not be read as
  improvement. The independent evaluations decide, and they say no.
- Two seeds, 108 questions: a two-answer gap is small, but every loss measure points the same way.

Reward-guided training stays **off** (not the default). Stage 3 is not proposed on this evidence.
Any adjusted schedule would have to be judged on new seeds (and ideally other lessons), not
tuned on these two runs.

### Gentler reward-guided training, on new seeds (211 and 307)

The gentler schedule (fixed before any of its results were seen: known answers keep regular
practice, emphasis 0.6-2.0, no concept or variation boosts) against ordinary training, same
start checkpoint, courses and settings (net safeguard, the default), seeds new to the reward
experiments. Both arms of a seed ran from the same build.

| | Seed 211: ordinary | Seed 211: gentle | Seed 307: ordinary | Seed 307: gentle |
|---|---|---|---|---|
| Held-out answers right at the end (of 108; 3 at start) | **34** | 29 | 33 | **36** |
| Answers gained / lost | 31 / 0 | 27 / 1 | 31 / 1 | 34 / 1 |
| Answers lost after their own course (forgetting) | **0** | 3 | **0** | 5 |
| Final exam (protected), course 1 / course 2 | **0.63 / 0.12** | 0.59 / 0.12 | 0.55 / 0.14 | **0.63 / 0.18** |
| Candidates rejected as unstable | 60 | 70 | 96 | **48** |
| Ledger credit: progress / retention / generalization | 385 / 371 / 40 | 477 / 442 / 35 | 549 / 533 / 32 | 629 / 576 / 46 |

Findings: one seed each way (-5 and +3 answers; four seeds in all, the reward schedules are
behind on three). Gentle forgot more after each course on both seeds (3 and 5 answers against
0), and its ledger credit was higher on both, including seed 211 where its real results were
lower. The spread between seeds (29 to 36 answers) is as large as any difference between the
arms. Reward-guided training stays off.

### Ordinary training: dropout and general-reading rehearsal (seeds 7, 101, 211)

`examples/reward_compare.rs` (ordinary training only), after the clipping fix and the
candidate-tally correction: the step-15000 checkpoint, courses 02A+02B then 03A+03B, the
default settings (net safeguard), seeds 7, 101 and 211 for every arm. Settings fixed before
any result: dropout 0.1; reading share 0.1 of each epoch's rows (same rows per epoch). The
reading text was 4,000 simple synthetic stories (TinyStories could not be downloaded here);
none of the 329 graded texts appears in it.

| | Ordinary | Dropout 0.1 | Reading 0.1 |
|---|---|---|---|
| Held-out answers right at the end (of 108), seeds 7 / 101 / 211 | 40 / 37 / 37 | **41** / 37 / **39** | 27 / 34 / 38 |
| Total over the three seeds | 114 | **117** | 99 |
| Answers lost after their own course (sum) | **5** | 7 | 6 |
| Final exam course 1 (mean of seeds) | **0.660** | 0.647 | 0.608 |
| Final exam course 2 (mean of seeds) | **0.163** | 0.122 | 0.143 |
| Lessons' held-out loss at the end (mean) | 0.833 | **0.817** | 0.982 |

Candidate tally (all seeds, every reason counted): of about 1,750 candidates per arm, 7-9% were
selected; the commonest reason was losing more of the lesson's held-out answers than gained
(about 55% of candidates), then harming earlier lessons (loss 20-27%, answers 17-21%).

Findings:
- Dropout: +1, 0 and +2 answers and slightly lower held-out loss, but lower course-2 exam scores
  and slightly more forgetting. The differences are within the spread between seeds: no clear
  gain. It stays available and off.
- General-reading rehearsal: behind on seed 7 (27 against 40) and on average; it stays off. With
  real reading text (TinyStories) the result could differ; the synthetic stories are a weak
  stand-in.
- Ordinary training after the fixes, seed 211 (the one seed with an identical earlier run):
  37 answers against 34, held-out loss 0.91 against 0.96, course-1 exam 0.65 against 0.63. One
  seed: suggestive, not proven.

### Weight blending (seeds 7, 101, 211)

The same harness and settings; blend share 0.5 (fixed in advance). With blending off training
is unchanged, so the ordinary runs above are the baseline.

| | Ordinary | Blend 0.5 |
|---|---|---|
| Held-out answers right at the end (of 108), seeds 7 / 101 / 211 | 40 / 37 / 37 | **41 / 41 / 42** |
| Answers lost after their own course (sum) | 5 | **-2** (net regained) |
| Lessons' held-out loss at the end (mean) | 0.833 | **0.809** |
| Final exam course 1 (protected; mean of seeds) | **0.660** | 0.654 |
| Final exam course 2 (protected; mean of seeds) | **0.163** | 0.122 |
| Blends selected / epochs with a selection | - | 76 / 160 |

Findings:
- Blending gained answers on every seed (+1, +4, +5) with less forgetting and lower held-out
  loss, and blends were chosen in about half the epochs that accepted a change.
- **The protected exams do not confirm it**: course 1 is level and course 2 is lower. The
  lessons' held-out questions are also what each session selects candidates on, and blending
  adds a fifth candidate per epoch, so part of its gain on those questions can be selection on
  the same questions rather than learning. The exams are independent of selection.
- Blending stays available and off. Before more selection-based experiments, the held-out
  questions should be split into a selection part and a separate verdict part (planned
  improvement 5), so gains can be measured on questions no choice was made on.

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
