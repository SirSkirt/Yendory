# How the Engine teaches

`yendory_engine::learn` owns lesson parsing, training, evaluation and checkpoint selection.
Studio (and any other frontend) only chooses lessons and settings and displays the records.
Nothing here adds rules to inference: every reply comes from the network. Teacher answers are
training targets and evaluation keys only.

## Lessons (`learn::lessons`)

Unified Teacher Lesson files, parsed exactly like `reference/lessons.py` (the pipeline that
trained the step-19,000 checkpoint):

| Block | Training text | Held-out key |
|---|---|---|
| `[Dialogue Lesson]` / `[Understanding Lesson]` — `question  answer` (two spaces) | `U: question\nN: ` (weight 0.15) + `answer\n` (1.0) | the question |
| `[Reasoning Lesson]` — `Premise: … Question: … Supported: … Contradicted: …` | `U: premise question\nN: ` + `supported\n`; the contradicted answer is used only for the preference check | premise + question |
| `[Multi-Turn Conversation Lesson]` — `[session]` with `User:` / other speaker lines | the whole session as alternating `U: …\nN: ` / `reply\n` | the session text |
| Text before `[examples]` | `T: text\n` (weight 0.3), never held out | — |

About 10% of units in every block are **held out** by `crc32(lowercased, whitespace-collapsed
key) % 100 < 10`. The split depends only on the text, so reordering or re-importing lessons
never moves a question between training and test. Held-out units are never trained on and never
used to grow the vocabulary.

New word units are learned from the training text only (frequency ≥ 12, 3–24 characters);
capitalised words that are rarely lower-case are treated as names and stay character-level.
Existing word ids never change.

Verified: on the owner's 27 lessons the Rust pipeline reproduces the reference split, the
step-19,000 vocabulary word for word, and the published held-out scores of step 19,000 exactly
(see docs/STATUS.md).

## Evaluation (`learn::eval`)

- **Exact match:** greedy reply (up to 24 tokens) must equal the teacher answer.
  Conversations are scored per Nova turn with the true earlier turns as history.
- **Supported vs contradicted:** for Reasoning, is the supported answer more likely than the
  contradicted one?
- Held-out results measure the lesson's own patterns with new wording. They are **not** an
  independent exam: lesson-matched held-out data is used to choose checkpoints. Keep genuinely
  unseen tests separate (Studio's Exams).

## Teaching sessions (`learn::session::teach`)

One lesson at a time, starting from the accepted model.

1. Grow the vocabulary from the lesson's training text.
2. Score the starting model (the **baseline**) on fixed probes chosen once for the session:
   held-out units of this lesson (up to 40 per kind), a panel of earlier-lesson held-out units
   (up to 8), and earlier-lesson held-out loss rows.
3. For each epoch, build one batch plan (lesson units in a seeded order; with an archive, about
   10% of rows rehearse earlier lessons). Then train candidates from the **accepted** model on
   that same plan:
   - Foundation on: **dense** (right), **transformer** (left), **interconnect**, **joint**.
     Only the optimizer's update mask differs; every pathway runs in every forward pass, and
     each candidate carries its own copy of the optimizer state.
   - Foundation off: joint only.
4. Score each candidate as a whole model. A candidate is **eligible** when:
   - its held-out loss is at most 0.08 above both the baseline and the accepted model;
   - earlier lessons' held-out loss is at most 0.02 above its baseline;
   - every earlier-lesson panel answer the baseline produced is still produced;
   - (default on) every held-out answer of this lesson the accepted model produced is still produced.
5. An eligible candidate **improves** when it gets more held-out answers exactly right, or the
   same number with a lower held-out loss. The best improving candidate becomes the accepted
   model. If none qualifies, the accepted model is kept.
6. Stop after `patience` epochs (default 2) without a selection, or at the epoch limit.
7. If nothing was ever selected, the original model is returned unchanged, including its
   vocabulary and optimizer state.

Every candidate of every epoch produces a `CandidateRecord` (focus, train loss, scores,
eligibility, the exact reasons for rejection, selected or not). Every epoch produces an
`EpochRecord` with the accepted score and sample replies. They are delivered as `Event`s and can
be stored with `learn::append_jsonl`. The stop flag ends the session at the next batch; the
epoch in progress is discarded.

## Parity with Neural Nova's Foundation Training (app docs v0.12.5–v0.15.0)

| Neural Nova | Yendory Engine |
|---|---|
| Four complete-brain candidates (Dense / Transformer / Interconnect / Joint) cloned from the accepted checkpoint with optimizer state, same batches, only update masks differ | same |
| Best eligible whole brain selected per epoch; rejected candidates discarded; lesson without improvement keeps its baseline | same |
| Held-out loss tolerance 0.08 vs lesson-start baseline and last selection | same |
| Earlier lessons' held-out loss tolerance (default 0.02) | same |
| Earlier-lesson answer panel (4–8 units) must keep answers the baseline produced | same (8) |
| Previously correct held-out answers must remain correct | same (`protect_correct_answers`, default on) |
| Early stopping | `patience` epochs without selection (default 2) |
| Rehearsal of archived earlier lessons | ~10% of batch rows from the archive |
| Autonomous probes: first character, 12-character prefix, self-conditioned likelihood (0.35 weight) | **not ported**: these were character-level measures of the v1 brain. The Engine scores full greedy replies and teacher-forced loss on word-unit tokens. |
| "First five lessons" foundation counter, adaptive learning rate per attempt (halved rate / extra epochs on retry) | **not ported** yet; Studio can decide when to switch Foundation on or off. The learning rate follows warmup + cosine decay within a session. |
| Optional user benchmark (Test & Corrections) as a guard | **not ported** yet (Studio Exams, Phase 3) |
| Synthetic-curriculum rehearsal in `train_lessons.py` | replaced by rehearsal of the owner's own archived lessons |
