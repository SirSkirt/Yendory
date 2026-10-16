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

**Context length.** `max_tokens` (Nova-2: 256) is how many tokens the model reads at once.
Positions are encoded by rotation (RoPE), so no parameter depends on it: it can be chosen for a
new model or changed for an existing one (`Bicameral::with_context`), and a model uses a longer
context well once it has trained on text that long (longer reading windows; lessons with long
conversations are no longer cut to 256 tokens). Training stores only each attention row's
softmax maximum and normaliser and recomputes the weights in the backward pass, allocates
gradients only when the backward pass reaches them, and frees activations as soon as they are
used, so memory grows with the context, not its square; the compute of attention still grows
with the square. Measured at Nova-2's size on a 4-core machine (rows as long as the context):
128 tokens 990 tokens/s, 1,024 649, 4,096 273, 8,192 111. Studio conversations keep as many
recent turns as fit in the context, and replies may be up to a quarter of it (48–512 tokens).

**Vocabulary size.** A model holds `vocab − 128` word units. Nova-2 allocates all 256 slots up
front (vocab 384). A larger **vocabulary limit** (`word_limit` in `SessionConfig` and
`PretrainConfig`; any size up to the Engine's parameter limit) adds slots only as words are
learned (`Bicameral::with_word_capacity`), so a model allowed a million words that has learned
5,000 trains and replies like a 5,000-word model. New slots start from fresh embedding rows.
Nova-2's training scores every allocated slot, used or not (as the reference does), so
allocating unused slots costs speed: at Nova-2's size, measured on a 4-core machine, 16,384
slots train at 725 tokens/s, 262,144 at 53 and 1,000,000 at 7 (256 slots: 1,715).

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

## Reading: corpus pre-training (`learn::corpus::pretrain`)

Optional, before the lessons: the model reads ordinary text and learns next-token prediction on
it, so it starts the lessons already knowing how English words and sentences go together.

1. **Normalise** to ASCII (typographic quotes, dashes, ellipses and spaces become ASCII; common
   accented letters lose their accents; anything else is dropped and counted).
2. **Split into documents**: at `<|endoftext|>` when present (TinyStories), otherwise at blank
   lines. About 10% of documents are **held out** by the same stable hash as lesson units and
   are never trained on.
3. **Word units** are learned from the training documents only (frequency ≥ 12, names stay
   character-level), up to a total limit (default 192 of 256) so later lessons can still add
   their own words.
4. **Train** the whole model (Joint) on random windows of the training documents (default
   128 tokens, batch 16, warmup then cosine learning rate).
5. **Judge**: about 20 times over the run the held-out loss is measured on fixed held-out
   windows. A checkpoint is **accepted** only when it lowers the best held-out loss by at
   least 0.002; every accepted checkpoint is saved. After 4 checks in a row without an
   accepted checkpoint, reading stops early. The model returned is the best accepted
   checkpoint, or the original model (including its vocabulary) when none was accepted.
   Each check also records a greedy continuation of the start of a held-out document, written
   by the model, so progress can be read and not only measured.

Studio stores corpora with their SHA-256 fingerprint and records what each model read
(`meta.json` `read`, and every check in `history.jsonl`). The built-in source is TinyStories
(Hugging Face `roneneldan/TinyStories`), downloaded on the user's device and never redistributed
by Yendory; any plain-text file can be used instead.

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
| Foundation V2 retries: gentler rate, then a changed strategy with 12 epochs, then both with 6 extra epochs | **replaced** by diagnosed retries: the remedy follows the cause (see Courses), and the rate is lowered only for unstable training; a lesson still not passed moves to the end of the queue |
| "First five lessons" foundation counter | **not ported**; Studio lets the teacher switch Foundation on or off. |
| Optional user benchmark (Test & Corrections) as a guard | **not ported** yet (Studio Exams, Phase 3) |
| Synthetic-curriculum rehearsal in `train_lessons.py` | replaced by rehearsal of the owner's own archived lessons |

## Courses (`learn::course::teach_course`)

**Passed is not the same as improved.** Each lesson has a **pass mark** (default 50%): the share
of its held-out questions the model must answer exactly right. Within an attempt, a checkpoint is
accepted only if it answers more questions exactly right, or lowers the held-out loss by at least
0.01 (`min_loss_improvement`; smaller changes are noise). Accepted improvements are always kept,
but the lesson's outcome is:

- `passed`: the pass mark was reached (if the model already met it, no change is needed);
- `progressed`: improvements were kept, but the pass mark was not reached;
- `kept_previous`: nothing qualified; the model is unchanged.

**A lesson that does not pass is diagnosed, not blindly retried.** After each attempt below the
pass mark the trainer works out why, from the rejected candidates and from re-asking the
lesson's own *training* questions, and the next attempt applies the remedy for that cause
(up to 3 retries, `RetryPolicy`):

| Diagnosis | How it is recognised | Next attempt |
|---|---|---|
| unstable | at least half the rejected candidates made held-out loss worse | half the learning rate (the only case that lowers it; floor 0.000001) |
| forgetting | at least half were rejected for harming earlier lessons | more rehearsal of earlier lessons (25–40% of each batch) |
| interference | at least half were rejected for losing answers of this lesson the model already gave | balanced practice: questions it gets wrong twice and every question it knows once, half the learning rate |
| not learned yet | it answers fewer of its own training questions than the pass mark | practise exactly the training questions it gets wrong (twice; weak kinds three times) plus a quarter of the ones it knows, easiest first, 12 epochs |
| memorised, not generalised | training questions right, held-out ones wrong | once: broader study (all units and explanations, new order, +6 epochs); a second time: stop retrying, other lessons are needed |

A remedy that failed without any change is never run again unchanged: if the same diagnosis
returns and the next attempt would be set up identically, retries stop and the lesson waits for
other lessons. Every attempt keeps all the session guards and is recorded (`AttemptRecord`, with
the diagnosis that led to it).

Example (step 15,000, sample course, pass mark 95% to force retries): exact 3/19 → targeted
practice ("not learned yet") 5/19 → balanced practice ("interference") 5/19 → gentler balanced
practice 7/19. The lesson is reported as *progressed*, not passed. If the lesson still does not pass, it **moves to the end of the queue**
(outcome `deferred`) and the course continues: a later lesson may teach the skill it needs. It
is tried again once the rest of the queue is done, but only if the model changed in the meantime
(otherwise the result would be the same); then its final outcome is recorded.

A course is several lesson files taught in order (for example the files of a ZIP; reading the
ZIP is the frontend's job, `sort_files` orders by name). For every file the Engine records a
`FileRecord`:

- `sha256` of the exact text taught, and its size, so it is provable which content was taught
  (Nova issue B: "ZIP training repeated the pasted lesson");
- training and held-out unit counts, `start_step` and `end_step`;
- the pass mark, the attempts (each with its diagnosis), and `pass` (2 when it is the retry after
  being moved to the end of the queue);
- the outcome: `passed`, `progressed`, `deferred`, `kept_previous`, `duplicate` (same content
  as an earlier file; not taught again), `failed` (with the reason, e.g. no held-out units or
  non-ASCII text; the course continues), or `stopped`;
- the baseline and final scores.

Learned files join the archive, so later files rehearse them and are guarded against forgetting them.

## Measurements (`Bicameral::measure`)

For the brain visualizer and diagnostics, measured from a real forward pass of the last prompt
token (the KV cache records it when created with `KvCache::with_trace`; tracing does not change
any value):

- attention weights of every block execution (lower stage, and each thinking round of the upper
  stage), per head, over the context window;
- every right→left injection: the gate opening per dimension, the size of the injected vector
  and of the left state it joins;
- right-hemisphere activations (bottom layer and upper layer per round);
- the five most likely next tokens;
- ablation effects as total-variation distance of the next-token distribution: right→left cut
  (silencing the right hemisphere is the same operation for the output) and left→right cut.

Example on step 15,000, "Where is the key?": gates 0.42 / 0.18 / 0.18, right→left cut 0.996,
left→right cut 0.000 — matching the earlier finding that the efferent path has no measurable
effect in that model.

## Thinking tree (`Bicameral::think`)

Nova-2 has no explicit reasoning steps: for every token, the lower stage runs once and the
upper stage runs `loops` times (thinking rounds), with a right→left exchange before each round.
The thinking tree records that computation while the model generates a greedy reply. It is
exactly the reply the model gives (tested), and every value is measured:

- each step: the chosen token and its probability, the next most likely tokens it weighed, and
  the uncertainty of the decision (entropy, bits);
- **read-outs**: the output layer (final norm, tied embedding, bias) applied to the left state
  after the lower stage and after each thinking round, i.e. what the model would have said had
  it stopped there (the last read-out is the actual output);
- the gate openings of each exchange, the attention of the last block (the earlier tokens it
  looked at most), and the dense hemisphere's influence on this decision (total-variation change
  when its input to the left is cut);
- **paths not taken**: at the closest calls (runner-up at least 5% likely, up to 3 steps), the
  runner-up token continued greedily by the same model for up to 12 tokens.

Studio shows the tree below the brain scan; any step can be opened in the brain scan, which then
measures the exact forward pass that made that decision (`Bicameral::measure_ids`).

