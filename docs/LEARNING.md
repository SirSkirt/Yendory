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
| `[Reading]` — optional first line `Mode: neural`, `memory` or `combined`, then text | the text at weight 1.0 in units of about 800 characters (paragraph and sentence boundaries), never held out; `memory` text is not trained (Study Mode only) | — |
| `[Evaluation]` — `question  answer` | not trained: these are the file's held-out questions | the question |

`[Reading]` and `[Evaluation]` are Yendory additions (the reference pipeline ignores them; files
without them parse exactly as before). When a file has an `[Evaluation]` block, its questions
are the held-out units, nothing else is held out by hash, and a taught unit asking an
evaluation question is dropped so it cannot leak the answer.

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

## Lesson Designer (`learn::designer`)

The Studio's Lesson Designer (§ 4) turns an imported `.txt` dataset into a lesson draft of
blocks: **Reading** (text Nova reads: neural reading, Study Mode memory, or both), **Lesson**
(instruction), **Examples**, **Practice**, **Extra Practice**, **Evaluation** (the teacher's
held-out questions), **Unified text** (blocks of the lesson format kept as written) and
**Unsorted** (material the analysis could not place; never taught).

**Analysis** (`analyze`) reads the file's layout, nothing more: Markdown headings and
`Lesson 3:`-style lines start sections; `Q:`/`A:`, `Question:`/`Answer:`, `User:`/`Nova:`,
`Problem:`/`Solution:`, `question  answer`, tab and `->` pairs are questions with answers; a line
ending in `?` takes the short line directly below it as its answer, flagged "paired by position"
until the teacher checks it; a question without an answer stays without one (shown, not
trained). Worked examples (`Example:`, `Solution:`) become Examples; otherwise about a third of
a section's answered questions are proposed as Examples, the rest Practice, and a fifth of a
long list Extra Practice. Prose in a section with questions is instruction, in a section without
them reading. Lines with no letters are Unsorted. A dataset with more than 40 questions or 3,000
words is divided into lessons at its top-level headings, and no lesson gets more than 60
questions (optional). Unified lesson files are imported block by block. Every block keeps the
lines it came from and every question its line; the file's SHA-256 is kept with the draft.

**Origins.** Items and blocks record who wrote them (imported, generated, teacher), whether the
teacher edited them, and where each answer came from (the file, generated, the teacher,
missing). A generated answer must be verified before it becomes a training target; until then
the lesson cannot be saved for teaching and the item is left out of the file.

**Checks** (`validate`) flag, without changing anything: missing answers, unverified generated
answers, answers paired by position, wrong arithmetic (`3 + 4 = 8`) and answers that disagree
with a sum in the question (`What is 5 + 2?` / `8`), duplicate questions, a question that is
both evaluated and taught (an error: it would leak), taught questions nearly identical to a
held-out one (80% of words in common), held-out questions appearing in reading or instruction
text, too few held-out questions (fewer than 8 makes grades coarse; none means the lesson cannot
be graded), Unsorted material, empty blocks and Unified text that does not parse.

**The lesson file** (`export`): each Lesson block and the examples and practice after it become a
`[Dialogue Lesson]` (the instruction as its explanation); Reading becomes `[Reading]`, Extra
Practice `[Extra Practice]`, Evaluation `[Evaluation]`; blocks not used in teaching and Unsorted
blocks are left out. Markdown is for the teacher: the file gets plain text (headings, emphasis,
links and HTML removed with their text kept, bullets as `- `, table rows as cells separated by
`; `), converted to ASCII. The file is read back with the lesson parser to report what will be
taught. Finished lessons go to the Classroom list or into a course (new or existing); the file
name carries the lesson's stage word. Drafts are saved as they are edited and can be left
unfinished.

## Study Mode memory (`memory`)

Study Mode (Studio § 7, a switch) gives a model a **memory** of statements it chose to keep from
its reading. The memory is not part of the network, and nothing in it is counted as learned.

**Forming memories.** Each sentence (3–40 words, not a question) of a Reading block in `memory`
or `combined` mode is a candidate. The network reads each candidate on its own and its mean
loss per token (its surprise) ranks them: it keeps the sentences it cannot already predict,
those at or above the passage's median and above `KNOWN_LOSS` (0.5 nats per token), at most half
of the passage, so passages are never copied whole. The ranking is the network's; the median,
the 0.5 level and the half limit are fixed supporting rules. As the model learns, sentences it
predicts drop out of what it keeps. A sentence matching a held-out question or answer of the
lesson is never stored. The same sentence studied again is merged (counted, with its new
source); memories that share most words but differ by a negation are flagged as possible
contradictions for the teacher. Every memory keeps its sources (lesson, Reading fingerprint,
sentence number), the model step and its surprise when stored, how often it was retrieved, and
reflection's usefulness measurement. The teacher can edit, switch off (kept, not retrieved),
mark checked, delete or add memories; each decision is recorded with the statement and its
surprise, as labelled examples of what is worth remembering for a learned selection later.

**Models and checkpoints.** A model's memory belongs to it in the Studio (`memory.json` in the
model's folder) and is plain text, so it stays valid when the weights change: later lessons,
a restored best checkpoint or a changed vocabulary leave it as it is, and every memory records
the training step it was stored at. Exporting a `.nnova2` checkpoint does not carry the memory
(the checkpoint format is unchanged), and a model imported from a checkpoint starts with none.

**Retrieval** is an index: memories sharing words with the question (function words left out,
plural endings removed), weighted by how rare each word is among the memories, plus half that
weight for memories sharing a word with those (one step), up to 4, within a third to a half of
the context. "Does Max need water?" retrieves "Max is a dog" and "Animals need water", and
through them "Dogs are animals". Retrieved memories go in front of the question as lines
starting with `\x03` (never present in outside text; lesson files and typed messages have it
removed). Every retrieval is recorded (training, grading with memory, conversation, reflection,
review).

**Teaching with Study Mode** (`memory::Studying`, `teach_course_studying`, `CourseHost::study`):
before a lesson is taught its memory-mode reading is studied (`CourseEvent::Studied`); each
taught question with related memories gets a copy with them in front (the memory lines weigh 0,
they are never a training target) added to training next to the original, so the network
learns to use memories and to answer without them. Grading stays without memory: the session's
held-out score and the lesson grade are the network's alone. After the lesson its held-out
questions are asked again with memories in front, recorded as `with_memory` (remembered
knowledge) beside `final_score` (neurally learned). Reading in `combined` mode also trains the
weights; `memory` mode does not. With Study Mode off nothing is read or written, `memory` mode
reading does nothing and `combined` is neural reading only; saved memories are kept. The
Studio's conversation puts related memories in front of a message and shows them.

**The network can learn to use memories** (`examples/memory_probe.rs`): every training question
("What is Max?") had two memory lines in front, the fact asked about and an unrelated one, and
which animal a name is changed from example to example, so the memory was the only way to know.
A fresh 1.2M model, batch 16: after 250 steps 86% of new random facts answered right with the
memory lines and 11% without them (chance 12%); 88% / 9% after 1,500 steps.

## Interactive practice (`learn::practice`)

In a practice session (Studio § 7) Nova answers a lesson's practice questions herself: the
Practice blocks of a Lesson Designer lesson, or the taught questions of a Classroom lesson (up to
40), each answered greedily by the network (with Study Mode memories in front when it is on,
shown with the answer). The teacher marks each answer right or wrong and may write a correction
and a note; marks are saved as they are made, and nothing is trained until the teacher starts
the training pass. A marked item's training target is Nova's own answer when marked right, the
teacher's correction when marked wrong (or the lesson's answer when there is no correction);
which of the three is recorded for every item, so teacher corrections are never mixed up with
Nova's own answers. The training pass is an ordinary teaching session on those targets
(`session::teach_with`), judged by the lesson's held-out questions with the usual acceptance
rules; afterwards Nova answers the same questions again, and with Study Mode the held-out
questions are also scored with memory. Sessions are stored with every answer, mark, note,
target, before/after score and step, and the model's history records the pass.

## Reflection (`learn::reflect`)

Nova revisits what she has read: her Study Mode memories (shown to her as memory lines), or,
with Study Mode off, the sentences of a lesson's Reading blocks. Three parts are kept apart:

* **Yendory** proposes pairs of statements sharing a concept word and measures, with Nova's own
  network, how much one helps her predict the other: her loss on the second read alone minus
  her loss on it with the first in front (nats per token, the larger direction). The best
  pairs (by that measurement) are kept. With Study Mode the measurement is stored on the
  helping memory as its usefulness, and each pair is recorded as a retrieval.
* **Nova** is shown each pair with the question "What do these tell you together?" and writes an
  observation (greedy generation, at most 32 tokens). Yendory never writes or edits it.
* **The teacher** marks each observation right or wrong, may write their own version and a
  note, and may approve a training pass: at least three approved observations, every fourth
  held out to judge the pass (an ordinary teaching session; the pair is context, never a
  target). Targets keep their origin (Nova's observation, confirmed; or the teacher's
  version). Before and after: the held-out observations' score and loss, and Nova's loss on the
  reflected statements read alone; afterwards she reflects on the same pairs again.

## Review Training (`learn::review`)

Turns Study Mode memories into review material, as a Lesson Designer draft (needs Study Mode on
to prepare). The teacher chooses concepts (words shared by memories) or the lessons the memories
were read in. Simple sentence patterns ("X is a Y", "X has Y", "X can Y", any "X <verb> Y",
negations) give, per statement, a direct question as a worked example ("Is Max a dog?" / "Yes,
Max is a dog.") and held-out review questions in other words ("Is it true that Max is a dog?",
"What is Max?"); level 3 adds questions combining two memories ("Max is a dog" + "Dogs are
animals" -> "Is Max an animal?"), marked inferred, half as worked examples with their reason and
half held out. A held-out question never repeats an example. Every answer is generated and
unverified, with the memory (and lesson) it came from; nothing can be assessed or trained until
the teacher has checked each one in the Designer, where the material is edited like any lesson
(move questions between examples and held-out, add, delete, reword). Answers carry accepted
forms and rules (`Yes, Max is a dog. | Yes. | must: yes | never: not`), graded like course
sentences.

The **initial assessment** asks the held-out review questions without memory (and, separately,
with memory). **Review training** is an ordinary teaching session on the worked examples, judged
by the held-out questions, with no memory in front. The **assessment after** asks the same
questions; the comparison lists questions improved, regressed, still wrong and kept, and the
score per concept word. What she answers with memory (remembered) is always shown apart from
what she answers without it (neurally learned).

## Rewards (`learn::reward`)

Nova has no reinforcement learning: everything that changes her weights is supervised training
on target text. Rewards add a measurement (Stage 1) and a way to decide how much each unit is
practised (Stage 2). They never change what she is taught.

**The grader** (`grade_reply`) gives 0-100%, or *uncertain*. With an answer key: the key's
verdict (100%), its required words for partial credit (up to 60%), 0% for a forbidden word or a
conflicting keyword. Without one: exact or normalised (case and punctuation) equality is 100%;
the contradicted answer, the opposite meaning (negation) or nothing in common is 0%; a short
answer contained in a longer reply is 75%; anything else is **uncertain** and earns nothing.
Letters count only in lessons marked `Grading: spelling` (a first line in the lesson block) with
the experiment on: the share of the answer's prefix spelled right. Conversations: share of turns
right. The teacher's grade (Practice, 0-100%) replaces the automatic one.

**The ledger** (`RewardLedger`, per model, `<model dir>/rewards.json` in Studio) keeps, for each
question (lesson + canonical text) and its **reward cycle** (expected answer + the question's
concept words, so a reworded copy shares the cycle):

- **Progress**: one-time credit for a new milestone, network alone and with memories as separate
  tracks (memory track: half the budget). A milestone is a better grade, or in spelling lessons a
  longer correct prefix (by default 0.10 for the first letter, 0.09 for each further one).
  Partial milestones together earn at most half the budget; the complete answer earns the rest.
  The same answer again, in any capitalisation or punctuation, earns nothing; after 20 attempts
  partial credit stops (the complete answer still counts); a complete cycle never reopens by
  itself (the teacher can reset one). Never-trained questions earn no progress.
- **Retention**: at saved checkpoints only (the start of a session, the model after an attempt),
  once per checkpoint: the first correct answer of a taught question 0.6, then 0.2, 0.1, 0.05 ...
  (at least 0.02) at later checkpoints, only when training happened in between. A lost answer is
  recorded as forgotten; relearning continues the shrinking series, so forgetting and relearning
  cannot farm credit.
- **Generalization**: the first correct answer to a question Nova was never trained on (a
  lesson's held-out questions, Review Training's evaluation questions): 1.0, the largest credit.
- **Relevance** (diagnostic only): words from the question's own text (story, context) that a
  wrong answer uses, 0.01 each up to 0.05 per cycle, once per word; words in neither the source
  nor the answer are listed as invented. Never used by training.
- An optional value for a certainly wrong (0%) answer (0 by default, may be negative), once per
  distinct wrong answer; never applied to uncertain grades.

Correctness is recorded apart from credit: a correct answer with its progress used up stays
correct. Every answer is written to the audit log (last 20,000) with its grade and reason, the
credit, the milestones already credited, and credit withheld with the reason. Session summaries
feed the checks for reward hacking, which flag patterns to investigate: credit rising while
held-out accuracy does not, one wrong answer given to many questions, answers creeping forward
letter by letter, taught questions right but held-out ones not, recognised words without right
answers.

**Protected examinations.** A course's final exam is protected: `run_course` gives its questions
to `Rewards::protect`, and the ledger refuses (and only counts) any answer to them. The Exams
page never uses the ledger. Study Mode never stores held-out questions.

**When answers are recorded.** In a teaching session (`session::teach_rewarded`): the held-out
questions at the start and, when a change was accepted, at the end; a quiz of up to 64 taught
units and 16 units of earlier lessons at the start (a saved checkpoint). After an attempt that did
not pass, the course runner's diagnosis already asks every taught question; those answers are
recorded too. Practice records the teacher's grades and notes; Review Training records its
assessments (never-trained questions). Recording uses evaluation only and none of the training
random streams, so ordinary training is unchanged (`tests/reward.rs` checks the weights are
identical).

**Stage 2: reward-guided supervised training** (experimental, off by default). Before each
epoch Nova is quizzed on the lesson's taught units (the first epoch uses the start quiz), and
each unit gets an emphasis from its history (`RewardLedger::emphasis`, every reason listed):
1.0 plus up to 1.0 for the gap at its latest grade (a 0% answer is a full gap), up to 0.5 for repeated
failures, 0.75 when forgotten, 0.25 for partial progress not yet complete, 0.5 x the weakness of
its concept (other questions of the same pattern), 0.5 x how badly new variations of the concept
were answered at first sight, 0.25 for long answers; a correct answer is multiplied by 0.6, and
by 0.6 again per checkpoint it was retained (up to four). Units failing eight times in a row
without any progress are capped at 1.5 and listed as stuck (they need the teacher, not endless
repetition). Emphasis stays between 0.25 (never left out) and 3. The epoch's units are drawn with
copies in proportion to emphasis, scaled to the same total as ordinary training (stochastic
rounding, at most 4 copies), and rehearsal of earlier lessons favours units being forgotten.
The targets are unchanged (the lesson's verified answers, the teacher's corrections, answers
the teacher marked fully right), as are the candidates and every acceptance rule. Held-out and
never-trained questions never influence emphasis, so they stay an independent judge. A partly
right practice answer is never trained as written. `examples/reward_compare.rs` compares
Stage 2 with ordinary training (results in docs/STATUS.md).

## Nova Neuro (experimental, `model::neuro`)

A second architecture, chosen when a model is created; Nova Bicameral is the default and is
unchanged. Nova Neuro keeps the transformer core and replaces the Dense hemisphere with modules
that talk through a learned communication bus: a minGRU working memory, an associative memory
of learned patterns, and an Executive Controller that learns, per token and round, how strongly
to use each module. Everything in this guide (reading, lessons, courses, the Planner, Study Mode,
practice, reflection, review, rewards) works with it; Foundation candidates train the modules
("dense" in the records), the language core ("transformer"), the bus and controller
("interconnect"), or everything. It trains on the GPU (DirectX 12 or Vulkan), the CPU or both, as the device setting
chooses; replies are generated on the CPU. Design,
plan and status: docs/NEURO.md.

## Evaluation (`learn::eval`)

- **Exact match:** greedy reply (up to 24 tokens) must equal the teacher answer.
  Conversations are scored per Nova turn with the true earlier turns as history.
- **Supported vs contradicted:** for Reasoning, is the supported answer more likely than the
  contradicted one?
- Held-out results measure the lesson's own patterns with new wording. They are **not** an
  independent exam: lesson-matched held-out data is used to choose checkpoints. Keep genuinely
  unseen tests separate (Studio's Exams).

## Reading: corpus pre-training (`learn::corpus::pretrain`, `pretrain_mix`)

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
   least 0.002; every accepted checkpoint is saved. Training continues through checks that are
   not accepted (the best checkpoint is kept apart). After 6 checks in a row without an
   accepted checkpoint, reading stops early. Each check scores at least 24 held-out windows
   per corpus (192 shared between the corpora by default), so one corpus's result is not
   decided by a handful of windows. The model returned is the best accepted
   checkpoint, or the original model (including its vocabulary) when none was accepted.
   Each check also records a greedy continuation of the start of a held-out document, written
   by the model, so progress can be read and not only measured.

### Reading a mix (`learn::corpus::pretrain_mix`)

Several corpora can be read together. Each corpus has a **category** (stories and prose,
knowledge, conversation, reasoning) and a **share**; every training batch comes from one corpus,
chosen so that the batches follow the shares, interleaved (not one corpus after another, which
would make the model forget the earlier ones). Each corpus keeps its own held-out documents:
every check reports each corpus's held-out loss, Dense reliance and a sample written by the model
(a conversation's sample starts at the model's first turn, `... N: `). A checkpoint is accepted
when the share-weighted held-out loss improves.

- **Joint** (conventional): the whole brain reads every corpus.
- **Bicameral** (experimental): each corpus is read in the stage of its category; by default
  knowledge in Stage A (Dense + interconnect), conversation and prose in Stage B (transformer),
  reasoning in Stage C (cooperation). A checkpoint whose Dense contribution falls below half of
  the best checkpoint's is not accepted, once that contribution is clearly above noise
  (`GUARDED_RELIANCE`, 0.05; an untrained model's Dense reliance moves at random below it).
  Every rejected check states why.

### Sources

Studio stores corpora with their SHA-256 fingerprint, category, licence and credit, and records
what each model read (`meta.json` `read`, with every corpus's share, stage and licence; every
check in `history.jsonl`). Everything is downloaded on the user's device and never redistributed
by Yendory. Records are converted into Nova's format: conversations become `U:` / `N:` turns,
problems a question and its worked answer. Formats, licences and reachability were checked from a
GitHub runner, and CI downloads a little of every source (non-blocking).

| Source | Category | Licence | Format |
|---|---|---|---|
| Simple English Wikipedia (2023-11, `wikimedia/wikipedia`) | knowledge | CC BY-SA 3.0, GFDL | title and paragraphs, up to the reference sections |
| SODA (`allenai/soda`) | conversation | CC BY 4.0 | situation, then `U:` / `N:` turns |
| Synthetic-Persona-Chat (`google/Synthetic-Persona-Chat`) | conversation | CC BY 4.0 | `U:` / `N:` turns, name placeholders filled |
| bAbI task 1 (`facebook/babi_qa`) | reasoning | CC BY 3.0 | story so far, `U:` question, `N:` answer (only task 1 of 20 is available as data) |
| GSM8K (`openai/grade-school-math`) | reasoning | MIT | question, worked steps, "The answer is N." |
| ProofWriter (`tasksource/proofwriter`) | reasoning | CC BY 4.0 per AI2's release (the mirror states none) | facts and rules, statement, true / false / unknown |
| 22 children's books (Project Gutenberg) | prose | public domain in the USA | a few paragraphs per document; Gutenberg header, licence and illustration captions removed |
| TinyStories (`roneneldan/TinyStories`) | prose | CDLA-Sharing-1.0 | stories |

DailyDialog was considered and left out: no data-only copy can be downloaded (its Hugging Face
copy needs a loading script, the original link no longer serves the file). Large sources are
downloaded in part ("first N MB"); Studio shows the total against a 1 GB budget.

Hugging Face sources are read 100 rows per request through its dataset viewer, which limits
how fast anyone may ask ("too many requests", HTTP 429). Studio spaces its requests (one a
second), and when the server says to slow down it waits as long as the server asks (or 30 s,
doubling up to 5 minutes, for up to 20 minutes), shows that it is waiting, and continues more
slowly. If the server still refuses, the documents downloaded so far are kept (marked "first
N MB"). A free Hugging Face access token in the `HF_TOKEN` environment variable raises the
limit; it is sent only to huggingface.co.

### Practice generated on the device (`learn::generate`)

Question-and-answer text in Nova's format, with the reasoning written out: arithmetic, word
problems, comparisons and ordering, if-then and all/some logic (including "we cannot tell"
cases), small "who is where, who has what" worlds, and sequences; difficulty 1 to 3. Every
answer is computed from the example itself (the sums are added, the world is simulated, the rule
is followed), so it is correct by construction. It is read like any corpus (10% held out) and is
never used when the model replies.

## Teaching sessions (`learn::session::teach`)

One lesson at a time. Two models are kept: the **accepted** model (the best validated
checkpoint) and the **working** model that training continues from (weights, optimizer state
and step). They are the same until an epoch is flat.

1. Grow the vocabulary from the lesson's training text.
2. Score the starting model (the **baseline**) on fixed probes chosen once for the session:
   held-out units of this lesson (up to 40 per kind), a panel of earlier-lesson held-out units
   (up to 8), and earlier-lesson held-out loss rows.
3. For each epoch, build one batch plan (lesson units in a seeded order; with an archive, about
   10% of rows rehearse earlier lessons). Then train candidates from the **working** model on
   that same plan:
   - Foundation on: **dense** (right), **transformer** (left), **interconnect**, **joint**.
     Only the optimizer's update mask differs; every pathway runs in every forward pass, and
     each candidate carries its own copy of the optimizer state.
   - Foundation off: joint only.
4. Score each candidate as a whole model. A candidate is **eligible** when:
   - its held-out loss is at most 0.08 above both the baseline and the accepted model;
   - earlier lessons' held-out loss is at most 0.02 above its baseline;
   - every earlier-lesson panel answer the baseline produced is still produced;
   - (default on) it does not lose more of this lesson's held-out answers that the accepted
     model produced than it gains (the **net** safeguard, `protect_answers_net`, the default
     since it beat the stricter rule on two seeds; docs/STATUS.md). With net off, no answer
     may be lost at all. Courses saved before the change keep the setting they were saved with.
5. An eligible candidate **improves** when it gets more held-out answers exactly right, or the
   same number with a held-out loss at least 0.01 lower than the accepted model's
   (`min_loss_improvement`). The best improving candidate becomes the accepted model and the
   working model.
6. **Flat epochs.** When no candidate improves enough but some are safe (eligible, only not
   better enough yet), training continues from the best safe candidate: the epoch's learning
   is kept in the working model, so improvement can build up over several epochs until it
   passes the 0.01 threshold against the best checkpoint and is accepted. When no candidate is
   safe (held-out loss rose more than 0.08, earlier lessons or answers lost, Dense unused), the
   working model is **restored** to the accepted model. Each epoch records which it was
   (`selected`, `continued`, `restored`, `flat_epochs`).
7. Stop after `patience` flat epochs in a row (default 4; the Classroom's and a course's
   "Flat epochs before stopping"), or at the epoch limit. The session's result is the
   accepted model; training after the last accepted checkpoint is not kept.
8. If nothing was ever accepted, the original model is returned unchanged, including its
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
| Best eligible whole brain selected per epoch; rejected candidates discarded; lesson without improvement keeps its baseline | same, except that in a flat epoch training continues from the best safe candidate without accepting it (see Early stopping) |
| Held-out loss tolerance 0.08 vs lesson-start baseline and last selection | same |
| Earlier lessons' held-out loss tolerance (default 0.02) | same |
| Earlier-lesson answer panel (4–8 units) must keep answers the baseline produced | same (8) |
| Previously correct held-out answers must remain correct | same (`protect_correct_answers`, default on) |
| Early stopping | **changed**: training continues through flat epochs from the best safe candidate (the best checkpoint is kept apart); stop after 4 flat epochs in a row (configurable) |
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
- `deferred`: not passed after its retries; moved once to the end of the queue;
- `unresolved`: still not passed after its retries and its one move to the end of the queue.
  Its attempts are used up; any improvement it made is kept (`FileRecord::changed`). Records
  written before this outcome existed say `progressed` (improvements kept) or `kept_previous`.

Lesson retries are finite: at most 3 diagnosed retries, then at most one move to the end of the
queue (only if the model changed meanwhile), then `unresolved`. No lesson is retried endlessly.

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
practice 7/19. The lesson is not passed. If the lesson still does not pass, it **moves to the end of the queue**
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
- the outcome: `passed`, `deferred`, `unresolved`, `duplicate` (same content
  as an earlier file; not taught again), `failed` (with the reason, e.g. no held-out units or
  non-ASCII text; the course continues), or `stopped`;
- the baseline and final scores;
- **learning apart from the pass mark**: `own_before` / `own_after`, the share of the lesson's
  own (taught) questions answered exactly right before and after (up to 40 per kind). Studio
  shows held-out loss, held-out answers and own questions for every lesson, and points out
  small held-out sets (fewer than 20 questions, each worth more than 5% of the score). A lesson
  can be learned from without reaching its pass mark; the record shows both.

Studio warns (and does not prevent) when a model that has not read anything is taught: without
reading the model has no general English, so held-out questions that need it rarely pass.

Learned files join the archive, so later files rehearse them and are guarded against forgetting them.

## Course Builder and Lesson Planner (`learn::curriculum`, `learn::program`)

Courses with graded lessons, a final exam and automatic remediation, run one after another by a
plan, unattended. Nothing here changes the network or its checkpoints: it decides what is taught
when, with which settings, and how the result is graded.

### Building a course

File names decide order, stage and role (`curriculum::read_name`, whole words, any case):

- the **first number** in the name orders the lessons (2 before 10); files with the same number
  form a set, taught Dense → Transformer → Co-op → whole brain; files without a number come last;
- **Dense**, **Transformer** or **Co-op** (also `Coop`, `Cooperative`) choose Stage A, B or C
  (see *Bicameral education*); without a keyword the course's default stage is used (whole brain
  unless chosen). A name with two stage keywords must have its stage chosen by hand;
- **Practice** marks extra practice for the lesson with the same number and stage; an
  `[Extra Practice] … [/Extra Practice]` block inside a lesson file works too (ordinary teaching
  ignores it);
- **Exam** marks the course's final exam: sections `[Dense]`, `[Transformer]`, `[Co-op]`,
  `[Whole Brain]` holding ordinary lesson blocks, or no sections (one whole-brain exam).

Each lesson has its own **pass mark**. The course's **required grade** is the average of its
lessons' pass marks.

### Answer keys and grading (`learn::grading`)

In course lessons and exams an answer may list accepted forms and rules separated by `|`; the
first form is the training target:

```
What is a cat?  animal | creature | !plant
Is a robin an animal?  Yes, a robin is an animal. | A robin is an animal. | must: robin animal | never: not
```

- **Keyword grading** (Dense lessons, the exam's Dense section): correct when the reply contains
  an accepted keyword (or phrase) as whole words, not negated, and no conflicting keyword that is
  not negated ("not", "no", "never", "nor", "without" within the two words before it).
  Conflicts are the `!word` entries plus, by default, the accepted keywords of the lesson's other
  questions, leaving out words of the question itself. "Animal", "Cat animal" and "Animal, not
  plant" are correct; "Animal plant" and "not an animal" are not.
- **Sentence grading** (Transformer, Co-op, whole brain): the reply must equal an accepted form
  after normalising case, punctuation, spacing and contractions ("isn't" = "is not"), or contain
  all `must:` words and no `never:` word. Overlapping words alone never give credit, so "The door
  is not open" never matches "The door is open". The reasoning check (supported answer more
  likely than the contradicted one) is recorded as evidence; it never turns a wrong reply right.

Ordinary Classroom lessons have no key and keep exact matching, as in the reference evaluation.

### Questions that are never trained on

- About 10% of each lesson's questions are its **evaluation questions** (the usual held-out hash).
- Without a teacher-written exam, a **second, independent hash** reserves about 10% of each
  lesson's remaining questions for the **final exam**, in the section of the lesson's stage. They
  are neither trained on nor used for the lesson grades. A section with fewer than 8 questions is
  warned about; questions are never reused to fill it.
- A teacher-written exam question that also appears in any lesson or practice file is left out
  of the exam (with a warning).
- Practice questions matching an evaluation or exam question are excluded (with a warning).

### The course grade

**Course grade = 70% lessons + 30% final exam.**

- Lessons: the average of each lesson's score on its evaluation questions, re-graded on the model
  **at the end of the course**. Each lesson also records its score *when taught*; the difference
  shows forgetting (or later improvement).
- Final exam: the sections' scores weighted (default equal; configurable per course; a section
  without questions shares its weight proportionally with the others). Every section is also
  graded with Dense's input cut, and the exam's Dense reliance (loss rise with Dense cut) is
  recorded. Neither counts towards passing.
- The course passes when the grade reaches the required grade and, if set, the exam reaches its
  optional minimum. Lessons below their own mark are listed even when the course passes.
- An optional **pre-test** gives the final exam before the course (ungraded, never trained on),
  to show how much the course taught.

### Remediation of a failed course

A failed course gets up to **3 remediation attempts**. Each one:

1. diagnoses the lessons below their mark (from lesson results only; exam questions never steer
   training): *forgotten* (passed when taught, below at course end), *not learned* (its training
   questions are not answered), or *not generalising*; *unstable* when the previous attempt made
   the grade worse. If every lesson met its mark (the exam pulled the grade down), the lessons
   with the lowest course-end scores are practised;
2. picks a strategy for the main cause that **has not been used yet in this run**:

   | Strategy | What the session trains on, and how |
   |---|---|
   | targeted practice | wrong answers twice, the extra practice, a quarter of what it knows; easiest first, ≥ 8 epochs, 25% rehearsal |
   | consolidation | the whole lesson and its extra practice, 40% rehearsal of the course's other lessons, ≥ 6 epochs |
   | broader study | all examples and explanations plus a varied two-thirds of the extra practice, +6 epochs |
   | fine-grained | half the batch size, Foundation candidates (each part of the stage on its own), +4 epochs |

   Preferred order: not learned → targeted, fine-grained, broader, consolidation; forgotten →
   consolidation, targeted, fine-grained, broader; not generalising → broader, consolidation,
   fine-grained, targeted; unstable → fine-grained, targeted, consolidation, broader;
3. **halves the learning rate** again (×0.5, ×0.25, ×0.125 of the course's rate; floor 0.000001);
4. teaches each weak lesson **in its own stage** (a Dense lesson stays a Dense lesson), with
   correct answers protected and all session guards on; extra practice joins the rehearsal
   archive once used;
5. re-grades the whole course.

The **best checkpoint** (highest course grade) is kept on disk. An attempt that lowers the grade
is reverted to it before the next attempt. A course still failing after 3 attempts stops the
plan on its best checkpoint: the plan **never moves on past a failed course**.

### Plans, resuming and restarts

A plan is an ordered list of saved courses for one model. Progress (the course run: queue,
records, grades, remediation step) is saved with the model after every lesson and every
remediation session, so a stopped or interrupted plan resumes where it was; the lesson in progress
starts again. If the model was changed elsewhere in the meantime, the plan asks for a restart.

Restarting a failed course **continues from the model's best checkpoint** (weights, optimizer
state and step are kept: restart the education, not the brain), with fresh educational progress
and three new remediation attempts, and re-grades every lesson. A course version is identified
by its content fingerprint (lesson files, stages, pass marks, practice, exam, section weights,
exam minimum; training settings are not part of it). A revised course always gets a fresh run.
An unchanged failed version gets **one** manual override (warned and recorded in the history);
after that it must be revised. Failed and earlier runs are kept for comparison.

## Training on the GPU (`gpu`, `learn::Trainer`, `learn::bench`)

With the `gpu` feature (the Studio enables it), training steps can run on the graphics card
through wgpu. Every device trains the same network and the same `.nnova2` model: weights,
optimizer state and step belong to the model, wherever they are computed, so a model can move
between devices at any time without losing anything.

**Devices** (`learn::set_device`):

- **CPU**: `batch_loss_and_gradients` + `adam_step`.
- **GPU — DirectX 12 / Vulkan / Metal** (`Device::Gpu(Api)`): one set of compute shaders
  (`gpu/kernels.wgsl`) and one recording of the network (the model's forward pass written once
  over the `Graph` trait), translated by wgpu for the chosen API. On Windows both DirectX 12
  and Vulkan are offered; the Windows build carries Microsoft's DXC shader compiler (feature
  `dxc`), because without it DirectX 12 falls back to the old FXC compiler, which compiles the
  kernels' loops poorly. Weights, gradients and AdamW state stay on the card during a run and
  are copied back before every check and save.
- **Hybrid — CPU + GPU** (experimental, `Device::Hybrid(Api)`): data-parallel. Each batch's
  rows are split between the GPU and the CPU in proportion to their measured speed (adjusted
  after every step); both compute their part's loss and gradient divided by the whole batch's
  weight, the parts are added (the batch's exact mean gradient), the CPU takes the AdamW step
  exactly as CPU training does, and the new weights are sent to the GPU. Splitting the brain
  itself (Dense on one device, the transformer on the other) was considered and left out:
  the hemispheres exchange activations in every block, so every step would cross the PCIe bus
  dozens of times each way.

**Failures.** If the GPU cannot be opened, training runs on the CPU and the Studio says why.
If a GPU step fails during a run, the model is read back from the GPU and training continues on
the CPU (in hybrid training nothing needs reading back); if the GPU's state cannot be read, the
run stops with an error and saved checkpoints are untouched. GPU errors the driver reports
(validation errors, a lost device) and crashes inside a GPU step are caught and turned into this
fallback with the error's text; they never end the training thread silently. In the Studio a
crash anywhere in teaching, reading or a plan ends the run with a message, never a frozen page.

**Both processors at once in GPU training**: lessons score each Foundation candidate on the CPU
while the GPU trains the next; reading computes a checkpoint's held-out losses on the GPU
(forward passes only, seconds) and writes its samples on the CPU while the GPU keeps reading.
Held-out losses on the CPU also run forward passes only (no gradients), about a third of the
work of a training step. Talking, exams and grading run on the CPU. `learn::set_cpu_threads(n)` limits the
CPU threads (0: all).

**Kernels.** Matrix products use 64 x 64 tiles with each thread's 4 x 4 results in vector
registers. Attention takes 64 query rows of one sequence and head per workgroup and reads keys
and values in tiles through workgroup memory, with a single-pass (running-maximum) softmax; for
the head size every Yendory size uses (32) the kernels hold a head in eight vec4 registers
instead of indexed per-thread arrays. The output layer's loss uses one 256-thread workgroup per
row; bias and norm-gain gradients sum rows in parallel chunks; the embedding gradient sums each
token's rows in parallel (rows sorted by token on the CPU). Input buffers are reused between
steps. All of this computes the same values as the CPU, only in a different order of
additions (see docs/STATUS.md).

**Benchmark** (`learn::bench::run`, Studio: Models → "Benchmark training on this computer").
The open model's copy is trained on every available device with identical batches (the Reading
settings' batch size and window): tokens per second, time per step, GPU memory, the hybrid
split it settled on, and the first step's loss compared with the CPU's (a device that disagrees
is never chosen). The fastest GPU is also profiled: CPU time preparing a step and the GPU time
of every kernel (GPU timestamps when the driver has them). The training device "Auto" then uses
the fastest measured device; hybrid is only chosen if it measured fastest.
`examples/gpu_profile.rs` and `examples/train_bench.rs` do the same from the command line.

## Yendory Nexus: actions and results (`nexus`)

Nexus is how a model takes actions in an environment (tool calls, game moves, later code) and
gets results back. It is part of the engine, not of the weights: a model file is unchanged and
nothing in it says Nexus exists. An environment announces itself in the context, and a model
uses it only if it was taught to. No tools or games ship yet; this is the infrastructure.

**Markers.** Three ASCII control characters mark Nexus lines: `\x02` an environment's
announcement, `\x05` an action the model writes, `\x06` a result Nexus writes. A turn looks
like this (shown readably):

```
Environment: Room: A room with a lamp. Actions: look - see the lamp; switch <on|off> - ...
U: is the lamp on?
N: Action: look
Result: The lamp is off.
No, it is off.
```

Outside text never carries markers: typed messages, lesson and exam files and results are
sanitized, and during generation the model cannot write a result or an announcement (an action
marker counts only at the start of a line). Only the model itself can request an action.

**Environments** implement `nexus::Environment` (`manifest`, `act`). A manifest names the
environment, says whether it is a game or a tool set, and lists its actions, each with the
access it needs: look, act or change. `Manifest::announcement` renders it in the one form
lessons and live sessions share.

**Permissions.** A rule per access: allow, ask or deny. Defaults: looking is allowed; acting
is allowed in a game and asked with tools; changing anything is asked. Asking goes to an
`Approver` (the person running the session); with no one to ask, the action is denied. A
denied or unknown action is answered with a result (`not permitted: save`), so a model can
learn from it.

**Limits** per turn: 8 actions, 64 tokens per line, results cut at 400 characters. Every
action is logged with its decision, result and time. Time limits and code sandboxing come
with the code environment (a WebAssembly interpreter with fuel and memory limits).

**Teaching.** In a Multi-Turn Conversation session, `Environment:`, `Action:` and `Result:`
lines become marked lines. The model is trained on its action lines and replies, never on
environment or result lines (loss weight 0: they come from the environment). Evaluation
checks every action and reply with the lesson's own results before it.
`tests/nexus.rs` teaches a small model a lamp environment that exists only in the test: it
then looks before answering and reports the room's actual state, its switch changes the
environment, saving is refused without approval and allowed with it, and without an
announcement it does not act.

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

## Bicameral education (experimental)

An optional way to teach each hemisphere its intended role: Dense as the store of learned
knowledge and associations, the transformer as the language and reasoning centre, and the
interconnect as the cable between them. It only changes which parameters may learn in a session
and adds one guard; the network, the training objective (next-token prediction) and the
checkpoints are unchanged, and nothing is written into the model by rules.

| Stage | Who learns | Why it teaches the role |
|---|---|---|
| A: Dense knowledge | Dense + interconnect (`Focus::DenseAndInterconnect`); transformer frozen | The frozen transformer cannot store what the text teaches, so lowering the loss on knowledge ("a dog is a kind of …") is only possible by Dense storing it and the interconnect carrying it. The cable learns with Dense so it can carry Dense's code; it has no hidden layer of its own (a gated linear projection), so it routes rather than stores. |
| B: Transformer language | The transformer; Dense and interconnect frozen but still sending | The transformer learns grammar and in-context reasoning (premises in the prompt) alongside Dense's signals, not around them. |
| C: Cooperation | Each epoch tries Dense + interconnect and the whole brain, keeps the better | Both learn together; a candidate that stops using Dense is rejected. |

**Foundation in a stage.** With Foundation on, each part the stage may train is also tried on
its own, and every candidate is scored as a whole brain: stage A tries Dense, the interconnect,
and Dense + interconnect; stage B the transformer; stage C Dense, the transformer, the
interconnect, Dense + interconnect and the whole brain. For every candidate the record shows the
whole brain's held-out loss and exact answers, the transformer alone (Dense's input cut), and
Dense's contribution (the difference); after every epoch a summary shows the whole brain as
accepted. In stage A the transformer-alone loss stays exactly the same for every candidate (the
transformer is frozen), which is a useful check that the masks hold.

**Dense reliance** is measured, not inferred: the rise in held-out loss when Dense's input to the
transformer is cut (the same ablation as the brain scan). Reading reports it at every check;
lessons report it for every candidate in an education stage. In every stage a checkpoint or
candidate is rejected if its Dense reliance falls below half of the accepted model's
(`keep_dense_reliance`). The Exams page grades a file with and without Dense, so the difference
shows what the answers owe to Dense.

Measured on a tiny synthetic fact corpus ("A dog is a kind of animal. …", 60,000 tokens, an
untrained Nova-2-size model): reading with the whole brain reached held-out loss 0.29 with Dense
reliance 0.40 — cutting Dense left most answers intact, so the transformer held the facts. Stage A
reached 1.68 with reliance 4.11: the facts were answered correctly ("animal", "flower",
"vehicle") and cutting Dense turned them into gibberish, so they lived in Dense. One small
experiment, not a general result.

Limits: Dense reads only the last 64 characters as character bits with separate weights per
position, so it learns facts tied to how words look in that window rather than word-level
concepts; word embeddings belong to the transformer and stay frozen in stage A (new word units
added during stage A keep their initial rows until the transformer trains). Specialisation is
encouraged, not guaranteed: joint training can move knowledge into the transformer, which the
reliance measurement and guard make visible.

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

