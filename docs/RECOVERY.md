# Recovery: leaving Nova training for hours

Since 0.8.1, Reading and Conversation practice save **recovery points** while they train, and
Studio asks what to do with a run that was interrupted (the app closed or crashed, Windows
restarted, the power went). This page says what is saved, what a continued run is, and what is
not covered yet.

## What is saved, and when

| File | Written | What it is |
|---|---|---|
| `models/<id>/model.nnova2` | when a check **accepts** a checkpoint | the accepted model (unchanged by anything below) |
| `models/<id>/runs/<run>/run.json` | at the start, at every recovery point, at the end | the run record: kind, model, the settings it was started with, datasets with SHA-256, progress, status, the two newest recovery points with their files' SHA-256 |
| `models/<id>/runs/<run>/slot-a/`, `slot-b/` | every *n* minutes (§ 1 Models, "Recovery point every", default 10) and when the run is stopped | `working.bin`: the working model with its AdamW moments and step; `state.json`: the run's position; `extra.bin`: Conversation's best balanced checkpoint, when that option is on |

The run's position (`learn::corpus::ReadingState`, `learn::conversation::ConversationState`) is
everything the run needs to carry on as if it had never stopped: the step reached, the random
generator's state (so the next batches are the ones the uninterrupted run would have drawn), the
interleaving of corpora, the judge's best loss, misses, accepted and balanced checks, the
curriculum stage with its learning-rate back-offs, the practice pool, the training-loss sums
since the last check, the baseline measurements, and the settings with a fingerprint of every
document read.

Before a recovery point is written, Studio checks the free space on the disk: the point is
skipped unless the disk keeps room to save the accepted model again plus 256 MB, so a recovery
point never fills the disk the model is saved on. A point that is skipped or cannot be written is
reported and counted on the run; training continues and the previous point stays valid.

Recovery points alternate between the two slots, so writing one never touches the newest
complete one. Each file is written to a temporary name, flushed to the disk and renamed (on Linux
and macOS the folder is flushed too); the run record is written last and names the files with
their SHA-256. A point whose files do not match their sums is not used: the one before it is
(Studio says so), and if both are damaged Studio says that too. The accepted model is never
replaced by a recovery point.

## Interrupted training

A run record still marked as training when Studio starts belongs to an interrupted run. Studio
shows **Interrupted training detected** with the model, the kind of run, when it started, its last
recovery point and progress and its datasets, and offers:

- **Continue training**: opens the model and continues from the recovery point, with the run's
  own settings (not the page's current ones) and seed;
- **Decide later**: it stays in § 1 Models, *Unfinished training*;
- **Discard…** (asks first): deletes its recovery points; the model keeps everything accepted.

Studio never continues or discards a run by itself. A stopped run (Stop pressed) has a recovery
point too and can be continued the same way. While a model has an unfinished run, other training
on that model is refused with a message (it would make the recovery point no longer fit the
model); continue or discard it first.

A step of a plan is continued by **Resume** on the Planner: the step continues from its own
recovery point instead of starting again.

## What "continues exactly" means

Before continuing, the run checks that the settings are the same, that the datasets are the same
(their fingerprint, and Studio checks every file's SHA-256), that the recovery point's model fits
the saved model, and that the saved model is either the one the run had accepted at that point or
a checkpoint it accepted afterwards. Anything else is refused with a reason.

Tested: on the CPU a reading or a conversation run stopped or killed after any recovery point and
continued ends with **the same weights, optimizer state and checks, bit for bit**, as a run that
was never interrupted (`crates/yendory-engine/tests/recovery.rs`; Conversation through a
curriculum stage change with a new practice pool pending; Studio's own files read back from disk
after a simulated crash in `apps/studio/src-tauri/src/recovery.rs`). On a GPU the arithmetic is
not bit-exact from one run to the next anyway; a continued run agrees with an uninterrupted one to
floating-point rounding and checks may fall a few steps differently while one is still being
measured.

A checkpoint accepted after the last recovery point (saved before the interruption) is measured
again when the run continues and kept as the best, so it is never lost or overwritten by a worse
one. For Conversation, the curriculum continues from the recovery point's stage.

The training since the last recovery point is lost: at most one interval (10 minutes by default),
or nothing when the run was stopped.

## Not covered yet

- **Classroom** keeps its own recovery from 0.6 (each accepted epoch's checkpoint and schedule in
  `checkpoints/`; restore the newest on § 1, then "Continue the interrupted run"): an epoch
  interrupted part-way starts again. At start-up Studio now says when a model has such
  checkpoints waiting (*Interrupted Classroom session*) and opens the model on request.
- Downloads and imports are not part of a run's recovery (0.9 stage 4).
