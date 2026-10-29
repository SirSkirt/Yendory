# Yendory Studio 0.9.0: audit and plan

Audit of the repository at 0.8.0 (2026-10-29), against the 0.9.0 request (reliability, Hugging Face
datasets, updates). "Exists" means it is in the code and tested; "partial" means part of what was
asked is there.

## What exists today

| Area | State |
|---|---|
| **Checkpoint file** (`.nnova2` / `.nneuro`, `model/checkpoint.rs`) | Weights, **AdamW moments**, step and learned words in one file. Every save already carries the optimizer state. Written atomically by Studio (`write_atomic`: temp file, `sync_all`, rename). |
| **Classroom recovery** | Exists at the **epoch boundary**: every accepted epoch is saved to `models/<id>/checkpoints/accepted-step-…-epoch-NN.nnova2` with its schedule (`CheckpointInfo`: epoch, learning rate, back-off); Studio offers "Continue the interrupted run" (`meta.resume`). Rejected epochs are never saved over the accepted model. The random seed is not saved (a resumed run continues the schedule, not the exact batch order). |
| **Reading recovery** | Partial: each **accepted** check is saved to the model at once (`on_accepted`), so a crash loses at most the work since the last accepted check, but a new run starts from scratch (tokens already read are not counted; no "continue"). Nothing is saved between accepted checks. |
| **Conversation recovery** | Partial: like Reading (accepted checks saved at once); no continue, the curriculum stage is not kept. |
| **Planner** | Partial: plans and each step's status are saved (`plans/<id>`); Resume runs the stopped step again from the model's best checkpoint. |
| **Settings** | Device settings and the Reading page in `settings.json` (app data folder); Classroom and Conversation pages since 0.8.0; course settings in each course; plans in their files. Not kept: which conversation datasets are ticked, Courses page defaults outside a course. No schema version. |
| **Hugging Face token** | Partial: one token, **stored in plain text** in `settings.json`; sent only to huggingface.co. No account check, no disconnect beyond clearing it. |
| **Downloads** | The 0.7.9 queue (one at a time, waiting list, remove, cancel, no duplicates) is **in memory only**: a restart loses it. Datasets come through the dataset viewer's `/rows` API (JSON pages of 100 rows), not files, so "range requests" do not apply; the natural resume point is the row offset. 501 is final (0.7.8). |
| **Imports** | Partial: a `.txt` corpus for Reading; conversation files `.txt` / `.json` / `.jsonl` with automatic layout detection. No Parquet, CSV, ZIP, Hugging Face repository import, inspection or column mapping. |
| **Updater** | Partial: plugin, signature check, commands and page controls are in 0.8.0, and the release workflow signs and publishes `latest.json` when the secret is present. **The 0.8.0 build was not signed** (the workflow saw no `TAURI_SIGNING_PRIVATE_KEY`), so no update is published yet and nothing has been tested end to end. |
| **Training history** | Partial: every event goes to `models/<id>/history.jsonl` and each model's `meta.json` lists what it was taught, read and practised. No run IDs, no history page, no comparison, no checkpoint names or protection. |
| **Dataset library** | Partial: the Reading page lists corpora on the device (size, documents, licence, delete); the Conversation page a table with filter results. No enable/disable, reimport, inspection or dependency protection. |

## Recovery guarantees that can honestly be given

- **Never lose more than one recovery interval of training** (default 10 minutes, a setting), in
  Reading and Conversation, by saving the *working* model (weights + AdamW state + step) with
  its progress (tokens done, stage, best loss) atomically, separate from the accepted model.
- **Classroom**: resume from the last accepted or last completed epoch (exists; extended with the
  run record and the recovery dialog). Mid-epoch recovery is not offered: an epoch trains several
  candidates from one plan and judges them together.
- **Planner**: resume at the step that was running, which itself resumes as above.
- **Not exact**: the batch order after a resume is not the order the uninterrupted run would have
  used (the data are drawn with a new random seed), so a resumed run is equivalent in kind, not
  identical step for step. Stated in the UI. The accepted model is never replaced by a recovery
  snapshot unless it passes the same checks.
- A crash during a save cannot destroy the previous good save: every file is written to a
  temporary name, synced, and renamed over the old one; the recovery record is written last.

## Order of work (each stage pushed and tested on its own)

1. **Recovery** (critical): run records for every training run (mode, model, settings snapshot,
   dataset fingerprints, progress), periodic recovery snapshots for Reading and Conversation,
   "Interrupted training" on start-up with details / resume / new run / discard (confirmed),
   validation before resuming (files present, fingerprints, model compatibility, disk space).
   Engine support for continuing a Reading or Conversation run (tokens done, best loss, stage,
   schedule position). Tests: kill-and-resume simulations in the Engine, atomic-write tests.
2. **Settings**: a versioned settings schema with migration, the remaining selections (ticked
   datasets), and the active run's own settings snapshot kept apart from the page's.
3. **Hugging Face**: token moved to the operating system's credential store (Windows Credential
   Manager, the Secret Service on Linux; plain-file fallback only where none exists, said so),
   account check (`/api/whoami-v2`), disconnect, gated-dataset messages.
4. **Downloads**: the queue saved to disk and restored (not restarted without asking), resume from
   the saved row offset, pause / retry / reorder, error handling per status.
5. **Imports**: a Hugging Face repository or file URL, local files and ZIP (JSON, JSONL, CSV,
   TXT, Parquet), inspection (files, splits, columns, sample rows), column mapping with a preview,
   validation and provenance. Python loading scripts are never run.
6. **History and checkpoints**: run list, details, comparison of two runs, named and protected
   checkpoints.
7. **Dataset library** and **updater** hardening; release notes and documentation.

## Progress

- **0.8.1, stage 1 (recovery)**: done for Reading and Conversation, and for plan steps of either
  kind (docs/RECOVERY.md). Run records, recovery points on an interval and on stop (two
  checksummed slots), the *Interrupted training detected* dialog, *Unfinished training* on § 1,
  continuing with the run's own settings, other training on that model refused until the run is
  continued or discarded. Continuing is exact on the CPU (tested bit for bit, including a crash
  read back from disk). Classroom's own epoch checkpoints are announced at start-up too, and a
  recovery point is skipped (and reported) when it would leave too little free disk space.

## Blockers and limits

- **Updater secret**: the builds did not receive `TAURI_SIGNING_PRIVATE_KEY`. It must be a
  *repository* secret (Settings → Secrets and variables → Actions → **Repository secrets**, not an
  environment secret or a variable), named exactly `TAURI_SIGNING_PRIVATE_KEY`.
- **End-to-end tests on Windows** (installing over a release, force-killing Studio, rebooting)
  cannot be run from the development container; they are simulated in tests and need a check on
  the owner's PC.
- **Hugging Face sign-in through the browser** needs an OAuth app registered by the owner on
  Hugging Face; until then, an access token (read-only scope recommended) is the supported way.
