# Settings

Studio keeps its settings in `settings.json` in its data folder (on Windows
`%APPDATA%\org.yendory.studio`, the same folder the models, datasets, chats and plans are in;
updating installs over Studio and keeps it).

## What is kept

| Where | Kept |
|---|---|
| § 1 Models | training device, CPU threads, GPU checks, fitted reply limits, the measured benchmark, the recovery-point interval |
| § 2 Reading | every field, the reading mode and stage map, the ticked practice families, each corpus's tick and share |
| § 3 Classroom | every field and switch (safeguards, loss allowances, answer-based checks, comparison variants…), and the lesson list (`classroom_lessons.json`) |
| § 4 Conversation | every field and switch (the quality filter's switches among them), which datasets are ticked (a dataset added later is ticked, as before), and each one's "use up to" size |
| Courses, Planner | each course and plan in its own file, with its steps, settings, status and position |
| Study, Rewards, memory | their own settings keys |

Fields save themselves when changed (after a short pause); there is no Save button. A saved value
that a control cannot hold any more (a choice that no longer exists, a number out of range) is
not applied: the control keeps its default and the diagnostics log says so.

A run's settings are kept with the run (docs/RECOVERY.md): changing a page while a run is
interrupted does not change what the run continues with.

## Versions, backups, damage

- `schemaVersion` says which layout the file has (2 since 0.8.2). Opening an older file keeps a
  copy of it (`settings.v1.json`) and adds the version; nothing else changes. Doing it again
  changes nothing. Settings a newer Studio wrote are kept as they are.
- Every change keeps the file as it was before as `settings.bak.json`.
- A `settings.json` that cannot be read is never written over. Studio uses the backup (or, with
  none, its defaults), says so, and when a setting is next saved keeps the damaged file as
  `settings.damaged-<time>.json`.

The Hugging Face token is not a setting: it is kept in the system's credential store
(docs/HUGGINGFACE.md).
