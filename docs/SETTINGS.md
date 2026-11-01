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

## Where each setting is (0.9.2 layout)

Studio has a sidebar with three sections, a gear for Global Settings, and a round Chat button.
Models are opened by Yendory Hub; Studio's Models page is kept but not in the sidebar (until
the Hub opens models in Studio, *Open a model…* appears at the top when none is open).

| Place | Settings | Applies to |
|---|---|---|
| Global Settings ▸ Appearance | theme (system, light, dark), interface scale, reduced motion (`appearance` in the settings file) | Studio, this computer |
| Global Settings ▸ Hardware and performance | training device, CPU threads, checking answers on the GPU, fitting the reply limit to each answer, the training benchmark, memory limits | Studio |
| Global Settings ▸ Training safety | recovery point interval | Studio (Reading and Conversation runs) |
| Global Settings ▸ Accounts | Hugging Face token (in the system's credential store) | Studio |
| Global Settings ▸ Updates, Notifications and logs | update check; Studio's message log | Studio |
| Train ▸ Reading ▸ Reading settings | tokens, batch, learning rate, window, training mode, words learned | Studio; a run keeps its own copy |
| Train ▸ Classroom ▸ Classroom settings | epochs, batch, learning rate, pass mark, safeguards, probation and the rest of the Classroom's options | Studio; a run keeps its own copy |
| Train ▸ Conversation ▸ Conversation settings | the quality filter at practice, curriculum, practice and checks | Studio; a run keeps its own copy |
| Datasets ▸ … ▸ Datasets settings | resuming the download queue when Studio starts | Studio |
| Analyze ▸ Brain ▸ Brain settings | measuring influence by default, paths not taken; saved traces (per model) | Studio / the model |
| Chat ▸ Settings (inside the chat) | each chat's history, reply length, sampling, memories, brain-trace recording | that chat |

Which corpora a reading run reads and which datasets conversation practice uses are ticked on
the Datasets pages. Moving the controls changed no setting and no behaviour: every control keeps
its id and is saved under the same key as before.

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
