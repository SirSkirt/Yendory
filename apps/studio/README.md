# Yendory Studio

The desktop and Android app for creating, teaching and examining Bicameral models. It is a thin
GUI over the Yendory Engine (`crates/yendory-engine`): every reply, score and measurement is
computed by the Engine; the window only displays them.

| Section | What it does |
|---|---|
| **1 Models** | Create an untrained model with any name, import a `.nnova2` checkpoint (e.g. Nova), open, rename, delete, export (`.nnova2`, verified after writing). |
| **2 Reading** | Pre-train on a text corpus before the lessons: download TinyStories (or the first part of its training file) or use any `.txt` file. One document in ten is held out; only checkpoints that lower the held-out loss are kept, and every check shows a sample written by the model. Corpora are stored in `corpora/` with their fingerprint. |
| **3 Classroom** | Add lesson files (`.txt`) or a course (`.zip`), see each file's taught / held-out units, then teach with Foundation candidates, rehearsal and answer protection. Live progress shows every candidate of every epoch and why it was or was not selected. The model is saved after every learned file. |
| **4 Exams** | Talk to the model; grade a lesson file as an exam (flags questions the model was taught). |
| **5 Brain** | Neural Nova's hemisphere shapes in the notebook colours, driven by real measurements: attention per block and thinking round, gate openings, right-hemisphere activations, and how much each pathway changes the output. Below it, the **thinking tree** of a reply: each token chosen and the alternatives weighed, what the model would have said after each thinking round, the dense hemisphere's influence, and where close runner-ups would have led. Any step opens in the brain scan. |
| **6 Library** | The open model's record: what it read, each lesson file's fingerprint, steps and outcome, and each epoch's candidates. |

Models live in the app's data folder (`models/<id>/model.nnova2`, `meta.json`, `archive.json`,
`history.jsonl`). Uninstalling the app can delete that folder: export models you want to keep.

## Build

```
cd apps/studio
npm ci
npx tauri dev          # run from source
npx tauri build        # installers for this computer
```

Linux needs `libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev` (and `patchelf` for AppImage).

## Releases

`.github/workflows/studio-release.yml` builds a Windows installer, Linux `.deb` / `.AppImage`
and an Android `.apk` on GitHub's machines and publishes them as a **pre-release** on the
Releases page:

- every push that changes Studio or the Engine → `studio-v<version>-dev.<run>` (the newest 5
  development pre-releases are kept, older ones are deleted);
- a pushed tag `studio-vX.Y.Z` → a pre-release with that tag;
- a manual run from the Actions tab → the tag you enter, or a development tag.
Android builds are all signed with one committed debug key (`keys/README.md`); no secrets needed.

## Known limits

- Android keeps teaching only while Studio is open in the foreground (no background service yet).
- Lessons must be ASCII text (the tokenizer is ASCII).
- Long conversations (over 256 tokens) re-read the whole context window for every token: slow on phones (decision D10).
