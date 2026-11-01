# Yendory Hub

The front door of the Yendory suite, for Windows and Linux. Make or import a model, keep your
models on shelves, and open them in Yendory Studio and the other programs. The Hub installs and
updates those programs too.

| Part | What it does |
|---|---|
| **The shelf** | Every model stands as a notebook. Spine height follows the model's size; the coloured band shows the architecture (blue and red for Bicameral, teal for Neuro); a red ribbon means it is partway through a training plan in Studio. Pick a book to pull it out. |
| **The card** | The pulled-out model: steps, parameters, words, context, what it read, lessons taught, its training plan, and its newest events from Studio and the Hub. From here: Open in Studio, Move to shelf, Rename, Export, Delete. |
| **Sidebar** | Search, your shelves (make, rename and remove them; a removed shelf's models move to the first), the programs with their state (installed, open, update), Downloads & updates, Settings. It folds into a rail of marks. |
| **New model** | Name, architecture (Bicameral or a Neuro preset), size, vocabulary and context, made by the Engine exactly as Studio makes them. Studio's builder makes Neuro designs of your own. |
| **Downloads & updates** | Installs or updates Studio from the release page (Windows: its installer; Linux: an AppImage kept in the Hub's folder). Every download is checked against the project's updater key before it runs. The Hub updates itself the same way. |
| **Settings** | The models folder and the Hub's folder, and where each program was found (choose its file if it is somewhere else). |

Models live in Studio's models folder, and every program reads them from there; see
[docs/HANDOFF.md](../../docs/HANDOFF.md) for the folder and how a program is started with a
model (`--model <id>`). The Hub never trains a model; Studio does.

Rename and Delete are refused while Studio is running, because Studio would undo the change when
it saves the model it has open. Opening a model in Studio while Studio is already open is
refused too (two copies of Studio would write the same files).

## Build

```
cd apps/hub
npm ci
npx tauri dev          # run from source
npx tauri build        # installers for this computer
```

Linux needs `libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev` (and `patchelf` for AppImage).
`cargo test --release` in `src-tauri` checks model creation against the Engine, Studio's records,
shelves, version comparison and signature checking.

## Releases

`.github/workflows/hub-release.yml` builds a Windows installer and Linux `.deb` / `.AppImage`
and publishes them as a pre-release: `hub-v<version>-dev.<run>` for every push that changes the
Hub or the Engine (the newest 5 are kept), or a pushed `hub-vX.Y.Z` tag. With the updater key
secret, it also writes `latest.json` to the `hub-updates` release for the Hub's own updates.

## Known limits

- Studio does not read `--model` yet: it starts on § 1 Models and the model is chosen there.
- The Hub is desktop only. On Android every app's data is private, so a Hub there couldn't see
  Studio's models.
- Studio installed from a `.deb` is updated by installing the new `.deb`; the Hub shows it but
  doesn't replace system packages.
