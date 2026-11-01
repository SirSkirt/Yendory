# Yendory Hub

The front door of the Yendory suite, for Windows and Linux, and its installer. Make or import a
model, keep your models on shelves, and open them in Yendory Studio and the other programs. The
Hub installs and updates the Yendory Network, Studio and itself, in its own window.

| Part | What it does |
|---|---|
| **The shelf** | Every model stands as a notebook. Spine height follows the model's size; the coloured band shows the architecture (blue and red for Bicameral, teal for Neuro); a red ribbon means it is partway through a training plan in Studio. Pick a book to pull it out. |
| **The card** | The pulled-out model: steps, parameters, words, context, what it read, lessons taught, its training plan, which program has it open, and its newest events. From here: Open in Studio, Move to shelf, Rename, Export, Delete. |
| **Sidebar** | Search, your shelves, the programs with their state (installed, open, update, on its own), Downloads & updates, Settings. It folds into a rail of marks. |
| **New model** | Name, architecture (Bicameral or a Neuro preset), size, vocabulary and context, made by the Yendory Network exactly as Studio makes them. |
| **The Yendory Network** | The background service that runs the Engine. On first start the Hub offers to install it; every model is listed, made and changed through it (docs/NETWORK.md). |
| **Downloads & updates** | The Hub, the Network, Studio and Tools: installed version, newest, and Install / Update / Start / Remove. Every package's signature is checked before it is installed (docs/PACKAGES.md). |
| **Switch to the Hub** | A Studio installed on its own (its installer, a `.deb`, the Hub 0.1 AppImage, or a file you chose) is found, and the Hub offers to replace it with its own copy. Studio's data is moved out of reach first and put back after. |
| **Settings** | The models, programs and Hub folders, the Network's version, and where each program was found. |

The Hub holds no Engine of its own: it talks to the Yendory Network (`dependency/client`).

## Build

```
cd apps/hub
npm ci
npx tauri dev          # run from source (set YENDORY_NETWORK to a built yendory-network)
npx tauri build        # installers for this computer
```

Linux needs `libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev` (and `patchelf` for AppImage).
`cargo test --release` in `src-tauri` checks the card's records, shelves, package unpacking,
signatures and the switch's data handling. The Network's own tests run from the repository root:
`cargo test --release -p yendory-network`.

A build with `--features dev-overrides` reads `YENDORY_RELEASES` and `YENDORY_UPDATER_KEY`, to test
installing, switching and updating against a local web server and a test key
(docs/PACKAGES.md). Release builds never have it.

## Releases

`.github/workflows/hub-release.yml` builds the Hub's installers (Windows `-setup.exe`, Linux
`.deb` and `.AppImage`) as a pre-release `hub-v<version>-dev.<run>`, the Hub's and the Network's
signed packages, and `package.json` on `hub-updates` and `network-updates`.

## Known limits

- Studio doesn't work through the Network yet, so while it runs the Hub refuses renames and
  deletes (Studio would undo them), and its training doesn't show live in the Hub.
- Studio publishes no Hub package yet: installing or switching it waits for its first one.
- Desktop only: on Android every app's data is private, so a Hub there couldn't see Studio's
  models.
- A Hub installed from the `.deb` updates with the system's password prompt.
