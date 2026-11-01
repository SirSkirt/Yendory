# Yendory packages: how the Hub installs and updates the suite

The Hub is the only thing people download (its `-setup.exe`, `.deb` or `.AppImage`). Everything
else, and the Hub's own updates, arrive as **packages**: a signed zip and a `package.json`
describing it. The Hub installs them itself, in its own window, without any installer of their
own.

## package.json

Each component publishes the newest on its fixed release `<id>-updates`
(`https://github.com/SirSkirt/Yendory/releases/download/<id>-updates/package.json`):

```json
{
  "format": 1,
  "id": "studio",
  "name": "Yendory Studio",
  "version": "0.9.4",
  "date": "2026-11-01T15:06:15Z",
  "notes": "…",
  "requires": { "network": "0.1.0", "hub": "0.1.1" },
  "platforms": {
    "windows-x86_64": { "url": "…/yendory-studio_0.9.4_windows-x86_64.zip", "signature": "…", "size": 123, "kind": "zip", "run": "yendory-studio.exe" },
    "linux-x86_64":   { "url": "…/yendory-studio_0.9.4_linux-x86_64.zip",   "signature": "…", "size": 456, "kind": "zip", "run": "Yendory-Studio.AppImage" }
  }
}
```

- `id`: `hub`, `network`, `studio`, `tools`. A missing `package.json` (404) means not released yet.
- `requires`: the oldest version of other components it works with. The Hub installs or updates
  those first; `hub` it can't, so it asks the owner to update the Hub first.
- `platforms`: `windows-x86_64`, `linux-x86_64`; for the Hub itself also
  `linux-x86_64-appimage` and `linux-x86_64-deb` (it updates the way it was installed).
- `signature`: the minisign signature from `tauri signer sign` with the project's updater key
  (repository secret `TAURI_SIGNING_PRIVATE_KEY`). The Hub refuses anything unsigned or signed
  otherwise.
- `run`: the file to start, a plain path inside the zip.
- An optional `icon.png` beside it is used for the menu shortcut.

`.github/scripts/package.py` makes the zip (keeping executable bits) and `package.json`.

## What a program in a package should be

- **Windows:** the program's `.exe` from `tauri build` (`target/release/<name>.exe`), plus any
  files it needs beside it. It must not need an installer: no registry entries of its own; data
  in its Tauri data folder (`%APPDATA%\<identifier>`), never beside the program. WebView2 is
  there already: the Hub's installer makes sure of it.
- **Linux:** the program's `.AppImage` (it carries its own libraries), named in `run`.
- It must not update itself: the Hub does (Studio 0.9.3 already hides its own updater).
- Started with `--model <id>`, it opens that model (docs/HANDOFF.md).

## How the Hub installs one

1. Downloads the zip for this computer and checks its signature (and size).
2. Unpacks it into `programs/<id>/<version>.partial/` in the Hub's local folder
   (`%LOCALAPPDATA%\org.yendory.hub\programs` on Windows, `~/.local/share/org.yendory.hub/programs`
   on Linux), refusing paths outside it, then renames it to `<version>/`.
3. For the Network: runs `yendory-network --version` to check it is that version, asks the old
   Network to stop (refused while a program uses it), and writes `install.json`.
   For a program: refused while it runs; writes its Start menu shortcut (Windows) or
   applications-menu entry (Linux).
4. Records it in `programs/programs.json`, and removes the older version's folder.

Nothing is changed until step 4: a failed download, a bad signature or a refusal leaves the
installed version as it was.

## The Hub updating itself

The Hub downloads its own package the same way, then replaces itself in place: the running file
steps aside (`.old`, removed at the next start), the new one takes its name, and "Restart to
finish" starts it. Windows: the per-user install folder; the Apps & features version is updated
too. Linux AppImage: the AppImage file. Linux `.deb`: the new `.deb` is installed with the
system's password prompt (`pkexec dpkg -i`), the only window that isn't the Hub's.

Hub 0.1.0 updated through Tauri's updater; `latest.json` on `hub-updates` stays so it can reach
this Hub once.

## Switching a program installed on its own

The Hub finds a Studio installed by its own installer (Windows), a system package (`.deb`), the
Hub 0.1 AppImage, or a file chosen in Settings, and offers to switch it (switch.rs):

1. downloads and checks the Hub's package first (a failure changes nothing);
2. moves Studio's data folders aside (a rename; recorded in `switch.json` so an interrupted switch
   is put right at the next start);
3. removes the old copy (Windows: its uninstaller, silently; `.deb`: `pkexec apt-get remove`);
4. puts the data folders back;
5. installs the Hub's copy and its menu shortcut.

Models, datasets, chats and settings are never in reach of the old uninstaller.

## Testing without the release page

A Hub built with the `dev-overrides` feature (never in releases) reads `YENDORY_RELEASES` (another
base address for the `<id>-updates/package.json` files) and `YENDORY_UPDATER_KEY` (another public
key). With a test key from `tauri signer generate`, `package.py` and a local web server, the whole
flow (install, switch, update, self-update) runs on one machine.
