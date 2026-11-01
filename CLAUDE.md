# Notes for development sessions

## Branches and parts

This is the **`yendory-tools`** branch: Yendory Tools (`apps/tools`), where trained models play
games (and later use apps) through Nexus (docs/GAMES.md). The suite's other parts live on their
own branches: Yendory Hub and the shared `dependency/` (the Engine, the Yendory Network, the
client library) on `yendory-hub`, Yendory Studio on `yendory-studio`.

Tools holds no copy of the Engine or the Network: it uses `yendory-client` and `yendory-engine`
from yendory-hub, pinned to one commit in `apps/tools/src-tauri/Cargo.toml`; move the pin on
purpose. Every model is opened and every line it writes comes through the Network. Tools uses the
Engine only for what needs no model (Nexus's announcements and actions, checking lessons). A
change Tools needs in the Network is made on yendory-hub by whoever owns the Network there.

Models are handed between programs through Studio's models folder and `--model <id>`
(docs/HANDOFF.md on yendory-hub). Yendory Hub installs and updates Tools from its signed package
(docs/PACKAGES.md on yendory-hub); Tools has no installer and doesn't update itself.

## Version numbers

Tools is at **0.1.0**. A push that changes Tools raises its last digit by one, and the last digit
carries over after 9: 0.1.9 → 0.2.0 → … → 0.9.9 → 1.0.0. The new number goes in the same push,
in every place that holds it:

- `apps/tools/package.json`
- `apps/tools/package-lock.json` (the two entries for `yendory-tools`)
- `apps/tools/src-tauri/tauri.conf.json`
- `apps/tools/src-tauri/Cargo.toml`
- `apps/tools/src-tauri/Cargo.lock` (update with cargo, e.g. `cargo metadata`)

The release workflow publishes Tools only when its version is new (`tools-v<version>` and
`package.json` on `tools-updates`).

## Switches must work

When the owner asks for something to be switchable (a checkbox, an option, a setting), the
switch must change what actually happens. Never build another path that does the same job
regardless of the switch (for example a download that already removes what a practice-time
checkbox is meant to leave out), so that the switch becomes decorative. If one switch only works
together with another, keep them consistent in the page and say so beside them, or disable it
with the reason. When an existing safeguard or rule makes a new switch inert, raise it with the
owner instead of shipping it.
