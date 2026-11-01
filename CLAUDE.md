# Notes for development sessions

## Branches

This is the **`yendory-hub`** branch: Yendory Hub (`apps/hub`) and the Engine. Yendory Studio
lives on the `yendory-studio` branch. The Engine is shared: change it on `yendory-studio` and
bring the change here, so the two copies don't drift. Models are handed between the programs
through Studio's models folder (docs/HANDOFF.md); keep that contract when changing either side.

## Version numbers

The current Hub version is **0.1.0**. Every update (each set of changes pushed for the owner)
raises the last digit by one, and the last digit carries over after 9:

0.1.0 → 0.1.1 → … → 0.1.9 → 0.2.0 → … → 0.9.9 → 1.0.0

The new number goes in the same push as the changes, in every place that holds it:

- `apps/hub/package.json`
- `apps/hub/package-lock.json` (the two entries for `yendory-hub`)
- `apps/hub/src-tauri/tauri.conf.json`
- `apps/hub/src-tauri/Cargo.toml`
- `apps/hub/src-tauri/Cargo.lock` (update with cargo, e.g. `cargo metadata`)

The Engine keeps its own version (`crates/yendory-engine/Cargo.toml`, set on `yendory-studio`).
The Hub release workflow names its builds after `apps/hub/package.json`
(`hub-v<version>-dev.<run>`).

## Switches must work

When the owner asks for something to be switchable (a checkbox, an option, a setting), the
switch must change what actually happens. Never build another path that does the same job
regardless of the switch (for example a download that already removes what a practice-time
checkbox is meant to leave out), so that the switch becomes decorative. If one switch only works
together with another, keep them consistent in the page and say so beside them, or disable it
with the reason. When an existing safeguard or rule makes a new switch inert, raise it with the
owner instead of shipping it.
