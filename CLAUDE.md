# Notes for development sessions

## Version numbers

The current version is **0.8.7**. Every update (each set of changes pushed for the owner) raises
the last digit by one, and the last digit carries over after 9:

0.4.0 → 0.4.1 → … → 0.4.9 → 0.5.0 → 0.5.1 → … → 0.9.9 → 1.0.0

The new number goes in the same push as the changes, in every place that holds it:

- `apps/studio/package.json`
- `apps/studio/package-lock.json` (the two entries for `yendory-studio`)
- `apps/studio/src-tauri/tauri.conf.json`
- `apps/studio/src-tauri/Cargo.toml`
- `crates/yendory-engine/Cargo.toml`
- `Cargo.lock` and `apps/studio/src-tauri/Cargo.lock` (update with cargo, e.g. `cargo metadata`)

The Studio release workflow names its builds after `apps/studio/package.json`
(`studio-v<version>-dev.<run>`).

## Switches must work

When the owner asks for something to be switchable (a checkbox, an option, a setting), the
switch must change what actually happens. Never build another path that does the same job
regardless of the switch (for example a download that already removes what a practice-time
checkbox is meant to leave out), so that the switch becomes decorative. If one switch only works
together with another, keep them consistent in the page and say so beside them, or disable it
with the reason. When an existing safeguard or rule makes a new switch inert, raise it with the
owner instead of shipping it.
