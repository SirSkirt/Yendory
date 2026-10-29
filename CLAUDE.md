# Notes for development sessions

## Version numbers

The current version is **0.7.9**. Every update (each set of changes pushed for the owner) raises
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
