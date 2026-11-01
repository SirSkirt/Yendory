# Notes for development sessions

## Branches

This is the **`yendory-studio`** branch: Yendory Studio (`apps/studio`). The Engine, the Yendory
Network and its client live once, on `yendory-hub` (`dependency/`); Studio builds against them as
git dependencies pinned to a commit (`rev` in `apps/studio/src-tauri/Cargo.toml`), and the pin is
moved on purpose. There is no Engine copy here: Engine and Network changes are made in
`dependency/` on `yendory-hub` (docs/INTEGRATION.md).

Rules for working on `yendory-hub` from Studio's side:

1. Touch only `dependency/network`, `dependency/client`, `dependency/engine` and docs/NETWORK.md
   (`apps/hub` and the workflows belong to the Hub's sessions). Small commits; pull/rebase
   before every push.
2. Every push that changes `dependency/` raises the Network's version
   (`dependency/network/Cargo.toml` and the root `Cargo.lock`); CI publishes `network-v<version>`
   when it is new.
3. Network API 1 stays additive; changing an existing method means API 2, and the Hub must know.
4. Release order: push the Network change, let CI publish it, then release the Studio that needs
   it with `requires.network` set to that version (studio-release.yml).
5. Before pushing there: `cargo test --release -p yendory-network -p yendory-client` and the
   Engine tests.

Models are handed between the programs through Studio's models folder (`docs/HANDOFF.md` on the
Hub branch); keep that contract when changing either side.

## Version numbers

The current version is **0.9.5**. Every update (each set of changes pushed for the owner) raises
the last digit by one, and the last digit carries over after 9:

0.4.0 → 0.4.1 → … → 0.4.9 → 0.5.0 → 0.5.1 → … → 0.9.9 → 1.0.0

The new number goes in the same push as the changes, in every place that holds it:

- `apps/studio/package.json`
- `apps/studio/package-lock.json` (the two entries for `yendory-studio`)
- `apps/studio/src-tauri/tauri.conf.json`
- `apps/studio/src-tauri/Cargo.toml`
- `apps/studio/src-tauri/Cargo.lock` (update with cargo, e.g. `cargo metadata`)

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
