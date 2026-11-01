# Notes for development sessions

## Branches and parts

This is the **`yendory-hub`** branch: Yendory Hub (`apps/hub`) and the shared parts in
`dependency/`: the Engine (`dependency/engine`), the Yendory Network service that runs it for
every program (`dependency/network`, docs/NETWORK.md) and the client library programs use to
reach it (`dependency/client`). Yendory Studio lives on the `yendory-studio` branch; it is moving
onto the Network (docs/NETWORK.md, "Studio on the Network"). Keep one Engine: there is no second
copy to change. Models are handed between programs through Studio's models folder
(docs/HANDOFF.md); the Hub installs everything from signed packages (docs/PACKAGES.md).

## Version numbers

The Hub is at **0.1.1** and the Yendory Network at **0.1.0**. Every update (each set of changes
pushed for the owner) raises the Hub's last digit by one; the Network's goes up by one in a push
that changes `dependency/` (the Network or the Engine it carries). The last digit carries over
after 9: 0.1.9 → 0.2.0 → … → 0.9.9 → 1.0.0.

The new number goes in the same push as the changes, in every place that holds it:

- Hub: `apps/hub/package.json`, `apps/hub/package-lock.json` (the two entries for `yendory-hub`),
  `apps/hub/src-tauri/tauri.conf.json`, `apps/hub/src-tauri/Cargo.toml`,
  `apps/hub/src-tauri/Cargo.lock` (update with cargo, e.g. `cargo metadata`).
- Network: `dependency/network/Cargo.toml` and the root `Cargo.lock`.

The Engine keeps its own version (`dependency/engine/Cargo.toml`). The release workflow names
the Hub's builds after `apps/hub/package.json` (`hub-v<version>-dev.<run>`) and publishes the
Network only when its version is new (`network-v<version>`).

## Switches must work

When the owner asks for something to be switchable (a checkbox, an option, a setting), the
switch must change what actually happens. Never build another path that does the same job
regardless of the switch (for example a download that already removes what a practice-time
checkbox is meant to leave out), so that the switch becomes decorative. If one switch only works
together with another, keep them consistent in the page and say so beside them, or disable it
with the reason. When an existing safeguard or rule makes a new switch inert, raise it with the
owner instead of shipping it.
