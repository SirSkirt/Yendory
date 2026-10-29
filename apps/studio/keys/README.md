# Android debug signing key

`yendory-debug.jks` is the **one debug key** for every Android build of Yendory Studio. CI signs
each APK with it, so a new build installs over the previous one **without uninstalling**
(uninstalling would delete the app's models). No repository secrets are needed.

| | |
|---|---|
| Store / key password | `yendory-debug` |
| Alias | `yendory-debug` |
| Certificate SHA-256 | `32:3C:7C:50:6F:D8:A1:60:12:F4:96:60:AF:71:CA:CF:A9:88:9A:02:4A:DD:1C:EC:84:11:DC:45:C6:93:24:E1` |

The key is public on purpose: it identifies "a Yendory debug build", not its author. Anyone can
sign an APK with it, so install only APKs you downloaded from this repository's Actions runs
or Releases. The workflow checks every APK carries this exact certificate.

# Desktop updater key

Studio on Windows and Linux (AppImage) can update itself (§ 1 Models → "Check for updates";
`src-tauri/src/update.rs`). It reads `latest.json` from the `studio-updates` release, downloads
the new installer, and installs it **only if its signature matches the public key** in
`src-tauri/tauri.conf.json` (`plugins.updater.pubkey`). Installing over the old version keeps
the app's data folder (models, datasets, chats, settings).

Unlike the Android key, this key's **private half is not in the repository**: it is the
repository secret `TAURI_SIGNING_PRIVATE_KEY` (no password; `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`
may be left unset). Without the secret, CI builds as before and publishes no update manifest, so
Studio simply finds no update. If the private key is ever lost, make a new pair
(`npx tauri signer generate -w yendory-updater.key`), put the new public key in
`tauri.conf.json`, the private key in the secret, and install that build once by hand.
