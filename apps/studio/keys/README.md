# Updater key

Studio on Windows and Linux (AppImage) can update itself (§ 1 Models → "Check for updates";
`src-tauri/src/update.rs`). It reads `latest.json` from the `studio-updates` release, downloads
the new installer, and installs it **only if its signature matches the public key** in
`src-tauri/tauri.conf.json` (`plugins.updater.pubkey`). Installing over the old version keeps
the app's data folder (models, datasets, chats, settings).

This key's **private half is not in the repository**: it is the
repository secret `TAURI_SIGNING_PRIVATE_KEY` (no password; `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`
may be left unset). Without the secret, CI builds as before and publishes no update manifest, so
Studio simply finds no update. If the private key is ever lost, make a new pair
(`npx tauri signer generate -w yendory-updater.key`), put the new public key in
`tauri.conf.json`, the private key in the secret, and install that build once by hand.
