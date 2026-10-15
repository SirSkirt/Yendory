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
