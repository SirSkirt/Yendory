# Android signing keys

`yendory-dev.jks` is a **public development key** (store and key password `yendory-dev`,
alias `yendory-dev`). It exists so that every APK built by CI is signed with the same
certificate, which lets a new build install over an older one **without uninstalling**
(uninstalling an Android app deletes its data, including your models).

Because this key is public, anyone could sign an APK that Android accepts as an update.
That is acceptable for development builds you install yourself; it is not acceptable for
public distribution. For that, add a private key as repository secrets and the release
workflow uses it instead:

| Secret | Value |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | `base64 -w0 your-key.jks` |
| `ANDROID_KEYSTORE_PASSWORD` | the store password |
| `ANDROID_KEY_ALIAS` | the key alias |
| `ANDROID_KEY_PASSWORD` | the key password |

Switching keys means the next APK cannot update the old one: export your models first.

Development certificate SHA-256:
`32:3C:7C:50:6F:D8:A1:60:12:F4:96:60:AF:71:CA:CF:A9:88:9A:02:4A:DD:1C:EC:84:11:DC:45:C6:93:24:E1`
