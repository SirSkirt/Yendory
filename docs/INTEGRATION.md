# Bringing Studio into the Hub branch

Yendory Studio lives on the `yendory-studio` branch; Yendory Hub on `yendory-hub` (the default
branch). Both branches use the same layout, so Studio drops into the Hub branch unchanged:

| Path | What | Owned by |
|---|---|---|
| `apps/studio` | Yendory Studio (Tauri 2 app: `src-tauri/`, `web/`) | `yendory-studio` |
| `crates/yendory-engine` | The Engine, shared by Studio and the Hub | `yendory-studio` (changed here, brought to the Hub) |
| `apps/hub` | Yendory Hub | `yendory-hub` |

Studio finds the Engine at `../../../crates/yendory-engine` (from `apps/studio/src-tauri`), which
is where the Hub branch keeps it too.

## Taking Studio (and the Engine) into the Hub branch

```
git fetch origin yendory-studio
git checkout yendory-hub
git checkout origin/yendory-studio -- apps/studio crates/yendory-engine
git commit -m "Studio <version> and Engine from yendory-studio"
```

The same two paths, taken again, bring later versions. Nothing else on `yendory-studio` is needed
by the Hub: its `docs/`, `tests/` and workflows describe and check Studio and the Engine.

## Builds and installing

Every push to `yendory-studio` that changes Studio or the Engine builds Studio
(`studio-v<version>-dev.<run>`): its Hub packages, signed zips of the program itself
(`yendory-studio_<version>_windows-x86_64.zip` with `yendory-studio.exe`,
`yendory-studio_<version>_linux-x86_64.zip` with `Yendory-Studio.AppImage`), and `package.json`
with `icon.png` on the fixed `studio-updates` release (docs/PACKAGES.md on the Hub branch;
`.github/scripts/package.py` is the Hub branch's, unchanged). The Hub installs and updates Studio
from there, checking each package against the project's updater key. The installers
(`-setup.exe`, `.deb`, `.AppImage`) and `latest.json` are still published for anyone without the
Hub. Studio does not update itself (its updater code is kept, not shown).

## Handing a model to Studio

`yendory-studio --model <id>` opens that model (Studio 0.9.3 and later); see `docs/HANDOFF.md` on
the Hub branch. Started again while it runs, Studio hands the id to the running copy instead of
starting a second one.
