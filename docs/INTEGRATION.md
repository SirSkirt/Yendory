# Studio, the Engine and the Yendory Network

Yendory Studio lives on the `yendory-studio` branch (`apps/studio`). Yendory Hub lives on
`yendory-hub` (the default branch), together with the parts every program shares, once:

| Path on `yendory-hub` | What |
|---|---|
| `dependency/engine` | The Yendory Engine (Rust library) |
| `dependency/network` | The Yendory Network: the background service that runs the Engine for every program (docs/NETWORK.md) |
| `dependency/client` | How a program finds, starts and talks to the Network |
| `apps/hub` | Yendory Hub |

## Building Studio

Studio takes the Engine (for work that needs no model: tokenizer, lessons, parsing, filters) and
the client from `yendory-hub`, pinned to one commit in `apps/studio/src-tauri/Cargo.toml`:

```toml
yendory-engine = { git = "https://github.com/SirSkirt/Yendory", rev = "<commit>", features = ["gpu"] }
yendory-client = { git = "https://github.com/SirSkirt/Yendory", rev = "<commit>" }
```

To move to a newer Engine or Network, change `rev` (all three places) and run
`cargo update -p yendory-engine -p yendory-client` in `apps/studio/src-tauri`. Engine changes are
made on `yendory-hub`, in `dependency/engine`; there is no copy here.

## At run time

Yendory Hub installs the Network (a package, docs/PACKAGES.md). Studio connects to it when it
starts (`Client::start_and_connect`, which starts the copy the Hub installed), and shows how in
Global Settings ▸ Yendory Network. Without it, Studio does everything itself.

What goes through the Network so far (step 1 of docs/NETWORK.md, "Studio on the Network"):

- the model Studio has open is opened there too (no other program renames or deletes it);
- chat replies (`model.chat`, streamed; Stop reaches it with `model.stop`), except replies that
  record a brain trace, which Studio writes itself;
- after Studio saves a model (it still trains on its own), the Network reads it again
  (`models.reload`).

## Builds and installing

Every push to `yendory-studio` that changes Studio builds it (`studio-v<version>-dev.<run>`): its
Hub packages, signed zips of the program itself (`yendory-studio_<version>_windows-x86_64.zip`
with `yendory-studio.exe`, `yendory-studio_<version>_linux-x86_64.zip` with
`Yendory-Studio.AppImage`), and `package.json` with `icon.png` on the fixed `studio-updates`
release (docs/PACKAGES.md on the Hub branch; `.github/scripts/package.py` is the Hub branch's,
unchanged). `package.json` names the Network version Studio needs (`requires.network`), so the
Hub installs that Network first: a Network change is pushed and published before the Studio that
needs it. The installers (`-setup.exe`, `.deb`, `.AppImage`) and `latest.json` are still
published for anyone without the Hub. Studio does not update itself.

## Handing a model to Studio

`yendory-studio --model <id>` opens that model (Studio 0.9.3 and later); see `docs/HANDOFF.md` on
the Hub branch. Started again while it runs, Studio hands the id to the running copy instead of
starting a second one.
