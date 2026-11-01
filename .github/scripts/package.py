#!/usr/bin/env python3
"""Yendory packages (docs/PACKAGES.md): makes a component's zip and its package.json.

  package.py zip OUT.zip FILE[=NAME_IN_ZIP] ...
      A zip of the files (executable bits kept), each at the top of the zip.

  package.py manifest --id ID --name NAME --version V [--notes TEXT] [--requires ID=VERSION ...]
                      --platform KEY=ZIP,RUN,URL ... > package.json
      ZIP is the local file (its signature is ZIP.sig, from `tauri signer sign`), RUN the file to
      start inside it, URL where the release page serves it. KEY is windows-x86_64,
      linux-x86_64, or for the Hub itself also linux-x86_64-appimage and linux-x86_64-deb.
"""
import argparse
import datetime
import json
import os
import stat
import sys
import zipfile


def make_zip(out, files):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for spec in files:
            path, _, name = spec.partition("=")
            name = name or os.path.basename(path)
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            mode = os.stat(path).st_mode
            # Unix permissions in the upper bits (the Hub restores them on Linux).
            info.external_attr = (stat.S_IFREG | (0o755 if mode & 0o111 else 0o644)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(path, "rb") as f:
                z.writestr(info, f.read())


def manifest(args):
    platforms = {}
    for spec in args.platform:
        key, _, rest = spec.partition("=")
        local, run, url = rest.split(",", 2)
        with open(local + ".sig") as f:
            signature = f.read().strip()
        platforms[key] = {
            "url": url,
            "signature": signature,
            "size": os.path.getsize(local),
            "kind": "deb" if local.endswith(".deb") else "zip",
            "run": run,
        }
    requires = dict(r.split("=", 1) for r in args.requires)
    return {
        "format": 1,
        "id": args.id,
        "name": args.name,
        "version": args.version,
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": args.notes,
        "requires": requires,
        "platforms": platforms,
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "zip":
        if len(sys.argv) < 4:
            sys.exit(__doc__)
        make_zip(sys.argv[2], sys.argv[3:])
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--id", required=True)
    m.add_argument("--name", required=True)
    m.add_argument("--version", required=True)
    m.add_argument("--notes", default="")
    m.add_argument("--requires", action="append", default=[])
    m.add_argument("--platform", action="append", default=[], required=True)
    args = p.parse_args()
    json.dump(manifest(args), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
