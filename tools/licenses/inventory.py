#!/usr/bin/env python3
"""Rust dependency licence inventory for the shipped targets (normal + build deps, no dev deps).

  python3 tools/licenses/inventory.py            # writes docs/release/dependency-licenses.md
  python3 tools/licenses/inventory.py --check    # exit 1 if the file is stale

Stdlib only; uses `cargo metadata` per target triple and `pnpm licenses list` for the npm packages (needs pnpm and `pnpm install`).
"""
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/release/dependency-licenses.md"
TARGETS = ["aarch64-apple-darwin", "x86_64-pc-windows-msvc"]
PLATFORM_PKG = re.compile(r"-(darwin|linux|win32|freebsd|android|openharmony)-")  # optional per-OS binaries: same licence as their parent
COPYLEFT = re.compile(r"\b(MPL|LGPL|GPL|AGPL|EPL|CDDL|CPL|OSL|SSPL)", re.I)


def shipped(target):
    md = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--filter-platform", target], cwd=ROOT))
    pk = {p["id"]: p for p in md["packages"]}
    nodes = {n["id"]: n for n in md["resolve"]["nodes"]}
    seen, todo = set(), list(md["workspace_members"])
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]["deps"]:
            if any(k["kind"] in (None, "build") for k in d["dep_kinds"]):
                todo.append(d["pkg"])
    members = set(md["workspace_members"])
    return {(pk[i]["name"], pk[i]["version"]): pk[i] for i in seen - members}


def npm():
    """npm packages: runtime (`--prod`) listed in full, build-time (dev) packages by licence; copyleft-ish ones flagged."""
    def get(*a):
        try:
            out = subprocess.check_output(["pnpm", "licenses", "list", "--json", *a], cwd=ROOT, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            sys.exit("pnpm licenses list failed: install pnpm and run `pnpm install --frozen-lockfile` first")
        return {lic: sorted({p["name"] + " " + v for p in pk for v in p["versions"] if not PLATFORM_PKG.search(p["name"])}) for lic, pk in json.loads(out).items()}

    prod, allp = get("--prod"), get()
    prod_names = {n for v in prod.values() for n in v}
    dev = {lic: [n for n in v if n not in prod_names] for lic, v in allp.items()}
    dev = {lic: v for lic, v in dev.items() if v}
    lines = ["## npm packages (generated from `pnpm licenses list`)", "", "Per-OS binary packages (`*-darwin-arm64` and similar) carry the licence of their parent package and are omitted so the list does not depend on the machine.", "", "Runtime (bundled into the front end):", ""]
    lines += [f"- {l}: " + ", ".join(v) for l, v in sorted(prod.items())]
    lines += ["", "Build-time only (not shipped in the app), by licence:", ""]
    lines += [f"- {l} ({len(v)}): " + ", ".join(v) for l, v in sorted(dev.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    flagged = {l: v for l, v in dev.items() if COPYLEFT.search(l)}
    if flagged:
        lines += ["", "Build-time packages under file-level copyleft (`lightningcss` is Vite's CSS minifier; its output is not a derivative): " + ", ".join(f"{l}: {', '.join(v)}" for l, v in flagged.items())]
    return "\n".join(lines) + "\n\n"


NATIVE = """## Bundled native binaries (maintained by hand in `tools/licenses/inventory.py`)

Not Cargo crates, so `cargo metadata` does not see them. Update this table when a pin changes.

| Component | Version / pin | Licence | Ships in | Notices to ship | Source of the claim |
|---|---|---|---|---|---|
| Microsoft ONNX Runtime (`libonnxruntime.1.31.0.dylib`; Windows `onnxruntime.dll`, `onnxruntime_providers_shared.dll`) | 1.31.0, archives pinned by sha256 in `tools/ort-fetch/MANIFEST.tsv` (see `docs/release/onnxruntime.md`) | MIT (Copyright Microsoft Corporation) plus third-party components listed in `ThirdPartyNotices.txt` (338,088 bytes in the macOS archive) | release builds, loaded at run time from the app bundle | `onnxruntime/LICENSE` and `onnxruntime/ThirdPartyNotices.txt` from the archive, unmodified | `LICENSE` file in the fetched archive (read 2026-10-18); GitHub licence of `microsoft/onnxruntime`: MIT. The third-party list was not reviewed entry by entry |
| ONNX Runtime used by dev/CI builds (pyke build downloaded by `ort-sys`) | follows `ort =2.0.0-rc.14` | not shipped | development and CI only | none | `docs/release/onnxruntime.md` |
| MLX (`libmlx.dylib` 21 MB, `libjaccl.dylib` 2 MB, `mlx.metallib` 216 MB) | PyPI wheel `mlx==0.32.4` today; a pinned source build is the intended release path (`docs/research/mlx-cpp-ffi.md`) | MIT (wheel metadata `License: MIT`, `licenses/LICENSE`); GitHub licence of `ml-explore/mlx` and `mlx-c`: MIT | macOS arm64 builds that enable the MLX backend | MLX `LICENSE` | wheel `METADATA` (read 2026-10-18); the wheel's bundled third-party code was not audited. The libraries link only Apple system frameworks (Metal, Foundation, QuartzCore, Accelerate) and libc++ per `otool -L` |

Model weights and the language model are separate artifacts with their own terms: `docs/research/licensing-review.md`.
"""


def main():
    per = {t: shipped(t) for t in TARGETS}
    allp = {}
    for t, d in per.items():
        for k, p in d.items():
            allp.setdefault(k, (p, set()))[1].add(t.split("-")[2])
    by_lic = defaultdict(list)
    for (n, v), (p, ts) in sorted(allp.items()):
        by_lic[p["license"] or ("file:" + str(p["license_file"]) if p["license_file"] else "MISSING")].append(f"{n} {v} ({'/'.join(sorted(ts))})")
    flagged = {l: c for l, c in by_lic.items() if COPYLEFT.search(l) or l.startswith(("MISSING", "file:"))}
    lines = [
        "# Rust dependency licences (generated)",
        "",
        "Generated by `tools/licenses/inventory.py` from `cargo metadata` (normal and build dependencies of every workspace member, no dev",
        f"dependencies) for {', '.join(TARGETS)}. Do not edit by hand. {len(allp)} third-party crates. Licence strings are the crates' declared",
        "SPDX expressions; this is not a legal review.",
        "Bundled native binaries and npm packages are covered in the sections below.",
        "",
        "Compatibility: every declared licence below is permissive (MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0, Unlicense, CC0, CDLA-Permissive-2.0, 0BSD)",
        "or file-level copyleft (MPL-2.0). None is GPL/LGPL/AGPL, so the combined work can be distributed under Apache-2.0 as long as the notices",
        "are kept (see `NOTICE`) and any modified MPL-2.0 file is published. `cargo deny check licenses` enforces the allow list in `deny.toml`.",
        "",
        "## Needs attention",
        "",
    ]
    if not flagged:
        lines.append("None.")
    for l, c in sorted(flagged.items()):
        lines += [f"- `{l}`: " + ", ".join(c)]
    lines += ["", "Obligations: MPL-2.0 is file-level copyleft. Linking is fine; if we modify an MPL-licensed file we must publish that file's source. Ship the licence text and notices. See `docs/release/` notes on the compliance screen.", "", "## All licences", ""]
    for l in sorted(by_lic, key=lambda x: (-len(by_lic[x]), x)):
        lines += [f"### {l} ({len(by_lic[l])})", "", ", ".join(by_lic[l]), ""]
    text = "\n".join(lines) + "\n" + npm() + NATIVE
    if "--check" in sys.argv:
        sys.exit(0 if OUT.exists() and OUT.read_text() == text else 1)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(allp)} crates, {len(by_lic)} licence expressions, {len(flagged)} flagged")


main()
