"""Read safetensors headers over HTTP range requests (no weight download) and write raw JSON to out/.

  uv run python st_headers.py   # pinned revisions from docs/research/PINNED.md
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

EG2 = ("google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b", ["model.safetensors"])
LARGE = ("tc3oliver/TingMix-Large", "75071a1260fb21874f150df6ebd53716ac423f13", ["model.safetensors", "encoder_delta.safetensors"])
OUT = Path(__file__).parent / "out"


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def header(repo, rev, name):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    h = json.loads(rng(url, 8, 8 + n - 1))
    return {"url": url, "header_bytes": n, "data_start": 8 + n, "header": h}


def main():
    OUT.mkdir(exist_ok=True)
    for repo, rev, names in (EG2, LARGE):
        for name in names:
            d = header(repo, rev, name)
            p = OUT / f"header_{repo.split('/')[1]}_{name}.json"
            p.write_text(json.dumps(d, indent=1))
            ts = {k: v for k, v in d["header"].items() if k != "__metadata__"}
            tot = sum(eval("*".join(map(str, v["shape"])) or "1") for v in ts.values())
            print(repo, name, len(ts), "tensors", tot, "elements", d["header"].get("__metadata__"), file=sys.stderr)


main()
