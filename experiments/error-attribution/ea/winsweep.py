"""Window-length sweep: is the error rate a function of how much audio the model sees at once?

    uv run python -m ea.winsweep NAME [NAME ...] [--windows 4,6,8,10,12,15,20,25] [--overlap 2]

Each region is decoded in windows of W seconds that overlap by `overlap` s (frames taken from the window where they are
most central, as in ea.decode), then greedy and LM-decoded and scored against the region reference. Same audio and
decoders for every W; only the window length changes. Writes out/winsweep.json."""

import argparse

from embed2asr.lm import beam_decode

from .common import ALPHA, BEAM, BETA, OUT, counts, load_json, read_wav, save_json, summarize, tokens, video_dir
from .decode import load, stitched
from .decoders import _G, _init
from .gt import reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--windows", default="4,6,8,10,12,15,20,25")
    p.add_argument("--overlap", type=float, default=2.0)
    a = p.parse_args()
    asr = load("cpu")
    _init()
    tok, lm, units = _G["tok"], _G["lm"], _G["units"]
    rows = []
    for name in a.names:
        meta = load_json(video_dir(name) / "regions.json")
        ref = reference(name)
        for r in meta["regions"]:
            rt = tokens(" ".join(ref[c["id"]][0] for c in r["clips"]))
            x = read_wav(video_dir(name) / "regions" / f"{r['id']}.wav")
            for w in (float(v) for v in a.windows.split(",")):
                lp = stitched(asr, x, w, w - a.overlap)
                ids = lp.argmax(-1)
                g = tok.decode([int(i) for k, i in enumerate(ids) if i != 0 and (k == 0 or i != ids[k - 1])])
                l = beam_decode(lp, tok, lm, ALPHA, BETA, BEAM, units=units)
                rows.append(dict(video=name, region=r["id"], window_s=w, greedy=counts(rt, tokens(g)), lm=counts(rt, tokens(l))))
            print(name, r["id"], flush=True)
    save_json(OUT / "winsweep.json", dict(overlap_s=a.overlap, rows=rows))
    print("| window s | greedy MER | greedy D | lm MER | lm D |\n|---|---|---|---|---|")
    for w in sorted({r["window_s"] for r in rows}):
        g = summarize([r["greedy"] for r in rows if r["window_s"] == w])
        l = summarize([r["lm"] for r in rows if r["window_s"] == w])
        print(f"| {w:g} | {g['mer']:.3f} | {g['D_rate']:.3f} | {l['mer']:.3f} | {l['D_rate']:.3f} |")


if __name__ == "__main__":
    main()
