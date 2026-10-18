"""Score the app's live path at several segment caps -> out/livecaps.json and a Markdown table.

    uv run python -m ea.livecaps NAME [NAME ...]

Reads out/NAME/app_live.json (the preset, 25 s) and out/NAME/app_live_cap<S>.json written by run_all.sh."""

import re
import sys

from .common import OUT, counts, out_file, load_json, save_json, summarize, tokens, video_dir
from .gt import reference


def main():
    names = sys.argv[1:]
    caps = sorted({int(m[1]) for n in names for p in video_dir(n).glob("app_live_cap*.json") if (m := re.search(r"cap(\d+)", p.name))})
    rows = []
    for cap, video in [(c, v) for c in caps + [25] for v in [None] + (names if len(names) > 1 else [])]:
        cs, nseg = [], 0
        for n in [video] if video else names:
            meta, ref = load_json(video_dir(n) / "regions.json"), reference(n)
            live = load_json(video_dir(n) / ("app_live.json" if cap == 25 else f"app_live_cap{cap}.json"))
            for r in meta["regions"]:
                cs.append(counts(tokens(" ".join(ref[c["id"]][0] for c in r["clips"])), tokens(" ".join(f[2] for f in live[r["id"]]))))
                nseg += len(live[r["id"]])
        rows.append(dict(cap_s=cap, video=video or "all", segments=nseg, **{k: v for k, v in summarize(cs).items() if k in ("mer", "S_rate", "D_rate", "I_rate", "err_zh", "err_en")}))
    save_json(out_file("livecaps.json"), rows)
    print("| live segment cap (s) | videos | MER | D | S | segments |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['cap_s']}{' (app today)' if r['cap_s'] == 25 else ''} | {r['video']} | {r['mer']:.3f} | {r['D_rate']:.3f} | {r['S_rate']:.3f} | {r['segments']} |")


if __name__ == "__main__":
    main()
