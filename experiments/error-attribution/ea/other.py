"""Score another engine's clip texts with the same references and normalisation as the TingMix decoders (section 2 of
docs/research/error-attribution.md), next to TingMix-Large's app decoder on the same clips.

    uv run python -m ea.other LABEL NAME=hyps.json [NAME=hyps.json ...]

Each hyps.json maps clip id -> text, either at the top level or under "texts" (the `sensevoice_dir` example writes that).
The engine's output goes through the same OpenCC s2t step as the references, so Simplified output is not penalised."""

import sys

from .common import counts, load_json, summarize, tokens
from .gt import reference


def main():
    label, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    rows = {label: [], "tingmix-large lm": []}
    per_video = {}
    for name, path in pairs:
        doc = load_json(path)
        texts = doc.get("texts", doc)
        base = load_json(f"out/{name}/hyps_orig.json")
        ref = reference(name)
        vid = {label: [], "tingmix-large lm": []}
        for cid, (text, _status) in ref.items():
            rt = tokens(text)
            for k, h in ((label, texts[cid]), ("tingmix-large lm", base[cid]["lm"])):
                c = counts(rt, tokens(h))
                rows[k].append(c)
                vid[k].append(c)
        per_video[name] = vid
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    print("| engine | clips | MER | CER zh | WER en | S | D | I |")
    print("|---|---|---|---|---|---|---|---|")
    for scope, data in [("both", rows)] + list(per_video.items()):
        for k, rs in data.items():
            s = summarize(rs)
            print(f"| {k} ({scope}) | {len(rs)} | {f(s['mer'])} | {f(s['err_zh'])} | {f(s['err_en'])} | {f(s['S_rate'])} | {f(s['D_rate'])} | {f(s['I_rate'])} |")


if __name__ == "__main__":
    main()
