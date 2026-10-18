"""Score every decoder and chunking against the references -> out/results.json and a Markdown summary on stdout.

    uv run python -m ea.score NAME [NAME ...]

A. Decoders on reference-boundary clips (same audio, same slices): greedy, lm, v3, and N-best oracles (the hypothesis
   closest to the reference among the N best; a diagnostic bound, never an inference result).
B. Chunkings per region (lm decoder unless noted): reference-boundary clips joined, fixed 25 s, 25 s windows overlapping by
   5 s, and the app's live path (adaptive VAD, MLX).
C. Reference sources: uploader subtitles and Whisper drafts scored against the human-verified text, where it exists."""

import sys

from .common import OUT, counts, out_file, edits, lang_class, load_json, save_json, summarize, tokens, video_dir
from .gt import reference

NS = (8, 16, 32)


def oracle(ref_t, hyps):
    return min(hyps, key=lambda h: edits(ref_t, tokens(h))) if hyps else ""


def clip_rows(name, variant="orig"):
    d = video_dir(name)
    meta = load_json(d / "regions.json")
    hyps = load_json(d / f"hyps_{variant}.json")
    ref = reference(name)
    rows = []
    for r in meta["regions"]:
        for c in r["clips"]:
            text, status = ref[c["id"]]
            rt = tokens(text)
            h = hyps[c["id"]]
            out = {"greedy": h["greedy"], "lm": h["lm"], "v3": h["v3"]}
            for n in NS:
                out[f"oracle_lm{n}"] = oracle(rt, h[f"nbest_lm{n}"])
                out[f"oracle_ac{n}"] = oracle(rt, h[f"nbest_ac{n}"])
            rows.append(dict(video=name, region=r["id"], clip=c["id"], start=c["start"], end=c["end"], ref=text, status=status,
                             lang=lang_class(rt) if rt else "empty", ref_tokens=len(rt), hyp=out,
                             counts={k: counts(rt, tokens(v)) for k, v in out.items()}))
    return rows


def region_rows(name):
    d = video_dir(name)
    meta = load_json(d / "regions.json")
    hyps = load_json(d / "hyps_orig.json")
    live = load_json(d / "app_live.json")
    ref = reference(name)
    rows = []
    for r in meta["regions"]:
        rt = tokens(" ".join(ref[c["id"]][0] for c in r["clips"]))
        out = {
            "refclips_lm": " ".join(hyps[c["id"]]["lm"] for c in r["clips"]),
            "refclips_greedy": " ".join(hyps[c["id"]]["greedy"] for c in r["clips"]),
            "fixed25_lm": hyps[f"{r['id']}_fixed25"]["lm"],
            "fixed25_greedy": hyps[f"{r['id']}_fixed25"]["greedy"],
            "overlap_lm": hyps[f"{r['id']}_overlap"]["lm"],
            "overlap_greedy": hyps[f"{r['id']}_overlap"]["greedy"],
            "app_live_lm": " ".join(f[2] for f in live[r["id"]]),
        }
        rows.append(dict(video=name, region=r["id"], counts={k: counts(rt, tokens(v)) for k, v in out.items()}))
    return rows


def refsrc_rows(name):
    d = video_dir(name)
    meta = load_json(d / "regions.json")
    gt = load_json(d / "gt.json") if (d / "gt.json").exists() else {}
    rows = []
    for r in meta["regions"]:
        for c in r["clips"]:
            if c["id"] in gt:
                rt = tokens(gt[c["id"]])
                src = {"whisper": c["whisper_text"]}
                if meta["cue_source"] == "subtitles":
                    src["subtitles"] = c["cue_text"]
                rows.append({k: counts(rt, tokens(v)) for k, v in src.items()})
    return rows


def table(rows, keys, label):
    lines = [f"| {label} | MER | CER zh | WER en | S | D | I | ref units |", "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        s = summarize([r[k] for r in rows])
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {k} | {f(s['mer'])} | {f(s['err_zh'])} | {f(s['err_en'])} | {f(s['S_rate'])} | {f(s['D_rate'])} | {f(s['I_rate'])} | {s['N_zh'] + s['N_en']} |")
    return "\n".join(lines)


def main():
    names = sys.argv[1:]
    clips = [r for n in names for r in clip_rows(n)]
    regions = [r for n in names for r in region_rows(n)]
    refsrc = [r for n in names for r in refsrc_rows(n)]
    keys = ["greedy", "lm", "v3"] + [f"oracle_lm{n}" for n in NS] + [f"oracle_ac{n}" for n in NS]
    status = {s: sum(r["status"] == s for r in clips) for s in ("verified", "provisional", "subtitles")}
    md = [f"References: {status['verified']} verified clips, {status['provisional']} provisional (Whisper draft), {status['subtitles']} uploader subtitles.", ""]
    for n in names + ["all"]:
        sel = [r for r in clips if n == "all" or r["video"] == n]
        md += [f"### A. Decoders, {n} ({len(sel)} clips)", "", table([r["counts"] for r in sel], keys, "decoder"), ""]
    for lang in ("zh", "mixed", "en"):
        sel = [r for r in clips if r["lang"] == lang]
        if sel:
            md += [f"### A. By clip language: {lang} ({len(sel)} clips)", "", table([r["counts"] for r in sel], ["greedy", "lm", "v3", "oracle_lm32", "oracle_ac32"], "decoder"), ""]
    for n in names + ["all"]:
        sel = [r for r in regions if n == "all" or r["video"] == n]
        md += [f"### B. Chunking, {n} ({len(sel)} regions)", "", table([r["counts"] for r in sel], list(regions[0]["counts"]), "chunking"), ""]
    if refsrc:
        md += [f"### C. Reference sources vs verified ({len(refsrc)} clips)", "", table(refsrc, sorted({k for r in refsrc for k in r}), "source"), ""]
    save_json(out_file("results.json"), dict(status=status, clips=clips, regions=regions, refsrc=refsrc))
    print("\n".join(md))


if __name__ == "__main__":
    main()
