"""Error spans, their evidence-based causes, and per-clip correlates -> out/attribution.json and out/worst20.md.

    uv run python -m ea.attribute NAME [NAME ...]

An error span is a maximal run of non-matching alignment ops of the app's decoder (`lm`) against the reference. For each
span the same reference positions are looked up in the other decodes of the same clip, which turns causes into tests:

- `decoder`: the span is right in some LM-beam 32-best hypothesis (search / LM scoring could have found it);
- `lm`: the span is right in greedy CTC but wrong with the LM (the LM made it worse);
- `background`: right after vocal separation (`vocals` variant, same decoder);
- `rate`: right after slowing the audio to 0.8x (`slow` variant, same decoder);
- `unexplained`: none of the above recovers it. This is a residual, not a test: it holds acoustic-model errors and also
  errors of the reference itself.

Several tests can pass for one span; the first in the order above that passes is its primary cause. A null perturbation
(`shift`: the clip delayed by 10 ms, which changes nothing a listener would hear) measures how often a span flips to
correct by chance; `background` and `rate` only mean something above that rate. Descriptive tags (`function_words`,
`latin`, `deletion_only`) record what the span contains; they are features, not causes. N-best oracles are a diagnostic
only."""

import collections
import sys

from .common import OUT, align, out_file, is_latin, load_json, save_json, tokens, video_dir
from .gt import reference

# high-frequency Mandarin function words and discourse particles (single characters)
FUNCTION = set("那的個你這一是了啊呢喔吧嗎就也在有我他它她們要會說然後其實嗯欸哦啦耶齁蛤")


def spans(ref, hyp):
    """-> list of (ref_lo, ref_hi, ops) for maximal error runs; ref indices are half-open."""
    out, i, cur, lo = [], 0, [], 0
    for op, r, h in align(ref, hyp):
        if op == "=":
            if cur:
                out.append((lo, i, cur))
                cur = []
        else:
            if not cur:
                lo = i
            cur.append((op, r, h))
        if r is not None:
            i += 1
    if cur:
        out.append((lo, i, cur))
    return out


def correct_at(ref, hyp, lo, hi):
    """True if hyp matches ref[lo:hi] exactly. Insertions inside the range fail it; insertions at its edges belong to the
    neighbouring context, except for a pure-insertion span (lo == hi), which is correct only if nothing is inserted there."""
    i = 0  # reference index the next op is at
    for op, r, h in align(ref, hyp):
        if (op in "SD" and lo <= i < hi) or (op == "I" and (lo < i < hi or (lo == hi and i == lo))):
            return False
        if r is not None:
            i += 1
    return True


def hyp_text(ref, hyp, lo, hi):
    """The hypothesis tokens aligned to ref[lo:hi] (including insertions inside), for display."""
    i, out = 0, []
    for op, r, h in align(ref, hyp):
        if (lo <= i < hi or (op == "I" and lo <= i <= hi)) and h is not None:
            out.append(h)
        if r is not None:
            i += 1
    return "".join(t if not is_latin(t) else f" {t} " for t in out).strip()


def show(ts):
    return "".join(t if not is_latin(t) else f" {t} " for t in ts).replace("  ", " ").strip()


def _r(x, y):
    import numpy as np

    return round(float(np.corrcoef(x, y)[0, 1]), 2)


def main():
    names = sys.argv[1:]
    unigram = collections.Counter()
    allspans, clips = [], []
    for name in names:
        d = video_dir(name)
        meta = load_json(d / "regions.json")
        ref = reference(name)
        H = {v: load_json(d / f"hyps_{v}.json") for v in ("orig", "vocals", "slow", "shift") if (d / f"hyps_{v}.json").exists()}
        feats = load_json(d / "clip_features.json") if (d / "clip_features.json").exists() else {}
        for r in meta["regions"]:
            for c in r["clips"]:
                rt = tokens(ref[c["id"]][0])
                unigram.update(rt)
                h = H["orig"][c["id"]]
                lmt = tokens(h["lm"])
                dur = c["end"] - c["start"]
                n_err = sum(len(s[2]) for s in spans(rt, lmt))
                clips.append(dict(video=name, clip=c["id"], seconds=dur, units_per_s=len(rt) / dur, errors=n_err, ref_units=len(rt),
                                  vocal_to_rest_db=feats.get(c["id"], {}).get("vocal_to_rest_db"),
                                  err_vocals=sum(len(s[2]) for s in spans(rt, tokens(H["vocals"][c["id"]]["lm"]))) if "vocals" in H else None,
                                  err_slow=sum(len(s[2]) for s in spans(rt, tokens(H["slow"][c["id"]]["lm"]))) if "slow" in H else None))
                for lo, hi, ops in spans(rt, lmt):
                    tests = {
                        "decoder": any(correct_at(rt, tokens(x), lo, hi) for x in h["nbest_lm32"]),
                        "lm": correct_at(rt, tokens(h["greedy"]), lo, hi),
                        "background": "vocals" in H and correct_at(rt, tokens(H["vocals"][c["id"]]["lm"]), lo, hi),
                        "rate": "slow" in H and correct_at(rt, tokens(H["slow"][c["id"]]["lm"]), lo, hi),
                    }
                    shift = "shift" in H and correct_at(rt, tokens(H["shift"][c["id"]]["lm"]), lo, hi)
                    cause = next((k for k, v in tests.items() if v), "unexplained")
                    gt = rt[lo:hi]
                    kinds = collections.Counter(op for op, _, _ in ops)
                    pos = c["start"] + dur * (lo / max(len(rt), 1))
                    allspans.append(dict(
                        video=name, clip=c["id"], approx_time_s=round(pos, 1), size=len(ops), ops=dict(kinds), cause=cause, tests={**tests, "shift": shift},
                        ref=show(gt), ref_context=show(rt[max(0, lo - 6):hi + 6]), lm=hyp_text(rt, lmt, lo, hi),
                        greedy=hyp_text(rt, tokens(h["greedy"]), lo, hi),
                        top5=[hyp_text(rt, tokens(x), lo, hi) for x in h["nbest_lm32"][:5]],
                        status=ref[c["id"]][1],
                        tags=[t for t, ok in (("function_words", gt and all(g in FUNCTION for g in gt)), ("latin", any(is_latin(g) for g in gt)),
                                              ("deletion_only", set(kinds) == {"D"})) if ok],
                    ))
    for s in allspans:
        if s["ref"] and not any(is_latin(t) for t in tokens(s["ref"])):
            s["min_ref_char_count"] = min(unigram[t] for t in tokens(s["ref"]))
    total = sum(s["size"] for s in allspans)
    by_cause = collections.Counter()
    recovered = collections.Counter()  # error units each test recovers on its own (not exclusive)
    for s in allspans:
        by_cause[s["cause"]] += s["size"]
        for k, v in s["tests"].items():
            recovered[k] += s["size"] * bool(v)
    tagged = {t: sum(s["size"] for s in allspans if t in s["tags"]) for t in ("function_words", "latin", "deletion_only")}
    corr = {}
    for v in names:
        cv = [c for c in clips if c["video"] == v and c["vocal_to_rest_db"] is not None]
        if len(cv) > 2:
            er = [c["errors"] / max(c["ref_units"], 1) for c in cv]
            corr[v] = dict(vocal_db=_r([c["vocal_to_rest_db"] for c in cv], er), units_per_s=_r([c["units_per_s"] for c in cv], er),
                           vocal_db_range=[round(min(c["vocal_to_rest_db"] for c in cv), 1), round(max(c["vocal_to_rest_db"] for c in cv), 1)])
    save_json(out_file("attribution.json"), dict(total_error_units=total, by_cause=dict(by_cause), recovered_by_test=dict(recovered), by_tag=tagged, correlations=corr, spans=allspans, clips=clips))
    worst = sorted(allspans, key=lambda s: -s["size"])[:20]
    md = ["| # | video / clip | approx. time | size | ops | cause | reference | app (lm) | greedy | 32-best top 3 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(worst, 1):
        md.append(f"| {i} | {s['video']} {s['clip']} | {s['approx_time_s']} s | {s['size']} | {s['ops']} | {s['cause']} | {s['ref_context']} → **{s['ref']}** | {s['lm']} | {s['greedy']} | {' / '.join(s['top5'][:3])} |")
    out_file("worst20.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"error units {total}; by primary cause:", {k: f"{v} ({v / total:.0%})" for k, v in by_cause.most_common()})
    for v in names:
        sv = [s for s in allspans if s["video"] == v]
        tv = sum(s["size"] for s in sv)
        cv = collections.Counter()
        for s in sv:
            cv[s["cause"]] += s["size"]
        print(f"  {v}: {tv} error units;", {k: f"{n} ({n / tv:.0%})" for k, n in cv.most_common()})
    print("recovered by each test (not exclusive):", {k: f"{v} ({v / total:.0%})" for k, v in recovered.most_common()})
    print("clip error rate correlations within each video:", corr)
    print("by tag:", {k: f"{v} ({v / total:.0%})" for k, v in tagged.items()})


if __name__ == "__main__":
    main()
